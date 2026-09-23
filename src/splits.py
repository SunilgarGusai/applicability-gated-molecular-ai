"""Group-aware molecular split utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import train_test_split


@dataclass
class SplitIndices:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def random_split(y: np.ndarray, task: str, seed: int, ratios=(0.7, 0.15, 0.15)) -> SplitIndices:
    idx = np.arange(len(y))
    strat = y if task == "classification" else None
    train, rem = train_test_split(idx, train_size=ratios[0], random_state=seed, stratify=strat)
    rem_y = y[rem]
    valid_fraction = ratios[1] / (ratios[1] + ratios[2])
    strat2 = rem_y if task == "classification" else None
    valid, test = train_test_split(rem, train_size=valid_fraction, random_state=seed + 1337, stratify=strat2)
    return SplitIndices(np.sort(train), np.sort(valid), np.sort(test))


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True) or smiles


def _assign_groups(groups: list[list[int]], n: int, seed: int, ratios=(0.7, 0.15, 0.15)) -> SplitIndices:
    rng = np.random.default_rng(seed)
    decorated = [(len(g), rng.random(), g) for g in groups]
    decorated.sort(key=lambda x: (-x[0], x[1]))
    targets = np.asarray(ratios) * n
    bins = [[], [], []]
    sizes = np.zeros(3, dtype=int)
    for _, _, group in decorated:
        deficits = targets - sizes
        feasible = np.where(deficits > 0)[0]
        if len(feasible):
            dest = int(feasible[np.argmax(deficits[feasible] / np.maximum(targets[feasible], 1))])
        else:
            dest = int(np.argmin(sizes / np.maximum(targets, 1)))
        bins[dest].extend(group)
        sizes[dest] += len(group)
    return SplitIndices(*(np.sort(np.asarray(b, dtype=int)) for b in bins))


def scaffold_split(smiles: np.ndarray, seed: int, ratios=(0.7, 0.15, 0.15)) -> SplitIndices:
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        buckets.setdefault(scaffold_key(str(s)), []).append(i)
    return _assign_groups(list(buckets.values()), len(smiles), seed, ratios)


def butina_groups(smiles: np.ndarray, cutoff: float = 0.6) -> list[list[int]]:
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
    fps = [gen.GetFingerprint(m) for m in mols]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True, reordering=True)
    return [list(map(int, c)) for c in clusters]


def butina_split(smiles: np.ndarray, seed: int, cutoff: float = 0.6, ratios=(0.7, 0.15, 0.15)) -> SplitIndices:
    return _assign_groups(butina_groups(smiles, cutoff=cutoff), len(smiles), seed, ratios)
