"""Data acquisition-independent standardization and feature generation.

All transformations are deterministic and labels are never used to construct features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

ALLOWED_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotB", "RingCount",
    "AromaticRings", "FractionCSP3", "HeavyAtomCount", "FormalCharge",
    "HeteroAtoms", "NHOHCount", "NOCount", "LabuteASA", "BalabanJ",
    "BertzCT", "MolMR", "AromaticProportion", "MetalFlag",
]

def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    return chooser.choose(mol)

def standardize_smiles(smiles: str) -> tuple[str | None, Chem.Mol | None, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "missing_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, None, "parse_failed"
        mol = rdMolStandardize.Cleanup(mol)
        mol = _largest_fragment(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return can, mol, None
    except Exception as exc:
        return None, None, f"standardization_failed:{type(exc).__name__}"

def descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    heavy = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    metal_flag = float(any(a.GetAtomicNum() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()))
    values = [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol), mol.GetNumHeavyAtoms(), Chem.GetFormalCharge(mol),
        Lipinski.NumHeteroatoms(mol), Lipinski.NHOHCount(mol), Lipinski.NOCount(mol),
        rdMolDescriptors.CalcLabuteASA(mol), Descriptors.BalabanJ(mol), Descriptors.BertzCT(mol),
        Crippen.MolMR(mol), aromatic_atoms / heavy, metal_flag,
    ]
    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr

def morgan_fingerprint(mol: Chem.Mol, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def esol_rule(desc: np.ndarray) -> float:
    idx = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}
    return 0.16 - 0.63 * float(desc[idx["MolLogP"]]) - 0.0062 * float(desc[idx["MolWt"]]) + 0.066 * float(desc[idx["RotB"]]) - 0.74 * float(desc[idx["AromaticProportion"]])

def clark_score(desc: np.ndarray) -> float:
    idx = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}
    return 0.152 * float(desc[idx["MolLogP"]]) - 0.0148 * float(desc[idx["TPSA"]]) + 0.139

@dataclass
class Audit:
    dataset: str
    raw_rows: int
    missing_or_invalid: int
    duplicate_groups: int
    conflicting_groups: int
    final_rows: int
    task: str
    target: str

def _aggregate_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    counts = df.groupby("canonical_smiles").size()
    duplicate_groups = int((counts > 1).sum())
    grouped = df.groupby("canonical_smiles", as_index=False).agg(target=("target", "mean"), original_count=("target", "size"), source_id=("source_id", "first"))
    return grouped, duplicate_groups, 0

def _aggregate_classification(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    counts = df.groupby("canonical_smiles").size()
    duplicate_groups = int((counts > 1).sum())
    nunique = df.groupby("canonical_smiles")["target"].nunique()
    conflict_smiles = set(nunique[nunique > 1].index)
    clean = df[~df["canonical_smiles"].isin(conflict_smiles)].copy()
    grouped = clean.groupby("canonical_smiles", as_index=False).agg(target=("target", "first"), original_count=("target", "size"), source_id=("source_id", "first"))
    return grouped, duplicate_groups, len(conflict_smiles)

def load_raw(root: Path) -> dict[str, tuple[pd.DataFrame, str]]:
    raw = root / "data" / "raw"
    out = {}
    esol = pd.read_csv(raw / "delaney-processed.csv")
    out["ESOL"] = (pd.DataFrame({"source_id": esol["Compound ID"].astype(str), "smiles": esol["smiles"].astype(str), "target": pd.to_numeric(esol["measured log solubility in mols per litre"], errors="coerce")}), "regression")
    bbbp = pd.read_csv(raw / "BBBP.csv")
    out["BBBP"] = (pd.DataFrame({"source_id": bbbp["num"].astype(str), "smiles": bbbp["smiles"].astype(str), "target": pd.to_numeric(bbbp["p_np"], errors="coerce")}), "classification")
    lipo = pd.read_csv(raw / "Lipophilicity.csv")
    out["Lipophilicity"] = (pd.DataFrame({"source_id": lipo["CMPD_CHEMBLID"].astype(str), "smiles": lipo["smiles"].astype(str), "target": pd.to_numeric(lipo["exp"], errors="coerce")}), "regression")
    rows = []
    with open(raw / "FreeSolv_database.txt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [x.strip() for x in line.rstrip().split(";")]
            if len(fields) >= 5:
                rows.append({"source_id": fields[0], "smiles": fields[1], "target": float(fields[3])})
    out["FreeSolv"] = (pd.DataFrame(rows), "regression")
    try:
        from B3DB import B3DB_DATA_DICT
        ext = B3DB_DATA_DICT["B3DB_classification"].copy()
        labels = ext["BBB+/BBB-"].astype(str).str.strip().map({"BBB+": 1, "BBB-": 0, "+": 1, "-": 0})
        out["B3DB_external"] = (pd.DataFrame({"source_id": ext["NO."].astype(str), "smiles": ext["SMILES"].astype(str), "target": labels}), "classification")
    except Exception:
        pass
    return out

def process_all(root: Path, n_bits: int = 1024) -> list[Audit]:
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    audits = []
    for name, (raw_df, task) in load_raw(root).items():
        raw_rows = len(raw_df)
        work = raw_df.dropna(subset=["smiles", "target"]).copy()
        invalid = raw_rows - len(work)
        can_smiles, reasons = [], []
        for s in work["smiles"]:
            can, _, reason = standardize_smiles(s)
            can_smiles.append(can); reasons.append(reason)
        work["canonical_smiles"] = can_smiles; work["reason"] = reasons
        invalid += int(work["canonical_smiles"].isna().sum())
        work = work[work["canonical_smiles"].notna()].copy()
        grouped, dup, conflicts = _aggregate_classification(work) if task == "classification" else _aggregate_regression(work)
        descriptors, fps, rule_values = [], [], []
        for s in grouped["canonical_smiles"]:
            mol = Chem.MolFromSmiles(s); assert mol is not None
            d = descriptor_vector(mol); descriptors.append(d); fps.append(morgan_fingerprint(mol, n_bits=n_bits))
            rule_values.append(esol_rule(d) if name == "ESOL" else (clark_score(d) if name in {"BBBP", "B3DB_external"} else np.nan))
        desc_arr, fp_arr = np.vstack(descriptors), np.vstack(fps)
        out_df = grouped[["canonical_smiles", "target", "source_id", "original_count"]].copy()
        for j, col in enumerate(DESCRIPTOR_NAMES): out_df[col] = desc_arr[:, j]
        out_df["rule_value"] = np.asarray(rule_values, dtype=float)
        out_df.to_csv(processed_dir / f"{name}.csv", index=False)
        np.savez_compressed(processed_dir / f"{name}_features.npz", descriptors=desc_arr, fingerprints=fp_arr, target=out_df["target"].to_numpy(dtype=np.float32), smiles=out_df["canonical_smiles"].to_numpy(dtype=str), rule=out_df["rule_value"].to_numpy(dtype=np.float32))
        target_name = "logS" if name == "ESOL" else ("BBB class" if "B3DB" in name or name == "BBBP" else ("logD" if name == "Lipophilicity" else "hydration free energy"))
        audits.append(Audit(name, raw_rows, invalid, dup, conflicts, len(out_df), task, target_name))
    with open(processed_dir / "audit.json", "w", encoding="utf-8") as fh: json.dump([asdict(a) for a in audits], fh, indent=2)
    pd.DataFrame([asdict(a) for a in audits]).to_csv(processed_dir / "audit.csv", index=False)
    return audits

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    audit = process_all(project_root)
    print(pd.DataFrame([asdict(a) for a in audit]).to_string(index=False))
