"""Compact graph, fusion, and rule-residual networks for CPU-reproducible experiments."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]

def one_hot(value, options):
    return [float(value == x) for x in options] + [float(value not in options)]

def mol_to_graph(smiles: str, desc: np.ndarray, target: float, rule: float, applicability: float, idx: int) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    xs = []
    for atom in mol.GetAtoms():
        feat = []
        feat += one_hot(atom.GetAtomicNum(), ATOM_TYPES)
        feat += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        feat += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        feat += [atom.GetFormalCharge() / 3.0, float(atom.GetIsAromatic()), atom.GetTotalNumHs() / 4.0, float(atom.IsInRing())]
        xs.append(feat)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); edges.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(x=torch.tensor(xs, dtype=torch.float32), edge_index=edge_index, desc=torch.tensor(desc, dtype=torch.float32).view(1, -1), y=torch.tensor([target], dtype=torch.float32), rule=torch.tensor([rule], dtype=torch.float32), applicability=torch.tensor([applicability], dtype=torch.float32), sample_idx=torch.tensor([idx], dtype=torch.long))

class GraphDataset(Dataset):
    def __init__(self, graphs: Sequence[Data]): self.graphs = list(graphs)
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]

class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm that falls back to stored statistics for a singleton node batch."""
    def forward(self, input):
        if self.training and input.ndim >= 2 and input.shape[0] == 1:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)
        return super().forward(input)

class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden: int = 48):
        super().__init__(); self.conv1 = GCNConv(node_dim, hidden); self.conv2 = GCNConv(hidden, hidden); self.norm1 = SafeBatchNorm1d(hidden); self.norm2 = SafeBatchNorm1d(hidden)
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.norm1(self.conv1(x, edge_index))); x = torch.relu(self.norm2(self.conv2(x, edge_index))); return global_mean_pool(x, batch)

class MolecularNet(nn.Module):
    def __init__(self, node_dim: int, desc_dim: int, mode: str, task: str, hidden: int = 48):
        super().__init__(); self.mode = mode; self.task = task; self.graph = GraphEncoder(node_dim, hidden)
        self.desc = nn.Sequential(nn.Linear(desc_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU())
        rep_dim = hidden if mode == "gnn" else 2 * hidden
        self.pred_head = nn.Sequential(nn.Linear(rep_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        if mode in {"residual", "gated", "proposed"}: self.gate_head = nn.Sequential(nn.Linear(rep_dim + 1, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
    def representation(self, data):
        g = self.graph(data.x, data.edge_index, data.batch)
        if self.mode == "gnn": return g
        d = self.desc(data.desc); return torch.cat([g, d], dim=1)
    def forward(self, data):
        h = self.representation(data); raw = self.pred_head(h).squeeze(1); gate = torch.zeros_like(raw); rule = data.rule.view(-1)
        if self.mode == "fusion" or self.mode == "gnn": out = raw
        elif self.mode == "residual": out = rule + raw
        elif self.mode == "gated":
            gate = torch.sigmoid(self.gate_head(torch.cat([h, data.applicability.view(-1, 1)], dim=1)).squeeze(1)); out = raw + gate * (rule - raw)
        else:
            learned = torch.sigmoid(self.gate_head(torch.cat([h, data.applicability.view(-1, 1)], dim=1)).squeeze(1)); gate = data.applicability.view(-1) * learned; out = raw + gate * (rule - raw)
        return out, gate, raw

@dataclass
class TrainResult:
    model: MolecularNet
    best_epoch: int
    best_valid: float

def train_model(train_graphs, valid_graphs, mode: str, task: str, seed: int, gate_lambda: float = 0.0, epochs: int = 80, patience: int = 12, batch_size: int = 64, lr: float = 2e-3, weight_decay: float = 1e-5) -> TrainResult:
    torch.manual_seed(seed); np.random.seed(seed)
    node_dim = train_graphs[0].x.shape[1]; desc_dim = train_graphs[0].desc.shape[1]; model = MolecularNet(node_dim, desc_dim, mode=mode, task=task)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay); train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True); valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=False); loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict()); best_valid = float("inf"); best_epoch = 0; bad = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True); out, gate, _ = model(batch); loss = loss_fn(out, batch.y.view(-1))
            if mode == "proposed" and gate_lambda > 0: loss = loss + gate_lambda * torch.mean(batch.applicability.view(-1) * (1.0 - gate))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval(); losses = []
        with torch.no_grad():
            for batch in valid_loader:
                out, gate, _ = model(batch); val = loss_fn(out, batch.y.view(-1)); losses.append(float(val) * batch.num_graphs)
        score = sum(losses) / max(1, len(valid_graphs))
        if score < best_valid - 1e-5: best_valid, best_epoch, best_state, bad = score, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return TrainResult(model, best_epoch, best_valid)

def predict_model(model: MolecularNet, graphs, batch_size: int = 128):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False); out_all, gate_all, raw_all, idx_all = [], [], [], []; model.eval()
    with torch.no_grad():
        for batch in loader:
            out, gate, raw = model(batch); out_all.append(out.cpu().numpy()); gate_all.append(gate.cpu().numpy()); raw_all.append(raw.cpu().numpy()); idx_all.append(batch.sample_idx.view(-1).cpu().numpy())
    return np.concatenate(out_all), np.concatenate(gate_all), np.concatenate(raw_all), np.concatenate(idx_all)
