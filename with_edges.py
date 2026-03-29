#!/usr/bin/env python3
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.data import Data


# ---------------------------
# Dataset wrapper
# ---------------------------
class ZINCWithExternalNodeEncoding(torch.utils.data.Dataset):
    """
    Wraps a PyG ZINC split and:
      - replaces node features with a slice of external E
      - replaces labels with max degree of each graph
    """
    def __init__(self, base_dataset, E: torch.Tensor):
        super().__init__()
        self.ds = base_dataset
        self.E = E
        assert hasattr(self.ds, "slices") and "x" in self.ds.slices, \
            "Expected an InMemoryDataset with slices['x']."

        # sanity: total nodes match
        total_nodes = int(self.ds.data.x.size(0))
        if E.size(0) != total_nodes:
            raise ValueError(
                f"E has {E.size(0)} rows but dataset split has {total_nodes} total nodes."
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        data = self.ds.get(i)

        # Figure out node range [s:e) for graph i using slices['x']
        # slices['x'] is a 1D tensor of length len(ds)+1
        s = int(self.ds.slices["x"][i])
        e = int(self.ds.slices["x"][i + 1])

        # Replace node features by external encoding
        x = self.E[s:e]

        # Replace label with maximum degree (scalar regression target)
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
        y = deg.max().view(1)/4  # shape [1]

        # create fresh Data object
        data_new = Data(
            x=torch.zeros_like(x).float(),
            edge_index=data.edge_index,
            y=y,
            num_nodes=x.size(0),
        )

        return data_new


# ---------------------------
# Simple Graph Transformer
# ---------------------------
class GraphTransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            TransformerConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout, beta=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Middle layers
        for _ in range(n_layers - 1):
            self.convs.append(
                TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, beta=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        out = self.mlp(g).view(-1)      # [num_graphs]
        return out


# ---------------------------
# Utilities: load E from npz
# ---------------------------
def load_E_npz(npz_path: str):
    """
    Supports either:
      - keys: 'train', 'val', 'test' (recommended), each a [sum_nodes_split, k] array
      - OR a single key (e.g. 'E' or the first key) that contains all nodes for all splits,
        in order train -> val -> test (we'll split by node counts).
    """
    result = []
    matrices = np.load(npz_path)

    # for i in range(len(matrices)):
    #     m = matrices[f"idx_{i}"]
    #     result.append(torch.Tensor(m))

    result = torch.from_numpy(np.concatenate(list(matrices.values()), axis=0)).float()

    return result


# ---------------------------
# Train / Eval loops
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.view(-1).to(device)
        loss = torch.sqrt(F.mse_loss(pred, y))
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else math.nan


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.view(-1).to(device)
        loss = torch.sqrt(F.mse_loss(pred, y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else math.nan


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data/ZINC", help="Where to store/load ZINC")
    ap.add_argument("--npz", type=str, required=True, help="Path to node-encoding .npz")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load splits
    train_base = ZINC(root=args.root, subset=True, split="train")
    val_base   = ZINC(root=args.root, subset=True, split="val")
    test_base  = ZINC(root=args.root, subset=True, split="test")

    # Load E
    E_all = load_E_npz(args.npz)

    # Split "all" by node counts of splits (train -> val -> test)
    n_train_nodes = int(train_base.data.x.size(0))
    n_val_nodes = int(val_base.data.x.size(0))
    n_test_nodes = int(test_base.data.x.size(0))
    expected = n_train_nodes + n_val_nodes + n_test_nodes
    if E_all.size(0) != expected:
        raise ValueError(
            f"E_all rows ({E_all.size(0)}) != total nodes across splits ({expected}). "
            "Either provide per-split arrays (train/val/test) in the NPZ, or ensure ordering matches."
        )
    E_train = E_all[:n_train_nodes]
    E_val = E_all[n_train_nodes:n_train_nodes + n_val_nodes]
    E_test = E_all[n_train_nodes + n_val_nodes:]

    # Wrap datasets (inject external node enc + max-degree labels)
    train_ds = ZINCWithExternalNodeEncoding(train_base, E_train)
    val_ds   = ZINCWithExternalNodeEncoding(val_base, E_val)
    test_ds  = ZINCWithExternalNodeEncoding(test_base, E_test)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    k = 8

    enc = E_train[0:train_base.get(0).num_nodes]

    with torch.no_grad():
        enc_norm = util.normalize_enc_torch(enc)

    # Reconstruct adjacency from normalized encodings
    L_n = enc_norm[:, :k]
    R_n = enc_norm[:, k:]
    A_reconstructed = (L_n @ R_n.T > 0).float()

    # Model
    in_dim = int(E_train.size(1))
    model = GraphTransformerRegressor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | train MSE {tr:.4f} | val MSE {va:.4f} | best val {best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    te = evaluate(model, test_loader, device)
    print(f"Test MSE: {te:.4f}")


if __name__ == "__main__":
    main()
