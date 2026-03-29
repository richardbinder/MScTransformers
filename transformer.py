#!/usr/bin/env python3
import argparse
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

import util


class ZINCEncodingOnlyDataset(torch.utils.data.Dataset):
    """
    Produces graphs whose inputs are derived only from the external encoding.

    The base ZINC split is used exclusively for:
      - slicing the correct node range for each graph
      - computing the regression target (true maximum degree)
    """

    def __init__(
        self,
        base_dataset,
        encodings: torch.Tensor,
        factor_dim: Optional[int] = None,
        edge_threshold: float = 0.0,
        topk_neighbors: int = 0,
    ):
        super().__init__()
        self.ds = base_dataset
        self.encodings = encodings.float()
        self.factor_dim = factor_dim
        self.edge_threshold = edge_threshold
        self.topk_neighbors = topk_neighbors

        assert hasattr(self.ds, "slices") and "x" in self.ds.slices, (
            "Expected an InMemoryDataset with slices['x']."
        )

        total_nodes = int(self.ds.data.x.size(0))
        if self.encodings.size(0) != total_nodes:
            raise ValueError(
                f"Encoding rows ({self.encodings.size(0)}) do not match split node count ({total_nodes})."
            )

        if self.encodings.size(1) < 2:
            raise ValueError("Encoding dimension must be at least 2.")

        if self.factor_dim is None:
            if self.encodings.size(1) % 2 != 0:
                raise ValueError(
                    "Encoding width is odd. Pass --factor_dim explicitly to split it into left/right factors."
                )
            self.factor_dim = self.encodings.size(1) // 2

        if not (0 < self.factor_dim < self.encodings.size(1)):
            raise ValueError(
                f"Invalid factor_dim={self.factor_dim} for encoding width {self.encodings.size(1)}."
            )

    def __len__(self) -> int:
        return len(self.ds)

    def _encoding_slice(self, idx: int) -> torch.Tensor:
        start = int(self.ds.slices["x"][idx])
        end = int(self.ds.slices["x"][idx + 1])
        return self.encodings[start:end]

    def _build_graph_from_encoding(self, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_norm = util.normalize_enc_torch(enc)
        left = enc_norm[:, : self.factor_dim]
        right = enc_norm[:, self.factor_dim :]

        affinity = left @ right.T
        n = affinity.size(0)
        affinity.fill_diagonal_(float("-inf"))

        edge_mask = affinity > self.edge_threshold

        if self.topk_neighbors > 0 and n > 1:
            k = min(self.topk_neighbors, n - 1)
            _, top_idx = torch.topk(affinity, k=k, dim=1)
            top_mask = torch.zeros_like(edge_mask)
            top_mask.scatter_(1, top_idx, True)
            edge_mask = edge_mask | top_mask

        edge_mask = edge_mask | edge_mask.T
        edge_mask.fill_diagonal_(False)

        if edge_mask.sum() == 0 and n > 1:
            best = torch.argmax(affinity, dim=1)
            edge_mask[torch.arange(n), best] = True
            edge_mask = edge_mask | edge_mask.T
            edge_mask.fill_diagonal_(False)

        edge_index = edge_mask.nonzero(as_tuple=False).T.contiguous()
        edge_attr = affinity[edge_mask].unsqueeze(-1)

        proxy_degree = edge_mask.sum(dim=1, keepdim=True).float()
        pos_strength = affinity.clamp_min(0.0)
        pos_strength = torch.where(torch.isfinite(pos_strength), pos_strength, torch.zeros_like(pos_strength))
        proxy_strength = pos_strength.sum(dim=1, keepdim=True)

        denom = max(n - 1, 1)
        node_features = torch.cat(
            [
                enc_norm,
                proxy_degree / denom,
                proxy_strength / max(n, 1),
            ],
            dim=1,
        )

        return node_features, edge_index, edge_attr

    def __getitem__(self, idx: int) -> Data:
        base_graph = self.ds.get(idx)
        enc = self._encoding_slice(idx)
        x, edge_index, edge_attr = self._build_graph_from_encoding(enc)

        target = degree(
            base_graph.edge_index[0],
            num_nodes=base_graph.num_nodes,
            dtype=torch.float,
        ).max()

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target.view(1),
            num_nodes=x.size(0),
        )


class EncodingGraphTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}).")

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=1,
                    beta=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor):
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x + residual)

        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=-1)
        return self.readout(graph_repr).view(-1)


def load_encodings(npz_path: str) -> Dict[str, torch.Tensor]:
    arrays = np.load(npz_path)
    keys = set(arrays.files)

    if {"train", "val", "test"}.issubset(keys):
        return {split: torch.from_numpy(arrays[split]).float() for split in ("train", "val", "test")}

    merged = torch.from_numpy(np.concatenate([arrays[key] for key in arrays.files], axis=0)).float()
    return {"all": merged}


def split_encodings(encodings: Dict[str, torch.Tensor], train_base, val_base, test_base):
    if "train" in encodings:
        return encodings["train"], encodings["val"], encodings["test"]

    all_enc = encodings["all"]
    train_nodes = int(train_base.data.x.size(0))
    val_nodes = int(val_base.data.x.size(0))
    test_nodes = int(test_base.data.x.size(0))
    expected = train_nodes + val_nodes + test_nodes

    if all_enc.size(0) != expected:
        raise ValueError(
            f"Encoding rows ({all_enc.size(0)}) do not match total nodes across splits ({expected})."
        )

    train_enc = all_enc[:train_nodes]
    val_enc = all_enc[train_nodes : train_nodes + val_nodes]
    test_enc = all_enc[train_nodes + val_nodes :]
    return train_enc, val_enc, test_enc


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.smooth_l1_loss(pred, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae_values = []
    rmse_values = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.view(-1)
        mae_values.append(F.l1_loss(pred, target).item())
        rmse_values.append(torch.sqrt(F.mse_loss(pred, target)).item())

    mae = float(np.mean(mae_values)) if mae_values else math.nan
    rmse = float(np.mean(rmse_values)) if rmse_values else math.nan
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/ZINC", help="Directory for ZINC.")
    parser.add_argument("--npz", type=str, required=True, help="Path to the external encoding .npz file.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor_dim", type=int, default=None, help="Width of the left matrix factor.")
    parser.add_argument("--edge_threshold", type=float, default=0.0, help="Affinity threshold for reconstructed edges.")
    parser.add_argument("--topk_neighbors", type=int, default=4, help="Extra high-affinity neighbors kept per node.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience on validation RMSE.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_base = ZINC(root=args.root, subset=True, split="train")
    val_base = ZINC(root=args.root, subset=True, split="val")
    test_base = ZINC(root=args.root, subset=True, split="test")

    loaded = load_encodings(args.npz)
    train_enc, val_enc, test_enc = split_encodings(loaded, train_base, val_base, test_base)

    train_ds = ZINCEncodingOnlyDataset(
        train_base,
        train_enc,
        factor_dim=args.factor_dim,
        edge_threshold=args.edge_threshold,
        topk_neighbors=args.topk_neighbors,
    )
    val_ds = ZINCEncodingOnlyDataset(
        val_base,
        val_enc,
        factor_dim=args.factor_dim,
        edge_threshold=args.edge_threshold,
        topk_neighbors=args.topk_neighbors,
    )
    test_ds = ZINCEncodingOnlyDataset(
        test_base,
        test_enc,
        factor_dim=args.factor_dim,
        edge_threshold=args.edge_threshold,
        topk_neighbors=args.topk_neighbors,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = int(train_ds[0].x.size(1))
    model = EncodingGraphTransformer(
        in_dim=input_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae, val_rmse = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f} | best RMSE {best_val_rmse:.4f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch:03d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae, test_rmse = evaluate(model, test_loader, device)
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
