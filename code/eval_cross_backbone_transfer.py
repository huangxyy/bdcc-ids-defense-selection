"""
Cross-backbone transfer attack experiment.
Trains standard + PGD-AT models for MLP / CNN1D / FT-Transformer,
then measures cross-architecture adversarial transfer ASR.

Outputs: cross_backbone_transfer_results.csv  (Appendix Tables 10 & 11)

Usage:
  python code/cross_backbone_transfer.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
import ids_core as base


# ── Model classes ──────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(1)


class FTTransformer(nn.Module):
    """Matches backbone_ft_transformer.py: d_token=32, n_heads=2, d_ffn=64, vectorized tokenizer."""
    def __init__(self, input_dim: int, d_token: int = 32, n_heads: int = 2,
                 n_layers: int = 2, d_ffn: int = 64, dropout: float = 0.15):
        super().__init__()
        self.d_token = d_token
        # Vectorized feature tokenizer: single (input_dim, d_token) matrix
        self.feature_weight = nn.Parameter(torch.empty(input_dim, d_token))
        self.feature_bias = nn.Parameter(torch.empty(input_dim, d_token))
        nn.init.kaiming_uniform_(self.feature_weight)
        nn.init.zeros_(self.feature_bias)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B, D, 1) * (1, D, d_token) + bias -> (B, D, d_token)
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        return self.head(out[:, 0, :]).squeeze(1)


# ── Training helpers ───────────────────────────────────────────────────────

def train_standard(model, X_tr, y_tr, device, epochs=10, lr=1e-3, bs=512):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(),
                       torch.from_numpy(y_tr).float()),
        batch_size=bs, shuffle=True,
    )
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_pgd_at(model, X_tr, y_tr, device, epochs=8, lr=1e-3, bs=512,
                 epsilon=0.06, alpha=0.015, steps=20, mins=None, maxs=None):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mask_t = torch.ones(X_tr.shape[1], device=device)
    mins_t = torch.from_numpy(mins).float().to(device)
    maxs_t = torch.from_numpy(maxs).float().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(),
                       torch.from_numpy(y_tr).float()),
        batch_size=bs, shuffle=True,
    )
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            x_adv = base.pgd_attack(model, xb, yb, epsilon, alpha, steps,
                                    mask_t, mins_t, maxs_t)
            loss = F.binary_cross_entropy_with_logits(model(x_adv), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


# ── Transfer evaluation ────────────────────────────────────────────────────

def compute_transfer_asr(source_model, target_model, X_ev, y_ev,
                         device, epsilon=0.1, steps=20,
                         mins=None, maxs=None, batch_size=512):
    """Generate PGD adversarial examples from source, measure ASR on target."""
    mask_t = torch.ones(X_ev.shape[1], device=device)
    mins_t = torch.from_numpy(mins).float().to(device)
    maxs_t = torch.from_numpy(maxs).float().to(device)
    alpha = epsilon / max(steps, 1)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_ev).float(),
                       torch.from_numpy(y_ev).float()),
        batch_size=batch_size, shuffle=False,
    )
    # Generate adversarial examples from source model
    all_adv = []
    source_model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        x_adv = base.pgd_attack(source_model, xb, yb, epsilon, alpha, steps,
                                mask_t, mins_t, maxs_t)
        all_adv.append(x_adv.detach().cpu().numpy())
    X_adv = np.concatenate(all_adv)

    # Evaluate adversarial examples on target model
    target_model.eval()
    adv_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_adv).float(),
                       torch.from_numpy(y_ev).float()),
        batch_size=batch_size, shuffle=False,
    )
    preds_adv = []
    with torch.no_grad():
        for xb, _ in adv_loader:
            logits = target_model(xb.to(device))
            preds_adv.append(
                (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            )
    y_pred_adv = np.concatenate(preds_adv)

    # Clean predictions on target model
    clean_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_ev).float(),
                       torch.from_numpy(y_ev).float()),
        batch_size=batch_size, shuffle=False,
    )
    preds_clean = []
    with torch.no_grad():
        for xb, _ in clean_loader:
            logits = target_model(xb.to(device))
            preds_clean.append(
                (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            )
    y_pred_clean = np.concatenate(preds_clean)

    # ASR = fraction of correctly-classified that become misclassified
    y_int = y_ev.astype(int)
    correct_mask = y_pred_clean == y_int
    n_correct = correct_mask.sum()
    if n_correct == 0:
        return 0.0
    flipped = (y_pred_adv[correct_mask] != y_int[correct_mask]).sum()
    return float(flipped / n_correct)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-backbone transfer attack")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-rows", type=int, default=20000)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_df, test_df = base.load_unsw_split(
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "test.csv"),
    )
    X_train, y_train, X_test, y_test, meta = base.build_features(train_df, test_df)
    n_features = X_train.shape[1]
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)

    # Subsample for evaluation
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), min(args.eval_rows, len(X_test)), replace=False)
    X_ev, y_ev = X_test[idx], y_test[idx]

    seeds = [7, 13, 21, 42, 100]
    backbone_factories = {
        "MLP": lambda: base.MLP(n_features, (128, 64, 32), 0.15),
        "1D-CNN": lambda: CNN1D(n_features, 0.15),
        "FT-Transformer": lambda: FTTransformer(n_features),
    }
    backbone_names = list(backbone_factories.keys())
    conditions = ["clean", "pgd_at"]

    results = []
    total_t0 = time.time()

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train all backbone x condition models
        models = {}
        for bname, factory in backbone_factories.items():
            t0 = time.time()
            m_std = factory()
            m_std = train_standard(m_std, X_train, y_train, device)
            print(f"  {bname:16s} standard: {time.time()-t0:.1f}s")
            models[(bname, "clean")] = m_std

            t0 = time.time()
            m_at = factory()
            m_at = train_pgd_at(m_at, X_train, y_train, device,
                                mins=mins, maxs=maxs)
            print(f"  {bname:16s} PGD-AT:   {time.time()-t0:.1f}s")
            models[(bname, "pgd_at")] = m_at

        # Cross-backbone transfer
        for cond in conditions:
            for src in backbone_names:
                for tgt in backbone_names:
                    if src == tgt:
                        continue
                    asr = compute_transfer_asr(
                        models[(src, cond)], models[(tgt, cond)],
                        X_ev, y_ev, device,
                        epsilon=0.1, steps=20, mins=mins, maxs=maxs,
                    )
                    print(f"  [{cond:6s}] {src:16s} -> {tgt:16s}: ASR={asr*100:.2f}%")
                    results.append({
                        "seed": seed, "condition": cond,
                        "source": src, "target": tgt,
                        "transfer_asr": asr,
                    })

    elapsed = time.time() - total_t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    raw_path = os.path.join(out_dir, "cross_backbone_transfer_raw.csv")
    df.to_csv(raw_path, index=False)

    mean_df = (df.groupby(["condition", "source", "target"])["transfer_asr"]
               .mean().reset_index())
    mean_path = os.path.join(out_dir, "cross_backbone_transfer_mean.csv")
    mean_df.to_csv(mean_path, index=False)

    # Print formatted LaTeX-ready tables
    for cond in conditions:
        label = "Standard (Clean)" if cond == "clean" else "PGD-AT"
        print(f"\n=== Transfer ASR Matrix: {label} ===")
        sub = mean_df[mean_df.condition == cond]
        header = f"{'Source / Target':16s}"
        for tgt in backbone_names:
            header += f" | {tgt:16s}"
        print(header)
        print("-" * len(header))
        for src in backbone_names:
            row_str = f"{src:16s}"
            for tgt in backbone_names:
                if src == tgt:
                    row_str += f" | {'---':>16s}"
                else:
                    val = sub[(sub.source == src) & (sub.target == tgt)]
                    if len(val) > 0:
                        row_str += f" | {val.iloc[0]['transfer_asr']*100:>15.2f}%"
                    else:
                        row_str += f" | {'N/A':>16s}"
            print(row_str)

    print(f"\nResults saved to:\n  {raw_path}\n  {mean_path}")


if __name__ == "__main__":
    main()
