"""Core library for the backbone-conditioned Pareto analysis of IDS defenses.

Shared by every experiment script in this repository:

  * ExperimentConfig       - all hyperparameters in one place
  * data pipeline          - loading, feature engineering, evaluation subsets
  * attacks                - FGSM, PGD, C&W (L2), APGD-CE
  * defense training       - the six candidate strategies
  * metrics and statistics - F1, ASR, cost efficiency, paired tests
  * the MLP backbone and the MLP experiment entry point

Run directly to reproduce the MLP backbone:

    python ids_core.py --train-path data/train.csv \\
        --test-path data/test.csv --output-dir outputs/mlp --device cuda

The 1D-CNN and FT-Transformer backbones live in backbone_cnn1d.py and
backbone_ft_transformer.py, both of which import this module.
"""
from __future__ import annotations

import argparse
import os
import copy
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed every stochastic component AND make GPU kernels deterministic.

    cuDNN selects reduction orders non-deterministically by default, so two runs
    with the same seed can differ in the third decimal place. That matters here:
    several Pareto decisions hinge on margins of that size. With the settings
    below, repeated runs are bit-identical.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentConfig:
    train_path: str
    test_path: str
    output_dir: str = "outputs/default"
    training_budget_mode: str = "legacy"
    batch_size: int = 1024
    baseline_epochs: int = 10
    adv_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.15
    eval_attack_rows: int = 20000
    adv_epsilon: float = 0.06
    adv_alpha: float = 0.015
    adv_steps: int = 20
    epsilon_list: tuple[float, ...] = (0.02, 0.05, 0.10)
    seeds: tuple[int, ...] = (7, 13, 21, 42, 100)
    sensitivity_top_ratio: float = 0.3
    sensitivity_ratio_list: tuple[float, ...] = (0.2, 0.3, 0.4)
    sensitivity_batches: int = 16
    transfer_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    validity_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    category_attack: str = "pgd"
    category_epsilon: float = 0.1
    ratio_attack: str = "pgd"
    ratio_attack_epsilon: float = 0.1
    top_attack_categories: int = 6
    reference_models: tuple[str, ...] = ("log_reg", "hist_gbdt")
    # TRADES
    trades_beta: float = 6.0
    # --- optional extra methods (see the Optional extra defense methods section) ---
    extra_methods: tuple[str, ...] = ()
    progressive_ratios: tuple[float, ...] = (0.20, 0.30, 0.45, 0.60)
    sa_trades_gamma: float = 1.0
    dst_update_interval: int = 2
    dst_ema_alpha: float = 0.7
    # Free AT
    free_at_replay: int = 4
    # Class-aware constrained AT
    class_aware_minority_weight: float = 3.0
    # C&W attack
    # PGD evaluation step size as a fraction of epsilon.
    # 0.05 = eps/20, the value used for every reported result.
    # Set to 0.10 to match the "alpha = eps/10" wording of the manuscript.
    eval_pgd_alpha_ratio: float = 0.05
    cw_steps: int = 30
    cw_lr: float = 0.01
    cw_c: float = 1.0
    # APGD attack
    apgd_steps: int = 50
    apgd_rho: float = 0.75
    device: str = "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(1)


def load_unsw_split(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    target_col = "label"
    drop_cols = [col for col in ["id", "attack_cat"] if col in train_df.columns]
    feature_cols = [col for col in train_df.columns if col not in drop_cols + [target_col]]

    categorical_cols = [col for col in feature_cols if train_df[col].dtype.kind == "O"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    x_train = encoder.fit_transform(train_df[feature_cols])
    x_test = encoder.transform(test_df[feature_cols])
    y_train = train_df[target_col].astype(np.float32).to_numpy()
    y_test = test_df[target_col].astype(np.float32).to_numpy()

    numeric_dim = len(numeric_cols)
    total_dim = x_train.shape[1]
    numeric_mask = np.zeros(total_dim, dtype=np.float32)
    numeric_mask[:numeric_dim] = 1.0

    numeric_mins = x_train[:, :numeric_dim].min(axis=0)
    numeric_maxs = x_train[:, :numeric_dim].max(axis=0)
    feature_names = list(numeric_cols)
    cat_feature_names = []
    if categorical_cols:
        one_hot = encoder.named_transformers_["cat"]
        cat_feature_names = one_hot.get_feature_names_out(categorical_cols).tolist()
    feature_names.extend(cat_feature_names)

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "numeric_mask": numeric_mask,
        "numeric_mins": numeric_mins,
        "numeric_maxs": numeric_maxs,
    }
    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test, metadata



def stratified_subset_indices(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    indices = np.arange(len(y))
    if len(y) <= size:
        return indices
    _, subset_indices = train_test_split(
        indices,
        test_size=size,
        random_state=seed,
        stratify=y,
    )
    return np.sort(subset_indices)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    if not contiguous.flags.writeable:
        contiguous = contiguous.copy()
    return torch.from_numpy(contiguous)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(numpy_to_torch(x), numpy_to_torch(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def count_out_of_range(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                       tol: float = 1e-4) -> tuple[int, float]:
    """Count samples that lie meaningfully outside the training [mins, maxs] box.

    clamp_numeric() projects perturbations back into that box, so a test sample
    that already lies outside it is moved by the projection itself, and that
    displacement can exceed epsilon. This quantifies how often that can happen.

    A relative tolerance is applied because in float32 a value can sit a few ULP
    outside the bound; without it, essentially every sample is reported as
    out-of-range while the actual displacement is exactly zero. Pass tol=0.0 for
    the strict comparison.

    Returns (n_out_of_range, fraction). Use max_violation() for the magnitude.
    """
    d = x[:, : mins.shape[0]]
    span = np.maximum(np.abs(mins), np.abs(maxs)) + 1.0
    eps = tol * span
    bad = np.any((d < mins[None, :] - eps) | (d > maxs[None, :] + eps), axis=1)
    return int(bad.sum()), float(bad.mean())


def max_violation(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> float:
    """Largest amount by which any continuous feature leaves the training box.

    This is the displacement that clamp_numeric() will apply to the untouched
    input, i.e. movement caused by constraint (ii) rather than by the attacker.
    """
    d = x[:, : mins.shape[0]]
    return float(np.maximum(np.maximum(mins[None, :] - d, d - maxs[None, :]), 0.0).max())


def clamp_numeric(x_adv: torch.Tensor, x_orig: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Project onto the box [mins, maxs]; non-perturbable coordinates are restored.

    NOTE: enforces constraint (ii) - perturbed features stay inside the range seen
    on the training set. Where (ii) and (iii) (||delta||_inf <= epsilon) conflict,
    this resolves in favour of (ii). See count_out_of_range().
    """
    x_adv = torch.where(mask > 0, x_adv, x_orig)
    numeric_dim = mins.shape[0]
    clipped_numeric = torch.max(torch.min(x_adv[:, :numeric_dim], maxs), mins)
    x_adv = torch.cat([clipped_numeric, x_adv[:, numeric_dim:]], dim=1)
    return x_adv


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    mask: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss()
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = criterion(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign() * mask
    x_adv = x_adv.detach() + perturbation
    return clamp_numeric(x_adv, x, mins, maxs, mask)


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    mask: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss()
    x_adv = x.detach().clone()
    random_noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon) * mask
    x_adv = x_adv + random_noise
    x_adv = clamp_numeric(x_adv, x, mins, maxs, mask)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        step = alpha * x_adv.grad.sign() * mask
        x_adv = x_adv.detach() + step
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon) * mask
        x_adv = x + delta
        x_adv = clamp_numeric(x_adv, x, mins, maxs, mask)
    return x_adv.detach()


def cw_attack(model, x, y, mask, mins, maxs, steps, lr, c_const):
    """Carlini-Wagner L2 attack for binary classification with feature constraints."""
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        x_candidate = x + delta * mask
        x_candidate = clamp_numeric(x_candidate, x, mins, maxs, mask)
        logits = model(x_candidate)
        target_sign = 1.0 - 2.0 * y
        margin_loss = torch.clamp(-logits * target_sign, min=0.0)
        l2_loss = (delta * mask).pow(2).sum(dim=1)
        loss = l2_loss.mean() + c_const * margin_loss.mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        x_adv = x + delta.detach() * mask
        x_adv = clamp_numeric(x_adv, x, mins, maxs, mask)
    return x_adv


def apgd_attack(model, x, y, epsilon, steps, mask, mins, maxs, rho=0.75):
    """Adaptive PGD (APGD-CE) - simplified AutoAttack component."""
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    batch_size_local = x.shape[0]
    x_adv = x.detach().clone()
    noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon) * mask
    x_adv = clamp_numeric(x_adv + noise, x, mins, maxs, mask)
    alpha = 2.0 * epsilon / max(steps, 1)
    momentum = torch.zeros_like(x)
    best_loss = torch.full((batch_size_local,), -float("inf"), device=x.device)
    best_x_adv = x_adv.clone()
    check_points = set()
    p_init = 0.22
    for j in range(1, steps):
        if j / steps >= p_init:
            check_points.add(j)
            p_init += max(p_init * 0.03, 0.06)
    prev_loss_sum = None
    for step_i in range(steps):
        x_adv.requires_grad_(True)
        loss_per = criterion(model(x_adv), y)
        loss_per.sum().backward()
        grad = x_adv.grad.detach()
        model.zero_grad(set_to_none=True)
        momentum = rho * momentum + (1.0 - rho) * grad
        step = alpha * momentum.sign() * mask
        x_adv = x_adv.detach() + step
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon) * mask
        x_adv = clamp_numeric(x + delta, x, mins, maxs, mask)
        with torch.no_grad():
            cur_loss = criterion(model(x_adv), y)
            improved = cur_loss > best_loss
            best_loss = torch.where(improved, cur_loss, best_loss)
            best_x_adv = torch.where(improved.unsqueeze(1), x_adv, best_x_adv)
        if step_i in check_points:
            cur_sum = float(cur_loss.sum())
            if prev_loss_sum is not None and cur_sum <= prev_loss_sum * 1.001:
                alpha *= 0.5
                x_adv = best_x_adv.clone()
                momentum.zero_()
            prev_loss_sum = cur_sum
    return best_x_adv.detach()


def adaptive_attack(model, xb, yb, epsilon, alpha, steps, defense_mask_t, mins_t, maxs_t):
    """DEPRECATED - this is plain PGD, not an adaptive attack.

    Kept only so that older scripts keep importing. It performs NO defence-aware
    adaptation: the body is a direct call to pgd_attack with identical parameters
    and the same mask. For a genuine adaptive evaluation use `adaptive_attacks.py`,
    which provides a decision-aligned loss with restarts, a gradient-free
    cross-check, and a complement attack against the defence training mask.
    """
    warnings.warn("adaptive_attack() is a PGD alias and is NOT an adaptive attack; "
                  "use adaptive_attacks.py instead.", DeprecationWarning, stacklevel=2)
    return pgd_attack(model, xb, yb, epsilon, alpha, steps, defense_mask_t, mins_t, maxs_t)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, clean_pred: np.ndarray | None = None) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.0
    if clean_pred is not None:
        originally_correct = clean_pred == y_true
        if originally_correct.any():
            attack_success = (y_pred[originally_correct] != y_true[originally_correct]).mean()
            metrics["attack_success_rate"] = float(attack_success)
        else:
            metrics["attack_success_rate"] = 0.0
    else:
        metrics["attack_success_rate"] = 0.0
    return metrics


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    probs = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def predict_probabilities_from_array(model: nn.Module, x_eval: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dummy_y = np.zeros(len(x_eval), dtype=np.float32)
    loader = make_loader(x_eval, dummy_y, batch_size=batch_size, shuffle=False)
    return predict_probabilities(model, loader, device)


def predict_probabilities_any(model, x_eval: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    if isinstance(model, nn.Module):
        return predict_probabilities_from_array(model, x_eval, batch_size=batch_size, device=device)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_eval)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities
    raise TypeError(f"Unsupported model type for probability prediction: {type(model)!r}")


def train_reference_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    enabled_models: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, float]]:
    model_registry: dict[str, object] = {}
    train_time_registry: dict[str, float] = {}
    y_train_int = y_train.astype(np.int32)

    if "log_reg" in enabled_models:
        log_reg = LogisticRegression(
            max_iter=800,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
        start = time.perf_counter()
        log_reg.fit(x_train, y_train_int)
        train_time_registry["log_reg"] = time.perf_counter() - start
        model_registry["log_reg"] = log_reg

    if "hist_gbdt" in enabled_models:
        hist_gbdt = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=40,
            random_state=seed,
        )
        start = time.perf_counter()
        hist_gbdt.fit(x_train, y_train_int)
        train_time_registry["hist_gbdt"] = time.perf_counter() - start
        model_registry["hist_gbdt"] = hist_gbdt

    return model_registry, train_time_registry


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def measure_inference_efficiency(
    model: nn.Module,
    x_eval: np.ndarray,
    batch_size: int,
    device: torch.device,
    repeats: int = 3,
) -> dict[str, float]:
    probe_rows = int(min(len(x_eval), batch_size * 8))
    probe_x = x_eval[:probe_rows]
    dummy_y = np.zeros(probe_rows, dtype=np.float32)
    loader = make_loader(probe_x, dummy_y, batch_size=batch_size, shuffle=False)

    predict_probabilities(model, loader, device)
    start = time.perf_counter()
    for _ in range(repeats):
        predict_probabilities(model, loader, device)
    elapsed = time.perf_counter() - start
    mean_seconds = elapsed / max(repeats, 1)
    return {
        "inference_probe_rows": probe_rows,
        "inference_seconds": mean_seconds,
        "inference_ms_per_sample": (mean_seconds * 1000.0 / probe_rows) if probe_rows else 0.0,
        "inference_samples_per_second": (probe_rows / mean_seconds) if mean_seconds > 0 else 0.0,
    }


def measure_inference_efficiency_any(
    model,
    x_eval: np.ndarray,
    batch_size: int,
    device: torch.device,
    repeats: int = 3,
) -> dict[str, float]:
    probe_rows = int(min(len(x_eval), batch_size * 8))
    probe_x = x_eval[:probe_rows]

    if isinstance(model, nn.Module):
        return measure_inference_efficiency(
            model=model,
            x_eval=probe_x,
            batch_size=batch_size,
            device=device,
            repeats=repeats,
        )

    predict_probabilities_any(model, probe_x, batch_size=batch_size, device=device)
    start = time.perf_counter()
    for _ in range(repeats):
        predict_probabilities_any(model, probe_x, batch_size=batch_size, device=device)
    elapsed = time.perf_counter() - start
    mean_seconds = elapsed / max(repeats, 1)
    return {
        "inference_probe_rows": probe_rows,
        "inference_seconds": mean_seconds,
        "inference_ms_per_sample": (mean_seconds * 1000.0 / probe_rows) if probe_rows else 0.0,
        "inference_samples_per_second": (probe_rows / mean_seconds) if mean_seconds > 0 else 0.0,
    }


def compute_attack_validity_metrics(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    numeric_mask: np.ndarray,
    numeric_mins: np.ndarray,
    numeric_maxs: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, float]:
    numeric_indices = np.where(numeric_mask > 0)[0]
    protected_indices = np.where(numeric_mask <= 0)[0]

    if len(numeric_indices):
        clean_numeric = x_clean[:, numeric_indices]
        adv_numeric = x_adv[:, numeric_indices]
        numeric_in_range = (adv_numeric >= (numeric_mins - tol)) & (adv_numeric <= (numeric_maxs + tol))
        numeric_valid_per_sample = numeric_in_range.all(axis=1)
        numeric_valid_rate = float(numeric_valid_per_sample.mean())
        numeric_delta = adv_numeric - clean_numeric
        changed_numeric = np.abs(numeric_delta) > tol
        boundary_hits = (
            np.isclose(adv_numeric, numeric_mins[None, :], atol=tol)
            | np.isclose(adv_numeric, numeric_maxs[None, :], atol=tol)
        ) & changed_numeric
        mean_abs_delta = float(np.abs(numeric_delta).mean())
        mean_l2_delta = float(np.linalg.norm(numeric_delta, axis=1).mean())
        max_abs_delta = float(np.abs(numeric_delta).max())
        changed_numeric_ratio = float(changed_numeric.mean())
        boundary_clip_ratio = float(boundary_hits.mean())
    else:
        numeric_valid_per_sample = np.ones(len(x_clean), dtype=bool)
        numeric_valid_rate = 1.0
        mean_abs_delta = 0.0
        mean_l2_delta = 0.0
        max_abs_delta = 0.0
        changed_numeric_ratio = 0.0
        boundary_clip_ratio = 0.0

    if len(protected_indices):
        clean_protected = x_clean[:, protected_indices]
        adv_protected = x_adv[:, protected_indices]
        protected_equal = np.isclose(clean_protected, adv_protected, atol=tol)
        protected_integrity_per_sample = protected_equal.all(axis=1)
        protected_integrity_rate = float(protected_integrity_per_sample.mean())
        protected_feature_change_ratio = float((~protected_equal).mean())
    else:
        protected_integrity_per_sample = np.ones(len(x_clean), dtype=bool)
        protected_integrity_rate = 1.0
        protected_feature_change_ratio = 0.0

    overall_validity_rate = float((numeric_valid_per_sample & protected_integrity_per_sample).mean())
    return {
        "numeric_valid_rate": numeric_valid_rate,
        "protected_integrity_rate": protected_integrity_rate,
        "overall_validity_rate": overall_validity_rate,
        "mean_numeric_abs_delta": mean_abs_delta,
        "mean_numeric_l2_delta": mean_l2_delta,
        "max_numeric_abs_delta": max_abs_delta,
        "changed_numeric_feature_ratio": changed_numeric_ratio,
        "protected_feature_change_ratio": protected_feature_change_ratio,
        "boundary_clip_ratio": boundary_clip_ratio,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    adv_training: bool,
    adv_builder,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if adv_training:
                model.eval()
                x_adv = adv_builder(model, xb, yb)
                model.train()
                batch_x = torch.cat([xb, x_adv], dim=0)
                batch_y = torch.cat([yb, yb], dim=0)
            else:
                batch_x = xb
                batch_y = yb
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    return model


def summarize_training_budget(
    *,
    pretrain_seconds: float,
    continuation_seconds: float = 0.0,
    mask_build_seconds: float = 0.0,
    include_pretrain_in_total: bool = True,
    include_mask_in_total: bool = False,
) -> dict[str, float]:
    # Keep both the decomposed timing and the headline timing used for comparisons.
    total_seconds = continuation_seconds
    if include_pretrain_in_total:
        total_seconds += pretrain_seconds
    if include_mask_in_total:
        total_seconds += mask_build_seconds
    return {
        "pretrain_seconds": pretrain_seconds,
        "continuation_seconds": continuation_seconds,
        "mask_build_seconds": mask_build_seconds,
        "train_seconds": total_seconds,
    }


def train_model_trades(model, train_loader, epochs, device, config,
                       mask_t, mins_t, maxs_t):
    """TRADES: BCE(clean) + beta * KL(clean || adv)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    criterion_bce = nn.BCEWithLogitsLoss()
    beta = config.trades_beta
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_clean = model(xb)
            loss_bce = criterion_bce(logits_clean, yb)
            probs_clean = torch.sigmoid(logits_clean).detach()
            model.eval()
            x_adv = xb.detach().clone()
            noise = torch.empty_like(x_adv).uniform_(
                -config.adv_epsilon, config.adv_epsilon) * mask_t
            x_adv = clamp_numeric(x_adv + noise, xb, mins_t, maxs_t, mask_t)
            for _ in range(config.adv_steps):
                x_adv.requires_grad_(True)
                probs_adv = torch.sigmoid(model(x_adv))
                p = probs_clean.clamp(1e-7, 1 - 1e-7)
                q = probs_adv.clamp(1e-7, 1 - 1e-7)
                kl = p*(p.log()-q.log()) + (1-p)*((1-p).log()-(1-q).log())
                model.zero_grad(set_to_none=True)
                kl.sum().backward()
                step = config.adv_alpha * x_adv.grad.sign() * mask_t
                x_adv = x_adv.detach() + step
                delta = torch.clamp(x_adv-xb, min=-config.adv_epsilon,
                                    max=config.adv_epsilon) * mask_t
                x_adv = clamp_numeric(xb+delta, xb, mins_t, maxs_t, mask_t)
            model.train()
            paf = torch.sigmoid(model(x_adv.detach())).clamp(1e-7, 1-1e-7)
            p = probs_clean.clamp(1e-7, 1-1e-7)
            kl_f = p*(p.log()-paf.log()) + (1-p)*((1-p).log()-(1-paf).log())
            loss = loss_bce + beta * kl_f.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_model_free_at(model, train_loader, epochs, device, config,
                        mask_t, mins_t, maxs_t):
    """Free Adversarial Training: replay each minibatch m times."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    replay = config.free_at_replay
    model.to(device)
    delta_global = None
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            bs = xb.shape[0]
            if delta_global is None or delta_global.shape != xb.shape:
                delta_global = torch.zeros_like(xb)
            d = delta_global[:bs]
            for _ in range(replay):
                d_var = d.clone().detach().requires_grad_(True)
                x_adv = clamp_numeric(xb + d_var * mask_t, xb, mins_t, maxs_t, mask_t)
                optimizer.zero_grad(set_to_none=True)
                loss = 0.5 * criterion(model(xb), yb) + 0.5 * criterion(model(x_adv), yb)
                loss.backward()
                optimizer.step()
                g = d_var.grad.detach() if d_var.grad is not None else torch.zeros_like(d)
                d = torch.clamp(d + config.adv_alpha * g.sign() * mask_t,
                                min=-config.adv_epsilon, max=config.adv_epsilon)
            delta_global[:bs] = d.detach()
    return model



# =============================================================================
# Optional extra defense methods
#
# These are NOT part of the six defenses evaluated in the paper. They are
# kept in full so the implementations remain available and reproducible, but
# they are only trained when explicitly requested:
#
#     --extra-methods progressive,sa_trades,dst_sa_trades
#
# With the default (empty) setting the code trains exactly the six defenses
# the paper reports, and nothing else.
# =============================================================================

def progressive_class_aware_masks(model, train_loader, device, numeric_mask,
                                  feature_names, ratios, max_batches):
    """Build a list of progressively wider masks from class-aware sensitivity.

    Returns list of (ratio, mask_np, n_features) for each stage.
    Sensitivity is computed once; only the top-k threshold changes.
    """
    criterion = nn.BCEWithLogitsLoss()
    grad_pos, grad_neg = [], []
    model.eval()
    for batch_idx, (xb, yb) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = xb.grad.detach().abs()
        pos_mask = yb > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            grad_pos.append(grads[pos_mask].mean(dim=0).cpu().numpy())
        if neg_mask.any():
            grad_neg.append(grads[neg_mask].mean(dim=0).cpu().numpy())
    scores_pos = np.mean(np.stack(grad_pos), axis=0) if grad_pos else np.zeros(len(feature_names))
    scores_neg = np.mean(np.stack(grad_neg), axis=0) if grad_neg else np.zeros(len(feature_names))
    numeric_indices = np.where(numeric_mask > 0)[0]

    masks = []
    for r in ratios:
        k = max(1, int(math.ceil(len(numeric_indices) * r)))
        ranked_pos = numeric_indices[np.argsort(scores_pos[numeric_indices])[::-1]][:k]
        ranked_neg = numeric_indices[np.argsort(scores_neg[numeric_indices])[::-1]][:k]
        selected = np.unique(np.concatenate([ranked_pos, ranked_neg]))
        m = np.zeros(len(feature_names), dtype=np.float32)
        m[selected] = 1.0
        masks.append((r, m, int(m.sum())))
    return masks


def train_model_progressive_class_aware(model, train_loader, epochs, device,
                                        config, masks_schedule, mins_t, maxs_t,
                                        minority_weight):
    """Progressive Class-Aware Constrained AT: mask expands across epochs.

    masks_schedule: list of (ratio, mask_np, n_features) from progressive_class_aware_masks.
    The training epochs are evenly divided among the stages.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    model.to(device)
    n_stages = len(masks_schedule)
    epochs_per_stage = max(1, epochs // n_stages)

    for stage_idx, (ratio, mask_np, n_feat) in enumerate(masks_schedule):
        mask_t = torch.from_numpy(mask_np).to(device)
        stage_epochs = epochs_per_stage if stage_idx < n_stages - 1 else (epochs - stage_idx * epochs_per_stage)
        for ep in range(stage_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                model.eval()
                x_adv = pgd_attack(model, xb, yb, config.adv_epsilon,
                                   config.adv_alpha, config.adv_steps,
                                   mask_t, mins_t, maxs_t)
                model.train()
                batch_x = torch.cat([xb, x_adv], dim=0)
                batch_y = torch.cat([yb, yb], dim=0)
                logits = model(batch_x)
                w = torch.where(batch_y > 0.5, minority_weight, 1.0)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y, weight=w)
                loss.backward()
                optimizer.step()
    return model


def compute_sensitivity_weights(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    numeric_mask: np.ndarray,
    max_batches: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute continuous per-feature sensitivity weights in [0, 1].

    Instead of a binary mask, each numeric feature gets a weight proportional
    to its gradient sensitivity raised to the power *gamma*.  Non-numeric
    features always receive weight 0.
    """
    criterion = nn.BCEWithLogitsLoss()
    grad_scores: list[np.ndarray] = []
    model.eval()
    for batch_idx, (xb, yb) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_scores.append(xb.grad.detach().abs().mean(dim=0).cpu().numpy())
    scores = np.mean(np.stack(grad_scores), axis=0)
    numeric_indices = np.where(numeric_mask > 0)[0]
    max_score = float(scores[numeric_indices].max()) + 1e-12
    weights = np.zeros_like(scores, dtype=np.float32)
    weights[numeric_indices] = (scores[numeric_indices] / max_score) ** gamma
    return weights


def pgd_attack_weighted(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps_weights: torch.Tensor,
    alpha_weights: torch.Tensor,
    steps: int,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    """PGD attack with per-feature epsilon budgets.

    *eps_weights*  – (dim,) tensor, the maximum perturbation allowed for each
    feature.  Features with weight 0 are never perturbed.
    *alpha_weights* – (dim,) tensor, per-step perturbation size per feature.
    """
    criterion = nn.BCEWithLogitsLoss()
    active_mask = (eps_weights > 0).float()
    x_adv = x.detach().clone()
    noise = torch.empty_like(x_adv).uniform_(-1.0, 1.0) * eps_weights
    x_adv = x_adv + noise
    x_adv = clamp_numeric(x_adv, x, mins, maxs, active_mask)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        step = alpha_weights * x_adv.grad.sign()
        x_adv = x_adv.detach() + step
        delta = x_adv - x
        delta = torch.max(torch.min(delta, eps_weights), -eps_weights)
        x_adv = x + delta
        x_adv = clamp_numeric(x_adv, x, mins, maxs, active_mask)
    return x_adv.detach()


def train_model_sa_trades(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    config,
    sensitivity_weights_np: np.ndarray,
    mins_t: torch.Tensor,
    maxs_t: torch.Tensor,
) -> nn.Module:
    """Sensitivity-Aware TRADES: per-feature epsilon + KL divergence loss.

    Combines TRADES' principled clean-robust decomposition with
    sensitivity-weighted per-feature perturbation budgets so that more
    vulnerable features receive proportionally larger adversarial pressure
    while less important features remain nearly unperturbed.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    criterion_bce = nn.BCEWithLogitsLoss()
    beta = config.trades_beta
    eps_weights = torch.from_numpy(
        sensitivity_weights_np * config.adv_epsilon,
    ).float().to(device)
    alpha_weights = torch.from_numpy(
        sensitivity_weights_np * config.adv_alpha,
    ).float().to(device)
    active_mask = (eps_weights > 0).float()
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_clean = model(xb)
            loss_bce = criterion_bce(logits_clean, yb)
            probs_clean = torch.sigmoid(logits_clean).detach()
            # Generate adv via weighted PGD on KL
            model.eval()
            x_adv = xb.detach().clone()
            noise = torch.empty_like(x_adv).uniform_(-1.0, 1.0) * eps_weights
            x_adv = clamp_numeric(x_adv + noise, xb, mins_t, maxs_t, active_mask)
            for _ in range(config.adv_steps):
                x_adv.requires_grad_(True)
                probs_adv = torch.sigmoid(model(x_adv))
                p = probs_clean.clamp(1e-7, 1 - 1e-7)
                q = probs_adv.clamp(1e-7, 1 - 1e-7)
                kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
                model.zero_grad(set_to_none=True)
                kl.sum().backward()
                step = alpha_weights * x_adv.grad.sign()
                x_adv = x_adv.detach() + step
                delta = torch.max(torch.min(x_adv - xb, eps_weights), -eps_weights)
                x_adv = clamp_numeric(xb + delta, xb, mins_t, maxs_t, active_mask)
            model.train()
            paf = torch.sigmoid(model(x_adv.detach())).clamp(1e-7, 1 - 1e-7)
            p = probs_clean.clamp(1e-7, 1 - 1e-7)
            kl_f = p * (p.log() - paf.log()) + (1 - p) * ((1 - p).log() - (1 - paf).log())
            loss = loss_bce + beta * kl_f.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def _quick_sensitivity_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> np.ndarray:
    """Lightweight sensitivity re-estimation (gradient abs mean)."""
    criterion = nn.BCEWithLogitsLoss()
    accum: list[np.ndarray] = []
    model.eval()
    for idx, (xb, yb) in enumerate(loader):
        if idx >= max_batches:
            break
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        loss = criterion(model(xb), yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        accum.append(xb.grad.detach().abs().mean(dim=0).cpu().numpy())
    return np.mean(np.stack(accum), axis=0)


def _scores_to_weights(
    scores: np.ndarray,
    numeric_mask: np.ndarray,
    gamma: float,
) -> np.ndarray:
    numeric_indices = np.where(numeric_mask > 0)[0]
    max_s = float(scores[numeric_indices].max()) + 1e-12
    w = np.zeros_like(scores, dtype=np.float32)
    w[numeric_indices] = (scores[numeric_indices] / max_s) ** gamma
    return w


def train_model_dst_sa_trades(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    config,
    initial_weights_np: np.ndarray,
    numeric_mask: np.ndarray,
    mins_t: torch.Tensor,
    maxs_t: torch.Tensor,
) -> nn.Module:
    """Dynamic Sensitivity Tracking + SA-TRADES.

    Re-estimates feature sensitivity every *dst_update_interval* epochs using
    an EMA update, then runs SA-TRADES with the refreshed per-feature
    perturbation budget.  This avoids the staleness of a one-shot frozen mask.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    criterion_bce = nn.BCEWithLogitsLoss()
    beta = config.trades_beta
    update_interval = config.dst_update_interval
    ema_alpha = config.dst_ema_alpha
    gamma = config.sa_trades_gamma
    current_weights = initial_weights_np.copy()
    model.to(device)
    for epoch in range(epochs):
        # --- Dynamic sensitivity re-estimation ---
        if epoch > 0 and epoch % update_interval == 0:
            new_scores = _quick_sensitivity_scores(
                model, train_loader, device, max_batches=config.sensitivity_batches,
            )
            new_weights = _scores_to_weights(new_scores, numeric_mask, gamma)
            current_weights = ema_alpha * new_weights + (1 - ema_alpha) * current_weights
        eps_weights = torch.from_numpy(current_weights * config.adv_epsilon).float().to(device)
        alpha_weights = torch.from_numpy(current_weights * config.adv_alpha).float().to(device)
        active_mask = (eps_weights > 0).float()
        # --- SA-TRADES training for this epoch ---
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_clean = model(xb)
            loss_bce = criterion_bce(logits_clean, yb)
            probs_clean = torch.sigmoid(logits_clean).detach()
            model.eval()
            x_adv = xb.detach().clone()
            noise = torch.empty_like(x_adv).uniform_(-1.0, 1.0) * eps_weights
            x_adv = clamp_numeric(x_adv + noise, xb, mins_t, maxs_t, active_mask)
            for _ in range(config.adv_steps):
                x_adv.requires_grad_(True)
                probs_adv = torch.sigmoid(model(x_adv))
                p = probs_clean.clamp(1e-7, 1 - 1e-7)
                q = probs_adv.clamp(1e-7, 1 - 1e-7)
                kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
                model.zero_grad(set_to_none=True)
                kl.sum().backward()
                step = alpha_weights * x_adv.grad.sign()
                x_adv = x_adv.detach() + step
                delta = torch.max(torch.min(x_adv - xb, eps_weights), -eps_weights)
                x_adv = clamp_numeric(xb + delta, xb, mins_t, maxs_t, active_mask)
            model.train()
            paf = torch.sigmoid(model(x_adv.detach())).clamp(1e-7, 1 - 1e-7)
            p = probs_clean.clamp(1e-7, 1 - 1e-7)
            kl_f = p * (p.log() - paf.log()) + (1 - p) * ((1 - p).log() - (1 - paf).log())
            loss = loss_bce + beta * kl_f.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def stratified_subset(x: np.ndarray, y: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= size:
        return x, y
    _, x_sub, _, y_sub = train_test_split(
        x,
        y,
        test_size=size,
        random_state=seed,
        stratify=y,
    )
    return x_sub, y_sub


def train_model_class_aware_constrained(model, train_loader, epochs, device,
                                        config, selected_mask_t, mins_t, maxs_t,
                                        minority_weight):
    """Class-aware constrained AT: weighted BCE favoring minority class."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            x_adv = pgd_attack(model, xb, yb, config.adv_epsilon,
                               config.adv_alpha, config.adv_steps,
                               selected_mask_t, mins_t, maxs_t)
            model.train()
            batch_x = torch.cat([xb, x_adv], dim=0)
            batch_y = torch.cat([yb, yb], dim=0)
            logits = model(batch_x)
            w = torch.where(batch_y > 0.5, minority_weight, 1.0)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, weight=w)
            loss.backward()
            optimizer.step()
    return model


def class_aware_sensitivity_mask(model, train_loader, device, numeric_mask,
                                 feature_names, top_ratio, max_batches):
    """Compute sensitivity per class, take union of top-k from each."""
    criterion = nn.BCEWithLogitsLoss()
    grad_pos, grad_neg = [], []
    model.eval()
    for batch_idx, (xb, yb) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = xb.grad.detach().abs()
        pos_mask = yb > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            grad_pos.append(grads[pos_mask].mean(dim=0).cpu().numpy())
        if neg_mask.any():
            grad_neg.append(grads[neg_mask].mean(dim=0).cpu().numpy())
    scores_pos = np.mean(np.stack(grad_pos), axis=0) if grad_pos else np.zeros(len(feature_names))
    scores_neg = np.mean(np.stack(grad_neg), axis=0) if grad_neg else np.zeros(len(feature_names))
    numeric_indices = np.where(numeric_mask > 0)[0]
    k = max(1, int(math.ceil(len(numeric_indices) * top_ratio)))
    # Top-k from each class
    ranked_pos = numeric_indices[np.argsort(scores_pos[numeric_indices])[::-1]][:k]
    ranked_neg = numeric_indices[np.argsort(scores_neg[numeric_indices])[::-1]][:k]
    selected = np.unique(np.concatenate([ranked_pos, ranked_neg]))
    mask = np.zeros(len(feature_names), dtype=np.float32)
    mask[selected] = 1.0
    scores_combined = np.maximum(scores_pos, scores_neg)
    table = pd.DataFrame({
        "feature": feature_names,
        "sensitivity_pos": scores_pos,
        "sensitivity_neg": scores_neg,
        "sensitivity_combined": scores_combined,
        "is_numeric": numeric_mask.astype(bool),
        "selected_class_aware": mask.astype(bool),
    }).sort_values("sensitivity_combined", ascending=False)
    return mask, table


def sensitivity_mask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    numeric_mask: np.ndarray,
    feature_names: list[str],
    top_ratio: float,
    max_batches: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    criterion = nn.BCEWithLogitsLoss()
    grad_scores = []
    model.eval()

    for batch_idx, (xb, yb) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_scores.append(xb.grad.detach().abs().mean(dim=0).cpu().numpy())

    scores = np.mean(np.stack(grad_scores), axis=0)
    numeric_indices = np.where(numeric_mask > 0)[0]
    k = max(1, int(math.ceil(len(numeric_indices) * top_ratio)))
    ranked_numeric = numeric_indices[np.argsort(scores[numeric_indices])[::-1]]
    selected = ranked_numeric[:k]
    mask = np.zeros_like(scores, dtype=np.float32)
    mask[selected] = 1.0

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "sensitivity_score": scores,
            "is_numeric": numeric_mask.astype(bool),
            "selected_for_constrained_training": mask.astype(bool),
        }
    ).sort_values("sensitivity_score", ascending=False)
    return mask, table


# ---------------------------------------------------------------------------
# SA-TRADES & DST-AT: new method innovations
# ---------------------------------------------------------------------------



def generate_attack_outputs(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_name: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    device: torch.device,
    batch_size: int,
    return_inputs: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask_t = torch.from_numpy(attack_mask.astype(np.float32)).to(device)
    mins_t = torch.from_numpy(mins.astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(maxs.astype(np.float32)).to(device)
    eval_loader = DataLoader(
        TensorDataset(numpy_to_torch(x_eval), numpy_to_torch(y_eval)),
        batch_size=batch_size,
        shuffle=False,
    )
    input_batches = []
    prob_batches = []

    model.eval()
    for xb_np, yb_np in eval_loader:
        xb = xb_np.to(device)
        yb = yb_np.to(device)
        if attack_name == "clean":
            x_adv = xb
        elif attack_name == "fgsm":
            x_adv = fgsm_attack(model, xb, yb, epsilon, mask_t, mins_t, maxs_t)
        elif attack_name == "pgd":
            alpha = max(epsilon * CONFIG.eval_pgd_alpha_ratio, 1e-4)
            x_adv = pgd_attack(model, xb, yb, epsilon, alpha, CONFIG.adv_steps, mask_t, mins_t, maxs_t)
        elif attack_name == "cw":
            x_adv = cw_attack(model, xb, yb, mask_t, mins_t, maxs_t,
                              steps=CONFIG.cw_steps, lr=CONFIG.cw_lr, c_const=CONFIG.cw_c)
        elif attack_name == "apgd":
            x_adv = apgd_attack(model, xb, yb, epsilon, CONFIG.apgd_steps,
                                mask_t, mins_t, maxs_t, rho=CONFIG.apgd_rho)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        if return_inputs:
            input_batches.append(x_adv.detach().cpu().numpy())
        with torch.no_grad():
            prob_batches.append(torch.sigmoid(model(x_adv)).cpu().numpy())

    attacked_inputs = np.concatenate(input_batches) if return_inputs else None
    attacked_probs = np.concatenate(prob_batches)
    return attacked_inputs, attacked_probs


def evaluate_single_attack_setting(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_name: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    clean_probs = predict_probabilities_from_array(model, x_eval, batch_size=batch_size, device=device)
    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    _, attacked_probs = generate_attack_outputs(
        model=model,
        x_eval=x_eval,
        y_eval=y_eval,
        attack_name=attack_name,
        epsilon=epsilon,
        attack_mask=attack_mask,
        mins=mins,
        maxs=maxs,
        device=device,
        batch_size=batch_size,
        return_inputs=False,
    )
    clean_metrics = compute_metrics(y_eval, clean_probs)
    attack_metrics = compute_metrics(y_eval, attacked_probs, clean_pred=clean_pred)
    return {
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_recall": clean_metrics["recall"],
        "clean_f1": clean_metrics["f1"],
        "robust_accuracy": attack_metrics["accuracy"],
        "robust_recall": attack_metrics["recall"],
        "robust_f1": attack_metrics["f1"],
        "attack_success_rate": attack_metrics["attack_success_rate"],
    }


def evaluate_transferability(
    source_name: str,
    source_model: nn.Module,
    target_models: dict[str, object],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_name: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    attacked_inputs, _ = generate_attack_outputs(
        model=source_model,
        x_eval=x_eval,
        y_eval=y_eval,
        attack_name=attack_name,
        epsilon=epsilon,
        attack_mask=attack_mask,
        mins=mins,
        maxs=maxs,
        device=device,
        batch_size=batch_size,
        return_inputs=True,
    )
    rows = []
    for target_name, target_model in target_models.items():
        clean_probs = predict_probabilities_any(target_model, x_eval, batch_size=batch_size, device=device)
        clean_pred = (clean_probs >= 0.5).astype(np.int32)
        attacked_probs = predict_probabilities_any(target_model, attacked_inputs, batch_size=batch_size, device=device)
        rows.append(
            {
                "source_model": source_name,
                "target_model": target_name,
                "attack": attack_name,
                "epsilon": epsilon,
                **compute_metrics(y_eval, attacked_probs, clean_pred=clean_pred),
            }
        )
    return pd.DataFrame(rows)


def evaluate_attack_categories(
    model_name: str,
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_categories: np.ndarray,
    selected_categories: list[str],
    attack_name: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    clean_probs = predict_probabilities_from_array(model, x_eval, batch_size=batch_size, device=device)
    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    _, attacked_probs = generate_attack_outputs(
        model=model,
        x_eval=x_eval,
        y_eval=y_eval,
        attack_name=attack_name,
        epsilon=epsilon,
        attack_mask=attack_mask,
        mins=mins,
        maxs=maxs,
        device=device,
        batch_size=batch_size,
        return_inputs=False,
    )
    attacked_pred = (attacked_probs >= 0.5).astype(np.int32)
    rows = []
    for category in selected_categories:
        category_mask = (attack_categories == category) & (y_eval == 1)
        sample_count = int(category_mask.sum())
        if sample_count == 0:
            continue
        clean_recall = float((clean_pred[category_mask] == 1).mean())
        adv_recall = float((attacked_pred[category_mask] == 1).mean())
        rows.append(
            {
                "model": model_name,
                "attack": attack_name,
                "epsilon": epsilon,
                "attack_cat": category,
                "samples": sample_count,
                "clean_recall": clean_recall,
                "adv_recall": adv_recall,
                "recall_drop": clean_recall - adv_recall,
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    epsilon_list: Iterable[float],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    clean_loader = make_loader(x_eval, y_eval, batch_size=batch_size, shuffle=False)
    clean_probs = predict_probabilities(model, clean_loader, device)
    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    rows = [{
        "attack": "clean",
        "epsilon": 0.0,
        **compute_metrics(y_eval, clean_probs),
    }]
    for attack_name in ["fgsm", "pgd", "cw", "apgd"]:
        for epsilon in epsilon_list:
            _, probs_adv = generate_attack_outputs(
                model=model,
                x_eval=x_eval,
                y_eval=y_eval,
                attack_name=attack_name,
                epsilon=epsilon,
                attack_mask=attack_mask,
                mins=mins,
                maxs=maxs,
                device=device,
                batch_size=batch_size,
                return_inputs=False,
            )
            rows.append(
                {
                    "attack": attack_name,
                    "epsilon": epsilon,
                    **compute_metrics(y_eval, probs_adv, clean_pred=clean_pred),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_curve(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for attack_name in ["fgsm", "pgd"]:
        attack_df = df[df["attack"] == attack_name]
        for model_name in attack_df["model"].unique():
            subset = attack_df[attack_df["model"] == model_name].sort_values("epsilon")
            plt.plot(subset["epsilon"], subset[metric], marker="o", label=f"{model_name}-{attack_name.upper()}")
    plt.xlabel("Epsilon")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} under adversarial perturbations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ablation(df: pd.DataFrame, output_path: Path) -> None:
    clean = df[df["attack"] == "clean"].groupby("model", as_index=False)["f1"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(clean["model"], clean["f1"], color=["#4472C4", "#ED7D31", "#70AD47"])
    plt.ylim(0, 1)
    plt.ylabel("F1")
    plt.title("Clean-set F1 comparison across training strategies")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_transfer_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No transfer results for selected filter", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    pivot = df.pivot(index="source_model", columns="target_model", values="f1")
    row_labels = list(pivot.index)
    col_labels = list(pivot.columns)
    matrix = pivot.to_numpy()

    fig_width = max(7, 1.25 * len(col_labels) + 2)
    fig_height = max(5, 1.0 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=matrix.min(), vmax=matrix.max())
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Target model")
    ax.set_ylabel("Source model")
    ax.set_title("Transfer-attack F1 heatmap")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ratio_ablation(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("ratio")
    plt.figure(figsize=(7, 5))
    plt.plot(plot_df["ratio"], plot_df["clean_f1"], marker="o", label="Clean F1")
    plt.plot(plot_df["ratio"], plot_df["robust_f1"], marker="s", label="Robust F1")
    plt.xlabel("Sensitivity top ratio")
    plt.ylabel("F1")
    plt.title("Constrained adversarial training ratio ablation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_efficiency_tradeoff(efficiency_df: pd.DataFrame, result_df: pd.DataFrame, output_path: Path) -> None:
    robust_df = result_df[(result_df["attack"] == "pgd") & (np.isclose(result_df["epsilon"], 0.1))][["model", "f1"]]
    robust_df = robust_df.rename(columns={"f1": "robust_f1"})
    clean_df = result_df[result_df["attack"] == "clean"][["model", "f1"]].rename(columns={"f1": "clean_f1"})
    plot_df = efficiency_df.merge(robust_df, on="model").merge(clean_df, on="model")

    colors = {
        "standard": "#4472C4",
        "adv_training": "#ED7D31",
        "constrained_adv": "#70AD47",
    }

    plt.figure(figsize=(7, 5))
    for row in plot_df.itertuples():
        plt.scatter(
            row.train_seconds,
            row.robust_f1,
            s=180,
            color=colors.get(row.model, "#5B9BD5"),
            alpha=0.9,
        )
        plt.annotate(
            f"{row.model}\nclean={row.clean_f1:.3f}",
            (row.train_seconds, row.robust_f1),
            xytext=(6, 6),
            textcoords="offset points",
        )
    plt.xlabel("Training time (s)")
    plt.ylabel("Robust F1 under PGD, ε=0.10")
    plt.title("Robustness-efficiency trade-off")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_results(
    results: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_cols is None:
        group_cols = ["model", "attack", "epsilon"]
    metric_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    mean_df = results.groupby(group_cols, as_index=False)[metric_cols].mean()
    std_df = results.groupby(group_cols, as_index=False)[metric_cols].std().fillna(0.0)
    return mean_df, std_df


def run_experiment(config: ExperimentConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[run] output_dir={output_dir} device={config.device} "
        f"training_budget_mode={config.training_budget_mode} seeds={config.seeds}",
        flush=True,
    )
    device = torch.device(config.device)
    print(f"[run] resolved_device={device}", flush=True)

    train_df, test_df = load_unsw_split(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)

    eval_indices = stratified_subset_indices(y_test, config.eval_attack_rows, seed=2026)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    eval_attack_categories = test_df["attack_cat"].fillna("Unknown").to_numpy()[eval_indices]
    top_attack_categories = (
        pd.Series(eval_attack_categories[eval_y == 1])
        .value_counts()
        .head(config.top_attack_categories)
        .index.tolist()
    )

    all_results = []
    sensitivity_tables = []
    transfer_results = []
    reference_clean_results = []
    category_results = []
    ratio_results = []
    validity_results = []
    efficiency_results = []
    full_test_clean_results = []

    for seed in config.seeds:
        set_seed(seed)
        matched_budget_mode = config.training_budget_mode == "matched_continuation"
        print(f"[seed {seed}] started", flush=True)

        train_loader = make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        input_dim = x_train.shape[1]
        mins_t = torch.from_numpy(metadata["numeric_mins"].astype(np.float32)).to(device)
        maxs_t = torch.from_numpy(metadata["numeric_maxs"].astype(np.float32)).to(device)
        numeric_mask = metadata["numeric_mask"]
        numeric_mask_t = torch.from_numpy(numeric_mask.astype(np.float32)).to(device)

        baseline_model = MLP(input_dim, config.hidden_dims, config.dropout)
        baseline_start = time.perf_counter()
        baseline_model = train_model(
            baseline_model,
            train_loader,
            epochs=config.baseline_epochs,
            device=device,
            adv_training=False,
            adv_builder=None,
        )
        baseline_train_seconds = time.perf_counter() - baseline_start
        print(f"[seed {seed}] baseline done in {baseline_train_seconds:.2f}s", flush=True)

        standard_model = baseline_model
        standard_continuation_seconds = 0.0
        if matched_budget_mode:
            standard_model = copy.deepcopy(baseline_model)
            standard_start = time.perf_counter()
            standard_model = train_model(
                standard_model,
                train_loader,
                epochs=config.adv_epochs,
                device=device,
                adv_training=False,
                adv_builder=None,
            )
            standard_continuation_seconds = time.perf_counter() - standard_start
            print(
                f"[seed {seed}] matched standard continuation done in "
                f"{standard_continuation_seconds:.2f}s",
                flush=True,
            )

        standard_adv_model = copy.deepcopy(baseline_model)
        adv_builder = lambda model, xb, yb: pgd_attack(
            model,
            xb,
            yb,
            config.adv_epsilon,
            config.adv_alpha,
            config.adv_steps,
            numeric_mask_t,
            mins_t,
            maxs_t,
        )
        standard_adv_start = time.perf_counter()
        standard_adv_model = train_model(
            standard_adv_model,
            train_loader,
            epochs=config.adv_epochs,
            device=device,
            adv_training=True,
            adv_builder=adv_builder,
        )
        standard_adv_train_seconds = time.perf_counter() - standard_adv_start
        print(f"[seed {seed}] adv_training done in {standard_adv_train_seconds:.2f}s", flush=True)

        selected_mask_start = time.perf_counter()
        selected_mask, sensitivity_table = sensitivity_mask(
            baseline_model,
            train_loader,
            device=device,
            numeric_mask=numeric_mask,
            feature_names=metadata["feature_names"],
            top_ratio=config.sensitivity_top_ratio,
            max_batches=config.sensitivity_batches,
        )
        selected_mask_seconds = time.perf_counter() - selected_mask_start
        print(
            f"[seed {seed}] constrained mask built in {selected_mask_seconds:.2f}s "
            f"(features={int(selected_mask.sum())})",
            flush=True,
        )
        sensitivity_table.insert(0, "seed", seed)
        sensitivity_tables.append(sensitivity_table)
        selected_mask_t = torch.from_numpy(selected_mask.astype(np.float32)).to(device)

        constrained_model = copy.deepcopy(baseline_model)
        constrained_adv_builder = lambda model, xb, yb: pgd_attack(
            model,
            xb,
            yb,
            config.adv_epsilon,
            config.adv_alpha,
            config.adv_steps,
            selected_mask_t,
            mins_t,
            maxs_t,
        )
        constrained_start = time.perf_counter()
        constrained_model = train_model(
            constrained_model,
            train_loader,
            epochs=config.adv_epochs,
            device=device,
            adv_training=True,
            adv_builder=constrained_adv_builder,
        )
        constrained_train_seconds = time.perf_counter() - constrained_start
        print(f"[seed {seed}] constrained_adv done in {constrained_train_seconds:.2f}s", flush=True)

        # --- TRADES ---
        trades_model = copy.deepcopy(baseline_model)
        trades_start = time.perf_counter()
        trades_model = train_model_trades(
            trades_model, train_loader, config.adv_epochs, device, config,
            numeric_mask_t, mins_t, maxs_t,
        )
        trades_train_seconds = time.perf_counter() - trades_start
        print(f"[seed {seed}] trades done in {trades_train_seconds:.2f}s", flush=True)

        # --- Free AT ---
        free_at_model = copy.deepcopy(baseline_model)
        free_at_start = time.perf_counter()
        free_at_model = train_model_free_at(
            free_at_model, train_loader, config.adv_epochs, device, config,
            numeric_mask_t, mins_t, maxs_t,
        )
        free_at_train_seconds = time.perf_counter() - free_at_start
        print(f"[seed {seed}] free_at done in {free_at_train_seconds:.2f}s", flush=True)

        # --- Class-Aware Constrained AT ---
        ca_mask_start = time.perf_counter()
        ca_mask, _ = class_aware_sensitivity_mask(
            baseline_model, train_loader, device=device,
            numeric_mask=numeric_mask,
            feature_names=metadata["feature_names"],
            top_ratio=config.sensitivity_top_ratio,
            max_batches=config.sensitivity_batches,
        )
        ca_mask_seconds = time.perf_counter() - ca_mask_start
        print(
            f"[seed {seed}] class-aware mask built in {ca_mask_seconds:.2f}s "
            f"(features={int(ca_mask.sum())})",
            flush=True,
        )
        ca_mask_t = torch.from_numpy(ca_mask.astype(np.float32)).to(device)
        class_aware_model = copy.deepcopy(baseline_model)
        class_aware_start = time.perf_counter()
        class_aware_model = train_model_class_aware_constrained(
            class_aware_model, train_loader, config.adv_epochs, device, config,
            ca_mask_t, mins_t, maxs_t,
            minority_weight=config.class_aware_minority_weight,
        )
        class_aware_train_seconds = time.perf_counter() - class_aware_start
        print(f"[seed {seed}] class_aware_constrained done in {class_aware_train_seconds:.2f}s", flush=True)

        # ---- Optional extra methods ------------------------------------------
        # Nothing in this section runs unless the method is named in
        # --extra-methods. The six defenses the paper reports are unaffected.
        # Each extra method is registered exactly like the six, so it flows
        # through the same evaluation, cost and selection code paths.
        extra_cost_registry: dict[str, object] = {}
        extra_model_registry: dict[str, object] = {}
        extra_perturb_registry: dict[str, float] = {}
        extra_feature_registry: dict[str, int] = {}

        if "progressive" in config.extra_methods:
            prog_masks = progressive_class_aware_masks(
                baseline_model, train_loader, device=device,
                numeric_mask=numeric_mask,
                feature_names=metadata["feature_names"],
                ratios=config.progressive_ratios,
                max_batches=config.sensitivity_batches,
            )
            progressive_model = copy.deepcopy(baseline_model)
            _t0 = time.perf_counter()
            progressive_model = train_model_progressive_class_aware(
                progressive_model, train_loader, config.adv_epochs, device, config,
                prog_masks, mins_t, maxs_t,
                minority_weight=config.class_aware_minority_weight,
            )
            _dt = time.perf_counter() - _t0
            print(f"[seed {seed}] progressive_class_aware done in {_dt:.2f}s", flush=True)
            extra_model_registry["progressive_class_aware"] = progressive_model
            extra_cost_registry["progressive_class_aware"] = summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=_dt,
                include_pretrain_in_total=matched_budget_mode,
            )
            extra_perturb_registry["progressive_class_aware"] = float(config.progressive_ratios[-1])
            extra_feature_registry["progressive_class_aware"] = int(prog_masks[-1][2])

        if {"sa_trades", "dst_sa_trades"} & set(config.extra_methods):
            sa_weights = compute_sensitivity_weights(
                baseline_model, train_loader, device=device,
                numeric_mask=numeric_mask,
                max_batches=config.sensitivity_batches,
                gamma=config.sa_trades_gamma,
            )
            sa_perturb = float(np.sum(sa_weights > 0) / max(numeric_mask.sum(), 1.0))
            sa_features = int(np.sum(sa_weights > 0))

            if "sa_trades" in config.extra_methods:
                sa_trades_model = copy.deepcopy(baseline_model)
                _t0 = time.perf_counter()
                sa_trades_model = train_model_sa_trades(
                    sa_trades_model, train_loader, config.adv_epochs, device, config,
                    sa_weights, mins_t, maxs_t,
                )
                _dt = time.perf_counter() - _t0
                print(f"[seed {seed}] sa_trades done in {_dt:.2f}s", flush=True)
                extra_model_registry["sa_trades"] = sa_trades_model
                extra_cost_registry["sa_trades"] = summarize_training_budget(
                    pretrain_seconds=baseline_train_seconds,
                    continuation_seconds=_dt,
                    include_pretrain_in_total=matched_budget_mode,
                )
                extra_perturb_registry["sa_trades"] = sa_perturb
                extra_feature_registry["sa_trades"] = sa_features

            if "dst_sa_trades" in config.extra_methods:
                dst_model = copy.deepcopy(baseline_model)
                _t0 = time.perf_counter()
                dst_model = train_model_dst_sa_trades(
                    dst_model, train_loader, config.adv_epochs, device, config,
                    sa_weights.copy(), numeric_mask, mins_t, maxs_t,
                )
                _dt = time.perf_counter() - _t0
                print(f"[seed {seed}] dst_sa_trades done in {_dt:.2f}s", flush=True)
                extra_model_registry["dst_sa_trades"] = dst_model
                extra_cost_registry["dst_sa_trades"] = summarize_training_budget(
                    pretrain_seconds=baseline_train_seconds,
                    continuation_seconds=_dt,
                    include_pretrain_in_total=matched_budget_mode,
                )
                extra_perturb_registry["dst_sa_trades"] = sa_perturb
                extra_feature_registry["dst_sa_trades"] = sa_features

        if extra_model_registry:
            print(f"[seed {seed}] extra methods trained: {sorted(extra_model_registry)}", flush=True)


        reference_model_registry, reference_train_time_registry = train_reference_models(
            x_train=x_train,
            y_train=y_train,
            seed=seed,
            enabled_models=config.reference_models,
        )
        print(f"[seed {seed}] reference models done", flush=True)

        train_cost_registry = {
            "standard": summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=standard_continuation_seconds,
                include_pretrain_in_total=True,
            ),
            "adv_training": summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=standard_adv_train_seconds,
                include_pretrain_in_total=matched_budget_mode,
            ),
            "constrained_adv": summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=constrained_train_seconds,
                mask_build_seconds=selected_mask_seconds,
                include_pretrain_in_total=matched_budget_mode,
                include_mask_in_total=matched_budget_mode,
            ),
            "trades": summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=trades_train_seconds,
                include_pretrain_in_total=matched_budget_mode,
            ),
            "free_at": summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=free_at_train_seconds,
                include_pretrain_in_total=matched_budget_mode,
            ),
            "class_aware_constrained": summarize_training_budget(
                pretrain_seconds=baseline_train_seconds,
                continuation_seconds=class_aware_train_seconds,
                mask_build_seconds=ca_mask_seconds,
                include_pretrain_in_total=matched_budget_mode,
                include_mask_in_total=matched_budget_mode,
            ),
        }
        model_registry = {
            "standard": standard_model,
            "adv_training": standard_adv_model,
            "constrained_adv": constrained_model,
            "trades": trades_model,
            "free_at": free_at_model,
            "class_aware_constrained": class_aware_model,
        }
        train_time_registry = {
            model_name: details["train_seconds"]
            for model_name, details in train_cost_registry.items()
        }
        ca_ratio = float(ca_mask.sum() / max(numeric_mask.sum(), 1.0))
        perturb_ratio_registry = {
            "standard": 0.0,
            "adv_training": 1.0,
            "constrained_adv": float(selected_mask.sum() / max(numeric_mask.sum(), 1.0)),
            "trades": 1.0,
            "free_at": 1.0,
            "class_aware_constrained": ca_ratio,
        }
        selected_feature_registry = {
            "standard": 0,
            "adv_training": int(numeric_mask.sum()),
            "constrained_adv": int(selected_mask.sum()),
            "trades": int(numeric_mask.sum()),
            "free_at": int(numeric_mask.sum()),
            "class_aware_constrained": int(ca_mask.sum()),
        }
        # Fold in the optional extra methods, if any were trained.
        train_cost_registry.update(extra_cost_registry)
        model_registry.update(extra_model_registry)
        perturb_ratio_registry.update(extra_perturb_registry)
        selected_feature_registry.update(extra_feature_registry)
        transfer_target_registry = {**model_registry, **reference_model_registry}
        print(f"[seed {seed}] evaluation started", flush=True)

        for model_name, model in model_registry.items():
            clean_probs = predict_probabilities_from_array(model, eval_x, batch_size=config.batch_size, device=device)
            clean_pred = (clean_probs >= 0.5).astype(np.int32)
            efficiency_results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "parameter_count": count_parameters(model),
                    "train_seconds": train_time_registry[model_name],
                    "pretrain_seconds": train_cost_registry[model_name]["pretrain_seconds"],
                    "continuation_seconds": train_cost_registry[model_name]["continuation_seconds"],
                    "mask_build_seconds": train_cost_registry[model_name]["mask_build_seconds"],
                    "training_attack_feature_ratio": perturb_ratio_registry[model_name],
                    "training_attack_feature_count": selected_feature_registry[model_name],
                    **measure_inference_efficiency(
                        model=model,
                        x_eval=eval_x,
                        batch_size=config.batch_size,
                        device=device,
                    ),
                }
            )

            result_df = evaluate_model(
                model=model,
                x_eval=eval_x,
                y_eval=eval_y,
                attack_mask=numeric_mask,
                mins=metadata["numeric_mins"],
                maxs=metadata["numeric_maxs"],
                epsilon_list=config.epsilon_list,
                device=device,
                batch_size=config.batch_size,
            )
            result_df.insert(0, "seed", seed)
            result_df.insert(1, "model", model_name)
            all_results.append(result_df)

            # Full test set clean evaluation
            full_test_probs = predict_probabilities_from_array(
                model, x_test, batch_size=config.batch_size, device=device)
            full_test_metrics = compute_metrics(y_test, full_test_probs)
            full_test_clean_results.append({
                "seed": seed, "model": model_name, "subset": "full_test",
                **full_test_metrics,
            })

            for attack_name, epsilon in config.validity_attack_settings:
                attacked_inputs, attacked_probs = generate_attack_outputs(
                    model=model,
                    x_eval=eval_x,
                    y_eval=eval_y,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    attack_mask=numeric_mask,
                    mins=metadata["numeric_mins"],
                    maxs=metadata["numeric_maxs"],
                    device=device,
                    batch_size=config.batch_size,
                    return_inputs=True,
                )
                validity_results.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "attack": attack_name,
                        "epsilon": epsilon,
                        **compute_metrics(
                            eval_y,
                            attacked_probs,
                            clean_pred=clean_pred,
                        ),
                        **compute_attack_validity_metrics(
                            x_clean=eval_x,
                            x_adv=attacked_inputs,
                            numeric_mask=numeric_mask,
                            numeric_mins=metadata["numeric_mins"],
                            numeric_maxs=metadata["numeric_maxs"],
                        ),
                    }
                )

            category_df = evaluate_attack_categories(
                model_name=model_name,
                model=model,
                x_eval=eval_x,
                y_eval=eval_y,
                attack_categories=eval_attack_categories,
                selected_categories=top_attack_categories,
                attack_name=config.category_attack,
                epsilon=config.category_epsilon,
                attack_mask=numeric_mask,
                mins=metadata["numeric_mins"],
                maxs=metadata["numeric_maxs"],
                device=device,
                batch_size=config.batch_size,
            )
            category_df.insert(0, "seed", seed)
            category_results.append(category_df)

        for reference_name, reference_model in reference_model_registry.items():
            reference_probs = predict_probabilities_any(
                reference_model,
                eval_x,
                batch_size=config.batch_size,
                device=device,
            )
            reference_clean_results.append(
                {
                    "seed": seed,
                    "model": reference_name,
                    **compute_metrics(eval_y, reference_probs),
                    "train_seconds": reference_train_time_registry[reference_name],
                    **measure_inference_efficiency_any(
                        model=reference_model,
                        x_eval=eval_x,
                        batch_size=config.batch_size,
                        device=device,
                    ),
                }
            )

        for attack_name, epsilon in config.transfer_attack_settings:
            for source_name, source_model in model_registry.items():
                transfer_df = evaluate_transferability(
                    source_name=source_name,
                    source_model=source_model,
                    target_models=transfer_target_registry,
                    x_eval=eval_x,
                    y_eval=eval_y,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    attack_mask=numeric_mask,
                    mins=metadata["numeric_mins"],
                    maxs=metadata["numeric_maxs"],
                    device=device,
                    batch_size=config.batch_size,
                )
                transfer_df.insert(0, "seed", seed)
                transfer_results.append(transfer_df)

        for ratio in config.sensitivity_ratio_list:
            if math.isclose(ratio, config.sensitivity_top_ratio):
                ratio_model = constrained_model
                ratio_mask = selected_mask
            else:
                ratio_mask, _ = sensitivity_mask(
                    baseline_model,
                    train_loader,
                    device=device,
                    numeric_mask=numeric_mask,
                    feature_names=metadata["feature_names"],
                    top_ratio=ratio,
                    max_batches=config.sensitivity_batches,
                )
                ratio_mask_t = torch.from_numpy(ratio_mask.astype(np.float32)).to(device)
                ratio_model = copy.deepcopy(baseline_model)
                ratio_adv_builder = lambda model, xb, yb, local_mask=ratio_mask_t: pgd_attack(
                    model,
                    xb,
                    yb,
                    config.adv_epsilon,
                    config.adv_alpha,
                    config.adv_steps,
                    local_mask,
                    mins_t,
                    maxs_t,
                )
                ratio_model = train_model(
                    ratio_model,
                    train_loader,
                    epochs=config.adv_epochs,
                    device=device,
                    adv_training=True,
                    adv_builder=ratio_adv_builder,
                )

            ratio_metrics = evaluate_single_attack_setting(
                model=ratio_model,
                x_eval=eval_x,
                y_eval=eval_y,
                attack_name=config.ratio_attack,
                epsilon=config.ratio_attack_epsilon,
                attack_mask=numeric_mask,
                mins=metadata["numeric_mins"],
                maxs=metadata["numeric_maxs"],
                device=device,
                batch_size=config.batch_size,
            )
            ratio_results.append(
                {
                    "seed": seed,
                    "ratio": ratio,
                    "selected_feature_count": int(ratio_mask.sum()),
                    **ratio_metrics,
                }
            )
        print(f"[seed {seed}] completed", flush=True)

    results_df = pd.concat(all_results, ignore_index=True)
    sensitivity_df = pd.concat(sensitivity_tables, ignore_index=True)
    mean_df, std_df = summarize_results(results_df)
    transfer_df = pd.concat(transfer_results, ignore_index=True)
    transfer_mean_df, transfer_std_df = summarize_results(
        transfer_df,
        group_cols=["source_model", "target_model", "attack", "epsilon"],
    )
    reference_clean_df = pd.DataFrame(reference_clean_results)
    reference_metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "attack_success_rate",
        "train_seconds",
        "inference_probe_rows",
        "inference_seconds",
        "inference_ms_per_sample",
        "inference_samples_per_second",
    ]
    reference_clean_mean_df = reference_clean_df.groupby(["model"], as_index=False)[reference_metric_cols].mean()
    reference_clean_std_df = reference_clean_df.groupby(["model"], as_index=False)[reference_metric_cols].std().fillna(0.0)
    reference_transfer_mean_df = transfer_mean_df[transfer_mean_df["target_model"].isin(config.reference_models)].copy()
    reference_transfer_std_df = transfer_std_df[transfer_std_df["target_model"].isin(config.reference_models)].copy()
    category_df = pd.concat(category_results, ignore_index=True)
    category_mean_df = (
        category_df.groupby(["model", "attack", "epsilon", "attack_cat"], as_index=False)
        .agg(
            samples=("samples", "mean"),
            clean_recall=("clean_recall", "mean"),
            adv_recall=("adv_recall", "mean"),
            recall_drop=("recall_drop", "mean"),
        )
        .sort_values(["model", "recall_drop"], ascending=[True, False])
    )
    ratio_df = pd.DataFrame(ratio_results)
    category_mean_df["samples"] = category_mean_df["samples"].round().astype(int)
    ratio_mean_df = ratio_df.groupby(["ratio", "selected_feature_count"], as_index=False).agg(
        clean_accuracy=("clean_accuracy", "mean"),
        clean_recall=("clean_recall", "mean"),
        clean_f1=("clean_f1", "mean"),
        robust_accuracy=("robust_accuracy", "mean"),
        robust_recall=("robust_recall", "mean"),
        robust_f1=("robust_f1", "mean"),
        attack_success_rate=("attack_success_rate", "mean"),
    )
    validity_df = pd.DataFrame(validity_results)
    validity_mean_df = (
        validity_df.groupby(["model", "attack", "epsilon"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            attack_success_rate=("attack_success_rate", "mean"),
            numeric_valid_rate=("numeric_valid_rate", "mean"),
            protected_integrity_rate=("protected_integrity_rate", "mean"),
            overall_validity_rate=("overall_validity_rate", "mean"),
            mean_numeric_abs_delta=("mean_numeric_abs_delta", "mean"),
            mean_numeric_l2_delta=("mean_numeric_l2_delta", "mean"),
            max_numeric_abs_delta=("max_numeric_abs_delta", "mean"),
            changed_numeric_feature_ratio=("changed_numeric_feature_ratio", "mean"),
            protected_feature_change_ratio=("protected_feature_change_ratio", "mean"),
            boundary_clip_ratio=("boundary_clip_ratio", "mean"),
        )
    )
    efficiency_df = pd.DataFrame(efficiency_results)
    efficiency_mean_df = (
        efficiency_df.groupby(["model"], as_index=False)
        .agg(
            parameter_count=("parameter_count", "mean"),
            train_seconds=("train_seconds", "mean"),
            pretrain_seconds=("pretrain_seconds", "mean"),
            continuation_seconds=("continuation_seconds", "mean"),
            mask_build_seconds=("mask_build_seconds", "mean"),
            training_attack_feature_ratio=("training_attack_feature_ratio", "mean"),
            training_attack_feature_count=("training_attack_feature_count", "mean"),
            inference_probe_rows=("inference_probe_rows", "mean"),
            inference_seconds=("inference_seconds", "mean"),
            inference_ms_per_sample=("inference_ms_per_sample", "mean"),
            inference_samples_per_second=("inference_samples_per_second", "mean"),
        )
    )
    standard_train_seconds = float(
        efficiency_mean_df.loc[efficiency_mean_df["model"] == "standard", "train_seconds"].iloc[0]
    )
    standard_inference_ms = float(
        efficiency_mean_df.loc[efficiency_mean_df["model"] == "standard", "inference_ms_per_sample"].iloc[0]
    )
    efficiency_mean_df["relative_train_cost_vs_standard"] = efficiency_mean_df["train_seconds"] / standard_train_seconds
    efficiency_mean_df["relative_inference_latency_vs_standard"] = (
        efficiency_mean_df["inference_ms_per_sample"] / standard_inference_ms
    )

    results_df.to_csv(output_dir / "raw_results.csv", index=False)
    mean_df.to_csv(output_dir / "mean_results.csv", index=False)
    std_df.to_csv(output_dir / "std_results.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_scores.csv", index=False)
    transfer_df.to_csv(output_dir / "transfer_raw_results.csv", index=False)
    transfer_mean_df.to_csv(output_dir / "transfer_mean_results.csv", index=False)
    transfer_std_df.to_csv(output_dir / "transfer_std_results.csv", index=False)
    reference_clean_df.to_csv(output_dir / "reference_model_clean_raw.csv", index=False)
    reference_clean_mean_df.to_csv(output_dir / "reference_model_clean_mean.csv", index=False)
    reference_clean_std_df.to_csv(output_dir / "reference_model_clean_std.csv", index=False)
    reference_transfer_mean_df.to_csv(output_dir / "reference_transfer_mean_results.csv", index=False)
    reference_transfer_std_df.to_csv(output_dir / "reference_transfer_std_results.csv", index=False)
    category_df.to_csv(output_dir / "category_raw_results.csv", index=False)
    category_mean_df.to_csv(output_dir / "category_mean_results.csv", index=False)
    ratio_df.to_csv(output_dir / "ratio_ablation_raw.csv", index=False)
    ratio_mean_df.to_csv(output_dir / "ratio_ablation_mean.csv", index=False)
    validity_df.to_csv(output_dir / "attack_validity_raw.csv", index=False)
    validity_mean_df.to_csv(output_dir / "attack_validity_mean.csv", index=False)
    efficiency_df.to_csv(output_dir / "efficiency_raw.csv", index=False)
    efficiency_mean_df.to_csv(output_dir / "efficiency_mean.csv", index=False)

    plot_metric_curve(mean_df, "f1", output_dir / "f1_curve.png")
    plot_metric_curve(mean_df, "recall", output_dir / "recall_curve.png")
    plot_ablation(mean_df, output_dir / "clean_f1_bar.png")
    transfer_heatmap_df = transfer_mean_df[
        (transfer_mean_df["attack"] == "pgd") & (np.isclose(transfer_mean_df["epsilon"], 0.1))
    ]
    plot_transfer_heatmap(transfer_heatmap_df, output_dir / "transfer_pgd_heatmap.png")
    plot_ratio_ablation(ratio_mean_df, output_dir / "ratio_ablation.png")
    plot_efficiency_tradeoff(efficiency_mean_df, mean_df, output_dir / "efficiency_tradeoff.png")
    print("[run] aggregations and plots completed", flush=True)

    # --- Statistical significance tests ---
    sig_rows = []
    comparisons = [
        ("constrained_adv", "standard"),
        ("constrained_adv", "adv_training"),
        ("class_aware_constrained", "constrained_adv"),
        ("trades", "adv_training"),
        ("free_at", "adv_training"),
    ]
    for attack_col in ["clean"] + [f"{a}_{e}" for a in ["fgsm","pgd","cw","apgd"] for e in config.epsilon_list]:
        parts = attack_col.split("_") if attack_col != "clean" else ["clean", "0.0"]
        if attack_col == "clean":
            atk_name, atk_eps = "clean", 0.0
        else:
            # parse e.g. "pgd_0.1" or "apgd_0.02"
            idx = attack_col.rfind("_")
            atk_name = attack_col[:idx]
            atk_eps = float(attack_col[idx+1:])
        sub = results_df[(results_df["attack"]==atk_name) & (np.isclose(results_df["epsilon"], atk_eps))]
        if len(sub) == 0:
            continue
        for m1, m2 in comparisons:
            v1 = sub[sub["model"]==m1].sort_values("seed")["f1"].values
            v2 = sub[sub["model"]==m2].sort_values("seed")["f1"].values
            if len(v1) < 2 or len(v2) < 2 or len(v1) != len(v2):
                continue
            for metric in ["f1", "attack_success_rate"]:
                a = sub[sub["model"]==m1].sort_values("seed")[metric].values
                b = sub[sub["model"]==m2].sort_values("seed")[metric].values
                if len(a) < 2 or len(b) < 2:
                    continue
                try:
                    t_stat, t_p = scipy_stats.ttest_rel(a, b)
                except Exception:
                    t_stat, t_p = float("nan"), float("nan")
                try:
                    w_stat, w_p = scipy_stats.wilcoxon(a, b)
                except Exception:
                    w_stat, w_p = float("nan"), float("nan")
                sig_rows.append({
                    "comparison": f"{m1}_vs_{m2}",
                    "attack": atk_name, "epsilon": atk_eps, "metric": metric,
                    "t_statistic": t_stat, "t_pvalue": t_p,
                    "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_p,
                })
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(output_dir / "significance_tests.csv", index=False)

    # --- Full test set clean metrics ---
    full_test_df = pd.DataFrame(full_test_clean_results)
    if len(full_test_df) > 0:
        fc_metric_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
        full_clean_mean = full_test_df.groupby(["model", "subset"], as_index=False)[fc_metric_cols].mean()
        full_clean_std = full_test_df.groupby(["model", "subset"], as_index=False)[fc_metric_cols].std().fillna(0)
        full_test_df.to_csv(output_dir / "full_test_clean_raw.csv", index=False)
        full_clean_mean.to_csv(output_dir / "full_test_clean_mean.csv", index=False)
        full_clean_std.to_csv(output_dir / "full_test_clean_std.csv", index=False)

    summary = {
        "config": asdict(config),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(eval_y)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "numeric_feature_count": int(len(metadata["numeric_cols"])),
        "categorical_feature_count": int(len(metadata["categorical_cols"])),
        "transformed_feature_count": int(len(metadata["feature_names"])),
        "top_attack_categories": top_attack_categories,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

# =============================================================================
# Command-line interface
#
# Every field of ExperimentConfig has a matching command-line flag, generated
# from the dataclass itself. Adding a field to ExperimentConfig automatically
# adds a flag here, so the CLI can never drift out of sync with the config.
#
#   scalar fields       --adv-epsilon 0.05
#   tuple[int]          --hidden-dims 256,128,64
#   tuple[float]        --epsilon-list 0.02,0.05,0.10
#   tuple[str]          --reference-models log_reg,hist_gbdt
#   tuple[(str,float)]  --transfer-attack-settings fgsm:0.05,pgd:0.10
#
# Run with --help to see the full list, or --print-config to dump the effective
# configuration as JSON.
# =============================================================================

CONFIG_HELP: dict[str, str] = {
    "output_dir": "directory for all result files",
    "training_budget_mode": "legacy = each defense trained independently; "
                            "matched_continuation = shared clean pre-training plus an equal continuation budget",
    "batch_size": "mini-batch size",
    "baseline_epochs": "clean pre-training epochs",
    "adv_epochs": "adversarial (continuation) epochs per defense",
    "learning_rate": "Adam learning rate",
    "weight_decay": "Adam weight decay",
    "hidden_dims": "MLP hidden layer widths",
    "dropout": "dropout probability",
    "eval_attack_rows": "number of test rows used for the attack evaluation (stratified, shared by all defenses)",
    "adv_epsilon": "L-inf perturbation budget used DURING TRAINING (differs from the evaluation budgets)",
    "adv_alpha": "PGD step size used during training",
    "adv_steps": "PGD steps used during training",
    "epsilon_list": "evaluation perturbation budgets",
    "seeds": "random seeds; results are aggregated over them",
    "sensitivity_top_ratio": "fraction of features kept by the sensitivity mask",
    "sensitivity_ratio_list": "mask ratios swept during the sensitivity analysis",
    "sensitivity_batches": "mini-batches used to estimate feature sensitivity",
    "transfer_attack_settings": "attack:epsilon pairs used for the transferability matrix",
    "validity_attack_settings": "attack:epsilon pairs used for the attack-validity check",
    "category_attack": "attack used for the per-category (worst-class) evaluation",
    "category_epsilon": "epsilon for the per-category evaluation",
    "ratio_attack": "attack used for the perturbation-ratio analysis",
    "ratio_attack_epsilon": "epsilon for the perturbation-ratio analysis",
    "top_attack_categories": "how many attack categories enter the worst-class objective",
    "reference_models": "non-neural reference classifiers (log_reg, hist_gbdt)",
    "trades_beta": "TRADES trade-off coefficient",
    "free_at_replay": "Free AT replay multiplier m",
    "class_aware_minority_weight": "extra loss weight for minority attack categories",
    "extra_methods": "optional additional defenses to train as well: progressive, sa_trades, dst_sa_trades",
    "progressive_ratios": "mask ratios of the progressive class-aware schedule",
    "sa_trades_gamma": "sensitivity-weighting exponent of SA-TRADES",
    "dst_update_interval": "epochs between sensitivity re-estimates in DST-SA-TRADES",
    "dst_ema_alpha": "EMA smoothing factor for the DST sensitivity estimate",
    "cw_steps": "C&W L2 optimisation steps",
    "cw_lr": "C&W L2 learning rate",
    "cw_c": "C&W L2 confidence constant",
    "apgd_steps": "APGD-CE steps",
    "apgd_rho": "APGD-CE step-size schedule parameter",
    "device": "torch device, e.g. cpu or cuda",
    "eval_pgd_alpha_ratio": "PGD evaluation step size as a fraction of epsilon "
                            "(0.05 = eps/20, the value behind every reported result; 0.10 = eps/10)",
}

# Shorter flag names used by older scripts, kept working.
CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "class_aware_minority_weight": ("--class-aware-weight",),
}


def _tuple_kind(value: tuple) -> str:
    """Classify a tuple default so the CLI knows how to parse it."""
    if not value:
        return "str"
    first = value[0]
    if isinstance(first, tuple):
        return "pair"
    if isinstance(first, bool):
        return "str"
    if isinstance(first, int):
        return "int"
    if isinstance(first, float):
        return "float"
    return "str"


def _format_default(value: object) -> str:
    if isinstance(value, tuple):
        if value and isinstance(value[0], tuple):
            return ",".join(f"{k}:{v}" for k, v in value)
        return ",".join(str(v) for v in value)
    return str(value)


def parse_tuple_arg(raw: str, kind: str) -> tuple:
    """Turn a comma-separated string into the tuple type the config expects.

    int / float / str :  "7,13,21"            -> (7, 13, 21)
    pair              :  "fgsm:0.05,pgd:0.1"  -> (("fgsm", 0.05), ("pgd", 0.1))
    """
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if kind == "pair":
        out = []
        for it in items:
            key, sep, val = it.partition(":")
            if not sep:
                raise argparse.ArgumentTypeError(
                    f"expected key:value pairs such as 'fgsm:0.05,pgd:0.10', got {it!r}")
            out.append((key.strip(), float(val)))
        return tuple(out)
    conv = {"int": int, "float": float, "str": str}[kind]
    return tuple(conv(x) for x in items)


def add_config_arguments(parser: argparse.ArgumentParser, skip: tuple[str, ...] = (),
                         defaults: dict[str, object] | None = None) -> None:
    """Attach one flag per ExperimentConfig field. Generated, never hand-maintained."""
    for name, field in ExperimentConfig.__dataclass_fields__.items():
        if name in skip:
            continue
        flag = "--" + name.replace("_", "-")
        aliases = list(CONFIG_ALIASES.get(name, ()))
        default = (defaults or {}).get(name, field.default)
        help_text = CONFIG_HELP.get(name, "")

        if isinstance(default, tuple):
            kind = _tuple_kind(default)
            example = _format_default(default)
            parser.add_argument(flag, *aliases, default=default, metavar="LIST",
                                help=f"{help_text}  [comma-separated; default: {example}]")
        elif isinstance(default, bool):
            parser.add_argument(flag, *aliases, default=default, type=_str2bool, nargs="?",
                                const=True, help=f"{help_text}  [default: {default}]")
        elif isinstance(default, int):
            parser.add_argument(flag, *aliases, default=default, type=int,
                                help=f"{help_text}  [default: {default}]")
        elif isinstance(default, float):
            parser.add_argument(flag, *aliases, default=default, type=float,
                                help=f"{help_text}  [default: {default}]")
        else:
            parser.add_argument(flag, *aliases, default=default, type=str,
                                help=f"{help_text}  [default: {default}]")


def _str2bool(raw: str) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def config_from_args(args: argparse.Namespace, **overrides: object) -> ExperimentConfig:
    """Build an ExperimentConfig from parsed arguments.

    Anything the user did not pass keeps the dataclass default. Tuple fields
    arrive as strings and are converted according to their default's element type.
    """
    kwargs: dict[str, object] = {}
    for name, field in ExperimentConfig.__dataclass_fields__.items():
        raw = getattr(args, name, None)
        if raw is None:
            continue
        default = field.default
        if isinstance(default, tuple) and isinstance(raw, str):
            raw = parse_tuple_arg(raw, _tuple_kind(default))
        kwargs[name] = raw
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def build_parser(description: str | None = None,
                 skip: tuple[str, ...] = (),
                 defaults: dict[str, object] | None = None,
                 require_paths: bool = True) -> argparse.ArgumentParser:
    """Standard parser: dataset paths plus every config field.

    defaults       per-script overrides for individual config fields, so each
                   backbone can keep its own default output directory, batch
                   size and so on while still exposing every parameter.
    require_paths  True for stand-alone scripts; False for the backbone scripts,
                   which fall back to data/train.csv and data/test.csv.
    """
    d = defaults or {}
    parser = argparse.ArgumentParser(
        description=description or "Adversarial robustness experiments for UNSW-NB15 intrusion detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every ExperimentConfig field can be set from the command line; see above.")
    parser.add_argument("--train-path", required=require_paths,
                        default=None if require_paths else d.get("train_path", "data/train.csv"),
                        help="UNSW-NB15 training partition (175,341 records)")
    parser.add_argument("--test-path", required=require_paths,
                        default=None if require_paths else d.get("test_path", "data/test.csv"),
                        help="UNSW-NB15 testing partition (82,332 records)")
    add_config_arguments(parser, skip=("train_path", "test_path") + tuple(skip), defaults=d)
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective configuration as JSON and exit")
    return parser


def emit_config(config: ExperimentConfig) -> None:
    """Print the effective configuration as JSON and exit. Used by --print-config."""
    print(json.dumps(asdict(config), indent=2, default=str))
    raise SystemExit(0)


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.print_config:
        emit_config(config)
    return config


if __name__ == "__main__":
    CONFIG = parse_args()
    run_experiment(CONFIG)
