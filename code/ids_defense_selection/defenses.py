"""Defense training: the six paper defenses, optional extras, and shared masks.

The six defenses all start from one shared clean pre-training checkpoint and
receive the same continuation budget; only the perturbation strategy differs,
which is what makes the training-cost comparison (objective phi3) meaningful.

Optional extra methods (Progressive Class-Aware Constrained AT, SA-TRADES,
DST-SA-TRADES) live in this module too and run only when requested through
``ExperimentConfig.extra_methods``.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from .attacks import clamp_numeric, pgd_attack
from .config import ExperimentConfig


def fit_reference_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    enabled_models: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, float]]:
    models: dict[str, object] = {}
    train_seconds: dict[str, float] = {}
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
        train_seconds["log_reg"] = time.perf_counter() - start
        models["log_reg"] = log_reg

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
        train_seconds["hist_gbdt"] = time.perf_counter() - start
        models["hist_gbdt"] = hist_gbdt

    return models, train_seconds

def fit_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    config: ExperimentConfig,
    adversarial: bool = False,
    attack_builder: Callable | None = None,
) -> nn.Module:
    """Train for `epochs` epochs; with `adversarial=True` on clean+adversarial pairs.

    `attack_builder(model, x, y)` must return the adversarial counterpart of a
    batch; it is only consulted when `adversarial` is True.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if adversarial:
                model.eval()
                x_adv = attack_builder(model, xb, yb)
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


def fit_trades(model, train_loader, epochs, device, config,
                       mask, mins, maxs):
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
                -config.adv_epsilon, config.adv_epsilon) * mask
            x_adv = clamp_numeric(x_adv + noise, xb, mins, maxs, mask)
            for _ in range(config.adv_steps):
                x_adv.requires_grad_(True)
                probs_adv = torch.sigmoid(model(x_adv))
                p = probs_clean.clamp(1e-7, 1 - 1e-7)
                q = probs_adv.clamp(1e-7, 1 - 1e-7)
                kl = p*(p.log()-q.log()) + (1-p)*((1-p).log()-(1-q).log())
                model.zero_grad(set_to_none=True)
                kl.sum().backward()
                step = config.adv_alpha * x_adv.grad.sign() * mask
                x_adv = x_adv.detach() + step
                delta = torch.clamp(x_adv-xb, min=-config.adv_epsilon,
                                    max=config.adv_epsilon) * mask
                x_adv = clamp_numeric(xb+delta, xb, mins, maxs, mask)
            model.train()
            paf = torch.sigmoid(model(x_adv.detach())).clamp(1e-7, 1-1e-7)
            p = probs_clean.clamp(1e-7, 1-1e-7)
            kl_f = p*(p.log()-paf.log()) + (1-p)*((1-p).log()-(1-paf).log())
            loss = loss_bce + beta * kl_f.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def fit_free_at(model, train_loader, epochs, device, config,
                        mask, mins, maxs):
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
                x_adv = clamp_numeric(xb + d_var * mask, xb, mins, maxs, mask)
                optimizer.zero_grad(set_to_none=True)
                loss = 0.5 * criterion(model(xb), yb) + 0.5 * criterion(model(x_adv), yb)
                loss.backward()
                optimizer.step()
                g = d_var.grad.detach() if d_var.grad is not None else torch.zeros_like(d)
                d = torch.clamp(d + config.adv_alpha * g.sign() * mask,
                                min=-config.adv_epsilon, max=config.adv_epsilon)
            delta_global[:bs] = d.detach()
    return model



# =============================================================================
# Optional extra defense methods
def compute_progressive_masks(model, train_loader, device, numeric_mask,
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


def fit_progressive_class_aware(model, train_loader, epochs, device,
                                        config, masks_schedule, mins, maxs,
                                        minority_weight):
    """Progressive Class-Aware Constrained AT: mask expands across epochs.

    masks_schedule: list of (ratio, mask_np, n_features) from compute_progressive_masks.
    The training epochs are evenly divided among the stages.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    model.to(device)
    n_stages = len(masks_schedule)
    epochs_per_stage = max(1, epochs // n_stages)

    for stage_idx, (ratio, mask_np, n_feat) in enumerate(masks_schedule):
        mask = torch.from_numpy(mask_np).to(device)
        stage_epochs = epochs_per_stage if stage_idx < n_stages - 1 else (epochs - stage_idx * epochs_per_stage)
        for ep in range(stage_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                model.eval()
                x_adv = pgd_attack(model, xb, yb, config.adv_epsilon,
                                   config.adv_alpha, config.adv_steps,
                                   mask, mins, maxs)
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


def fit_sa_trades(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    config,
    sensitivity_weights_np: np.ndarray,
    mins: torch.Tensor,
    maxs: torch.Tensor,
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
            x_adv = clamp_numeric(x_adv + noise, xb, mins, maxs, active_mask)
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
                x_adv = clamp_numeric(xb + delta, xb, mins, maxs, active_mask)
            model.train()
            paf = torch.sigmoid(model(x_adv.detach())).clamp(1e-7, 1 - 1e-7)
            p = probs_clean.clamp(1e-7, 1 - 1e-7)
            kl_f = p * (p.log() - paf.log()) + (1 - p) * ((1 - p).log() - (1 - paf).log())
            loss = loss_bce + beta * kl_f.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def estimate_sensitivity_scores(
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


def sensitivity_scores_to_weights(
    scores: np.ndarray,
    numeric_mask: np.ndarray,
    gamma: float,
) -> np.ndarray:
    numeric_indices = np.where(numeric_mask > 0)[0]
    max_s = float(scores[numeric_indices].max()) + 1e-12
    w = np.zeros_like(scores, dtype=np.float32)
    w[numeric_indices] = (scores[numeric_indices] / max_s) ** gamma
    return w


def fit_dst_sa_trades(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    config,
    initial_weights_np: np.ndarray,
    numeric_mask: np.ndarray,
    mins: torch.Tensor,
    maxs: torch.Tensor,
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
            new_scores = estimate_sensitivity_scores(
                model, train_loader, device, max_batches=config.sensitivity_batches,
            )
            new_weights = sensitivity_scores_to_weights(new_scores, numeric_mask, gamma)
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
            x_adv = clamp_numeric(x_adv + noise, xb, mins, maxs, active_mask)
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
                x_adv = clamp_numeric(xb + delta, xb, mins, maxs, active_mask)
            model.train()
            paf = torch.sigmoid(model(x_adv.detach())).clamp(1e-7, 1 - 1e-7)
            p = probs_clean.clamp(1e-7, 1 - 1e-7)
            kl_f = p * (p.log() - paf.log()) + (1 - p) * ((1 - p).log() - (1 - paf).log())
            loss = loss_bce + beta * kl_f.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model

def fit_class_aware_constrained(model, train_loader, epochs, device,
                                        config, selected_mask_t, mins, maxs,
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
                               selected_mask_t, mins, maxs)
            model.train()
            batch_x = torch.cat([xb, x_adv], dim=0)
            batch_y = torch.cat([yb, yb], dim=0)
            logits = model(batch_x)
            w = torch.where(batch_y > 0.5, minority_weight, 1.0)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, weight=w)
            loss.backward()
            optimizer.step()
    return model


def compute_class_aware_sensitivity_mask(model, train_loader, device, numeric_mask,
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


def compute_sensitivity_mask(
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


DEFENSE_ORDER: tuple[str, ...] = (
    "standard",
    "adv_training",
    "constrained_adv",
    "trades",
    "free_at",
    "class_aware_constrained",
)

OPTIONAL_DEFENSES: tuple[str, ...] = ("progressive", "sa_trades", "dst_sa_trades")


@dataclass
class TrainedDefenses:
    """Everything a runner needs after training all defense candidates once."""

    baseline_model: nn.Module
    models: dict[str, nn.Module]
    train_cost: dict[str, dict[str, float]]
    perturb_ratio: dict[str, float]
    perturb_features: dict[str, int]
    selected_mask: np.ndarray
    class_aware_mask: np.ndarray
    sensitivity_table: pd.DataFrame

    def defense_masks(self) -> dict[str, np.ndarray]:
        """Training-time masks of the two feature-constrained defenses."""
        return {
            "constrained_adv": self.selected_mask,
            "class_aware_constrained": self.class_aware_mask,
        }


def train_all_defenses(
    model_factory: Callable[[], nn.Module],
    config: ExperimentConfig,
    train_loader: DataLoader,
    device: torch.device,
    metadata: dict,
    seed: int,
) -> TrainedDefenses:
    """Train the six defenses (plus any optional extras) under one protocol.

    All candidates start from the same clean pre-training checkpoint and receive
    an identical continuation budget in ``matched_continuation`` mode, so only
    the perturbation strategy differs between them.
    """
    matched_budget = config.training_budget_mode == "matched_continuation"
    numeric_mask = metadata["numeric_mask"]
    numeric_mask_t = torch.from_numpy(numeric_mask.astype(np.float32)).to(device)
    mins_t = torch.from_numpy(metadata["numeric_mins"].astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(metadata["numeric_maxs"].astype(np.float32)).to(device)

    # --- shared clean pre-training ---------------------------------------
    baseline_model = model_factory()
    start = time.perf_counter()
    baseline_model = fit_supervised(baseline_model, train_loader, config.baseline_epochs,
                                    device, config)
    baseline_seconds = time.perf_counter() - start

    standard_model = baseline_model
    standard_continuation_seconds = 0.0
    if matched_budget:
        standard_model = copy.deepcopy(baseline_model)
        start = time.perf_counter()
        standard_model = fit_supervised(standard_model, train_loader, config.adv_epochs,
                                        device, config)
        standard_continuation_seconds = time.perf_counter() - start

    # --- PGD adversarial training ----------------------------------------
    adv_model = copy.deepcopy(baseline_model)

    def adv_attack(model, xb, yb):
        return pgd_attack(model, xb, yb, config.adv_epsilon, config.adv_alpha,
                          config.adv_steps, numeric_mask_t, mins_t, maxs_t)

    start = time.perf_counter()
    adv_model = fit_supervised(adv_model, train_loader, config.adv_epochs, device, config,
                               adversarial=True, attack_builder=adv_attack)
    adv_seconds = time.perf_counter() - start

    # --- constrained AT (top-k sensitivity mask) --------------------------
    start = time.perf_counter()
    selected_mask, sensitivity_table = compute_sensitivity_mask(
        baseline_model,
        train_loader,
        device=device,
        numeric_mask=numeric_mask,
        feature_names=metadata["feature_names"],
        top_ratio=config.sensitivity_top_ratio,
        max_batches=config.sensitivity_batches,
    )
    mask_seconds = time.perf_counter() - start
    selected_mask_t = torch.from_numpy(selected_mask.astype(np.float32)).to(device)

    constrained_model = copy.deepcopy(baseline_model)

    def constrained_attack(model, xb, yb):
        return pgd_attack(model, xb, yb, config.adv_epsilon, config.adv_alpha,
                          config.adv_steps, selected_mask_t, mins_t, maxs_t)

    start = time.perf_counter()
    constrained_model = fit_supervised(constrained_model, train_loader, config.adv_epochs,
                                       device, config, adversarial=True,
                                       attack_builder=constrained_attack)
    constrained_seconds = time.perf_counter() - start

    # --- TRADES -----------------------------------------------------------
    trades_model = copy.deepcopy(baseline_model)
    start = time.perf_counter()
    trades_model = fit_trades(trades_model, train_loader, config.adv_epochs, device, config,
                              numeric_mask_t, mins_t, maxs_t)
    trades_seconds = time.perf_counter() - start

    # --- Free AT ----------------------------------------------------------
    free_at_model = copy.deepcopy(baseline_model)
    start = time.perf_counter()
    free_at_model = fit_free_at(free_at_model, train_loader, config.adv_epochs, device, config,
                                numeric_mask_t, mins_t, maxs_t)
    free_at_seconds = time.perf_counter() - start

    # --- class-aware constrained AT --------------------------------------
    start = time.perf_counter()
    class_aware_mask, _ = compute_class_aware_sensitivity_mask(
        baseline_model,
        train_loader,
        device=device,
        numeric_mask=numeric_mask,
        feature_names=metadata["feature_names"],
        top_ratio=config.sensitivity_top_ratio,
        max_batches=config.sensitivity_batches,
    )
    class_aware_mask_seconds = time.perf_counter() - start
    class_aware_mask_t = torch.from_numpy(class_aware_mask.astype(np.float32)).to(device)

    class_aware_model = copy.deepcopy(baseline_model)
    start = time.perf_counter()
    class_aware_model = fit_class_aware_constrained(
        class_aware_model, train_loader, config.adv_epochs, device, config,
        class_aware_mask_t, mins_t, maxs_t, config.class_aware_minority_weight)
    class_aware_seconds = time.perf_counter() - start

    models: dict[str, nn.Module] = {
        "standard": standard_model,
        "adv_training": adv_model,
        "constrained_adv": constrained_model,
        "trades": trades_model,
        "free_at": free_at_model,
        "class_aware_constrained": class_aware_model,
    }
    train_cost: dict[str, dict[str, float]] = {
        "standard": summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=standard_continuation_seconds,
            include_pretrain_in_total=True,
        ),
        "adv_training": summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=adv_seconds,
            include_pretrain_in_total=matched_budget,
        ),
        "constrained_adv": summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=constrained_seconds,
            mask_build_seconds=mask_seconds,
            include_pretrain_in_total=matched_budget,
            include_mask_in_total=matched_budget,
        ),
        "trades": summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=trades_seconds,
            include_pretrain_in_total=matched_budget,
        ),
        "free_at": summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=free_at_seconds,
            include_pretrain_in_total=matched_budget,
        ),
        "class_aware_constrained": summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=class_aware_seconds,
            mask_build_seconds=class_aware_mask_seconds,
            include_pretrain_in_total=matched_budget,
            include_mask_in_total=matched_budget,
        ),
    }
    numeric_total = max(float(numeric_mask.sum()), 1.0)
    perturb_ratio: dict[str, float] = {
        "standard": 0.0,
        "adv_training": 1.0,
        "constrained_adv": float(selected_mask.sum() / numeric_total),
        "trades": 1.0,
        "free_at": 1.0,
        "class_aware_constrained": float(class_aware_mask.sum() / numeric_total),
    }
    perturb_features: dict[str, int] = {
        "standard": 0,
        "adv_training": int(numeric_mask.sum()),
        "constrained_adv": int(selected_mask.sum()),
        "trades": int(numeric_mask.sum()),
        "free_at": int(numeric_mask.sum()),
        "class_aware_constrained": int(class_aware_mask.sum()),
    }

    # --- optional extra methods ------------------------------------------
    # Nothing in this block runs unless the method is named in --extra-methods.
    if "progressive" in config.extra_methods:
        progressive_masks = compute_progressive_masks(
            baseline_model,
            train_loader,
            device=device,
            numeric_mask=numeric_mask,
            feature_names=metadata["feature_names"],
            ratios=config.progressive_ratios,
            max_batches=config.sensitivity_batches,
        )
        progressive_model = copy.deepcopy(baseline_model)
        start = time.perf_counter()
        progressive_model = fit_progressive_class_aware(
            progressive_model, train_loader, config.adv_epochs, device, config,
            progressive_masks, mins_t, maxs_t, config.class_aware_minority_weight)
        progressive_seconds = time.perf_counter() - start
        models["progressive_class_aware"] = progressive_model
        train_cost["progressive_class_aware"] = summarize_training_budget(
            pretrain_seconds=baseline_seconds,
            continuation_seconds=progressive_seconds,
            include_pretrain_in_total=matched_budget,
        )
        perturb_ratio["progressive_class_aware"] = float(config.progressive_ratios[-1])
        perturb_features["progressive_class_aware"] = int(progressive_masks[-1][2])

    if {"sa_trades", "dst_sa_trades"} & set(config.extra_methods):
        sensitivity_weights = compute_sensitivity_weights(
            baseline_model,
            train_loader,
            device=device,
            numeric_mask=numeric_mask,
            max_batches=config.sensitivity_batches,
            gamma=config.sa_trades_gamma,
        )
        weighted_count = int(np.sum(sensitivity_weights > 0))
        weighted_ratio = float(weighted_count / numeric_total)

        if "sa_trades" in config.extra_methods:
            sa_model = copy.deepcopy(baseline_model)
            start = time.perf_counter()
            sa_model = fit_sa_trades(sa_model, train_loader, config.adv_epochs, device, config,
                                     sensitivity_weights, mins_t, maxs_t)
            sa_seconds = time.perf_counter() - start
            models["sa_trades"] = sa_model
            train_cost["sa_trades"] = summarize_training_budget(
                pretrain_seconds=baseline_seconds,
                continuation_seconds=sa_seconds,
                include_pretrain_in_total=matched_budget,
            )
            perturb_ratio["sa_trades"] = weighted_ratio
            perturb_features["sa_trades"] = weighted_count

        if "dst_sa_trades" in config.extra_methods:
            dst_model = copy.deepcopy(baseline_model)
            start = time.perf_counter()
            dst_model = fit_dst_sa_trades(
                dst_model, train_loader, config.adv_epochs, device, config,
                sensitivity_weights.copy(), numeric_mask, mins_t, maxs_t)
            dst_seconds = time.perf_counter() - start
            models["dst_sa_trades"] = dst_model
            train_cost["dst_sa_trades"] = summarize_training_budget(
                pretrain_seconds=baseline_seconds,
                continuation_seconds=dst_seconds,
                include_pretrain_in_total=matched_budget,
            )
            perturb_ratio["dst_sa_trades"] = weighted_ratio
            perturb_features["dst_sa_trades"] = weighted_count

    sensitivity_table.insert(0, "seed", seed)
    return TrainedDefenses(
        baseline_model=baseline_model,
        models=models,
        train_cost=train_cost,
        perturb_ratio=perturb_ratio,
        perturb_features=perturb_features,
        selected_mask=selected_mask,
        class_aware_mask=class_aware_mask,
        sensitivity_table=sensitivity_table,
    )
