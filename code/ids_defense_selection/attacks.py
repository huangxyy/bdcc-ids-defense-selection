"""Adversarial attacks and the feature-validity constraints of the paper.

All attacks optimise the single-logit binary model under three constraints:

  (i)   only continuous features may change (the one-hot block is frozen),
  (ii)  perturbed features must stay inside the [min, max] box observed on the
        training set,
  (iii) ||delta||_inf <= epsilon.

When (ii) and (iii) conflict, :func:`clamp_numeric` resolves in favour of (ii);
:func:`count_out_of_range` and :func:`max_violation` quantify how often that
happens.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .data import numpy_to_torch


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


def clamp_numeric(x_adv: torch.Tensor, x_orig: torch.Tensor, mins: torch.Tensor,
                  maxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    """Single-step fast gradient sign method."""
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
    """Projected gradient descent with a random start (Madry et al.)."""
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
    """Adaptive PGD (APGD-CE) - simplified AutoAttack component.

    Uses momentum, the AutoAttack step-size schedule and per-sample best-loss
    tracking so the strongest found example is kept.
    """
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


def compute_attack_validity_metrics(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    numeric_mask: np.ndarray,
    numeric_mins: np.ndarray,
    numeric_maxs: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Measure how well the three attack constraints held for every sample."""
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


#: Attack names accepted by :func:`generate_adversarial_examples`.
ATTACK_NAMES = ("clean", "fgsm", "pgd", "cw", "apgd")


def generate_adversarial_examples(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack: str,
    epsilon: float,
    config: ExperimentConfig,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    device: torch.device,
    batch_size: int,
    return_inputs: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Run one attack over an evaluation set and return probabilities (and inputs).

    Returns (adversarial_inputs_or_None, predicted_probabilities).  All attack
    hyperparameters are read from *config*; the configuration is never taken from
    module-level state, so several backbones with different budgets can run in
    the same process without interfering.
    """
    if attack not in ATTACK_NAMES:
        raise ValueError(f"Unsupported attack: {attack!r}; choose from {ATTACK_NAMES}")

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
        if attack == "clean":
            x_adv = xb
        elif attack == "fgsm":
            x_adv = fgsm_attack(model, xb, yb, epsilon, mask_t, mins_t, maxs_t)
        elif attack == "pgd":
            alpha = max(epsilon * config.eval_pgd_alpha_ratio, 1e-4)
            x_adv = pgd_attack(model, xb, yb, epsilon, alpha, config.eval_pgd_steps,
                               mask_t, mins_t, maxs_t)
        elif attack == "cw":
            x_adv = cw_attack(model, xb, yb, mask_t, mins_t, maxs_t,
                              steps=config.cw_steps, lr=config.cw_lr, c_const=config.cw_c)
        else:  # apgd
            x_adv = apgd_attack(model, xb, yb, epsilon, config.apgd_steps,
                                mask_t, mins_t, maxs_t, rho=config.apgd_rho)

        if return_inputs:
            input_batches.append(x_adv.detach().cpu().numpy())
        with torch.no_grad():
            prob_batches.append(torch.sigmoid(model(x_adv)).cpu().numpy())

    attacked_inputs = np.concatenate(input_batches) if return_inputs else None
    attacked_probs = np.concatenate(prob_batches)
    return attacked_inputs, attacked_probs
