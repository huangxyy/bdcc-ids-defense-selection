"""Adaptive attack suite for the IDS defense-selection framework.

WHY THIS FILE EXISTS
--------------------
The six candidate defenses are ALL training-time defenses:
  Standard / PGD-AT / Constrained AT / TRADES / Free AT / Class-Aware Constrained AT
At inference time every one of them is a plain differentiable network.
There is no input preprocessing, no randomization, no gradient obfuscation.

Therefore the standard "adaptive attack" question ("does the defense survive an
adversary that knows the defense mechanism?") decomposes into three concrete,
testable questions:

  Q1  Is the white-box attack strong enough?            -> restarts + margin loss
  Q2  Is there gradient masking induced by training?    -> gradient diagnostics + NES
  Q3  Does the defense's TRAINING mask create a blind  -> complement attack
      spot the attacker can exploit?

Q3 is the genuinely DEFENSE-SPECIFIC adaptive strategy and is the novel part:
Constrained AT and Class-Aware Constrained AT only ever perturb the top-k most
sensitive features during training.  An adversary that knows this can deliberately
spend the whole budget on the COMPLEMENT -- the features the defense never trained
against.  If complement-ASR > full-ASR on those two defenses but not on the others,
the constrained family has a measurable blind spot.

USAGE
-----
    from ids_defense_selection.adaptive import (
        pgd_adaptive, nes_attack, gradient_diagnostics, evaluate_adaptive_suite)

All tensors follow the convention of the package:
    x        : (B, D) float32, already standardised
    mask     : (D,)   float32 in {0,1}, 1 = perturbable (continuous features only)
    mins/maxs: (D,)   float32, per-feature valid range observed on the TRAIN set
    logits   : (B,)   single-logit binary model, decision threshold 0
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .attacks import clamp_numeric


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def logits_of(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out.reshape(-1) if out.dim() > 1 else out


def attack_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                kind: str = "margin", kappa: float = 0.0) -> torch.Tensor:
    """Attacker objective. HIGHER = better for the attacker.

    margin : -t*z                (linear margin, t = 2y-1)   -- default, decision-aligned
    cw     : clamp(-t*z + kappa) (hinge / Carlini-Wagner style)
    ce     : BCEWithLogits       (the loss the defenses were TRAINED against)
    """
    z = logits_of(model, x)
    if kind == "ce":
        return nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
    t = 2.0 * y - 1.0
    m = -(t * z)
    return torch.clamp(m + kappa, min=0.0) if kind == "cw" else m


def _rand_start(x, epsilon, mask, mins, maxs):
    noise = (torch.rand_like(x) * 2.0 - 1.0) * epsilon * mask
    return clamp_numeric(x + noise, x, mins, maxs, mask)


# ----------------------------------------------------------------------
# Q1  strong white-box adaptive PGD  (restarts + decision-aligned loss)
# ----------------------------------------------------------------------

@torch.no_grad()
def _pick_better(model, x_new, x_best, y, best_loss, kind, kappa):
    new_loss = attack_loss(model, x_new, y, kind, kappa)
    if best_loss is None:
        return x_new.clone(), new_loss.clone()
    better = new_loss > best_loss
    x_best = x_best.clone()
    x_best[better] = x_new[better]
    best_loss = torch.where(better, new_loss, best_loss)
    return x_best, best_loss


def pgd_adaptive(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 epsilon: float, mask: torch.Tensor,
                 mins: torch.Tensor, maxs: torch.Tensor,
                 steps: int = 40, restarts: int = 5,
                 alpha: float | None = None,
                 loss_kind: str = "margin", kappa: float = 0.0,
                 seed: int | None = None) -> torch.Tensor:
    """PGD that ASCENDS a decision-aligned loss, with random restarts, keeping
    the worst-case (highest-loss) example found per sample.

    Differences vs. the original pgd_attack():
      * loss is the margin -t*z (or CW hinge), not the defense's training BCE
      * multiple random restarts, best-per-sample selection
      * adaptive step size (halved when the loss stops improving)
    """
    if seed is not None:
        torch.manual_seed(seed)
    if alpha is None:
        alpha = 2.0 * epsilon / max(steps, 1)

    x_best, best_loss = None, None
    for r in range(max(1, restarts)):
        x_adv = x.clone() if r == 0 else _rand_start(x, epsilon, mask, mins, maxs)
        cur_alpha = alpha
        prev_total = None
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = attack_loss(model, x_adv, y, loss_kind, kappa)
            model.zero_grad(set_to_none=True)
            loss.sum().backward()
            g = x_adv.grad.detach()
            new = x_adv.detach() + cur_alpha * g.sign() * mask
            delta = torch.clamp(new - x, -epsilon, epsilon) * mask
            x_adv = clamp_numeric(x + delta, x, mins, maxs, mask)
            # crude adaptive step: shrink if the objective stalls
            with torch.no_grad():
                tot = float(attack_loss(model, x_adv, y, loss_kind, kappa).sum())
                if prev_total is not None and tot <= prev_total * 1.001:
                    cur_alpha *= 0.5
                prev_total = tot
        x_best, best_loss = _pick_better(model, x_adv, x_best, y, best_loss, loss_kind, kappa) \
            if x_best is not None else (x_adv.clone(), None)
        if best_loss is None:
            with torch.no_grad():
                best_loss = attack_loss(model, x_best, y, loss_kind, kappa)
    return x_best.detach()


# ----------------------------------------------------------------------
# Q2a gradient-free cross-check  (NES / SPSA-style, forward passes only)
# ----------------------------------------------------------------------

def nes_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               epsilon: float, mask: torch.Tensor,
               mins: torch.Tensor, maxs: torch.Tensor,
               steps: int = 40, n_samples: int = 50, sigma: float = 1e-3,
               alpha: float | None = None,
               loss_kind: str = "margin", kappa: float = 0.0) -> torch.Tensor:
    """Gradient-free attack: estimate d(loss)/dx by antithetic NES, then PGD.

    Only forward passes are used.  If this attack reaches an ASR comparable to the
    white-box one, gradient masking is ruled out (Athalye et al., 2018).
    """
    if alpha is None:
        alpha = 2.0 * epsilon / max(steps, 1)
    x_adv = _rand_start(x, epsilon, mask, mins, maxs)
    half = max(1, n_samples // 2)
    for _ in range(steps):
        grad = torch.zeros_like(x_adv)
        with torch.no_grad():
            for _ in range(half):
                u = torch.randn_like(x_adv) * mask
                lp = attack_loss(model, clamp_numeric(x_adv + sigma * u, x, mins, maxs, mask),
                                 y, loss_kind, kappa)
                lm = attack_loss(model, clamp_numeric(x_adv - sigma * u, x, mins, maxs, mask),
                                 y, loss_kind, kappa)
                grad += (lp - lm).unsqueeze(1) * u
            grad /= float(n_samples * sigma)
            x_adv = x_adv + alpha * grad.sign() * mask
            delta = torch.clamp(x_adv - x, -epsilon, epsilon) * mask
            x_adv = clamp_numeric(x + delta, x, mins, maxs, mask)
    return x_adv


# ----------------------------------------------------------------------
# Q2b gradient diagnostics  (is the training procedure masking gradients?)
# ----------------------------------------------------------------------



def gradient_diagnostics(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         mask: torch.Tensor,
                         loss_kind: str = "margin") -> dict:
    """Input-gradient statistics.

    A defense that obfuscates gradients typically shows an anomalously small or
    near-zero input gradient.  We report the per-sample L-inf and L2 norms of
    d(loss)/dx restricted to the perturbable coordinates.
    """
    x = x.detach().clone().requires_grad_(True)
    loss = attack_loss(model, x, y, loss_kind)
    model.zero_grad(set_to_none=True)
    loss.sum().backward()
    g = (x.grad.detach() * mask)
    linf = g.abs().amax(dim=1).cpu().numpy()
    l2 = g.norm(dim=1).cpu().numpy()
    zero_frac = float((linf <= 1e-12).mean())
    return {"grad_linf_mean": float(linf.mean()),
            "grad_linf_median": float(np.median(linf)),
            "grad_l2_mean": float(l2.mean()),
            "zero_gradient_fraction": zero_frac}


# ----------------------------------------------------------------------
# Q3 the defense-specific adaptive strategy  (complement attack)
# ----------------------------------------------------------------------

def complement_mask(numeric_mask: torch.Tensor, defense_mask: torch.Tensor) -> torch.Tensor:
    """Features the defense NEVER perturbs during training, restricted to the
    valid (continuous) perturbation space.

    defense_mask is 1 on the top-k sensitive features the constrained defenses
    train against; the complement is what their training never covered.
    """
    comp = numeric_mask * (1.0 - defense_mask)
    if float(comp.sum()) < 1.0:          # degenerate: nothing left, fall back
        return numeric_mask.clone()
    return comp


def pgd_complement(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   epsilon: float, numeric_mask: torch.Tensor,
                   defense_mask: torch.Tensor,
                   mins: torch.Tensor, maxs: torch.Tensor,
                   steps: int = 40, restarts: int = 5,
                   loss_kind: str = "margin") -> torch.Tensor:
    """Attack ONLY the features the defense did not protect during training."""
    return pgd_adaptive(model, x, y, epsilon,
                        complement_mask(numeric_mask, defense_mask),
                        mins, maxs, steps=steps, restarts=restarts,
                        loss_kind=loss_kind)


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------

@torch.no_grad()
def _asr(model: nn.Module, x_adv: torch.Tensor, y: torch.Tensor,
         clean_pred: torch.Tensor) -> float:
    pred = (logits_of(model, x_adv) >= 0.0).float()
    correct = clean_pred == y
    if correct.sum() == 0:
        return 0.0
    return float((pred[correct] != y[correct]).float().mean().item())


def evaluate_adaptive_suite(models: dict, x_eval: np.ndarray, y_eval: np.ndarray,
                            numeric_mask: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                            defense_masks: dict | None, epsilon: float,
                            device: torch.device, batch_size: int = 1024,
                            steps: int = 40, restarts: int = 5,
                            nes_samples: int = 50, nes_subset: int = 2000,
                            seed: int = 2026) -> "list[dict]":
    """Run the full adaptive suite over all defenses and return tidy rows.

    Columns: model, epsilon, attack, asr, plus gradient diagnostics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask_t = torch.from_numpy(numeric_mask.astype(np.float32)).to(device)
    mins_t = torch.from_numpy(np.asarray(mins, dtype=np.float32)).to(device)
    maxs_t = torch.from_numpy(np.asarray(maxs, dtype=np.float32)).to(device)
    x_all = torch.from_numpy(np.ascontiguousarray(x_eval)).float()
    y_all = torch.from_numpy(np.ascontiguousarray(y_eval)).float()
    rows: list[dict] = []

    for name, model in models.items():
        model.eval()
        d_mask_np = (defense_masks or {}).get(name)
        d_mask_t = (torch.from_numpy(np.asarray(d_mask_np, dtype=np.float32)).to(device)
                    if d_mask_np is not None else mask_t)
        n = x_all.shape[0]
        clean_pred = torch.empty(n)
        accs = {"pgd_margin": [], "pgd_ce": [], "complement": [], "nes": []}
        diag_l2, diag_linf, diag_zero = [], [], []
        offset = 0
        for s in range(0, n, batch_size):
            xb = x_all[s:s + batch_size].to(device)
            yb = y_all[s:s + batch_size].to(device)
            with torch.no_grad():
                cp = (logits_of(model, xb) >= 0.0).float()
            clean_pred[s:s + xb.shape[0]] = cp.cpu()

            # Q1: strong white-box, two loss functions
            for tag, kind in (("pgd_margin", "margin"), ("pgd_ce", "ce")):
                xa = pgd_adaptive(model, xb, yb, epsilon, mask_t, mins_t, maxs_t,
                                  steps=steps, restarts=restarts, loss_kind=kind)
                accs[tag].append(_asr(model, xa, yb, cp))

            # Q3: complement attack (only meaningful when a defense mask exists)
            if d_mask_np is not None:
                xa = pgd_complement(model, xb, yb, epsilon, mask_t, d_mask_t,
                                    mins_t, maxs_t, steps=steps, restarts=restarts)
                accs["complement"].append(_asr(model, xa, yb, cp))

            # Q2b: gradient diagnostics
            d = gradient_diagnostics(model, xb, yb, mask_t)
            diag_l2.append(d["grad_l2_mean"])
            diag_linf.append(d["grad_linf_mean"])
            diag_zero.append(d["zero_gradient_fraction"])
            offset += xb.shape[0]

        # Q2a: gradient-free cross-check on a subset (forward-pass heavy)
        sub = min(nes_subset, n)
        xb = x_all[:sub].to(device)
        yb = y_all[:sub].to(device)
        with torch.no_grad():
            cp = (logits_of(model, xb) >= 0.0).float()
        xa = nes_attack(model, xb, yb, epsilon, mask_t, mins_t, maxs_t,
                        steps=max(10, steps // 2), n_samples=nes_samples)
        nes_asr = _asr(model, xa, yb, cp)

        rows.append({
            "model": name, "epsilon": epsilon,
            "asr_pgd_margin": float(np.mean(accs["pgd_margin"])),
            "asr_pgd_ce": float(np.mean(accs["pgd_ce"])),
            "asr_complement": float(np.mean(accs["complement"])) if accs["complement"] else float("nan"),
            "asr_nes_gradfree": nes_asr,
            "grad_l2_mean": float(np.mean(diag_l2)),
            "grad_linf_mean": float(np.mean(diag_linf)),
            "zero_gradient_fraction": float(np.mean(diag_zero)),
        })
    return rows


def build_defense_mask(model, train_loader, device, numeric_mask, feature_names,
                       top_ratio: float, max_batches: int):
    """Build the top-k sensitivity mask exactly as the constrained defenses do.

    Reuses :func:`ids_defense_selection.compute_sensitivity_mask` so the mask
    matches training.  Returns (mask, sensitivity_table).
    """
    from .defenses import compute_sensitivity_mask

    return compute_sensitivity_mask(
        model, train_loader, device=device, numeric_mask=numeric_mask,
        feature_names=feature_names, top_ratio=top_ratio, max_batches=max_batches,
    )
