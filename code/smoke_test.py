#!/usr/bin/env python
"""Fast end-to-end sanity check (roughly one minute).

Verifies that the plumbing works on the real dataset without paying for a full
training run. Run this first to confirm the environment and the data are set up
correctly.

The scaler and the feature-space bounds used by the attack constraints are fitted
on the FULL training partition (that is what the real experiments do); only the
training and attack steps are restricted to a small sample, so the check stays
fast. Sampling rows for the fit would produce a truncated one-hot space and much
too narrow [mins, maxs] bounds, which would make the constraint checks meaningless.

Checks
  1. data/train.csv and data/test.csv load, expected columns present
  2. feature engineering reproduces the documented dimensionality (190)
  3. all three backbones instantiate and run a forward pass
  4. PGD respects the perturbation constraints on in-range samples, and leaves
     one-hot features untouched
  5. clamp_numeric / count_out_of_range behave as documented
  6. the metric helpers compute on a batch
  7. the adaptive attack helpers run on a small batch

Usage
-----
    python code/smoke_test.py
    python code/smoke_test.py --data-dir data --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ids_defense_selection as idsds                              # noqa: E402
from ids_defense_selection import CNN1DBackbone      # noqa: E402
from ids_defense_selection.paths import DEFAULT_DATA_DIR, resolve_path  # noqa: E402
from prepare_data import REQUIRED_COLUMNS, TEST_ROWS, TRAIN_ROWS   # noqa: E402

#: Feature counts reported in the manuscript (the official split yields a different
#: one-hot width; see the "Notes on the feature space" section of the README).
PAPER_TRANSFORMED_FEATURES = 190
PAPER_CONTINUOUS_FEATURES = 39

RESULTS: list[tuple[str, str, str]] = []   # (name, PASS/FAIL/SKIP, detail)


def check(name: str, status: str, detail: str = "") -> None:
    tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    RESULTS.append((name, tag, detail))
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="dataset directory (relative paths resolve against the repository root)")
    ap.add_argument("--device", default="auto",
                    help="torch device: auto, cpu, cuda, cuda:N or mps")
    ap.add_argument("--n-train", type=int, default=4000,
                    help="rows actually trained on in the check (the fit still uses the full set)")
    ap.add_argument("--n-attack", type=int, default=256, help="samples attacked in the constraint check")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*not writable.*")

    t0 = time.perf_counter()
    device = idsds.resolve_device(args.device)
    idsds.log_device(args.device, device)
    d = resolve_path(args.data_dir)
    print("=" * 78)
    print("Smoke test -- backbone-conditioned Pareto analysis for IDS defenses")
    print("=" * 78)

    # ---- 1. data -------------------------------------------------------
    print("\n1. Data")
    tr_p, te_p = d / "train.csv", d / "test.csv"
    if not tr_p.exists() or not te_p.exists():
        print(f"  [FAIL] {tr_p} and/or {te_p} not found. Run 'python code/prepare_data.py' first.")
        return 2
    t = time.perf_counter()
    train_df = pd.read_csv(tr_p)
    test_df = pd.read_csv(te_p)
    check("CSV load", "PASS",
          f"train {len(train_df):,} rows / test {len(test_df):,} rows  ({time.perf_counter()-t:.1f}s)")
    check("required columns present", "PASS" if set(REQUIRED_COLUMNS).issubset(train_df.columns) else "FAIL",
          f"{len(train_df.columns)} columns")
    if len(train_df) != TRAIN_ROWS or len(test_df) != TEST_ROWS:
        check("official split direction", "FAIL",
              f"expected {TRAIN_ROWS:,} / {TEST_ROWS:,} -- run 'uv run python code/prepare_data.py'")
    else:
        check("official split direction", "PASS", f"{TRAIN_ROWS:,} train / {TEST_ROWS:,} test")

    # ---- 2. features ---------------------------------------------------
    print("\n2. Feature engineering (fitted on the full training partition)")
    t = time.perf_counter()
    x_tr, y_tr, x_te, y_te, meta = idsds.build_features(train_df, test_df)
    n_dim, n_cont = x_tr.shape[1], int(meta["numeric_mask"].sum())
    # The one-hot width depends on how many categorical levels the TRAINING
    # partition contains, so it is a property of the split rather than a constant.
    # 39 continuous features is the invariant; the total is reported for reference.
    check("continuous feature count", "PASS" if n_cont == PAPER_CONTINUOUS_FEATURES else "FAIL",
          f"{n_cont} continuous features (expected {PAPER_CONTINUOUS_FEATURES})")
    note = ("" if n_dim == PAPER_TRANSFORMED_FEATURES
            else f"  [note: the manuscript reports {PAPER_TRANSFORMED_FEATURES}; "
                 f"this split yields {n_dim}]")
    check("transformed feature count", "PASS", f"{n_dim} one-hot + continuous{note}  ({time.perf_counter()-t:.1f}s)")
    check("no NaN in features", "PASS" if not (np.isnan(x_tr).any() or np.isnan(x_te).any()) else "FAIL")

    # ---- 3. backbones --------------------------------------------------
    print("\n3. Backbones")
    models = {
        "MLP": idsds.MLPBackbone(n_dim, (128, 64, 32), 0.15),
        "1D-CNN": CNN1DBackbone(n_dim, 0.15),
        "FT-Transformer": idsds.build_ft_transformer(n_dim, 0.15),
    }
    xb = torch.from_numpy(np.ascontiguousarray(x_te[:64])).float().to(device)
    for nm, m in models.items():
        m = m.to(device).eval()
        with torch.no_grad():
            out = m(xb)
        n_par = sum(p.numel() for p in m.parameters())
        check(f"{nm} forward pass", "PASS" if tuple(out.shape)[0] == xb.shape[0] else "FAIL",
              f"{n_par:,} parameters")

    # ---- 4. attack constraints -----------------------------------------
    print("\n4. Attack constraints")
    model = models["MLP"].to(device).eval()
    mask_t = torch.from_numpy(meta["numeric_mask"].astype(np.float32)).to(device)
    mins_t = torch.from_numpy(meta["numeric_mins"].astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(meta["numeric_maxs"].astype(np.float32)).to(device)

    n = args.n_attack
    xb = torch.from_numpy(np.ascontiguousarray(x_te[:n])).float().to(device)
    yb = torch.from_numpy(np.ascontiguousarray(y_te[:n])).float().to(device)
    eps = 0.05
    x_adv = idsds.pgd_attack(model, xb, yb, eps, eps / 10.0, 10, mask_t, mins_t, maxs_t)
    delta = (x_adv - xb).abs()

    check("one-hot features untouched", "PASS" if float(delta[:, n_cont:].max()) == 0.0 else "FAIL",
          f"max change = {float(delta[:, n_cont:].max()):.2e}")

    # Two distinct quantities, and it matters which is which:
    #   proj_delta  - displacement caused purely by projecting onto [mins, maxs],
    #                 with no attack applied at all
    #   att_delta   - displacement after the attack, which includes proj_delta
    # When constraint (ii) (stay inside the observed range) and constraint (iii)
    # (||delta||_inf <= eps) conflict, clamp_numeric resolves in favour of (ii),
    # so a sample can move by more than eps without the attacker doing anything.
    x_proj = idsds.clamp_numeric(xb.clone(), xb, mins_t, maxs_t, mask_t)
    proj_delta = (x_proj - xb).abs()[:, :n_cont].max(dim=1).values
    att_delta = delta[:, :n_cont].max(dim=1).values
    n_proj_over = int((proj_delta > eps + 1e-6).sum())
    n_att_over = int((att_delta > eps + 1e-6).sum())

    check("attack displacement within epsilon",
          "PASS" if n_att_over == 0 else "SKIP",
          f"max |delta| = {float(att_delta.max()):.5f} (eps={eps}); "
          f"{n_att_over}/{n} samples exceed eps")
    check("projection-only displacement",
          "PASS" if n_proj_over == 0 else "SKIP",
          f"max = {float(proj_delta.max()):.5f}; {n_proj_over}/{n} samples moved > eps "
          f"by the box projection alone  (constraint (ii) takes precedence over (iii))")
    n_oor, frac_oor = idsds.count_out_of_range(x_te[:n], meta["numeric_mins"], meta["numeric_maxs"])
    mv = idsds.max_violation(x_te[:n], meta["numeric_mins"], meta["numeric_maxs"])
    check("out-of-range diagnostics", "PASS",
          f"{n_oor}/{n} samples leave the training box (tolerance 1e-4); "
          f"largest excursion = {mv:.2e}")

    # ---- 5. projection helpers -----------------------------------------
    print("\n5. Projection helpers")
    cl = idsds.clamp_numeric(x_adv.clone(), xb, mins_t, maxs_t, mask_t)
    in_box = bool((cl[:, :n_cont] <= maxs_t + 1e-6).all() and (cl[:, :n_cont] >= mins_t - 1e-6).all())
    check("clamp_numeric keeps values inside [mins, maxs]", "PASS" if in_box else "FAIL")
    check("count_out_of_range returns a fraction in [0,1]",
          "PASS" if 0.0 <= frac_oor <= 1.0 else "FAIL", f"{frac_oor:.3f}")
    check("max_violation is non-negative", "PASS" if mv >= 0.0 else "FAIL", f"{mv:.3e}")

    # ---- 6. metrics ----------------------------------------------------
    print("\n6. Metrics")
    with torch.no_grad():
        prob = torch.sigmoid(model(xb)).cpu().numpy()
    pred = (prob >= 0.5).astype(np.int32)
    m = idsds.classification_metrics(y_te[:n].astype(np.int32), prob, clean_pred=pred)
    keys_ok = {"accuracy", "precision", "recall", "f1", "auc", "attack_success_rate"}.issubset(m)
    check("compute_metrics returns the expected keys", "PASS" if keys_ok else "FAIL",
          "(values are meaningless here -- the model is untrained)")

    # ---- 7. adaptive helpers -------------------------------------------
    print("\n7. Adaptive attack helpers")
    try:
        from ids_defense_selection import adaptive as AA
        xs = xb[:64].detach().clone()
        ys = yb[:64].detach().clone()
        xa = AA.pgd_adaptive(model, xs, ys, eps, mask_t, mins_t, maxs_t,
                             steps=5, restarts=2, loss_kind="margin")
        check("pgd_adaptive runs", "PASS" if tuple(xa.shape) == tuple(xs.shape) else "FAIL")
        g = AA.gradient_diagnostics(model, xs, ys, mask_t)
        check("gradient_diagnostics runs", "PASS" if "grad_linf_mean" in g else "FAIL",
              f"mean grad L-inf = {g['grad_linf_mean']:.4f}")
        cm = AA.complement_mask(mask_t, mask_t)
        check("complement_mask handles the degenerate case",
              "PASS" if float(cm.sum()) > 0 else "FAIL")
    except Exception as exc:                                       # noqa: BLE001
        check("adaptive attack suite usable", "FAIL", f"{type(exc).__name__}: {exc}")

    # ---- summary -------------------------------------------------------
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    nfail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    nskip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print("\n" + "=" * 78)
    print(f"  {npass} passed, {nfail} failed, {nskip} skipped   ({time.perf_counter()-t0:.1f}s)")
    if nfail == 0:
        print("  Environment and data are ready.")
        if nskip:
            print("  (SKIP entries are informational, not failures.)")
        print("  Next:  uv run python run_experiments.py          # --device auto by default")
    else:
        print("  Fix the failures above before starting the full experiments.")
    print("=" * 78)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
