#!/usr/bin/env python
"""Bootstrap dominance probabilities over the seeds (local, no GPU).

Eq. (4) is an interval criterion, not a hypothesis test.  This script quantifies
how often each pairwise dominance relation and each Pareto-front membership
survives a bootstrap resampling of the ten seeds:

    uv run python scripts/bootstrap_dominance.py --outputs-root outputs

The admissibility thresholds (tau2=0.90, tau4=0.60 by default) are applied
inside every resample, mirroring the main decision criterion.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection import style as FS
from ids_defense_selection.selection import OBJECTIVE_COLUMNS, is_pareto_optimal

BACKBONES = (("mlp", "MLP", None), ("cnn1d", "1D-CNN", "phi4_cnn"),
             ("ft_transformer", "FT-Transformer", "phi4_ft"))
REF_ATTACK, REF_EPSILON = "pgd", 0.10


def _per_seed_objectives(directory: Path, phi4_dir: Path | None) -> pd.DataFrame:
    raw = pd.read_csv(directory / "raw_results.csv")
    efficiency = pd.read_csv(directory / "efficiency_raw.csv")
    category = pd.read_csv((phi4_dir or directory) / "category_raw_results.csv")

    clean = raw[(raw["attack"] == "clean") & np.isclose(raw["epsilon"], 0.0)]
    attacked = raw[(raw["attack"] == REF_ATTACK) &
                   np.isclose(raw["epsilon"], REF_EPSILON)]
    phi1 = clean.set_index(["seed", "model"])["f1"]
    phi2 = 1.0 - attacked.set_index(["seed", "model"])["attack_success_rate"]

    standard = (efficiency[efficiency["model"] == "standard"]
                .set_index("seed")["train_seconds"])
    cost = efficiency.copy()
    cost["relative"] = [
        row["train_seconds"] / standard.get(row["seed"], np.nan)
        for _, row in cost.iterrows()
    ]
    phi3 = 1.0 / cost.set_index(["seed", "model"])["relative"].clip(lower=1e-6)

    cat = category[(category["attack"] == REF_ATTACK) &
                   np.isclose(category["epsilon"], REF_EPSILON)]
    phi4 = cat.groupby(["seed", "model"])["adv_recall"].min()

    frame = pd.concat({"phi1": phi1, "phi2": phi2, "phi3": phi3, "phi4": phi4}, axis=1)
    return frame.dropna()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out-dir", default=None, help="default: <outputs-root>/tables")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tau2", type=float, default=0.90)
    parser.add_argument("--tau4", type=float, default=0.60)
    args = parser.parse_args()

    root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pair_rows, front_rows = [], []
    for subdir, label, phi4_subdir in BACKBONES:
        per_seed = _per_seed_objectives(root / subdir,
                                        root / phi4_subdir if phi4_subdir else None)
        seeds = sorted(per_seed.index.get_level_values("seed").unique())
        models = [model for model in FS.DEFENSE_ORDER
                  if model in per_seed.index.get_level_values("model")]
        per_seed = per_seed.reindex(pd.MultiIndex.from_product(
            [seeds, models], names=["seed", "model"]))
        values = per_seed[OBJECTIVE_COLUMNS].to_numpy(float)
        n_models = len(models)
        seed_positions = {seed: np.arange(i * n_models, (i + 1) * n_models)
                          for i, seed in enumerate(seeds)}
        dominance = np.zeros((len(models), len(models)))
        front = np.zeros(len(models))

        for _ in range(args.n_boot):
            picked = rng.choice(seeds, size=len(seeds), replace=True)
            rows = np.concatenate([seed_positions[seed] for seed in picked])
            means = values[rows].reshape(len(picked), n_models, 4).mean(axis=0)
            feasible = [(means[i, 1] >= args.tau2) and (means[i, 3] >= args.tau4)
                        for i in range(len(models))]
            for i in range(len(models)):
                if not feasible[i]:
                    continue
                for j in range(len(models)):
                    if i == j or not feasible[j]:
                        continue
                    if np.all(means[i] >= means[j]) and np.any(means[i] > means[j]):
                        dominance[i, j] += 1
            sub = means[[i for i in range(len(models)) if feasible[i]]]
            if len(sub):
                mask = is_pareto_optimal(sub)
                for position, keep in zip([i for i in range(len(models)) if feasible[i]], mask):
                    front[position] += int(keep)

        for i, model in enumerate(models):
            front_rows.append({"backbone": label, "model": FS.get_label(model),
                               "prob_on_front": front[i] / args.n_boot})
            for j, other in enumerate(models):
                if i != j:
                    pair_rows.append({"backbone": label, "dominant": FS.get_label(model),
                                      "dominated": FS.get_label(other),
                                      "prob": dominance[i, j] / args.n_boot})

        print(f"\n[{label}] front membership probability "
              f"(tau2={args.tau2}, tau4={args.tau4}, B={args.n_boot})")
        for position, model in enumerate(models):
            print(f"  {FS.get_label(model):12s} {front[position] / args.n_boot:.3f}")

    pd.DataFrame(pair_rows).to_csv(out_dir / "bootstrap_dominance_pairs.csv", index=False)
    pd.DataFrame(front_rows).to_csv(out_dir / "bootstrap_front_stability.csv", index=False)
    print(f"\nwrote {out_dir / 'bootstrap_dominance_pairs.csv'}")
    print(f"wrote {out_dir / 'bootstrap_front_stability.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
