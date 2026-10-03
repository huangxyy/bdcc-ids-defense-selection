#!/usr/bin/env python
"""Compare mean- and median-based aggregation of the multi-seed results.

The decision layer aggregates each objective over seeds.  Mean and median can
disagree when Pareto decisions hinge on 10^-3 margins, so this script rebuilds
the four objectives under both aggregations, re-derives the Pareto fronts and
the preference-based recommendations, and reports where they differ.

    uv run python scripts/check_aggregation_robustness.py --outputs-root outputs
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection import style as FS
from ids_defense_selection.selection import (
    OBJECTIVE_COLUMNS,
    THETA_PRESETS,
    is_pareto_optimal,
    select_defense,
)

BACKBONES = (("mlp", "MLP", None), ("cnn1d", "1D-CNN", "phi4_cnn"),
             ("ft_transformer", "FT-Transformer", "phi4_ft"))
REF_ATTACK, REF_EPSILON = "pgd", 0.10
TAU2, TAU4 = 0.90, 0.60


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


def _aggregate(per_seed: pd.DataFrame, how: str) -> pd.DataFrame:
    matrix = per_seed.reset_index().groupby("model")[OBJECTIVE_COLUMNS].agg(how)
    matrix = matrix.reindex([model for model in FS.DEFENSE_ORDER if model in matrix.index])
    return matrix


def _decide(matrix: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    feasible = matrix[(matrix["phi2"] >= TAU2) & (matrix["phi4"] >= TAU4)]
    if feasible.empty:
        return [], {name: "none" for name in THETA_PRESETS}
    mask = is_pareto_optimal(feasible[OBJECTIVE_COLUMNS].values)
    front = feasible[mask]
    names = front.index.tolist()
    selections = {
        name: FS.get_label(select_defense(front[OBJECTIVE_COLUMNS].values, names, theta))
        for name, theta in THETA_PRESETS.items()
    }
    return names, selections


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out-dir", default=None, help="default: <outputs-root>/tables")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, differences = [], 0
    for subdir, label, phi4_subdir in BACKBONES:
        directory = root / subdir
        phi4_dir = root / phi4_subdir if phi4_subdir else None
        per_seed = _per_seed_objectives(directory, phi4_dir)
        results = {}
        for how in ("mean", "median"):
            matrix = _aggregate(per_seed, how)
            front, selections = _decide(matrix)
            results[how] = (matrix, front, selections)
            rows.append({
                "backbone": label, "aggregation": how,
                "front": ", ".join(FS.get_label(name) for name in front),
                **{f"selected_{name}": value for name, value in selections.items()},
            })
        mean_front = {FS.get_label(name) for name in results["mean"][1]}
        median_front = {FS.get_label(name) for name in results["median"][1]}
        changed = mean_front ^ median_front
        if changed:
            differences += 1
        print(f"\n[{label}] mean front : {sorted(mean_front)}")
        print(f"[{label}] median front: {sorted(median_front)}")
        if changed:
            print(f"[{label}] membership differs: {sorted(changed)}")
        for name in THETA_PRESETS:
            if results["mean"][2][name] != results["median"][2][name]:
                print(f"[{label}] {name}: {results['mean'][2][name]} (mean) vs "
                      f"{results['median'][2][name]} (median)")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "aggregation_robustness.csv", index=False)
    print(f"\nwrote {out_dir / 'aggregation_robustness.csv'}")
    print(f"backbones whose front changes between mean and median: {differences}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
