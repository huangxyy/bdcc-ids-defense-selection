#!/usr/bin/env python
"""Classify the Pareto front into supported and unsupported efficient points."""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import sys
from pathlib import Path

import pandas as pd

from ids_defense_selection.supportedness import supported_solutions

BACKBONES = {"mlp": "mlp", "cnn1d": "cnn1d", "ft": "ft_transformer"}
OBJECTIVE_COLUMNS = ["phi1_clean_f1", "phi2_resilience", "phi3_cost_eff", "phi4_fairness"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--backbones", default="mlp,cnn1d,ft",
                        help="comma-separated subset of: mlp, cnn1d, ft")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    keys = [key.strip().lower() for key in args.backbones.split(",") if key.strip()]
    unknown = [key for key in keys if key not in BACKBONES]
    if unknown:
        print(f"unknown backbone(s): {unknown}; choose from {sorted(BACKBONES)}",
              file=sys.stderr)
        return 2

    for key in keys:
        run = root / BACKBONES[key]
        profile_path = run / "risk_profile_4d.csv"
        if not profile_path.is_file():
            print(f"[skip] {profile_path} is missing", file=sys.stderr)
            continue
        profile = pd.read_csv(profile_path)
        mask = profile["is_pareto_optimal"].astype(bool)
        if "admissible" in profile.columns:
            mask &= profile["admissible"].astype(bool)
        front = profile[mask].set_index("model")[OBJECTIVE_COLUMNS]
        table = supported_solutions(front)
        out_path = run / "supportedness.csv"
        table.to_csv(out_path, index=False)

        unsupported = table.loc[~table["supported"].astype(bool), "model"].tolist()
        print(f"\n[{BACKBONES[key]}] front size={len(table)} -> {out_path}")
        print(table[["model", "supported", "margin"]].round(6).to_string(index=False))
        if unsupported:
            print(f"  unsupported (unreachable by any theta): {unsupported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
