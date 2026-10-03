#!/usr/bin/env python
"""Multiple-comparison control and effect sizes for the paired defense tests.

Reads each backbone's ``raw_results.csv`` and writes
``significance_enhanced.csv`` with Holm/BH adjusted p-values, paired Cohen's
d_z and bootstrap confidence intervals for the mean difference:

    uv run python scripts/enhance_significance.py --outputs-root outputs
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection.stats import enhance_significance

BACKBONES = {"mlp": "mlp", "cnn1d": "cnn1d", "ft": "ft_transformer"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--backbones", default="mlp,cnn1d,ft",
                        help="comma-separated subset of: mlp, cnn1d, ft")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
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
        raw_path = run / "raw_results.csv"
        if not raw_path.is_file():
            print(f"[skip] {raw_path} is missing", file=sys.stderr)
            continue
        raw = pd.read_csv(raw_path)
        table = enhance_significance(raw, n_boot=args.n_boot,
                                     alpha=args.alpha, seed=args.seed)
        out_path = run / "significance_enhanced.csv"
        table.to_csv(out_path, index=False)

        pgd = table[(table["attack"] == "pgd") &
                    np.isclose(table["epsilon"], 0.10) &
                    (table["metric"] == "f1")].copy()
        top = pgd.reindex(pgd["cohens_dz"].abs().sort_values(ascending=False).index).head(3)
        print(f"\n[{BACKBONES[key]}] {len(table)} comparisons -> {out_path}")
        print(f"  significant after Holm (t / Wilcoxon): "
              f"{int(pgd['significant_holm_t'].sum())} / "
              f"{int(pgd['significant_holm_wilcoxon'].sum())} "
              f"of {len(pgd)} PGD eps=0.10 f1 comparisons")
        for row in top.itertuples():
            print(f"  {row.comparison:45s} diff={row.mean_diff:+.4f} "
                  f"dz={row.cohens_dz:+.2f} CI=[{row.ci_low:+.4f},{row.ci_high:+.4f}] "
                  f"p_holm={row.holm_t_pvalue:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
