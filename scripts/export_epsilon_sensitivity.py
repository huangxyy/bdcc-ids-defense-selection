#!/usr/bin/env python
"""Export the perturbation-budget sensitivity table and figure.

Reads the per-seed means/stds of the main runs and builds, for every backbone
and attack family, the ASR and resilience (φ2 = 1 − ASR) as a function of ε.
C&W is listed separately because its implementation does not depend on ε.

    uv run python scripts/export_epsilon_sensitivity.py --attack pgd
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ids_defense_selection import style as FS

BACKBONES = (("mlp", "MLP"), ("cnn1d", "1D-CNN"), ("ft_transformer", "FT-Transformer"))
ATTACKS = ("pgd", "fgsm", "apgd")


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out-dir", default=None, help="default: <outputs-root>/tables")
    parser.add_argument("--attack", default="pgd", choices=ATTACKS,
                        help="attack shown in the Markdown table and figure")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for subdir, label in BACKBONES:
        mean = _load(root / subdir / "mean_results.csv")
        std = _load(root / subdir / "std_results.csv")
        if mean is None or std is None:
            print(f"[skip] {subdir}: mean/std results missing")
            continue
        key = ["model", "attack", "epsilon"]
        merged = mean[key + ["f1", "attack_success_rate"]].merge(
            std[key + ["attack_success_rate"]], on=key,
            suffixes=("_mean", "_std"))
        merged = merged[merged["attack"].isin(ATTACKS)].copy()
        merged["backbone"] = label
        merged["phi2_mean"] = 1.0 - merged["attack_success_rate_mean"]
        merged["phi2_std"] = merged["attack_success_rate_std"]
        frames.append(merged)
    if not frames:
        print("no mean/std results found")
        return 1
    long = pd.concat(frames, ignore_index=True)
    long = long.rename(columns={
        "attack_success_rate_mean": "asr_mean",
        "attack_success_rate_std": "asr_std",
    })[["backbone", "attack", "model", "epsilon", "asr_mean", "asr_std",
        "phi2_mean", "phi2_std"]]
    long.to_csv(out_dir / "epsilon_sensitivity.csv", index=False)

    def _pct(mean: float, std: float) -> str:
        if pd.isna(std):
            return f"{100 * mean:.2f}"
        return f"{100 * mean:.2f} ± {100 * std:.2f}"

    lines = [f"# Perturbation-budget sensitivity ({args.attack.upper()})", "",
             "ASR in percent (mean ± std over seeds); φ2 = 1 − ASR.", ""]
    for _, label in BACKBONES:
        subset = long[(long["backbone"] == label) & (long["attack"] == args.attack)]
        if subset.empty:
            continue
        epsilons = sorted(subset["epsilon"].unique())
        header = ["Defense", *(f"ε={eps:g}" for eps in epsilons)]
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * len(header)) + "|"]
        order = {name: index for index, name in enumerate(FS.DEFENSE_ORDER)}
        models = sorted(subset["model"].unique(),
                        key=lambda name: order.get(name, 99))
        for model in models:
            row = [FS.get_label(model)]
            for eps in epsilons:
                cell = subset[(subset["model"] == model) &
                              np.isclose(subset["epsilon"], eps)]
                row.append(_pct(cell.iloc[0]["asr_mean"], cell.iloc[0]["asr_std"])
                           if not cell.empty else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["> C&W (L2) is not shown: its implementation is independent of ε "
              "(identical ASR at every budget).", ""]
    (out_dir / "epsilon_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")

    figure, axes = plt.subplots(1, len(BACKBONES), figsize=(3.35 * len(BACKBONES), 2.9),
                                squeeze=False)
    for axis, (_, label) in zip(axes[0], BACKBONES):
        subset = long[(long["backbone"] == label) & (long["attack"] == args.attack)]
        if subset.empty:
            axis.set_visible(False)
            continue
        order = {name: index for index, name in enumerate(FS.DEFENSE_ORDER)}
        for model in sorted(subset["model"].unique(), key=lambda n: order.get(n, 99)):
            cell = subset[subset["model"] == model].sort_values("epsilon")
            axis.plot(cell["epsilon"], 100 * cell["asr_mean"],
                      marker="o", markersize=3, linewidth=1.0,
                      color=FS.get_color(model), label=FS.get_label(model))
        axis.set_xlabel("ε", fontsize=8)
        axis.set_ylabel("ASR (%)", fontsize=8)
        axis.set_title(label, fontsize=9, pad=4)
        axis.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=6, ncol=2, loc="lower center")
    figure.suptitle(f"{args.attack.upper()} sensitivity to the perturbation budget",
                    fontsize=10)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(out_dir / f"epsilon_sensitivity_{args.attack}.png", dpi=200)
    plt.close(figure)
    print(f"wrote {out_dir / 'epsilon_sensitivity.csv'}")
    print(f"wrote {out_dir / 'epsilon_sensitivity.md'}")
    print(f"wrote {out_dir / f'epsilon_sensitivity_{args.attack}.png'}")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
