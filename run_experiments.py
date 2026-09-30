#!/usr/bin/env python
"""Reproduce the full experiment suite, one backbone after another.

Runs the three backbone experiments in sequence (running them in parallel would
make them compete for the GPU and invalidate the training-cost measurements,
which are part of objective phi3).

    python run_experiments.py --device cuda
    python run_experiments.py --backbones mlp,cnn --device cuda
    python run_experiments.py --dry-run

Outputs land in outputs/<backbone>/:
    mean_results.csv, std_results.csv, raw_results.csv,
    significance_tests.csv, efficiency_mean.csv, run_summary.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"

BACKBONES = {
    "mlp": {
        "script": "ids_core.py",
        "out": "outputs/mlp",
        "extra": [],
        "note": "MLP backbone (128-64-32)",
    },
    "cnn": {
        "script": "backbone_cnn1d.py",
        "out": "outputs/cnn1d",
        "extra": [],
        "note": "1D-CNN backbone",
    },
    "ft": {
        "script": "backbone_ft_transformer.py",
        "out": "outputs/ft_transformer",
        "extra": [],
        "note": "FT-Transformer backbone (slowest: allow several hours per seed)",
    },
}


def build_cmd(key: str, args) -> list[str]:
    spec = BACKBONES[key]
    cmd = [sys.executable, "-u", str(CODE / spec["script"]),
           "--train-path", str(Path(args.data_dir) / "train.csv"),
           "--test-path", str(Path(args.data_dir) / "test.csv"),
           "--output-dir", str(ROOT / spec["out"]),
           "--device", args.device,
           "--training-budget-mode", "matched_continuation"]
    cmd += spec["extra"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", default="mlp,cnn,ft",
                    help="comma-separated subset of: mlp, cnn, ft")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    keys = [k.strip().lower() for k in args.backbones.split(",") if k.strip()]
    unknown = [k for k in keys if k not in BACKBONES]
    if unknown:
        print(f"unknown backbone(s): {unknown}. choose from {list(BACKBONES)}")
        return 2

    print("=" * 78)
    print("Backbone-conditioned Pareto analysis -- full reproduction")
    print("=" * 78)
    print(f"  backbones : {', '.join(keys)}")
    print(f"  data dir  : {args.data_dir}")
    print(f"  device    : {args.device}")
    print()

    # Step 0: dataset layout. Never start a multi-hour run on a reversed split.
    print("-- step 0: dataset check " + "-" * 46)
    rc = subprocess.call([sys.executable, str(CODE / "prepare_data.py"), "--data-dir", args.data_dir])
    if rc != 0:
        print("\nDataset is not ready. Fix it and re-run.")
        return rc
    print()

    t_all = time.perf_counter()
    done: list[tuple[str, float, str]] = []
    for i, key in enumerate(keys, 1):
        spec = BACKBONES[key]
        cmd = build_cmd(key, args)
        print(f"-- step {i}/{len(keys)}: {key} -- {spec['note']} " + "-" * 20)
        print("   " + " ".join(cmd))
        if args.dry_run:
            print("   (dry run -- not executed)")
            continue
        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"   FAILED (exit {rc}) after {dt/60:.1f} min")
            return rc
        print(f"   done in {dt/60:.1f} min -> {spec['out']}/")
        done.append((key, dt, spec["out"]))
        print()

    if args.dry_run:
        print("dry run complete.")
        return 0

    print("=" * 78)
    print(f"All done in {(time.perf_counter()-t_all)/60:.1f} min")
    for key, dt, out in done:
        print(f"  {key:<5} {dt/60:>7.1f} min   {out}/")
    print()
    print("Next steps")
    print("  python code/eval_worst_class_recall_cnn_5seed.py --device cuda --out-dir outputs/phi4_cnn")
    print("  python code/eval_worst_class_recall_ft_5seed.py  --device cuda --out-dir outputs/phi4_ft")
    print("  python code/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
