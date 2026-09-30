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
sys.path.insert(0, str(CODE))

from ids_defense_selection.paths import (  # noqa: E402
    BACKBONE_OUTPUT_SUBDIRS,
    BACKBONE_RUNNERS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    resolve_path,
)

#: Backbone CLI key -> runner script, canonical output directory and description.
BACKBONES = {
    "mlp": {
        "script": BACKBONE_RUNNERS["mlp"],
        "out": DEFAULT_OUTPUT_ROOT / BACKBONE_OUTPUT_SUBDIRS["mlp"],
        "extra": [],
        "note": "MLP backbone (128-64-32)",
    },
    "cnn": {
        "script": BACKBONE_RUNNERS["cnn"],
        "out": DEFAULT_OUTPUT_ROOT / BACKBONE_OUTPUT_SUBDIRS["cnn"],
        "extra": [],
        "note": "1D-CNN backbone",
    },
    "ft": {
        "script": BACKBONE_RUNNERS["ft"],
        "out": DEFAULT_OUTPUT_ROOT / BACKBONE_OUTPUT_SUBDIRS["ft"],
        "extra": [],
        "note": "FT-Transformer backbone (slowest: allow several hours per seed)",
    },
}


def build_cmd(key: str, args, data_dir: Path) -> list[str]:
    spec = BACKBONES[key]
    cmd = [sys.executable, "-u", str(CODE / spec["script"]),
           "--train-path", str(data_dir / "train.csv"),
           "--test-path", str(data_dir / "test.csv"),
           "--output-dir", str(spec["out"]),
           "--device", args.device,
           "--training-budget-mode", "matched_continuation"]
    cmd += spec["extra"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", default="mlp,cnn,ft",
                    help="comma-separated subset of: mlp, cnn, ft")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--device", default="auto",
                    help="torch device passed to every backbone: auto, cpu, cuda, cuda:N or mps")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    data_dir = resolve_path(args.data_dir)
    args.data_dir = str(data_dir)

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
    if args.dry_run:
        print("   (dry run -- dataset check skipped)")
    else:
        rc = subprocess.call([sys.executable, str(CODE / "prepare_data.py"), "--data-dir", args.data_dir])
        if rc != 0:
            print("\nDataset is not ready. Fix it and re-run.")
            return rc
    print()

    t_all = time.perf_counter()
    done: list[tuple[str, float, str]] = []
    for i, key in enumerate(keys, 1):
        spec = BACKBONES[key]
        cmd = build_cmd(key, args, data_dir)
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
        done.append((key, dt, str(spec["out"])))
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
    print("  uv run python code/evaluate_phi4_cnn.py --device cuda --output-dir outputs/phi4_cnn")
    print("  uv run python code/evaluate_phi4_ft.py  --device cuda --output-dir outputs/phi4_ft")
    print("  uv run python code/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
