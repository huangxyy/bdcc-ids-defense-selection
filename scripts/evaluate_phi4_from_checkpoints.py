#!/usr/bin/env python
"""Compute phi4 (per-category adversarial recall) from saved checkpoints.

The main FT/CNN runs did not carry attack-category labels into their evaluation
set, so their phi4 source is missing.  This script rebuilds the stratified
evaluation subset, loads each seed's checkpoint and computes the per-category
recall without retraining:

    uv run python scripts/evaluate_phi4_from_checkpoints.py \
        --checkpoint-root outputs/cnn1d/checkpoints \
        --output-dir outputs/phi4_cnn --backbone cnn1d \
        --seeds 7,13,21,42,100,11,23,37,59,89 --device cuda

Outputs ``category_raw_results.csv`` (one row per seed) and
``category_mean_results.csv`` in the ``--output-dir``, matching the format the
decision layer reads.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from ids_defense_selection import (
    ExperimentConfig,
    build_features,
    load_checkpoint,
    load_unsw_nb15,
    log_device,
    prepare_attack_categories,
    rebuild_trained_defenses,
    resolve_device,
    stratified_subset_indices,
)
from ids_defense_selection.evaluation import evaluate_category_recall

ROOT = Path(__file__).resolve().parents[1]


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", required=True,
                        help="directory containing seed<seed>/ checkpoint bundles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", required=True,
                        choices=("mlp", "cnn1d", "ft_transformer"))
    parser.add_argument("--seeds", default="7,13,21,42,100,11,23,37,59,89")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-path", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--test-path", default=str(ROOT / "data" / "test.csv"))
    parser.add_argument("--eval-attack-rows", type=int, default=None)
    parser.add_argument("--eval-subset-seed", type=int, default=None)
    parser.add_argument("--category-attack", default=None)
    parser.add_argument("--category-epsilon", type=float, default=None)
    args = parser.parse_args()

    config = ExperimentConfig(
        train_path=args.train_path, test_path=args.test_path,
        output_dir=args.output_dir, device=args.device,
        **{key: value for key, value in {
            "eval_attack_rows": args.eval_attack_rows,
            "eval_subset_seed": args.eval_subset_seed,
            "category_attack": args.category_attack,
            "category_epsilon": args.category_epsilon,
        }.items() if value is not None},
    )
    device = resolve_device(config.device)
    log_device(config.device, device)
    print(f"[phi4] backbone={args.backbone} device={device} "
          f"subset={config.eval_attack_rows} attack={config.category_attack} "
          f"epsilon={config.category_epsilon}", flush=True)

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    indices = stratified_subset_indices(y_test, config.eval_attack_rows,
                                        seed=config.eval_subset_seed)
    x_eval, y_eval = x_test[indices], y_test[indices]
    categories, top_categories = prepare_attack_categories(
        test_df, indices, y_eval, config.top_attack_categories)

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    frames = []
    for seed in seeds:
        bundle = load_checkpoint(Path(args.checkpoint_root) / f"seed{seed}")
        if int(bundle["input_dim"]) != x_train.shape[1]:
            print(f"error: checkpoint expects {bundle['input_dim']} features but the "
                  f"data yields {x_train.shape[1]}", flush=True)
            return 2
        trained = rebuild_trained_defenses(bundle, device)
        for name, model in trained.models.items():
            frame = evaluate_category_recall(
                name, model, x_eval, y_eval, categories, top_categories,
                config.category_attack, config.category_epsilon,
                metadata["numeric_mask"], metadata["numeric_mins"],
                metadata["numeric_maxs"], config, device)
            frame.insert(0, "seed", seed)
            frames.append(frame)
        print(f"[phi4] seed {seed} done", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    mean = (
        raw.drop(columns=["seed"])
        .groupby(["model", "attack", "epsilon", "attack_cat"], as_index=False)
        .agg(samples=("samples", "mean"), clean_recall=("clean_recall", "mean"),
             adv_recall=("adv_recall", "mean"), recall_drop=("recall_drop", "mean"))
    )
    mean["samples"] = mean["samples"].round().astype(int)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "category_raw_results.csv", index=False)
    mean.to_csv(out_dir / "category_mean_results.csv", index=False)
    (out_dir / "phi4_summary.json").write_text(json.dumps({
        "backbone": args.backbone,
        "seeds": seeds,
        "code_revision": _git_revision(),
        "checkpoint_root": str(Path(args.checkpoint_root).expanduser().resolve()),
        "eval_subset_seed": config.eval_subset_seed,
        "eval_attack_rows": config.eval_attack_rows,
        "category_attack": config.category_attack,
        "category_epsilon": config.category_epsilon,
        "raw_rows": int(len(raw)),
        "mean_rows": int(len(mean)),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phi4] saved {len(raw)} raw rows / {len(mean)} mean rows to {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
