"""
Quick single-seed category evaluation for FT-Transformer.

Trains all 6 defense models with seed=42, evaluates per-attack-category
recall under PGD eps=0.10, and writes category_mean_results.csv to the
ft_transformer output directory.

Usage:
  python code/ft_category_eval.py --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ids_core as base
from backbone_ft_transformer import (
    FTTransformer,
    train_all_ft_models,
    MODEL_ORDER,
)

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "ft_transformer_matched_budget_run"


def main():
    parser = argparse.ArgumentParser(description="Quick phi4 estimation for FT-Transformer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", default=None,
                        help="output directory (defaults to outputs/<backbone>)")
    parser.add_argument("--train-adv-steps", type=int, default=7)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else out_dir

    device = torch.device(args.device)
    data_root = Path(__file__).resolve().parent.parent / "data"
    config = base.ExperimentConfig(
        train_path=str(data_root / "train.csv"),
        test_path=str(data_root / "test.csv"),
        output_dir=str(out_dir),
        training_budget_mode="matched_continuation",
        batch_size=args.batch_size,
        baseline_epochs=10,
        adv_epochs=8,
        seeds=(SEED,),
        device=args.device,
    )
    base.CONFIG = config

    # Save full eval adv_steps, use reduced for training
    eval_adv_steps = config.adv_steps
    config.adv_steps = args.train_adv_steps

    # ---- Data ----
    train_df, test_df = base.load_unsw_split(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = base.build_features(train_df, test_df)

    eval_indices = base.stratified_subset_indices(y_test, config.eval_attack_rows, seed=2026)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]

    # Attack categories from original test data
    eval_attack_categories = test_df["attack_cat"].fillna("Unknown").to_numpy()[eval_indices]
    top_attack_categories = (
        pd.Series(eval_attack_categories[eval_y == 1])
        .value_counts()
        .head(config.top_attack_categories)
        .index.tolist()
    )
    print(f"[phi4-eval] Top attack categories: {top_attack_categories}")

    # ---- Train ----
    base.set_seed(SEED)
    train_loader = base.make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)

    print(f"[phi4-eval] Training all 6 models with seed={SEED}...")
    t0 = time.perf_counter()
    (
        model_registry,
        train_cost_registry,
        perturb_ratio_registry,
        selected_feature_registry,
        eval_context,
        _sensitivity_table,
    ) = train_all_ft_models(
        config=config,
        train_loader=train_loader,
        input_dim=x_train.shape[1],
        device=device,
        metadata=metadata,
    )
    print(f"[phi4-eval] Training done in {time.perf_counter()-t0:.1f}s")

    # Restore full PGD steps for evaluation
    config.adv_steps = eval_adv_steps

    # ---- Category evaluation ----
    category_results = []
    for model_name in MODEL_ORDER:
        model = model_registry[model_name]
        cat_df = base.evaluate_attack_categories(
            model_name=model_name,
            model=model,
            x_eval=eval_x,
            y_eval=eval_y,
            attack_categories=eval_attack_categories,
            selected_categories=top_attack_categories,
            attack_name=config.category_attack,
            epsilon=config.category_epsilon,
            attack_mask=eval_context["numeric_mask"],
            mins=eval_context["numeric_mins"],
            maxs=eval_context["numeric_maxs"],
            device=device,
            batch_size=config.batch_size,
        )
        cat_df.insert(0, "seed", SEED)
        category_results.append(cat_df)
        phi4_val = cat_df["adv_recall"].min()
        print(f"  {model_name:30s}  phi4 = {phi4_val:.4f}  (min adv_recall across {len(cat_df)} categories)")

    category_df = pd.concat(category_results, ignore_index=True)
    # Since single seed, mean == raw values
    category_mean_df = (
        category_df.drop(columns=["seed"])
        .groupby(["model", "attack", "epsilon", "attack_cat"], as_index=False)
        .agg(
            samples=("samples", "mean"),
            clean_recall=("clean_recall", "mean"),
            adv_recall=("adv_recall", "mean"),
            recall_drop=("recall_drop", "mean"),
        )
        .sort_values(["model", "recall_drop"], ascending=[True, False])
    )
    category_mean_df["samples"] = category_mean_df["samples"].round().astype(int)

    # ---- Write ----
    out_raw = out_dir / "category_raw_results.csv"
    out_mean = out_dir / "category_mean_results.csv"
    category_df.to_csv(out_raw, index=False)
    category_mean_df.to_csv(out_mean, index=False)
    print(f"\n[phi4-eval] Saved: {out_raw}")
    print(f"[phi4-eval] Saved: {out_mean}")

    # ---- Summary ----
    print("\n=== FT-Transformer phi4 (single-seed estimate) ===")
    for model_name in MODEL_ORDER:
        subset = category_mean_df[category_mean_df["model"] == model_name]
        if not subset.empty:
            print(f"  {model_name:30s}  phi4 = {subset['adv_recall'].min():.6f}")


if __name__ == "__main__":
    main()
