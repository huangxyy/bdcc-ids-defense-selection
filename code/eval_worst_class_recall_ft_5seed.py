"""Worst-class adversarial recall (phi4) for FT-Transformer, five seeds.

Trains all six defenses once per seed and records per-attack-category recall
under PGD at epsilon = 0.10, then aggregates mean and standard deviation across
seeds. This is the script behind the phi4 column of Table 5.

    python eval_worst_class_recall_ft_5seed.py --device cuda --out-dir outputs/phi4_ft

Always pass --out-dir: without it the results go to a fixed directory and a
second run silently overwrites the first.
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import ids_core as base
from backbone_ft_transformer import FTTransformer, train_all_ft_models, MODEL_ORDER

# five seeds, matching the rest of the study
SEEDS = (7, 13, 21, 42, 100)
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "ft_transformer_matched_budget_run"

# ---------------------------------------------------------------- entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
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
        baseline_epochs=10, adv_epochs=8,
        seeds=SEEDS, device=args.device,
    )
    config.adv_steps = args.train_adv_steps
    base.CONFIG = config

    train_df, test_df = base.load_unsw_split(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = base.build_features(train_df, test_df)
    eval_indices = base.stratified_subset_indices(y_test, config.eval_attack_rows, seed=2026)
    eval_x = x_test[eval_indices]; eval_y = y_test[eval_indices]
    eval_attack_categories = test_df["attack_cat"].fillna("Unknown").to_numpy()[eval_indices]
    top_cats = pd.Series(eval_attack_categories[eval_y==1]).value_counts().head(config.top_attack_categories).index.tolist()
    print(f"[ft-multiseed] Categories: {top_cats}")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\n[ft-multiseed] SEED = {seed}\n{'='*60}")
        base.set_seed(seed)
        train_loader = base.make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        t0 = time.perf_counter()
        model_registry, _, _, _, eval_context, _ = train_all_ft_models(
            config=config, train_loader=train_loader,
            input_dim=x_train.shape[1], device=device, metadata=metadata,
        )
        print(f"[ft-multiseed] Seed {seed} training: {time.perf_counter()-t0:.1f}s")

        for model_name in MODEL_ORDER:
            model = model_registry[model_name]
            cat_df = base.evaluate_attack_categories(
                model_name=model_name, model=model,
                x_eval=eval_x, y_eval=eval_y,
                attack_categories=eval_attack_categories,
                selected_categories=top_cats,
                attack_name=config.category_attack,
                epsilon=config.category_epsilon,
                attack_mask=eval_context["numeric_mask"],
                mins=eval_context["numeric_mins"],
                maxs=eval_context["numeric_maxs"],
                device=device, batch_size=config.batch_size,
            )
            cat_df.insert(0, "seed", seed)
            all_results.append(cat_df)
            phi4 = cat_df["adv_recall"].min()
            print(f"  {model_name:30s} phi4={phi4:.4f}")

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(out_dir / "category_raw_results.csv", index=False)

    cat_mean = all_df.groupby(["model","attack","epsilon","attack_cat"], as_index=False).agg(
        samples=("samples","mean"),
        clean_recall=("clean_recall","mean"), clean_recall_std=("clean_recall","std"),
        adv_recall=("adv_recall","mean"), adv_recall_std=("adv_recall","std"),
        recall_drop=("recall_drop","mean"),
    ).sort_values(["model","recall_drop"], ascending=[True,False])
    cat_mean["samples"] = cat_mean["samples"].round().astype(int)
    cat_mean.to_csv(out_dir / "category_mean_results.csv", index=False)

    print(f"\n=== FT-Transformer phi4 (5-seed) ===")
    for m in MODEL_ORDER:
        sub = cat_mean[cat_mean["model"]==m]
        if not sub.empty:
            phi4 = sub["adv_recall"].min()
            phi4_std = sub.loc[sub["adv_recall"].idxmin(), "adv_recall_std"]
            print(f"  {m:30s} phi4={phi4:.6f} ± {phi4_std:.6f}")

    print(f"\nSaved to {out_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
