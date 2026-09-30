"""
Run CNN and FT-Transformer phi4 evaluation with 5 seeds, then aggregate.
Saves results directly to the backbone output directories.
"""
import sys, os, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import ids_core as base

SEEDS = (7, 13, 21, 42, 100)
BACKBONES = {
    "cnn": {
        "out_dir": ROOT / "outputs" / "cnn1d_matched_budget_run",
        "train_fn": None,  # set after import
        "model_cls": None,
        "model_order": None,
        "batch_size": 512,
    },
    "ft": {
        "out_dir": ROOT / "outputs" / "ft_transformer_matched_budget_run",
        "train_fn": None,
        "model_cls": None,
        "model_order": None,
        "batch_size": 256,
    },
}

def run_backbone(name, info, device="cuda"):
    device = torch.device(device)
    data_root = ROOT / "data"

    # Import backbone-specific modules
    if name == "cnn":
        from backbone_cnn1d import CNN1D, train_all_cnn_models, MODEL_ORDER
        info["train_fn"] = train_all_cnn_models
        info["model_order"] = MODEL_ORDER
    else:
        from backbone_ft_transformer import FTTransformer, train_all_ft_models, MODEL_ORDER
        info["train_fn"] = train_all_ft_models
        info["model_order"] = MODEL_ORDER

    config = base.ExperimentConfig(
        train_path=str(data_root / "train.csv"),
        test_path=str(data_root / "test.csv"),
        output_dir=str(info["out_dir"]),
        training_budget_mode="matched_continuation",
        batch_size=info["batch_size"],
        baseline_epochs=10, adv_epochs=8,
        seeds=(42,),  # placeholder, overridden per iteration
        device=str(device),
    )
    if name == "ft":
        config.adv_steps = 7

    base.CONFIG = config
    train_df, test_df = base.load_unsw_split(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = base.build_features(train_df, test_df)
    eval_indices = base.stratified_subset_indices(y_test, config.eval_attack_rows, seed=2026)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    eval_cats = test_df["attack_cat"].fillna("Unknown").to_numpy()[eval_indices]
    top_cats = pd.Series(eval_cats[eval_y==1]).value_counts().head(config.top_attack_categories).index.tolist()
    print(f"[{name}] Categories: {top_cats}")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*50}\n[{name}] SEED = {seed}\n{'='*50}")
        base.set_seed(seed)
        train_loader = base.make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        t0 = time.perf_counter()
        model_registry, _, _, _, eval_context, _ = info["train_fn"](
            config=config, train_loader=train_loader,
            input_dim=x_train.shape[1], device=device, metadata=metadata,
        )
        print(f"[{name}] Seed {seed} training: {time.perf_counter()-t0:.0f}s")

        for model_name in info["model_order"]:
            model = model_registry[model_name]
            cat_df = base.evaluate_attack_categories(
                model_name=model_name, model=model,
                x_eval=eval_x, y_eval=eval_y,
                attack_categories=eval_cats,
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

    # Aggregate
    all_df = pd.concat(all_results, ignore_index=True)
    out_dir = info["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "category_raw_results.csv", index=False)

    cat_mean = all_df.groupby(
        ["model","attack","epsilon","attack_cat"], as_index=False
    ).agg(
        samples=("samples","mean"),
        clean_recall=("clean_recall","mean"), clean_recall_std=("clean_recall","std"),
        adv_recall=("adv_recall","mean"), adv_recall_std=("adv_recall","std"),
        recall_drop=("recall_drop","mean"),
    ).sort_values(["model","recall_drop"], ascending=[True,False])
    cat_mean["samples"] = cat_mean["samples"].round().astype(int)
    cat_mean.to_csv(out_dir / "category_mean_results.csv", index=False)

    print(f"\n=== {name} phi4 (5-seed) ===")
    for m in info["model_order"]:
        sub = cat_mean[cat_mean["model"]==m]
        if not sub.empty:
            phi4 = sub["adv_recall"].min()
            idx = sub["adv_recall"].idxmin()
            std = sub.loc[idx, "adv_recall_std"]
            print(f"  {m:30s} phi4={phi4:.6f} +- {std:.6f}")

    return cat_mean

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["cnn","ft","both"], default="both")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.backbone in ("cnn","both"):
        run_backbone("cnn", BACKBONES["cnn"], args.device)
    if args.backbone in ("ft","both"):
        run_backbone("ft", BACKBONES["ft"], args.device)
    print("\nDone.")
