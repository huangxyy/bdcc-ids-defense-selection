"""Worst-class recall objective (phi4): per-attack-category evaluation.

Trains every defense once per configured seed and records the per-attack-category
recall under the configured (attack, epsilon) scenario.  ``phi4`` is the minimum
of those recalls, i.e. the worst class.  Shared by the per-backbone
``evaluate_phi4_*.py`` scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch import nn

from .config import ExperimentConfig
from .data import build_features, load_unsw_nb15, make_dataloader, set_seed, stratified_subset_indices
from .defenses import DEFENSE_ORDER, train_all_defenses
from .evaluation import evaluate_category_recall
from .experiment import prepare_attack_categories


def evaluate_phi4(
    config: ExperimentConfig,
    model_factory: Callable[[int], nn.Module],
    device: torch.device,
) -> pd.DataFrame:
    """Run the phi4 evaluation and write category_raw/mean_results.csv.

    Returns the seed-averaged per-category table.  Training uses
    ``config.adv_steps``; every evaluation attack uses ``config.eval_pgd_steps``,
    so the two budgets cannot be confused.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    eval_indices = stratified_subset_indices(y_test, config.eval_attack_rows,
                                             seed=config.eval_subset_seed)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    categories, top_categories = prepare_attack_categories(
        test_df, eval_indices, eval_y, config.top_attack_categories)
    print(f"[phi4] categories: {top_categories}", flush=True)
    print(
        f"[phi4] scenario={config.category_attack} eps={config.category_epsilon} "
        f"train_pgd_steps={config.adv_steps} eval_pgd_steps={config.eval_pgd_steps}",
        flush=True,
    )

    all_results: list[pd.DataFrame] = []
    for seed in config.seeds:
        print(f"\n{'=' * 60}\n[phi4] seed {seed}\n{'=' * 60}", flush=True)
        set_seed(seed)
        train_loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        trained = train_all_defenses(
            model_factory=lambda: model_factory(x_train.shape[1]),
            config=config,
            train_loader=train_loader,
            device=device,
            metadata=metadata,
            seed=seed,
        )

        for model_name, model in trained.models.items():
            category_df = evaluate_category_recall(
                model_name, model, eval_x, eval_y, categories, top_categories,
                config.category_attack, config.category_epsilon,
                metadata["numeric_mask"], metadata["numeric_mins"], metadata["numeric_maxs"],
                config, device,
            )
            category_df.insert(0, "seed", seed)
            all_results.append(category_df)
            print(f"  {model_name:30s} phi4 = {category_df['adv_recall'].min():.4f}", flush=True)

    category_df = pd.concat(all_results, ignore_index=True)
    category_mean_df = (
        category_df.groupby(["model", "attack", "epsilon", "attack_cat"], as_index=False)
        .agg(
            samples=("samples", "mean"),
            clean_recall=("clean_recall", "mean"),
            clean_recall_std=("clean_recall", "std"),
            adv_recall=("adv_recall", "mean"),
            adv_recall_std=("adv_recall", "std"),
            recall_drop=("recall_drop", "mean"),
        )
        .sort_values(["model", "recall_drop"], ascending=[True, False])
    )
    category_mean_df["samples"] = category_mean_df["samples"].round().astype(int)
    category_df.to_csv(output_dir / "category_raw_results.csv", index=False)
    category_mean_df.to_csv(output_dir / "category_mean_results.csv", index=False)

    print(f"\n=== phi4 ({len(config.seeds)} seed(s)) ===", flush=True)
    for model_name in DEFENSE_ORDER:
        subset = category_mean_df[category_mean_df["model"] == model_name]
        if not subset.empty:
            worst = subset.loc[subset["adv_recall"].idxmin()]
            print(
                f"  {model_name:30s} phi4={worst['adv_recall']:.6f} "
                f"(worst class: {worst['attack_cat']})",
                flush=True,
            )
    print(f"\nSaved to {output_dir}", flush=True)
    return category_mean_df
