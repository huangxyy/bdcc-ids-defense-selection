#!/usr/bin/env python
"""Run the FT-Transformer backbone experiment.

    uv run python code/run_ft_transformer.py --device cuda

The FT-Transformer is the slowest backbone, so its default training attack uses
7 PGD steps (``--adv-steps``, alias ``--train-adv-steps``) while every evaluation
attack still uses the full 20 steps (``--eval-pgd-steps``).  The two budgets are
recorded separately in run_summary.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection import (
    BackboneRunFrames,
    DEFAULT_DATA_DIR,
    FT_TRANSFORMER_KWARGS,
    EvaluationSet,
    FTTransformerBackbone,
    build_features,
    build_parser,
    config_from_args,
    count_parameters,
    default_output_dir,
    emit_config,
    evaluate_defenses,
    load_unsw_nb15,
    log_device,
    make_dataloader,
    resolve_path,
    resolve_device,
    set_seed,
    split_summary,
    stratified_subset_indices,
    train_all_defenses,
    write_backbone_outputs,
)


def main() -> None:
    parser = build_parser(
        "Run the FT-Transformer matched-budget IDS experiment.",
        defaults={
            "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
            "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
            "output_dir": str(default_output_dir("ft")),
            "training_budget_mode": "matched_continuation",
            "batch_size": 512,
            "adv_steps": 7,  # cheaper PGD used during training only
        },
        require_paths=False,
    )
    args = parser.parse_args()

    config = config_from_args(
        args,
        train_path=str(resolve_path(args.train_path)),
        test_path=str(resolve_path(args.test_path)),
        output_dir=str(resolve_path(args.output_dir)),
    )
    if args.print_config:
        emit_config(config)

    out_dir = Path(config.output_dir)
    device = resolve_device(config.device)
    log_device(config.device, device)
    print(
        f"[ft_transformer] output_dir={out_dir} device={device} "
        f"training_budget_mode={config.training_budget_mode} seeds={config.seeds} "
        f"train_pgd_steps={config.adv_steps} eval_pgd_steps={config.eval_pgd_steps}",
        flush=True,
    )

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    eval_indices = stratified_subset_indices(y_test, config.eval_attack_rows,
                                             seed=config.eval_subset_seed)
    eval_set = EvaluationSet(
        x_eval=x_test[eval_indices],
        y_eval=y_test[eval_indices],
        x_test=x_test,
        y_test=y_test,
        attack_mask=metadata["numeric_mask"],
        numeric_mins=metadata["numeric_mins"],
        numeric_maxs=metadata["numeric_maxs"],
    )

    results: list[pd.DataFrame] = []
    sensitivity: list[pd.DataFrame] = []
    efficiency: list[pd.DataFrame] = []
    validity: list[pd.DataFrame] = []
    full_test_clean: list[pd.DataFrame] = []
    full_test_attack: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []
    adaptive_frames: list[pd.DataFrame] = []

    for seed in config.seeds:
        set_seed(seed)
        train_loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        print(f"[ft_transformer][seed {seed}] training started", flush=True)
        trained = train_all_defenses(
            model_factory=lambda: FTTransformerBackbone(x_train.shape[1], **FT_TRANSFORMER_KWARGS),
            config=config,
            train_loader=train_loader,
            device=device,
            metadata=metadata,
            seed=seed,
        )
        print(f"[ft_transformer][seed {seed}] evaluation started", flush=True)
        evaluation = evaluate_defenses(
            trained, config, eval_set, device, seed,
            include_categories=True,
            full_test_attack_settings=config.full_test_attack_settings,
            full_test_attack_rows=config.full_test_attack_rows,
        )
        results.append(evaluation.metrics)
        sensitivity.append(trained.sensitivity_table)
        efficiency.append(evaluation.efficiency)
        validity.append(evaluation.validity)
        full_test_clean.append(evaluation.full_test_clean)
        full_test_attack.append(evaluation.full_test_attack)
        if not evaluation.adaptive.empty:
            adaptive_frames.append(evaluation.adaptive)
        if not evaluation.categories.empty:
            category_frames.append(evaluation.categories)
        print(f"[ft_transformer][seed {seed}] completed", flush=True)

    frames = BackboneRunFrames(
        results=pd.concat(results, ignore_index=True),
        sensitivity=pd.concat(sensitivity, ignore_index=True),
        efficiency=pd.concat(efficiency, ignore_index=True),
        validity=pd.concat(validity, ignore_index=True),
        full_test_clean=pd.concat(full_test_clean, ignore_index=True),
        full_test_attack=pd.concat(full_test_attack, ignore_index=True),
        adaptive=(pd.concat(adaptive_frames, ignore_index=True) if adaptive_frames
                  else pd.DataFrame()),
    )
    reference_model = FTTransformerBackbone(x_train.shape[1], **FT_TRANSFORMER_KWARGS)

    if category_frames:
        category_df = pd.concat(category_frames, ignore_index=True)
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
        category_df.to_csv(out_dir / "category_raw_results.csv", index=False)
        category_mean_df.to_csv(out_dir / "category_mean_results.csv", index=False)

    summary = {
        "architecture": (
            "FT-Transformer (feature_tokenizer: "
            f"{x_train.shape[1]}*Linear(1,{FT_TRANSFORMER_KWARGS['d_token']}), CLS token, "
            f"TransformerEncoder(n_layers={FT_TRANSFORMER_KWARGS['n_layers']}, "
            f"n_heads={FT_TRANSFORMER_KWARGS['n_heads']}, "
            f"d_ffn={FT_TRANSFORMER_KWARGS['d_ffn']}), head: LayerNorm+Linear)"
        ),
        "resolved_device": str(device),
        "dataset_split": split_summary(len(train_df), len(test_df)),
        "parameter_count": count_parameters(reference_model),
        "adv_steps_training": int(config.adv_steps),
        "adv_steps_evaluation": int(config.eval_pgd_steps),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(eval_set.y_eval)),
        "full_test_attack_rows": int(frames.full_test_attack["test_rows"].iloc[0])
        if not frames.full_test_attack.empty else 0,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "transformed_feature_count": int(len(metadata["feature_names"])),
    }
    aggregated = write_backbone_outputs(out_dir, config, frames, summary)

    key = aggregated["mean_results"]
    reference_epsilon = max(config.epsilon_list)
    key = key[(key["attack"] == "pgd") & np.isclose(key["epsilon"], reference_epsilon)]
    print(f"\n=== FT-Transformer key results (PGD eps={reference_epsilon:.2f} on eval subset) ===",
          flush=True)
    print(key[["model", "f1", "attack_success_rate"]].to_string(index=False), flush=True)
    print(f"\nAll saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
