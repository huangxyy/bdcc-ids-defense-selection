"""The MLP backbone experiment: the full reference pipeline end to end.

Trains the six defense candidates plus the optional extras, evaluates the full
attack suite, the transfer-attack matrix, the sensitivity-ratio ablation and the
non-neural reference models, and writes every table and figure into the output
directory.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attacks import pgd_attack
from .backbones import MLPBackbone
from .config import ExperimentConfig
from .data import (
    build_features,
    load_unsw_nb15,
    make_dataloader,
    set_seed,
    stratified_subset_indices,
)
from .defenses import compute_sensitivity_mask, fit_reference_models, fit_supervised, train_all_defenses
from .evaluation import (
    EvaluationSet,
    classification_metrics,
    evaluate_defenses,
    evaluate_model_on_attack,
    evaluate_transfer_attack,
    measure_inference_cost,
    predict_proba,
)
from .reporting import (
    compute_significance_tests,
    plot_clean_f1_bar,
    plot_efficiency_tradeoff,
    plot_metric_curve,
    plot_ratio_ablation,
    plot_transfer_heatmap,
    summarize_results,
)

#: Seed of the fixed stratified evaluation subset shared by every defense/backbone.
EVAL_SUBSET_SEED = 2026


def run_mlp_experiment(config: ExperimentConfig) -> None:
    """Train and evaluate every defense on the MLP backbone. Writes all outputs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    print(
        f"[mlp] output_dir={output_dir} device={config.device} "
        f"training_budget_mode={config.training_budget_mode} seeds={config.seeds}",
        flush=True,
    )

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)

    eval_indices = stratified_subset_indices(y_test, config.eval_attack_rows, seed=EVAL_SUBSET_SEED)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    eval_attack_categories, top_attack_categories = prepare_attack_categories(
        test_df, eval_indices, eval_y, config.top_attack_categories)
    print(f"[mlp] top attack categories: {top_attack_categories}", flush=True)

    mask = metadata["numeric_mask"]
    mins = metadata["numeric_mins"]
    maxs = metadata["numeric_maxs"]
    mins_t = torch.from_numpy(mins.astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(maxs.astype(np.float32)).to(device)

    metric_frames: list[pd.DataFrame] = []
    sensitivity_tables: list[pd.DataFrame] = []
    efficiency_frames: list[pd.DataFrame] = []
    validity_frames: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []
    full_test_frames: list[pd.DataFrame] = []
    transfer_frames: list[pd.DataFrame] = []
    reference_clean_rows: list[dict] = []
    ratio_rows: list[dict] = []

    for seed in config.seeds:
        set_seed(seed)
        print(f"[mlp][seed {seed}] training started", flush=True)
        train_loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        trained = train_all_defenses(
            model_factory=lambda: MLPBackbone(x_train.shape[1], config.hidden_dims, config.dropout),
            config=config,
            train_loader=train_loader,
            device=device,
            metadata=metadata,
            seed=seed,
        )
        sensitivity_tables.append(trained.sensitivity_table)
        print(f"[mlp][seed {seed}] evaluation started", flush=True)

        eval_set = EvaluationSet(
            x_eval=eval_x,
            y_eval=eval_y,
            x_test=x_test,
            y_test=y_test,
            attack_mask=mask,
            numeric_mins=mins,
            numeric_maxs=maxs,
            attack_categories=eval_attack_categories,
            top_categories=top_attack_categories,
        )
        evaluation = evaluate_defenses(trained, config, eval_set, device, seed)
        metric_frames.append(evaluation.metrics)
        efficiency_frames.append(evaluation.efficiency)
        validity_frames.append(evaluation.validity)
        category_frames.append(evaluation.categories)
        full_test_frames.append(evaluation.full_test_clean)

        # --- non-neural reference models ----------------------------------
        reference_models, reference_train_seconds = fit_reference_models(
            x_train, y_train, seed=seed, enabled_models=config.reference_models)
        for name, reference_model in reference_models.items():
            probs = predict_proba(reference_model, eval_x, batch_size=config.batch_size, device=device)
            reference_clean_rows.append({
                "seed": seed,
                "model": name,
                **classification_metrics(eval_y, probs),
                "train_seconds": reference_train_seconds[name],
                **measure_inference_cost(reference_model, eval_x, config.batch_size, device),
            })

        # --- transfer-attack matrix ---------------------------------------
        transfer_targets = {**trained.models, **reference_models}
        for attack, epsilon in config.transfer_attack_settings:
            for source_name, source_model in trained.models.items():
                transfer_df = evaluate_transfer_attack(
                    source_name, source_model, transfer_targets, eval_x, eval_y,
                    attack, epsilon, mask, mins, maxs, config, device,
                )
                transfer_df.insert(0, "seed", seed)
                transfer_frames.append(transfer_df)

        # --- sensitivity-ratio ablation -----------------------------------
        for ratio in config.sensitivity_ratio_list:
            if math.isclose(ratio, config.sensitivity_top_ratio):
                ratio_model = trained.models["constrained_adv"]
                ratio_mask = trained.selected_mask
            else:
                ratio_mask, _ = compute_sensitivity_mask(
                    trained.baseline_model,
                    train_loader,
                    device=device,
                    numeric_mask=mask,
                    feature_names=metadata["feature_names"],
                    top_ratio=ratio,
                    max_batches=config.sensitivity_batches,
                )
                ratio_mask_t = torch.from_numpy(ratio_mask.astype(np.float32)).to(device)

                def ratio_attack(model, xb, yb, local_mask=ratio_mask_t):
                    return pgd_attack(
                        model, xb, yb, config.adv_epsilon, config.adv_alpha,
                        config.adv_steps, local_mask, mins_t, maxs_t,
                    )

                ratio_model = copy.deepcopy(trained.baseline_model)
                ratio_model = fit_supervised(
                    ratio_model, train_loader, config.adv_epochs, device, config,
                    adversarial=True, attack_builder=ratio_attack,
                )

            ratio_metrics = evaluate_model_on_attack(
                ratio_model, eval_x, eval_y, config.ratio_attack,
                config.ratio_attack_epsilon, mask, mins, maxs, config, device,
            )
            ratio_rows.append({
                "seed": seed,
                "ratio": ratio,
                "selected_feature_count": int(ratio_mask.sum()),
                **ratio_metrics,
            })
        print(f"[mlp][seed {seed}] completed", flush=True)

    _write_mlp_outputs(
        config=config,
        output_dir=output_dir,
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
        top_attack_categories=top_attack_categories,
        metric_frames=metric_frames,
        sensitivity_tables=sensitivity_tables,
        efficiency_frames=efficiency_frames,
        validity_frames=validity_frames,
        category_frames=category_frames,
        full_test_frames=full_test_frames,
        transfer_frames=transfer_frames,
        reference_clean_rows=reference_clean_rows,
        ratio_rows=ratio_rows,
    )


def _write_mlp_outputs(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    metadata: dict,
    top_attack_categories: list[str],
    metric_frames: list[pd.DataFrame],
    sensitivity_tables: list[pd.DataFrame],
    efficiency_frames: list[pd.DataFrame],
    validity_frames: list[pd.DataFrame],
    category_frames: list[pd.DataFrame],
    full_test_frames: list[pd.DataFrame],
    transfer_frames: list[pd.DataFrame],
    reference_clean_rows: list[dict],
    ratio_rows: list[dict],
) -> None:
    """Aggregate every per-seed frame, write the CSV tables and draw the figures."""
    results_df = pd.concat(metric_frames, ignore_index=True)
    sensitivity_df = pd.concat(sensitivity_tables, ignore_index=True)
    mean_df, std_df = summarize_results(results_df)

    transfer_df = pd.concat(transfer_frames, ignore_index=True)
    transfer_mean_df, transfer_std_df = summarize_results(
        transfer_df, group_cols=["source_model", "target_model", "attack", "epsilon"])

    reference_clean_df = pd.DataFrame(reference_clean_rows)
    reference_metric_cols = [
        "accuracy", "precision", "recall", "f1", "attack_success_rate", "train_seconds",
        "inference_probe_rows", "inference_seconds", "inference_ms_per_sample",
        "inference_samples_per_second",
    ]
    reference_clean_mean_df = reference_clean_df.groupby(["model"], as_index=False)[reference_metric_cols].mean()
    reference_clean_std_df = (
        reference_clean_df.groupby(["model"], as_index=False)[reference_metric_cols].std().fillna(0.0))
    reference_transfer_mean_df = transfer_mean_df[
        transfer_mean_df["target_model"].isin(config.reference_models)].copy()
    reference_transfer_std_df = transfer_std_df[
        transfer_std_df["target_model"].isin(config.reference_models)].copy()

    category_df = pd.concat(category_frames, ignore_index=True) if category_frames else pd.DataFrame()
    if not category_df.empty:
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

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_mean_df = ratio_df.groupby(["ratio", "selected_feature_count"], as_index=False).agg(
        clean_accuracy=("clean_accuracy", "mean"),
        clean_recall=("clean_recall", "mean"),
        clean_f1=("clean_f1", "mean"),
        robust_accuracy=("robust_accuracy", "mean"),
        robust_recall=("robust_recall", "mean"),
        robust_f1=("robust_f1", "mean"),
        attack_success_rate=("attack_success_rate", "mean"),
    )

    validity_df = pd.concat(validity_frames, ignore_index=True)
    validity_mean_df = (
        validity_df.groupby(["model", "attack", "epsilon"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            attack_success_rate=("attack_success_rate", "mean"),
            numeric_valid_rate=("numeric_valid_rate", "mean"),
            protected_integrity_rate=("protected_integrity_rate", "mean"),
            overall_validity_rate=("overall_validity_rate", "mean"),
            mean_numeric_abs_delta=("mean_numeric_abs_delta", "mean"),
            mean_numeric_l2_delta=("mean_numeric_l2_delta", "mean"),
            max_numeric_abs_delta=("max_numeric_abs_delta", "mean"),
            changed_numeric_feature_ratio=("changed_numeric_feature_ratio", "mean"),
            protected_feature_change_ratio=("protected_feature_change_ratio", "mean"),
            boundary_clip_ratio=("boundary_clip_ratio", "mean"),
        )
    )

    efficiency_df = pd.concat(efficiency_frames, ignore_index=True)
    efficiency_mean_df = (
        efficiency_df.groupby(["model"], as_index=False)
        .agg(
            parameter_count=("parameter_count", "mean"),
            train_seconds=("train_seconds", "mean"),
            pretrain_seconds=("pretrain_seconds", "mean"),
            continuation_seconds=("continuation_seconds", "mean"),
            mask_build_seconds=("mask_build_seconds", "mean"),
            training_attack_feature_ratio=("training_attack_feature_ratio", "mean"),
            training_attack_feature_count=("training_attack_feature_count", "mean"),
            inference_probe_rows=("inference_probe_rows", "mean"),
            inference_seconds=("inference_seconds", "mean"),
            inference_ms_per_sample=("inference_ms_per_sample", "mean"),
            inference_samples_per_second=("inference_samples_per_second", "mean"),
        )
    )
    standard_train_seconds = float(
        efficiency_mean_df.loc[efficiency_mean_df["model"] == "standard", "train_seconds"].iloc[0])
    standard_inference_ms = float(
        efficiency_mean_df.loc[efficiency_mean_df["model"] == "standard", "inference_ms_per_sample"].iloc[0])
    efficiency_mean_df["relative_train_cost_vs_standard"] = (
        efficiency_mean_df["train_seconds"] / standard_train_seconds)
    efficiency_mean_df["relative_inference_latency_vs_standard"] = (
        efficiency_mean_df["inference_ms_per_sample"] / standard_inference_ms)

    results_df.to_csv(output_dir / "raw_results.csv", index=False)
    mean_df.to_csv(output_dir / "mean_results.csv", index=False)
    std_df.to_csv(output_dir / "std_results.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_scores.csv", index=False)
    transfer_df.to_csv(output_dir / "transfer_raw_results.csv", index=False)
    transfer_mean_df.to_csv(output_dir / "transfer_mean_results.csv", index=False)
    transfer_std_df.to_csv(output_dir / "transfer_std_results.csv", index=False)
    reference_clean_df.to_csv(output_dir / "reference_model_clean_raw.csv", index=False)
    reference_clean_mean_df.to_csv(output_dir / "reference_model_clean_mean.csv", index=False)
    reference_clean_std_df.to_csv(output_dir / "reference_model_clean_std.csv", index=False)
    reference_transfer_mean_df.to_csv(output_dir / "reference_transfer_mean_results.csv", index=False)
    reference_transfer_std_df.to_csv(output_dir / "reference_transfer_std_results.csv", index=False)
    if not category_df.empty:
        category_df.to_csv(output_dir / "category_raw_results.csv", index=False)
        category_mean_df.to_csv(output_dir / "category_mean_results.csv", index=False)
    ratio_df.to_csv(output_dir / "ratio_ablation_raw.csv", index=False)
    ratio_mean_df.to_csv(output_dir / "ratio_ablation_mean.csv", index=False)
    validity_df.to_csv(output_dir / "attack_validity_raw.csv", index=False)
    validity_mean_df.to_csv(output_dir / "attack_validity_mean.csv", index=False)
    efficiency_df.to_csv(output_dir / "efficiency_raw.csv", index=False)
    efficiency_mean_df.to_csv(output_dir / "efficiency_mean.csv", index=False)

    plot_metric_curve(mean_df, "f1", output_dir / "f1_curve.png")
    plot_metric_curve(mean_df, "recall", output_dir / "recall_curve.png")
    plot_clean_f1_bar(mean_df, output_dir / "clean_f1_bar.png")
    transfer_heatmap_df = transfer_mean_df[
        (transfer_mean_df["attack"] == "pgd") & (np.isclose(transfer_mean_df["epsilon"], 0.1))]
    plot_transfer_heatmap(transfer_heatmap_df, output_dir / "transfer_pgd_heatmap.png")
    plot_ratio_ablation(ratio_mean_df, output_dir / "ratio_ablation.png")
    plot_efficiency_tradeoff(efficiency_mean_df, mean_df, output_dir / "efficiency_tradeoff.png")
    print("[mlp] aggregations and plots completed", flush=True)

    significance_df = compute_significance_tests(results_df, config.epsilon_list)
    if not significance_df.empty:
        significance_df.to_csv(output_dir / "significance_tests.csv", index=False)

    full_test_df = pd.concat(full_test_frames, ignore_index=True)
    full_metric_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    full_clean_mean = full_test_df.groupby(["model", "subset"], as_index=False)[full_metric_cols].mean()
    full_clean_std = full_test_df.groupby(["model", "subset"], as_index=False)[full_metric_cols].std().fillna(0.0)
    full_test_df.to_csv(output_dir / "full_test_clean_raw.csv", index=False)
    full_clean_mean.to_csv(output_dir / "full_test_clean_mean.csv", index=False)
    full_clean_std.to_csv(output_dir / "full_test_clean_std.csv", index=False)

    summary = {
        "architecture": f"MLP {config.hidden_dims}",
        "config": asdict(config),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "numeric_feature_count": int(len(metadata["numeric_cols"])),
        "categorical_feature_count": int(len(metadata["categorical_cols"])),
        "transformed_feature_count": int(len(metadata["feature_names"])),
        "top_attack_categories": top_attack_categories,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def prepare_attack_categories(test_df: pd.DataFrame, eval_indices: np.ndarray,
                              eval_y: np.ndarray, top_n: int) -> tuple[np.ndarray, list[str]]:
    """Attack-category labels of the evaluation subset and the top-N categories."""
    categories = test_df["attack_cat"].fillna("Unknown").to_numpy()[eval_indices]
    top = (
        pd.Series(categories[eval_y == 1])
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    return categories, top
