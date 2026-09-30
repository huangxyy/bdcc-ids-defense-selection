"""1D-CNN backbone: training, evaluation and the full matched-budget experiment.

Implements CNN1D (two Conv1d layers -> global average pooling -> linear head) and
train_all_cnn_models(), which trains the six candidate defenses under the
matched-continuation protocol.

    python backbone_cnn1d.py --output-dir outputs/cnn1d --device cuda

Writes mean_results.csv, std_results.csv, raw_results.csv, significance_tests.csv,
efficiency_mean.csv and run_summary.json into the output directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch import nn

import ids_core as base


MODEL_ORDER = [
    "standard",
    "adv_training",
    "constrained_adv",
    "trades",
    "free_at",
    "class_aware_constrained",
]

FULL_TEST_ATTACK_SETTINGS = (("pgd", 0.10),)


class CNN1D(nn.Module):
    """1D-CNN classifier for tabular IDS features."""

    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(1)


# ---------------------------------------------------------------- paired significance tests
def write_significance_tests(results_df: pd.DataFrame, epsilon_list: tuple[float, ...], output_path: Path) -> None:
    sig_rows = []
    comparisons = [
        ("constrained_adv", "standard"),
        ("constrained_adv", "adv_training"),
        ("class_aware_constrained", "constrained_adv"),
        ("trades", "adv_training"),
        ("free_at", "adv_training"),
    ]
    attack_columns = ["clean"] + [f"{attack}_{eps}" for attack in ["fgsm", "pgd", "cw", "apgd"] for eps in epsilon_list]

    for attack_col in attack_columns:
        if attack_col == "clean":
            attack_name, epsilon = "clean", 0.0
        else:
            idx = attack_col.rfind("_")
            attack_name = attack_col[:idx]
            epsilon = float(attack_col[idx + 1 :])
        subset = results_df[
            (results_df["attack"] == attack_name) & (np.isclose(results_df["epsilon"], epsilon))
        ]
        if subset.empty:
            continue
        for model_a, model_b in comparisons:
            for metric in ["f1", "attack_success_rate"]:
                a = subset[subset["model"] == model_a].sort_values("seed")[metric].to_numpy()
                b = subset[subset["model"] == model_b].sort_values("seed")[metric].to_numpy()
                if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                    continue
                try:
                    t_statistic, t_pvalue = scipy_stats.ttest_rel(a, b)
                except Exception:
                    t_statistic, t_pvalue = float("nan"), float("nan")
                try:
                    wilcoxon_statistic, wilcoxon_pvalue = scipy_stats.wilcoxon(a, b)
                except Exception:
                    wilcoxon_statistic, wilcoxon_pvalue = float("nan"), float("nan")
                sig_rows.append(
                    {
                        "comparison": f"{model_a}_vs_{model_b}",
                        "attack": attack_name,
                        "epsilon": epsilon,
                        "metric": metric,
                        "t_statistic": t_statistic,
                        "t_pvalue": t_pvalue,
                        "wilcoxon_statistic": wilcoxon_statistic,
                        "wilcoxon_pvalue": wilcoxon_pvalue,
                    }
                )

    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(output_path, index=False)


# ---------------------------------------------------------------- six defenses, matched-continuation protocol
def train_all_cnn_models(
    config: base.ExperimentConfig,
    train_loader,
    input_dim: int,
    device: torch.device,
    metadata: dict,
):
    """Train the six defense strategies with the same matched-budget logic as the main experiment."""
    # Six defenses, one shared pre-training checkpoint, identical continuation budget.
    # Only the perturbation strategy differs, so cost differences are attributable.
    matched_budget_mode = config.training_budget_mode == "matched_continuation"
    numeric_mask = metadata["numeric_mask"]
    numeric_mask_t = torch.from_numpy(numeric_mask.astype(np.float32)).to(device)
    mins_t = torch.from_numpy(metadata["numeric_mins"].astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(metadata["numeric_maxs"].astype(np.float32)).to(device)

    baseline_model = CNN1D(input_dim, config.dropout)
    baseline_start = time.perf_counter()
    baseline_model = base.train_model(
        baseline_model,
        train_loader,
        epochs=config.baseline_epochs,
        device=device,
        adv_training=False,
        adv_builder=None,
    )
    baseline_train_seconds = time.perf_counter() - baseline_start

    standard_model = baseline_model
    standard_continuation_seconds = 0.0
    if matched_budget_mode:
        standard_model = copy.deepcopy(baseline_model)
        standard_start = time.perf_counter()
        standard_model = base.train_model(
            standard_model,
            train_loader,
            epochs=config.adv_epochs,
            device=device,
            adv_training=False,
            adv_builder=None,
        )
        standard_continuation_seconds = time.perf_counter() - standard_start

    adv_model = copy.deepcopy(baseline_model)
    adv_builder = lambda model, xb, yb: base.pgd_attack(
        model,
        xb,
        yb,
        config.adv_epsilon,
        config.adv_alpha,
        config.adv_steps,
        numeric_mask_t,
        mins_t,
        maxs_t,
    )
    adv_start = time.perf_counter()
    adv_model = base.train_model(
        adv_model,
        train_loader,
        epochs=config.adv_epochs,
        device=device,
        adv_training=True,
        adv_builder=adv_builder,
    )
    adv_train_seconds = time.perf_counter() - adv_start

    selected_mask_start = time.perf_counter()
    selected_mask, sensitivity_table = base.sensitivity_mask(
        baseline_model,
        train_loader,
        device=device,
        numeric_mask=numeric_mask,
        feature_names=metadata["feature_names"],
        top_ratio=config.sensitivity_top_ratio,
        max_batches=config.sensitivity_batches,
    )
    selected_mask_seconds = time.perf_counter() - selected_mask_start
    selected_mask_t = torch.from_numpy(selected_mask.astype(np.float32)).to(device)

    constrained_model = copy.deepcopy(baseline_model)
    constrained_builder = lambda model, xb, yb: base.pgd_attack(
        model,
        xb,
        yb,
        config.adv_epsilon,
        config.adv_alpha,
        config.adv_steps,
        selected_mask_t,
        mins_t,
        maxs_t,
    )
    constrained_start = time.perf_counter()
    constrained_model = base.train_model(
        constrained_model,
        train_loader,
        epochs=config.adv_epochs,
        device=device,
        adv_training=True,
        adv_builder=constrained_builder,
    )
    constrained_train_seconds = time.perf_counter() - constrained_start

    trades_model = copy.deepcopy(baseline_model)
    trades_start = time.perf_counter()
    trades_model = base.train_model_trades(
        trades_model,
        train_loader,
        config.adv_epochs,
        device,
        config,
        numeric_mask_t,
        mins_t,
        maxs_t,
    )
    trades_train_seconds = time.perf_counter() - trades_start

    free_at_model = copy.deepcopy(baseline_model)
    free_at_start = time.perf_counter()
    free_at_model = base.train_model_free_at(
        free_at_model,
        train_loader,
        config.adv_epochs,
        device,
        config,
        numeric_mask_t,
        mins_t,
        maxs_t,
    )
    free_at_train_seconds = time.perf_counter() - free_at_start

    ca_mask_start = time.perf_counter()
    ca_mask, _ = base.class_aware_sensitivity_mask(
        baseline_model,
        train_loader,
        device=device,
        numeric_mask=numeric_mask,
        feature_names=metadata["feature_names"],
        top_ratio=config.sensitivity_top_ratio,
        max_batches=config.sensitivity_batches,
    )
    ca_mask_seconds = time.perf_counter() - ca_mask_start
    ca_mask_t = torch.from_numpy(ca_mask.astype(np.float32)).to(device)

    class_aware_model = copy.deepcopy(baseline_model)
    class_aware_start = time.perf_counter()
    class_aware_model = base.train_model_class_aware_constrained(
        class_aware_model,
        train_loader,
        config.adv_epochs,
        device,
        config,
        ca_mask_t,
        mins_t,
        maxs_t,
        config.class_aware_minority_weight,
    )
    class_aware_train_seconds = time.perf_counter() - class_aware_start

    train_cost_registry = {
        "standard": base.summarize_training_budget(
            pretrain_seconds=baseline_train_seconds,
            continuation_seconds=standard_continuation_seconds,
            include_pretrain_in_total=True,
        ),
        "adv_training": base.summarize_training_budget(
            pretrain_seconds=baseline_train_seconds,
            continuation_seconds=adv_train_seconds,
            include_pretrain_in_total=matched_budget_mode,
        ),
        "constrained_adv": base.summarize_training_budget(
            pretrain_seconds=baseline_train_seconds,
            continuation_seconds=constrained_train_seconds,
            mask_build_seconds=selected_mask_seconds,
            include_pretrain_in_total=matched_budget_mode,
            include_mask_in_total=matched_budget_mode,
        ),
        "trades": base.summarize_training_budget(
            pretrain_seconds=baseline_train_seconds,
            continuation_seconds=trades_train_seconds,
            include_pretrain_in_total=matched_budget_mode,
        ),
        "free_at": base.summarize_training_budget(
            pretrain_seconds=baseline_train_seconds,
            continuation_seconds=free_at_train_seconds,
            include_pretrain_in_total=matched_budget_mode,
        ),
        "class_aware_constrained": base.summarize_training_budget(
            pretrain_seconds=baseline_train_seconds,
            continuation_seconds=class_aware_train_seconds,
            mask_build_seconds=ca_mask_seconds,
            include_pretrain_in_total=matched_budget_mode,
            include_mask_in_total=matched_budget_mode,
        ),
    }

    model_registry = {
        "standard": standard_model,
        "adv_training": adv_model,
        "constrained_adv": constrained_model,
        "trades": trades_model,
        "free_at": free_at_model,
        "class_aware_constrained": class_aware_model,
    }
    perturb_ratio_registry = {
        "standard": 0.0,
        "adv_training": 1.0,
        "constrained_adv": float(selected_mask.sum() / max(numeric_mask.sum(), 1.0)),
        "trades": 1.0,
        "free_at": 1.0,
        "class_aware_constrained": float(ca_mask.sum() / max(numeric_mask.sum(), 1.0)),
    }
    selected_feature_registry = {
        "standard": 0,
        "adv_training": int(numeric_mask.sum()),
        "constrained_adv": int(selected_mask.sum()),
        "trades": int(numeric_mask.sum()),
        "free_at": int(numeric_mask.sum()),
        "class_aware_constrained": int(ca_mask.sum()),
    }
    eval_context = {
        "numeric_mask": numeric_mask,
        "numeric_mins": metadata["numeric_mins"],
        "numeric_maxs": metadata["numeric_maxs"],
        "defense_masks": {
            # training masks of the two feature-constrained defenses;
            # consumed by adaptive_attacks.py for the complement attack
            "constrained_adv": selected_mask,
            "class_aware_constrained": ca_mask,
        },
    }
    return (
        model_registry,
        train_cost_registry,
        perturb_ratio_registry,
        selected_feature_registry,
        eval_context,
        sensitivity_table,
    )


# ---------------------------------------------------------------- entry point
def main() -> None:
    parser = base.build_parser(
        "Run the 1D-CNN matched-budget IDS experiment.",
        defaults={"output_dir": "outputs/cnn1d_matched_budget_run", "training_budget_mode": "matched_continuation"},
        require_paths=False,
    )
    parser.add_argument("--full-test-attack-rows", type=int, default=0,
                        help="if > 0, additionally attack this many rows of the full test set")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    out_dir = output_dir if output_dir.is_absolute() else (base_dir / output_dir)
    config = base.config_from_args(
        args,
        train_path=str((base_dir / args.train_path) if not Path(args.train_path).is_absolute() else Path(args.train_path)),
        test_path=str((base_dir / args.test_path) if not Path(args.test_path).is_absolute() else Path(args.test_path)),
        output_dir=str(out_dir),
    )
    if args.print_config:
        base.emit_config(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    base.CONFIG = config
    device = torch.device(config.device)

    print(
        f"[cnn1d] output_dir={out_dir} device={config.device} "
        f"training_budget_mode={config.training_budget_mode} seeds={config.seeds}",
        flush=True,
    )

    train_df, test_df = base.load_unsw_split(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = base.build_features(train_df, test_df)
    eval_indices = base.stratified_subset_indices(y_test, config.eval_attack_rows, seed=2026)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    if args.full_test_attack_rows and args.full_test_attack_rows > 0:
        full_attack_indices = base.stratified_subset_indices(y_test, args.full_test_attack_rows, seed=2027)
        full_attack_x = x_test[full_attack_indices]
        full_attack_y = y_test[full_attack_indices]
        full_attack_subset = "full_test_subset"
    else:
        full_attack_x = x_test
        full_attack_y = y_test
        full_attack_subset = "full_test"

    all_results = []
    sensitivity_tables = []
    efficiency_results = []
    validity_results = []
    full_test_clean_results = []
    full_test_attack_results = []

    for seed in config.seeds:
        base.set_seed(seed)
        train_loader = base.make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        input_dim = x_train.shape[1]
        print(f"[cnn1d][seed {seed}] training started", flush=True)

        (
            model_registry,
            train_cost_registry,
            perturb_ratio_registry,
            selected_feature_registry,
            eval_context,
            sensitivity_table,
        ) = train_all_cnn_models(
            config=config,
            train_loader=train_loader,
            input_dim=input_dim,
            device=device,
            metadata=metadata,
        )

        sensitivity_table.insert(0, "seed", seed)
        sensitivity_tables.append(sensitivity_table)
        print(f"[cnn1d][seed {seed}] evaluation started", flush=True)

        for model_name in MODEL_ORDER:
            model = model_registry[model_name]
            clean_probs = base.predict_probabilities_from_array(
                model,
                eval_x,
                batch_size=config.batch_size,
                device=device,
            )
            clean_pred = (clean_probs >= 0.5).astype(np.int32)
            efficiency_results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "parameter_count": base.count_parameters(model),
                    "train_seconds": train_cost_registry[model_name]["train_seconds"],
                    "pretrain_seconds": train_cost_registry[model_name]["pretrain_seconds"],
                    "continuation_seconds": train_cost_registry[model_name]["continuation_seconds"],
                    "mask_build_seconds": train_cost_registry[model_name]["mask_build_seconds"],
                    "training_attack_feature_ratio": perturb_ratio_registry[model_name],
                    "training_attack_feature_count": selected_feature_registry[model_name],
                    **base.measure_inference_efficiency(
                        model=model,
                        x_eval=eval_x,
                        batch_size=config.batch_size,
                        device=device,
                    ),
                }
            )

            result_df = base.evaluate_model(
                model=model,
                x_eval=eval_x,
                y_eval=eval_y,
                attack_mask=eval_context["numeric_mask"],
                mins=eval_context["numeric_mins"],
                maxs=eval_context["numeric_maxs"],
                epsilon_list=config.epsilon_list,
                device=device,
                batch_size=config.batch_size,
            )
            result_df.insert(0, "seed", seed)
            result_df.insert(1, "model", model_name)
            all_results.append(result_df)

            full_test_probs = base.predict_probabilities_from_array(
                model,
                x_test,
                batch_size=config.batch_size,
                device=device,
            )
            full_test_clean_pred = (full_test_probs >= 0.5).astype(np.int32)
            full_test_clean_results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "subset": "full_test",
                    **base.compute_metrics(y_test, full_test_probs),
                }
            )

            for attack_name, epsilon in config.validity_attack_settings:
                attacked_inputs, attacked_probs = base.generate_attack_outputs(
                    model=model,
                    x_eval=eval_x,
                    y_eval=eval_y,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    attack_mask=eval_context["numeric_mask"],
                    mins=eval_context["numeric_mins"],
                    maxs=eval_context["numeric_maxs"],
                    device=device,
                    batch_size=config.batch_size,
                    return_inputs=True,
                )
                validity_results.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "attack": attack_name,
                        "epsilon": epsilon,
                        **base.compute_metrics(eval_y, attacked_probs, clean_pred=clean_pred),
                        **base.compute_attack_validity_metrics(
                            x_clean=eval_x,
                            x_adv=attacked_inputs,
                            numeric_mask=eval_context["numeric_mask"],
                            numeric_mins=eval_context["numeric_mins"],
                            numeric_maxs=eval_context["numeric_maxs"],
                        ),
                    }
                )

            for attack_name, epsilon in FULL_TEST_ATTACK_SETTINGS:
                full_attack_clean_probs = base.predict_probabilities_from_array(
                    model,
                    full_attack_x,
                    batch_size=config.batch_size,
                    device=device,
                )
                full_attack_clean_pred = (full_attack_clean_probs >= 0.5).astype(np.int32)
                _, attacked_probs = base.generate_attack_outputs(
                    model=model,
                    x_eval=full_attack_x,
                    y_eval=full_attack_y,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    attack_mask=eval_context["numeric_mask"],
                    mins=eval_context["numeric_mins"],
                    maxs=eval_context["numeric_maxs"],
                    device=device,
                    batch_size=config.batch_size,
                    return_inputs=False,
                )
                full_test_attack_results.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "subset": full_attack_subset,
                        "attack": attack_name,
                        "epsilon": epsilon,
                        "test_rows": len(full_attack_y),
                        **base.compute_metrics(
                            full_attack_y,
                            attacked_probs,
                            clean_pred=full_attack_clean_pred,
                        ),
                    }
                )

        print(f"[cnn1d][seed {seed}] completed", flush=True)

    results_df = pd.concat(all_results, ignore_index=True)
    sensitivity_df = pd.concat(sensitivity_tables, ignore_index=True)
    mean_df, std_df = base.summarize_results(results_df)

    efficiency_df = pd.DataFrame(efficiency_results)
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
        efficiency_mean_df.loc[efficiency_mean_df["model"] == "standard", "train_seconds"].iloc[0]
    )
    standard_inference_ms = float(
        efficiency_mean_df.loc[efficiency_mean_df["model"] == "standard", "inference_ms_per_sample"].iloc[0]
    )
    efficiency_mean_df["relative_train_cost_vs_standard"] = (
        efficiency_mean_df["train_seconds"] / standard_train_seconds
    )
    efficiency_mean_df["relative_inference_latency_vs_standard"] = (
        efficiency_mean_df["inference_ms_per_sample"] / standard_inference_ms
    )

    validity_df = pd.DataFrame(validity_results)
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

    full_test_clean_df = pd.DataFrame(full_test_clean_results)
    full_test_metric_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    full_test_clean_mean_df = full_test_clean_df.groupby(["model", "subset"], as_index=False)[full_test_metric_cols].mean()
    full_test_clean_std_df = full_test_clean_df.groupby(["model", "subset"], as_index=False)[full_test_metric_cols].std().fillna(0.0)

    full_test_attack_df = pd.DataFrame(full_test_attack_results)
    full_test_attack_metric_cols = ["test_rows", "accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    full_test_attack_mean_df = full_test_attack_df.groupby(
        ["model", "subset", "attack", "epsilon"], as_index=False
    )[full_test_attack_metric_cols].mean()
    full_test_attack_std_df = full_test_attack_df.groupby(
        ["model", "subset", "attack", "epsilon"], as_index=False
    )[full_test_attack_metric_cols].std().fillna(0.0)

    results_df.to_csv(out_dir / "raw_results.csv", index=False)
    mean_df.to_csv(out_dir / "mean_results.csv", index=False)
    std_df.to_csv(out_dir / "std_results.csv", index=False)
    sensitivity_df.to_csv(out_dir / "sensitivity_scores.csv", index=False)
    efficiency_df.to_csv(out_dir / "efficiency_raw.csv", index=False)
    efficiency_mean_df.to_csv(out_dir / "efficiency_mean.csv", index=False)
    validity_df.to_csv(out_dir / "attack_validity_raw.csv", index=False)
    validity_mean_df.to_csv(out_dir / "attack_validity_mean.csv", index=False)
    full_test_clean_df.to_csv(out_dir / "full_test_clean_raw.csv", index=False)
    full_test_clean_mean_df.to_csv(out_dir / "full_test_clean_mean.csv", index=False)
    full_test_clean_std_df.to_csv(out_dir / "full_test_clean_std.csv", index=False)
    full_test_attack_df.to_csv(out_dir / "full_test_attack_raw.csv", index=False)
    full_test_attack_mean_df.to_csv(out_dir / "full_test_attack_mean.csv", index=False)
    full_test_attack_std_df.to_csv(out_dir / "full_test_attack_std.csv", index=False)

    base.plot_metric_curve(mean_df, "f1", out_dir / "f1_curve.png")
    base.plot_metric_curve(mean_df, "recall", out_dir / "recall_curve.png")
    base.plot_ablation(mean_df, out_dir / "clean_f1_bar.png")
    base.plot_efficiency_tradeoff(efficiency_mean_df, mean_df, out_dir / "efficiency_tradeoff.png")
    write_significance_tests(results_df, config.epsilon_list, out_dir / "significance_tests.csv")

    summary = {
        "architecture": "1D-CNN (Conv1d(1,32,3) -> Conv1d(32,64,3) -> AvgPool -> FC(64,32) -> FC(32,1))",
        "config": asdict(config),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(eval_y)),
        "full_test_attack_rows": int(len(full_attack_y)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "transformed_feature_count": int(len(metadata["feature_names"])),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== 1D-CNN key results (PGD eps=0.10 on eval subset) ===", flush=True)
    key = mean_df[(mean_df["attack"] == "pgd") & (np.isclose(mean_df["epsilon"], 0.10))]
    print(key[["model", "f1", "attack_success_rate"]].to_string(index=False), flush=True)
    print(f"\nAll saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
