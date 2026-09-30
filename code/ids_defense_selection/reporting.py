"""Aggregation, significance tests and the shared matplotlib figures."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .config import ExperimentConfig


def plot_metric_curve(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for attack_name in ["fgsm", "pgd"]:
        attack_df = df[df["attack"] == attack_name]
        for model_name in attack_df["model"].unique():
            subset = attack_df[attack_df["model"] == model_name].sort_values("epsilon")
            plt.plot(subset["epsilon"], subset[metric], marker="o", label=f"{model_name}-{attack_name.upper()}")
    plt.xlabel("Epsilon")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} under adversarial perturbations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_clean_f1_bar(df: pd.DataFrame, output_path: Path) -> None:
    clean = df[df["attack"] == "clean"].groupby("model", as_index=False)["f1"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(clean["model"], clean["f1"], color=["#4472C4", "#ED7D31", "#70AD47"])
    plt.ylim(0, 1)
    plt.ylabel("F1")
    plt.title("Clean-set F1 comparison across training strategies")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_transfer_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No transfer results for selected filter", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    pivot = df.pivot(index="source_model", columns="target_model", values="f1")
    row_labels = list(pivot.index)
    col_labels = list(pivot.columns)
    matrix = pivot.to_numpy()

    fig_width = max(7, 1.25 * len(col_labels) + 2)
    fig_height = max(5, 1.0 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=matrix.min(), vmax=matrix.max())
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Target model")
    ax.set_ylabel("Source model")
    ax.set_title("Transfer-attack F1 heatmap")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ratio_ablation(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("ratio")
    plt.figure(figsize=(7, 5))
    plt.plot(plot_df["ratio"], plot_df["clean_f1"], marker="o", label="Clean F1")
    plt.plot(plot_df["ratio"], plot_df["robust_f1"], marker="s", label="Robust F1")
    plt.xlabel("Sensitivity top ratio")
    plt.ylabel("F1")
    plt.title("Constrained adversarial training ratio ablation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_efficiency_tradeoff(efficiency_df: pd.DataFrame, result_df: pd.DataFrame, output_path: Path) -> None:
    robust_df = result_df[(result_df["attack"] == "pgd") & (np.isclose(result_df["epsilon"], 0.1))][["model", "f1"]]
    robust_df = robust_df.rename(columns={"f1": "robust_f1"})
    clean_df = result_df[result_df["attack"] == "clean"][["model", "f1"]].rename(columns={"f1": "clean_f1"})
    plot_df = efficiency_df.merge(robust_df, on="model").merge(clean_df, on="model")

    colors = {
        "standard": "#4472C4",
        "adv_training": "#ED7D31",
        "constrained_adv": "#70AD47",
    }

    plt.figure(figsize=(7, 5))
    for row in plot_df.itertuples():
        plt.scatter(
            row.train_seconds,
            row.robust_f1,
            s=180,
            color=colors.get(row.model, "#5B9BD5"),
            alpha=0.9,
        )
        plt.annotate(
            f"{row.model}\nclean={row.clean_f1:.3f}",
            (row.train_seconds, row.robust_f1),
            xytext=(6, 6),
            textcoords="offset points",
        )
    plt.xlabel("Training time (s)")
    plt.ylabel("Robust F1 under PGD, ε=0.10")
    plt.title("Robustness-efficiency trade-off")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_results(
    results: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and standard deviation over seeds for every metric column."""
    if group_cols is None:
        group_cols = ["model", "attack", "epsilon"]
    metric_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    mean_df = results.groupby(group_cols, as_index=False)[metric_cols].mean()
    std_df = results.groupby(group_cols, as_index=False)[metric_cols].std().fillna(0.0)
    return mean_df, std_df


#: Defense pairs compared in the paired significance tests.
DEFAULT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("constrained_adv", "standard"),
    ("constrained_adv", "adv_training"),
    ("class_aware_constrained", "constrained_adv"),
    ("trades", "adv_training"),
    ("free_at", "adv_training"),
)


def compute_significance_tests(
    results_df: pd.DataFrame,
    epsilon_list: tuple[float, ...],
    comparisons: tuple[tuple[str, str], ...] = DEFAULT_COMPARISONS,
) -> pd.DataFrame:
    """Paired t-test and Wilcoxon signed-rank test for every comparison cell.

    `results_df` is the per-seed raw results frame (columns: seed, model,
    attack, epsilon, ...).  One row is returned per (comparison, attack,
    epsilon, metric); the text report selects the cells that matter.
    """
    rows = []
    for attack in ("clean", "fgsm", "pgd", "cw", "apgd"):
        epsilons = [0.0] if attack == "clean" else list(epsilon_list)
        for epsilon in epsilons:
            subset = results_df[
                (results_df["attack"] == attack) & np.isclose(results_df["epsilon"], epsilon)
            ]
            if subset.empty:
                continue
            for model_a, model_b in comparisons:
                for metric in ("f1", "attack_success_rate"):
                    a = subset[subset["model"] == model_a].sort_values("seed")[metric].to_numpy()
                    b = subset[subset["model"] == model_b].sort_values("seed")[metric].to_numpy()
                    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                        continue
                    try:
                        t_statistic, t_pvalue = scipy_stats.ttest_rel(a, b)
                    except Exception:  # noqa: BLE001 - degenerate samples must not abort the run
                        t_statistic, t_pvalue = float("nan"), float("nan")
                    try:
                        wilcoxon_statistic, wilcoxon_pvalue = scipy_stats.wilcoxon(a, b)
                    except Exception:  # noqa: BLE001
                        wilcoxon_statistic, wilcoxon_pvalue = float("nan"), float("nan")
                    rows.append({
                        "comparison": f"{model_a}_vs_{model_b}",
                        "attack": attack,
                        "epsilon": epsilon,
                        "metric": metric,
                        "t_statistic": t_statistic,
                        "t_pvalue": t_pvalue,
                        "wilcoxon_statistic": wilcoxon_statistic,
                        "wilcoxon_pvalue": wilcoxon_pvalue,
                    })
    return pd.DataFrame(rows)


@dataclass
class BackboneRunFrames:
    """Per-seed frames collected by one backbone run."""

    results: pd.DataFrame
    sensitivity: pd.DataFrame
    efficiency: pd.DataFrame
    validity: pd.DataFrame
    full_test_clean: pd.DataFrame
    full_test_attack: pd.DataFrame


def write_backbone_outputs(
    output_dir: Path,
    config: ExperimentConfig,
    frames: BackboneRunFrames,
    summary: dict,
) -> dict[str, pd.DataFrame]:
    """Aggregate the per-seed frames and write the standard backbone outputs.

    Writes raw/mean/std metrics, sensitivity scores, efficiency, attack
    validity, full-test clean/attacked tables, significance tests and the
    summary figures.  Returns the aggregated frames for further use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_df, std_df = summarize_results(frames.results)

    efficiency_df = frames.efficiency
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

    validity_df = frames.validity
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

    full_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    full_clean_df = frames.full_test_clean
    full_clean_mean = full_clean_df.groupby(["model", "subset"], as_index=False)[full_cols].mean()
    full_clean_std = full_clean_df.groupby(["model", "subset"], as_index=False)[full_cols].std().fillna(0.0)

    full_attack_df = frames.full_test_attack
    if not full_attack_df.empty:
        attack_cols = ["test_rows", *full_cols]
        full_attack_mean = full_attack_df.groupby(
            ["model", "subset", "attack", "epsilon"], as_index=False)[attack_cols].mean()
        full_attack_std = full_attack_df.groupby(
            ["model", "subset", "attack", "epsilon"], as_index=False)[attack_cols].std().fillna(0.0)

    frames.results.to_csv(output_dir / "raw_results.csv", index=False)
    mean_df.to_csv(output_dir / "mean_results.csv", index=False)
    std_df.to_csv(output_dir / "std_results.csv", index=False)
    frames.sensitivity.to_csv(output_dir / "sensitivity_scores.csv", index=False)
    efficiency_df.to_csv(output_dir / "efficiency_raw.csv", index=False)
    efficiency_mean_df.to_csv(output_dir / "efficiency_mean.csv", index=False)
    validity_df.to_csv(output_dir / "attack_validity_raw.csv", index=False)
    validity_mean_df.to_csv(output_dir / "attack_validity_mean.csv", index=False)
    full_clean_df.to_csv(output_dir / "full_test_clean_raw.csv", index=False)
    full_clean_mean.to_csv(output_dir / "full_test_clean_mean.csv", index=False)
    full_clean_std.to_csv(output_dir / "full_test_clean_std.csv", index=False)
    if not full_attack_df.empty:
        full_attack_df.to_csv(output_dir / "full_test_attack_raw.csv", index=False)
        full_attack_mean.to_csv(output_dir / "full_test_attack_mean.csv", index=False)
        full_attack_std.to_csv(output_dir / "full_test_attack_std.csv", index=False)

    significance_df = compute_significance_tests(frames.results, config.epsilon_list)
    if not significance_df.empty:
        significance_df.to_csv(output_dir / "significance_tests.csv", index=False)

    plot_metric_curve(mean_df, "f1", output_dir / "f1_curve.png")
    plot_metric_curve(mean_df, "recall", output_dir / "recall_curve.png")
    plot_clean_f1_bar(mean_df, output_dir / "clean_f1_bar.png")
    plot_efficiency_tradeoff(efficiency_mean_df, mean_df, output_dir / "efficiency_tradeoff.png")

    summary_payload = {"config": asdict(config), **summary}
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "mean_results": mean_df,
        "std_results": std_df,
        "efficiency_mean": efficiency_mean_df,
        "validity_mean": validity_mean_df,
    }
