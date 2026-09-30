"""CIC-IDS2017 source-disjoint validation.

Stricter variant of run_cicids2017.py: the train/test partition is built so
that no source IP or capture block appears in both halves, which removes the
near-duplicate leakage that a naive random split of flow records would allow.
This is the variant whose numbers are the more defensible cross-dataset result.

Depends on run_cicids2017.py for the shared loading utilities.

    uv run python code/run_cicids2017_source_disjoint.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_cicids2017 as legacy_cic
import ids_defense_selection as idsds


MODEL_ORDER = [
    "standard",
    "adv_training",
    "constrained_adv",
    "trades",
    "free_at",
    "class_aware_constrained",
]

FULL_TEST_ATTACK_SETTINGS = idsds.DEFAULT_FULL_TEST_ATTACK_SETTINGS + (("apgd", 0.10),)


@dataclass
class CICStrictExperimentConfig:
    data_dir: str = "data/cicids2017"
    output_dir: str = "outputs/cicids2017_strict_matched_budget_run"
    chunk_size: int = 100_000
    blocks_per_day: int = 10
    train_block_ratio: float = 0.7
    batch_size: int = 1024
    training_budget_mode: str = "matched_continuation"
    baseline_epochs: int = 10
    adv_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.15
    eval_attack_rows: int = 30_000
    full_test_attack_rows: int = 0
    train_max_rows: int = 300_000
    adv_epsilon: float = 0.06
    adv_alpha: float = 0.015
    adv_steps: int = 20
    epsilon_list: tuple[float, float, float] = idsds.DEFAULT_EPSILON_LIST
    eval_pgd_steps: int = 20
    eval_pgd_alpha_ratio: float = 0.05
    seeds: tuple[int, ...] = idsds.DEFAULT_SEEDS
    sensitivity_top_ratio: float = 0.3
    sensitivity_batches: int = 16
    validity_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    family_attack: str = "pgd"
    family_epsilon: float = 0.1
    family_min_test_samples: int = 200
    trades_beta: float = 6.0
    free_at_replay: int = 4
    class_aware_minority_weight: float = 3.0
    # optional extra defense methods (off by default)
    extra_methods: tuple[str, ...] = ()
    progressive_ratios: tuple[float, ...] = (0.20, 0.30, 0.45, 0.60)
    sa_trades_gamma: float = 1.0
    dst_update_interval: int = 2
    dst_ema_alpha: float = 0.7
    cw_steps: int = 30
    cw_lr: float = 0.01
    cw_c: float = 1.0
    apgd_steps: int = 50
    apgd_rho: float = 0.75
    device: str = "cpu"
    split_seed: int = 2026
    train_sample_seed: int = 2027
    rebuild_cache: bool = False


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
    attack_columns = ["clean"] + [
        f"{attack}_{eps}" for attack in ["fgsm", "pgd", "cw", "apgd"] for eps in epsilon_list
    ]

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
                if len(a) < 2 or len(a) != len(b):
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


# ---------------------------------------------------------------- six defenses on the source-disjoint split
def train_all_models(
    config: CICStrictExperimentConfig,
    train_loader,
    input_dim: int,
    device: torch.device,
    metadata: dict,
    seed: int,
):
    """Train the six defenses with the shared matched-budget protocol.

    Thin adapter around :func:`ids_defense_selection.train_all_defenses` that
    keeps this script's historical return shape.
    """
    trained = idsds.train_all_defenses(
        model_factory=lambda: idsds.MLPBackbone(input_dim, config.hidden_dims, config.dropout),
        config=config,
        train_loader=train_loader,
        device=device,
        metadata=metadata,
        seed=seed,
    )
    train_time_registry = {
        name: cost["train_seconds"] for name, cost in trained.train_cost.items()
    }
    return (
        trained.models,
        train_time_registry,
        trained.perturb_ratio,
        trained.perturb_features,
        trained.sensitivity_table,
    )

# ---------------------------------------------------------------- per-flow provenance bookkeeping
def assign_block_ids(length: int, blocks_per_day: int) -> np.ndarray:
    effective_blocks = max(2, blocks_per_day)
    boundaries = np.linspace(0, length, effective_blocks + 1, dtype=int)
    block_ids = np.empty(length, dtype=np.int32)
    for block_id in range(effective_blocks):
        start = boundaries[block_id]
        end = boundaries[block_id + 1]
        block_ids[start:end] = block_id
    return block_ids


def load_day_with_provenance(path: Path, blocks_per_day: int) -> pd.DataFrame:
    day_name = path.stem.lower()
    df = pd.read_csv(path)
    df["AttackFamily"] = df["Label"].map(legacy_cic.normalize_attack_family)
    df["binary_label"] = (df["AttackFamily"] != "BENIGN").astype(np.float32)
    df["source_day"] = day_name
    df["source_row_id"] = np.arange(len(df), dtype=np.int64)
    df["source_block_id"] = assign_block_ids(len(df), blocks_per_day)
    return df


def proportional_group_targets(counts: pd.Series, max_total: int) -> dict[str, int]:
    if max_total <= 0 or counts.sum() <= max_total:
        return {str(key): int(value) for key, value in counts.items()}

    raw = counts / counts.sum() * max_total
    targets = np.floor(raw).astype(int)
    targets = targets.clip(lower=1)

    while targets.sum() > max_total:
        reducible = targets[targets > 1]
        if reducible.empty:
            break
        key = reducible.sort_values(ascending=False).index[0]
        targets.loc[key] -= 1

    remainder = (raw - np.floor(raw)).sort_values(ascending=False)
    while targets.sum() < max_total:
        for key in remainder.index:
            if targets.sum() >= max_total:
                break
            if targets.loc[key] < counts.loc[key]:
                targets.loc[key] += 1
        else:
            break

    targets = targets.clip(upper=counts)
    return {str(key): int(value) for key, value in targets.items()}


def sample_training_rows(train_df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(train_df) <= max_rows:
        return train_df.reset_index(drop=True)

    counts = train_df["AttackFamily"].value_counts().sort_index()
    targets = proportional_group_targets(counts, max_rows)
    sampled_frames = []
    for offset, (family, group) in enumerate(train_df.groupby("AttackFamily", sort=True)):
        target = targets.get(str(family), 0)
        if target <= 0:
            continue
        if len(group) <= target:
            sampled_frames.append(group)
        else:
            sampled_frames.append(group.sample(n=target, random_state=seed + offset))
    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_source_disjoint(config: CICStrictExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    csv_paths = legacy_cic.ensure_daily_csvs(Path(config.data_dir))
    train_blocks = max(1, min(config.blocks_per_day - 1, math.floor(config.blocks_per_day * config.train_block_ratio)))
    test_blocks = config.blocks_per_day - train_blocks
    if test_blocks <= 0:
        raise ValueError("train_block_ratio leaves no held-out blocks; increase blocks_per_day or lower train_block_ratio.")

    train_frames = []
    test_frames = []
    split_rows = []

    for path in csv_paths:
        day_df = load_day_with_provenance(path, config.blocks_per_day)
        day_train = day_df[day_df["source_block_id"] < train_blocks].reset_index(drop=True)
        day_test = day_df[day_df["source_block_id"] >= train_blocks].reset_index(drop=True)
        train_frames.append(day_train)
        test_frames.append(day_test)
        split_rows.append(
            {
                "source_day": path.stem.lower(),
                "raw_rows": int(len(day_df)),
                "train_rows": int(len(day_train)),
                "test_rows": int(len(day_test)),
                "train_positive_rate": float(day_train["binary_label"].mean()) if len(day_train) else 0.0,
                "test_positive_rate": float(day_test["binary_label"].mean()) if len(day_test) else 0.0,
            }
        )

    raw_train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    train_df = sample_training_rows(raw_train_df, config.train_max_rows, seed=config.train_sample_seed)

    split_summary = {
        "blocks_per_day": config.blocks_per_day,
        "train_blocks_per_day": train_blocks,
        "test_blocks_per_day": test_blocks,
        "train_sampling_applied": int(len(train_df) != len(raw_train_df)),
        "raw_train_rows": int(len(raw_train_df)),
        "sampled_train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "per_day_counts": split_rows,
        "train_family_counts": train_df["AttackFamily"].value_counts().sort_index().to_dict(),
        "test_family_counts": test_df["AttackFamily"].value_counts().sort_index().to_dict(),
    }
    return train_df, test_df, split_summary


def build_cicids_features_with_provenance(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    drop_cols = [
        "Label",
        "AttackFamily",
        "binary_label",
        "Timestamp",
        "Attempted Category",
        "Src IP dec",
        "Dst IP dec",
        "source_day",
        "source_row_id",
        "source_block_id",
    ]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    train_x = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_x = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_train = transformer.fit_transform(train_x).astype(np.float32)
    x_test = transformer.transform(test_x).astype(np.float32)
    y_train = train_df["binary_label"].astype(np.float32).to_numpy()
    y_test = test_df["binary_label"].astype(np.float32).to_numpy()

    numeric_mask = np.ones(x_train.shape[1], dtype=np.float32)
    metadata = {
        "numeric_cols": feature_cols,
        "categorical_cols": [],
        "feature_names": feature_cols,
        "numeric_mask": numeric_mask,
        "numeric_mins": x_train.min(axis=0),
        "numeric_maxs": x_train.max(axis=0),
    }
    return x_train, y_train, x_test, y_test, metadata


# ---------------------------------------------------------------- source-disjoint split machinery
def select_attack_families(eval_families: np.ndarray, eval_y: np.ndarray, min_samples: int) -> list[str]:
    counts = pd.Series(eval_families[eval_y == 1]).value_counts()
    return counts[counts >= min_samples].index.tolist()


# ---------------------------------------------------------------- full experiment for one seed
def run_experiment(config: CICStrictExperimentConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    train_df, test_df, split_summary = split_source_disjoint(config)
    x_train, y_train, x_test, y_test, metadata = build_cicids_features_with_provenance(train_df, test_df)

    eval_size = min(config.eval_attack_rows, len(y_test))
    eval_indices = idsds.stratified_subset_indices(y_test, eval_size, seed=config.split_seed)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    eval_families = test_df["AttackFamily"].to_numpy()[eval_indices]
    selected_families = select_attack_families(eval_families, eval_y, config.family_min_test_samples)

    full_attack_subset = "full_test"
    if config.full_test_attack_rows and config.full_test_attack_rows > 0:
        full_attack_size = min(config.full_test_attack_rows, len(y_test))
        full_attack_indices = idsds.stratified_subset_indices(y_test, full_attack_size, seed=config.split_seed + 1)
        full_attack_x = x_test[full_attack_indices]
        full_attack_y = y_test[full_attack_indices]
        full_attack_subset = "full_test_subset"
    else:
        full_attack_x = x_test
        full_attack_y = y_test

    all_results = []
    sensitivity_tables = []
    family_results = []
    validity_results = []
    efficiency_results = []
    full_test_clean_results = []
    full_test_attack_results = []

    for seed in config.seeds:
        idsds.set_seed(seed)
        train_loader = idsds.make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        input_dim = x_train.shape[1]
        model_registry, train_time_registry, perturb_ratio_registry, selected_feature_registry, sensitivity_table = (
            train_all_models(config, train_loader, input_dim, device, metadata, seed)
        )
        sensitivity_tables.append(sensitivity_table)

        for model_name, model in model_registry.items():
            clean_probs = idsds.predict_proba(model, eval_x, batch_size=config.batch_size, device=device)
            clean_pred = (clean_probs >= 0.5).astype(np.int32)

            efficiency_results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "parameter_count": idsds.count_parameters(model),
                    "train_seconds": train_time_registry[model_name],
                    "training_attack_feature_ratio": perturb_ratio_registry[model_name],
                    "training_attack_feature_count": selected_feature_registry[model_name],
                    **idsds.measure_inference_cost(
                        model=model,
                        x_eval=eval_x,
                        batch_size=config.batch_size,
                        device=device,
                    ),
                }
            )

            result_df = idsds.evaluate_attack_suite(
                model=model,
                x_eval=eval_x,
                y_eval=eval_y,
                attack_mask=metadata["numeric_mask"],
                mins=metadata["numeric_mins"],
                maxs=metadata["numeric_maxs"],
                config=config,
                device=device,
            )
            result_df.insert(0, "seed", seed)
            result_df.insert(1, "model", model_name)
            all_results.append(result_df)

            for attack_name, epsilon in config.validity_attack_settings:
                attacked_inputs, attacked_probs = idsds.generate_adversarial_examples(
                    model=model,
                    x_eval=eval_x,
                    y_eval=eval_y,
                    attack=attack_name,
                    epsilon=epsilon,
                    config=config,
                    attack_mask=metadata["numeric_mask"],
                    mins=metadata["numeric_mins"],
                    maxs=metadata["numeric_maxs"],
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
                        **idsds.classification_metrics(eval_y, attacked_probs, clean_pred=clean_pred),
                        **idsds.compute_attack_validity_metrics(
                            x_clean=eval_x,
                            x_adv=attacked_inputs,
                            numeric_mask=metadata["numeric_mask"],
                            numeric_mins=metadata["numeric_mins"],
                            numeric_maxs=metadata["numeric_maxs"],
                        ),
                    }
                )

            if selected_families:
                family_df = idsds.evaluate_category_recall(
                    model_name=model_name,
                    model=model,
                    x_eval=eval_x,
                    y_eval=eval_y,
                    attack_categories=eval_families,
                    selected_categories=selected_families,
                    attack=config.family_attack,
                    epsilon=config.family_epsilon,
                    attack_mask=metadata["numeric_mask"],
                    mins=metadata["numeric_mins"],
                    maxs=metadata["numeric_maxs"],
                    config=config,
                    device=device,
                )
                family_df.insert(0, "seed", seed)
                family_results.append(family_df)

            full_test_probs = idsds.predict_proba(
                model,
                x_test,
                batch_size=config.batch_size,
                device=device,
            )
            full_test_clean_results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "subset": "full_test",
                    **idsds.classification_metrics(y_test, full_test_probs),
                }
            )

            full_attack_clean_probs = idsds.predict_proba(
                model,
                full_attack_x,
                batch_size=config.batch_size,
                device=device,
            )
            full_attack_clean_pred = (full_attack_clean_probs >= 0.5).astype(np.int32)
            for attack_name, epsilon in FULL_TEST_ATTACK_SETTINGS:
                _, attacked_probs = idsds.generate_adversarial_examples(
                    model=model,
                    x_eval=full_attack_x,
                    y_eval=full_attack_y,
                    attack=attack_name,
                    epsilon=epsilon,
                    config=config,
                    attack_mask=metadata["numeric_mask"],
                    mins=metadata["numeric_mins"],
                    maxs=metadata["numeric_maxs"],
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
                        "test_rows": int(len(full_attack_y)),
                        **idsds.classification_metrics(full_attack_y, attacked_probs, clean_pred=full_attack_clean_pred),
                    }
                )

    results_df = pd.concat(all_results, ignore_index=True)
    mean_df, std_df = idsds.summarize_results(results_df)
    sensitivity_df = pd.concat(sensitivity_tables, ignore_index=True)

    family_df = pd.concat(family_results, ignore_index=True) if family_results else pd.DataFrame()
    if not family_df.empty:
        family_mean_df = (
            family_df.groupby(["model", "attack", "epsilon", "attack_cat"], as_index=False)
            .agg(
                samples=("samples", "mean"),
                clean_recall=("clean_recall", "mean"),
                adv_recall=("adv_recall", "mean"),
                recall_drop=("recall_drop", "mean"),
            )
            .sort_values(["model", "recall_drop"], ascending=[True, False])
        )
        family_mean_df["samples"] = family_mean_df["samples"].round().astype(int)
        family_mean_df = family_mean_df.rename(columns={"attack_cat": "attack_family"})
    else:
        family_mean_df = pd.DataFrame()

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

    efficiency_df = pd.DataFrame(efficiency_results)
    efficiency_mean_df = (
        efficiency_df.groupby(["model"], as_index=False)
        .agg(
            parameter_count=("parameter_count", "mean"),
            train_seconds=("train_seconds", "mean"),
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
    efficiency_mean_df["relative_train_cost_vs_standard"] = efficiency_mean_df["train_seconds"] / standard_train_seconds
    efficiency_mean_df["relative_inference_latency_vs_standard"] = (
        efficiency_mean_df["inference_ms_per_sample"] / standard_inference_ms
    )

    full_test_clean_df = pd.DataFrame(full_test_clean_results)
    full_test_metric_cols = ["accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    full_test_clean_mean_df = full_test_clean_df.groupby(["model", "subset"], as_index=False)[
        full_test_metric_cols
    ].mean()
    full_test_clean_std_df = full_test_clean_df.groupby(["model", "subset"], as_index=False)[
        full_test_metric_cols
    ].std().fillna(0.0)

    full_test_attack_df = pd.DataFrame(full_test_attack_results)
    full_test_attack_metric_cols = ["test_rows", "accuracy", "precision", "recall", "f1", "attack_success_rate", "auc"]
    full_test_attack_mean_df = full_test_attack_df.groupby(
        ["model", "subset", "attack", "epsilon"], as_index=False
    )[full_test_attack_metric_cols].mean()
    full_test_attack_std_df = full_test_attack_df.groupby(
        ["model", "subset", "attack", "epsilon"], as_index=False
    )[full_test_attack_metric_cols].std().fillna(0.0)

    results_df.to_csv(output_dir / "raw_results.csv", index=False)
    mean_df.to_csv(output_dir / "mean_results.csv", index=False)
    std_df.to_csv(output_dir / "std_results.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_scores.csv", index=False)
    validity_df.to_csv(output_dir / "attack_validity_raw.csv", index=False)
    validity_mean_df.to_csv(output_dir / "attack_validity_mean.csv", index=False)
    efficiency_df.to_csv(output_dir / "efficiency_raw.csv", index=False)
    efficiency_mean_df.to_csv(output_dir / "efficiency_mean.csv", index=False)
    full_test_clean_df.to_csv(output_dir / "full_test_clean_raw.csv", index=False)
    full_test_clean_mean_df.to_csv(output_dir / "full_test_clean_mean.csv", index=False)
    full_test_clean_std_df.to_csv(output_dir / "full_test_clean_std.csv", index=False)
    full_test_attack_df.to_csv(output_dir / "full_test_attack_raw.csv", index=False)
    full_test_attack_mean_df.to_csv(output_dir / "full_test_attack_mean.csv", index=False)
    full_test_attack_std_df.to_csv(output_dir / "full_test_attack_std.csv", index=False)

    if not family_df.empty:
        family_df.to_csv(output_dir / "family_raw_results.csv", index=False)
        family_mean_df.to_csv(output_dir / "family_mean_results.csv", index=False)

    write_significance_tests(results_df, config.epsilon_list, output_dir / "significance_tests.csv")

    idsds.plot_metric_curve(mean_df, "f1", output_dir / "f1_curve.png")
    idsds.plot_metric_curve(mean_df, "recall", output_dir / "recall_curve.png")
    idsds.plot_clean_f1_bar(mean_df, output_dir / "clean_f1_bar.png")
    idsds.plot_efficiency_tradeoff(efficiency_mean_df, mean_df, output_dir / "efficiency_tradeoff.png")
    if not family_mean_df.empty:
        legacy_cic.plot_family_recall_drop(family_mean_df, output_dir / "family_recall_drop.png")

    summary = {
        "config": asdict(config),
        "split_summary": split_summary,
        "eval_rows": int(len(eval_y)),
        "full_test_attack_rows": int(len(full_attack_y)),
        "full_test_attack_subset": full_attack_subset,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "feature_count": int(len(metadata["feature_names"])),
        "selected_attack_families": selected_families,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- CLI
def parse_args() -> CICStrictExperimentConfig:
    parser = argparse.ArgumentParser(description="Run strict source-disjoint CIC-IDS2017 adversarial experiments.")
    parser.add_argument("--data-dir", default=str(idsds.DEFAULT_DATA_DIR / "cicids2017"),
                        help="directory with the daily CIC-IDS2017 csv files "
                             "(relative paths resolve against the repository root)")
    parser.add_argument("--output-dir",
                        default=str(idsds.DEFAULT_OUTPUT_ROOT / "cicids2017_strict_matched_budget_run"))
    parser.add_argument("--blocks-per-day", type=int, default=10)
    parser.add_argument("--train-block-ratio", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--training-budget-mode", choices=["legacy", "matched_continuation"], default="matched_continuation")
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--adv-epochs", type=int, default=8)
    parser.add_argument("--eval-attack-rows", type=int, default=30000)
    parser.add_argument("--full-test-attack-rows", type=int, default=0)
    parser.add_argument("--train-max-rows", type=int, default=300000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(idsds.DEFAULT_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    return CICStrictExperimentConfig(
        data_dir=str(idsds.resolve_path(args.data_dir)),
        output_dir=str(idsds.resolve_path(args.output_dir)),
        blocks_per_day=args.blocks_per_day,
        train_block_ratio=args.train_block_ratio,
        batch_size=args.batch_size,
        training_budget_mode=args.training_budget_mode,
        baseline_epochs=args.baseline_epochs,
        adv_epochs=args.adv_epochs,
        eval_attack_rows=args.eval_attack_rows,
        full_test_attack_rows=args.full_test_attack_rows,
        train_max_rows=args.train_max_rows,
        seeds=tuple(args.seeds),
        device=args.device,
        rebuild_cache=args.rebuild_cache,
    )


if __name__ == "__main__":
    CONFIG = parse_args()
    run_experiment(CONFIG)
