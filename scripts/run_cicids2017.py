"""CIC-IDS2017 cross-dataset experiment (legacy variant).

Trains the same six defenses on CIC-IDS2017 and evaluates them under the shared
attack suite, to check whether the ranking obtained on UNSW-NB15 transfers to a
different traffic distribution.

Run:

    uv run python scripts/run_cicids2017.py --device cuda

The dataset is not shipped with this repository; see data/README.md.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ids_defense_selection as idsds


CICIDS_URLS = {
    "monday.csv": "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/monday.csv",
    "tuesday.csv": "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/tuesday.csv",
    "wednesday.csv": "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/wednesday.csv",
    "thursday.csv": "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/thursday.csv",
    "friday.csv": "https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/friday.csv",
}


FAMILY_TARGETS = {
    "BENIGN": 120_000,
    "DoS": 80_000,
    "PortScan": 50_000,
    "Infiltration": 30_000,
    "BruteForce": 12_000,
    "WebAttack": 8_000,
    "Botnet": 5_000,
    "Heartbleed": 500,
}


@dataclass
class CICExperimentConfig:
    data_dir: str = "data/cicids2017"
    sampled_csv: str = "data/cicids2017/cicids2017_sampled.csv"
    output_dir: str = "outputs/cicids2017_run"
    chunk_size: int = 100_000
    test_size: float = 0.3
    batch_size: int = 1024
    training_budget_mode: str = "legacy"
    baseline_epochs: int = 8
    adv_epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.15
    eval_attack_rows: int = 30000
    adv_epsilon: float = 0.06
    adv_alpha: float = 0.015
    adv_steps: int = 20
    epsilon_list: tuple[float, float, float] = idsds.DEFAULT_EPSILON_LIST
    eval_pgd_steps: int = 20
    seeds: tuple[int, ...] = idsds.DEFAULT_SEEDS
    sensitivity_top_ratio: float = 0.3
    sensitivity_batches: int = 16
    validity_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    transfer_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    family_attack: str = "pgd"
    family_epsilon: float = 0.1
    reference_models: tuple[str, ...] = ("log_reg", "hist_gbdt")
    # TRADES
    trades_beta: float = 6.0
    # Free AT
    free_at_replay: int = 4
    # Class-aware constrained AT
    class_aware_minority_weight: float = 3.0
    # C&W attack
    cw_steps: int = 30
    cw_lr: float = 0.01
    cw_c: float = 1.0
    # APGD attack
    apgd_steps: int = 50
    apgd_rho: float = 0.75
    eval_pgd_alpha_ratio: float = 0.05
    # optional extra defense methods (off by default)
    extra_methods: tuple[str, ...] = ()
    progressive_ratios: tuple[float, ...] = (0.20, 0.30, 0.45, 0.60)
    sa_trades_gamma: float = 1.0
    dst_update_interval: int = 2
    dst_ema_alpha: float = 0.7
    device: str = "auto"
    sample_seed: int = 2026
    rebuild_cache: bool = False


def normalize_attack_family(label: str) -> str:
    cleaned = str(label).strip()
    cleaned = cleaned.replace(" - Attempted", "")
    mapping = {
        "BENIGN": "BENIGN",
        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS Slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "DDoS": "DoS",
        "Heartbleed": "Heartbleed",
        "Web Attack - Brute Force": "WebAttack",
        "Web Attack - XSS": "WebAttack",
        "Web Attack - SQL Injection": "WebAttack",
        "Portscan": "PortScan",
        "Infiltration": "Infiltration",
        "Infiltration - Portscan": "Infiltration",
        "Botnet": "Botnet",
    }
    return mapping.get(cleaned, cleaned)


def ensure_daily_csvs(data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in CICIDS_URLS.items():
        path = data_dir / filename
        paths.append(path)
        if path.exists():
            continue
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
    return paths


def count_families(csv_paths: list[Path], chunk_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in csv_paths:
        for chunk in pd.read_csv(path, usecols=["Label"], chunksize=chunk_size):
            families = chunk["Label"].map(normalize_attack_family)
            value_counts = families.value_counts()
            for family, count in value_counts.items():
                counts[family] = counts.get(family, 0) + int(count)
    return counts


def sample_cicids_dataset(config: CICExperimentConfig) -> pd.DataFrame:
    sampled_path = Path(config.sampled_csv)
    if sampled_path.exists() and not config.rebuild_cache:
        return pd.read_csv(sampled_path)

    csv_paths = ensure_daily_csvs(Path(config.data_dir))
    family_counts = count_families(csv_paths, chunk_size=config.chunk_size)
    sample_probs = {
        family: min(1.0, FAMILY_TARGETS.get(family, 5_000) / count)
        for family, count in family_counts.items()
    }

    rng = np.random.default_rng(config.sample_seed)
    sampled_frames = []
    for path in csv_paths:
        for chunk in pd.read_csv(path, chunksize=config.chunk_size):
            chunk["AttackFamily"] = chunk["Label"].map(normalize_attack_family)
            family_parts = []
            for family, part in chunk.groupby("AttackFamily"):
                prob = sample_probs.get(family, 1.0)
                if prob >= 1.0:
                    family_parts.append(part)
                    continue
                keep_mask = rng.random(len(part)) < prob
                sampled_part = part.loc[keep_mask]
                if not sampled_part.empty:
                    family_parts.append(sampled_part)
            if family_parts:
                sampled_frames.append(pd.concat(family_parts, ignore_index=True))

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    exact_frames = []
    for family, part in sampled_df.groupby("AttackFamily"):
        target = FAMILY_TARGETS.get(family, min(len(part), 5_000))
        exact_frames.append(part.sample(n=min(target, len(part)), random_state=config.sample_seed))
    sampled_df = pd.concat(exact_frames, ignore_index=True)
    sampled_df["binary_label"] = (sampled_df["AttackFamily"] != "BENIGN").astype(np.float32)

    sampled_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(sampled_path, index=False)
    return sampled_df


def build_cicids_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    drop_cols = [
        "Label",
        "AttackFamily",
        "binary_label",
        "Timestamp",
        "Attempted Category",
        "Src IP dec",
        "Dst IP dec",
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


def plot_family_recall_drop(df: pd.DataFrame, output_path: Path) -> None:
    families = [family for family in df["attack_family"].unique() if family != "BENIGN"]
    models = list(df["model"].unique())
    x = np.arange(len(families))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for idx, model in enumerate(models):
        subset = df[df["model"] == model].set_index("attack_family").reindex(families)
        plt.bar(x + idx * width - width, subset["recall_drop"], width=width, label=model)
    plt.xticks(x, families, rotation=20)
    plt.ylabel("Recall drop")
    plt.title("Recall drop by attack family under PGD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_experiment(config: CICExperimentConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = idsds.resolve_device(config.device)
    idsds.log_device(config.device, device)

    sampled_df = sample_cicids_dataset(config)
    train_df, test_df = train_test_split(
        sampled_df,
        test_size=config.test_size,
        random_state=config.sample_seed,
        stratify=sampled_df["AttackFamily"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    x_train, y_train, x_test, y_test, metadata = build_cicids_features(train_df, test_df)
    eval_indices = idsds.stratified_subset_indices(y_test, config.eval_attack_rows, seed=config.sample_seed)
    eval_x = x_test[eval_indices]
    eval_y = y_test[eval_indices]
    eval_families = test_df["AttackFamily"].to_numpy()[eval_indices]
    selected_families = (
        pd.Series(eval_families[eval_y == 1])
        .value_counts()
        .head(6)
        .index.tolist()
    )

    all_results = []
    sensitivity_tables = []
    family_results = []
    validity_results = []
    efficiency_results = []
    reference_clean_results = []
    reference_transfer_results = []

    for seed in config.seeds:
        idsds.set_seed(seed)
        train_loader = idsds.make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        numeric_mask = metadata["numeric_mask"]

        trained = idsds.train_all_defenses(
            model_factory=lambda: idsds.MLPBackbone(x_train.shape[1], config.hidden_dims, config.dropout),
            config=config,
            train_loader=train_loader,
            device=device,
            metadata=metadata,
            seed=seed,
        )
        model_registry = trained.models
        train_time_registry = {
            name: cost["train_seconds"] for name, cost in trained.train_cost.items()
        }
        perturb_ratio_registry = trained.perturb_ratio
        selected_feature_registry = trained.perturb_features
        sensitivity_tables.append(trained.sensitivity_table)

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
                attack_mask=numeric_mask,
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
                    attack_mask=numeric_mask,
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
                            numeric_mask=numeric_mask,
                            numeric_mins=metadata["numeric_mins"],
                            numeric_maxs=metadata["numeric_maxs"],
                        ),
                    }
                )

            family_df = idsds.evaluate_category_recall(
                model_name=model_name,
                model=model,
                x_eval=eval_x,
                y_eval=eval_y,
                attack_categories=eval_families,
                selected_categories=selected_families,
                attack=config.family_attack,
                epsilon=config.family_epsilon,
                attack_mask=numeric_mask,
                mins=metadata["numeric_mins"],
                maxs=metadata["numeric_maxs"],
                config=config,
                device=device,
            )
            family_df.insert(0, "seed", seed)
            family_results.append(family_df)

        reference_model_registry, reference_train_time_registry = idsds.fit_reference_models(
            x_train=x_train,
            y_train=y_train,
            seed=seed,
            enabled_models=config.reference_models,
        )
        for reference_name, reference_model in reference_model_registry.items():
            reference_probs = idsds.predict_proba(
                reference_model,
                eval_x,
                batch_size=config.batch_size,
                device=device,
            )
            reference_clean_results.append(
                {
                    "seed": seed,
                    "model": reference_name,
                    **idsds.classification_metrics(eval_y, reference_probs),
                    "train_seconds": reference_train_time_registry[reference_name],
                    **idsds.measure_inference_cost(
                        model=reference_model,
                        x_eval=eval_x,
                        batch_size=config.batch_size,
                        device=device,
                    ),
                }
            )

        for attack_name, epsilon in config.transfer_attack_settings:
            for source_name, source_model in model_registry.items():
                transfer_df = idsds.evaluate_transfer_attack(
                    source_name=source_name,
                    source_model=source_model,
                    target_models=reference_model_registry,
                    x_eval=eval_x,
                    y_eval=eval_y,
                    attack=attack_name,
                    epsilon=epsilon,
                    attack_mask=numeric_mask,
                    mins=metadata["numeric_mins"],
                    maxs=metadata["numeric_maxs"],
                    config=config,
                    device=device,
                )
                transfer_df.insert(0, "seed", seed)
                reference_transfer_results.append(transfer_df)

    results_df = pd.concat(all_results, ignore_index=True)
    mean_df, std_df = idsds.summarize_results(results_df)
    sensitivity_df = pd.concat(sensitivity_tables, ignore_index=True)
    family_df = pd.concat(family_results, ignore_index=True)
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
    reference_clean_df = pd.DataFrame(reference_clean_results)
    reference_metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "attack_success_rate",
        "auc",
        "train_seconds",
        "inference_probe_rows",
        "inference_seconds",
        "inference_ms_per_sample",
        "inference_samples_per_second",
    ]
    reference_clean_mean_df = reference_clean_df.groupby(["model"], as_index=False)[reference_metric_cols].mean()
    reference_clean_std_df = reference_clean_df.groupby(["model"], as_index=False)[reference_metric_cols].std().fillna(0.0)
    reference_transfer_df = pd.concat(reference_transfer_results, ignore_index=True)
    reference_transfer_mean_df, reference_transfer_std_df = idsds.summarize_results(
        reference_transfer_df,
        group_cols=["source_model", "target_model", "attack", "epsilon"],
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

    results_df.to_csv(output_dir / "raw_results.csv", index=False)
    mean_df.to_csv(output_dir / "mean_results.csv", index=False)
    std_df.to_csv(output_dir / "std_results.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_scores.csv", index=False)
    family_df.to_csv(output_dir / "family_raw_results.csv", index=False)
    family_mean_df.to_csv(output_dir / "family_mean_results.csv", index=False)
    reference_clean_df.to_csv(output_dir / "reference_model_clean_raw.csv", index=False)
    reference_clean_mean_df.to_csv(output_dir / "reference_model_clean_mean.csv", index=False)
    reference_clean_std_df.to_csv(output_dir / "reference_model_clean_std.csv", index=False)
    reference_transfer_df.to_csv(output_dir / "reference_transfer_raw_results.csv", index=False)
    reference_transfer_mean_df.to_csv(output_dir / "reference_transfer_mean_results.csv", index=False)
    reference_transfer_std_df.to_csv(output_dir / "reference_transfer_std_results.csv", index=False)
    validity_df.to_csv(output_dir / "attack_validity_raw.csv", index=False)
    validity_mean_df.to_csv(output_dir / "attack_validity_mean.csv", index=False)
    efficiency_df.to_csv(output_dir / "efficiency_raw.csv", index=False)
    efficiency_mean_df.to_csv(output_dir / "efficiency_mean.csv", index=False)

    idsds.plot_metric_curve(mean_df, "f1", output_dir / "f1_curve.png")
    idsds.plot_metric_curve(mean_df, "recall", output_dir / "recall_curve.png")
    idsds.plot_clean_f1_bar(mean_df, output_dir / "clean_f1_bar.png")
    plot_family_recall_drop(family_mean_df, output_dir / "family_recall_drop.png")
    idsds.plot_efficiency_tradeoff(efficiency_mean_df, mean_df, output_dir / "efficiency_tradeoff.png")

    summary = {
        "config": asdict(config),
        "family_targets": FAMILY_TARGETS,
        "sampled_rows": int(len(sampled_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(eval_y)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "feature_count": int(len(metadata["feature_names"])),
        "selected_attack_families": selected_families,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def parse_args() -> CICExperimentConfig:
    parser = argparse.ArgumentParser(description="Run CIC-IDS2017 adversarial robustness experiments.")
    parser.add_argument("--data-dir", default=str(idsds.DEFAULT_DATA_DIR / "cicids2017"),
                        help="directory with the daily CIC-IDS2017 csv files "
                             "(relative paths resolve against the repository root)")
    parser.add_argument("--sampled-csv",
                        default=str(idsds.DEFAULT_DATA_DIR / "cicids2017" / "cicids2017_sampled.csv"))
    parser.add_argument("--output-dir",
                        default=str(idsds.DEFAULT_OUTPUT_ROOT / "cicids2017_run"))
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--baseline-epochs", type=int, default=8)
    parser.add_argument("--adv-epochs", type=int, default=6)
    parser.add_argument("--eval-attack-rows", type=int, default=30000)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated training seeds (default: 7,13,21,42,100)")
    parser.add_argument("--device", default="auto",
                        help="torch device: auto, cpu, cuda, cuda:N or mps")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    kwargs = dict(
        data_dir=str(idsds.resolve_path(args.data_dir)),
        sampled_csv=str(idsds.resolve_path(args.sampled_csv)),
        output_dir=str(idsds.resolve_path(args.output_dir)),
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        baseline_epochs=args.baseline_epochs,
        adv_epochs=args.adv_epochs,
        eval_attack_rows=args.eval_attack_rows,
        device=args.device,
        rebuild_cache=args.rebuild_cache,
    )
    if args.seeds:
        kwargs["seeds"] = tuple(
            int(part) for part in args.seeds.split(",") if part.strip())
    return CICExperimentConfig(**kwargs)


if __name__ == "__main__":
    CONFIG = parse_args()
    run_experiment(CONFIG)
