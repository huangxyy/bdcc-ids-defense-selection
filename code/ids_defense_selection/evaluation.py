"""Prediction, metrics, cost measurement and the shared evaluation loop.

The same routines serve the MLP, 1D-CNN, FT-Transformer and CIC-IDS2017
experiments, so every reported number is produced by identical code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .attacks import compute_attack_validity_metrics, generate_adversarial_examples
from .config import ExperimentConfig
from .data import make_dataloader, stratified_subset_indices
from .defenses import TrainedDefenses


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clean_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Accuracy / precision / recall / F1 / AUC, plus ASR when `clean_pred` is given.

    The attack success rate is the fraction of *originally correct* samples that
    the attack flips, which is the standard definition used throughout the paper.
    """
    y_pred = (y_prob >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.0
    if clean_pred is not None:
        originally_correct = clean_pred == y_true
        if originally_correct.any():
            attack_success = (y_pred[originally_correct] != y_true[originally_correct]).mean()
            metrics["attack_success_rate"] = float(attack_success)
        else:
            metrics["attack_success_rate"] = 0.0
    else:
        metrics["attack_success_rate"] = 0.0
    return metrics


def predict_proba(model, x_eval: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Positive-class probabilities for a torch backbone or an sklearn estimator."""
    if isinstance(model, nn.Module):
        return _predict_proba_loader(
            model,
            make_dataloader(x_eval, np.zeros(len(x_eval), dtype=np.float32),
                            batch_size=batch_size, shuffle=False),
            device,
        )
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_eval)
        return probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    raise TypeError(f"Unsupported model type for probability prediction: {type(model)!r}")


def _predict_proba_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    probs = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def measure_inference_cost(
    model,
    x_eval: np.ndarray,
    batch_size: int,
    device: torch.device,
    repeats: int = 3,
) -> dict[str, float]:
    """Wall-clock inference cost on a fixed probe subset (batch_size * 8 rows).

    Works for both torch backbones and sklearn reference models.
    """
    probe_rows = int(min(len(x_eval), batch_size * 8))
    probe_x = x_eval[:probe_rows]

    predict_proba(model, probe_x, batch_size=batch_size, device=device)
    start = time.perf_counter()
    for _ in range(repeats):
        predict_proba(model, probe_x, batch_size=batch_size, device=device)
    elapsed = time.perf_counter() - start
    mean_seconds = elapsed / max(repeats, 1)
    return {
        "inference_probe_rows": probe_rows,
        "inference_seconds": mean_seconds,
        "inference_ms_per_sample": (mean_seconds * 1000.0 / probe_rows) if probe_rows else 0.0,
        "inference_samples_per_second": (probe_rows / mean_seconds) if mean_seconds > 0 else 0.0,
    }


def evaluate_model_on_attack(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Clean-vs-attacked metrics for one (attack, epsilon) cell."""
    clean_probs = predict_proba(model, x_eval, batch_size=config.batch_size, device=device)
    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    _, attacked_probs = generate_adversarial_examples(
        model, x_eval, y_eval, attack, epsilon, config, attack_mask, mins, maxs,
        device, config.batch_size, return_inputs=False,
    )
    clean_metrics = classification_metrics(y_eval, clean_probs)
    attack_metrics = classification_metrics(y_eval, attacked_probs, clean_pred=clean_pred)
    return {
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_recall": clean_metrics["recall"],
        "clean_f1": clean_metrics["f1"],
        "robust_accuracy": attack_metrics["accuracy"],
        "robust_recall": attack_metrics["recall"],
        "robust_f1": attack_metrics["f1"],
        "attack_success_rate": attack_metrics["attack_success_rate"],
    }


def evaluate_transfer_attack(
    source_name: str,
    source_model: nn.Module,
    target_models: dict[str, object],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Attack every source model and measure the metrics it transfers to targets."""
    attacked_inputs, _ = generate_adversarial_examples(
        source_model, x_eval, y_eval, attack, epsilon, config, attack_mask, mins, maxs,
        device, config.batch_size, return_inputs=True,
    )
    rows = []
    for target_name, target_model in target_models.items():
        clean_probs = predict_proba(target_model, x_eval, batch_size=config.batch_size, device=device)
        clean_pred = (clean_probs >= 0.5).astype(np.int32)
        attacked_probs = predict_proba(target_model, attacked_inputs,
                                       batch_size=config.batch_size, device=device)
        rows.append(
            {
                "source_model": source_name,
                "target_model": target_name,
                "attack": attack,
                "epsilon": epsilon,
                **classification_metrics(y_eval, attacked_probs, clean_pred=clean_pred),
            }
        )
    return pd.DataFrame(rows)


def evaluate_category_recall(
    model_name: str,
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_categories: np.ndarray,
    selected_categories: list[str],
    attack: str,
    epsilon: float,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Per-attack-category recall before and after the attack (phi4 source)."""
    clean_probs = predict_proba(model, x_eval, batch_size=config.batch_size, device=device)
    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    _, attacked_probs = generate_adversarial_examples(
        model, x_eval, y_eval, attack, epsilon, config, attack_mask, mins, maxs,
        device, config.batch_size, return_inputs=False,
    )
    attacked_pred = (attacked_probs >= 0.5).astype(np.int32)
    rows = []
    for category in selected_categories:
        category_mask = (attack_categories == category) & (y_eval == 1)
        sample_count = int(category_mask.sum())
        if sample_count == 0:
            continue
        clean_recall = float((clean_pred[category_mask] == 1).mean())
        adv_recall = float((attacked_pred[category_mask] == 1).mean())
        rows.append(
            {
                "model": model_name,
                "attack": attack,
                "epsilon": epsilon,
                "attack_cat": category,
                "samples": sample_count,
                "clean_recall": clean_recall,
                "adv_recall": adv_recall,
                "recall_drop": clean_recall - adv_recall,
            }
        )
    return pd.DataFrame(rows)


def evaluate_attack_suite(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    attack_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Clean + FGSM/PGD/C&W/APGD at every epsilon in the evaluation budget."""
    clean_probs = predict_proba(model, x_eval, batch_size=config.batch_size, device=device)
    clean_pred = (clean_probs >= 0.5).astype(np.int32)
    rows = [{
        "attack": "clean",
        "epsilon": 0.0,
        **classification_metrics(y_eval, clean_probs),
    }]
    for attack in ("fgsm", "pgd", "cw", "apgd"):
        for epsilon in config.epsilon_list:
            _, attacked_probs = generate_adversarial_examples(
                model, x_eval, y_eval, attack, epsilon, config, attack_mask, mins, maxs,
                device, config.batch_size, return_inputs=False,
            )
            rows.append(
                {
                    "attack": attack,
                    "epsilon": epsilon,
                    **classification_metrics(y_eval, attacked_probs, clean_pred=clean_pred),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class EvaluationSet:
    """Evaluation data and attack constraints shared by every defense of a run."""

    x_eval: np.ndarray
    y_eval: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    attack_mask: np.ndarray
    numeric_mins: np.ndarray
    numeric_maxs: np.ndarray
    attack_categories: np.ndarray | None = None
    top_categories: list[str] = field(default_factory=list)


@dataclass
class EvaluationResults:
    """All frames one backbone run reports for its defense candidates."""

    metrics: pd.DataFrame
    efficiency: pd.DataFrame
    validity: pd.DataFrame
    categories: pd.DataFrame
    full_test_clean: pd.DataFrame
    full_test_attack: pd.DataFrame


def evaluate_defenses(
    trained: TrainedDefenses,
    config: ExperimentConfig,
    eval_set: EvaluationSet,
    device: torch.device,
    seed: int,
    *,
    include_categories: bool = True,
    full_test_attack_settings: tuple[tuple[str, float], ...] = (),
    full_test_attack_rows: int = 0,
    full_test_attack_seed: int | None = None,
) -> EvaluationResults:
    """Run the full evaluation protocol over every trained defense candidate.

    Produces the per-(model, attack, epsilon) metrics, the cost/latency table,
    the attack-validity checks, the per-category recall (objective phi4), and
    clean/attacked metrics on the full test partition.
    """
    results_rows: list[pd.DataFrame] = []
    efficiency_rows: list[dict] = []
    validity_rows: list[dict] = []
    category_rows: list[pd.DataFrame] = []
    full_clean_rows: list[dict] = []
    full_attack_rows: list[dict] = []

    if full_test_attack_rows > 0:
        attack_seed = (config.eval_subset_seed + 1) if full_test_attack_seed is None \
            else full_test_attack_seed
        full_indices = stratified_subset_indices(
            eval_set.y_test, full_test_attack_rows, seed=attack_seed)
        full_x = eval_set.x_test[full_indices]
        full_y = eval_set.y_test[full_indices]
        full_subset = "full_test_subset"
    else:
        full_x = eval_set.x_test
        full_y = eval_set.y_test
        full_subset = "full_test"

    for model_name, model in trained.models.items():
        clean_probs = predict_proba(model, eval_set.x_eval, batch_size=config.batch_size, device=device)
        clean_pred = (clean_probs >= 0.5).astype(np.int32)

        cost = trained.train_cost[model_name]
        efficiency_rows.append({
            "seed": seed,
            "model": model_name,
            "parameter_count": count_parameters(model),
            "train_seconds": cost["train_seconds"],
            "pretrain_seconds": cost["pretrain_seconds"],
            "continuation_seconds": cost["continuation_seconds"],
            "mask_build_seconds": cost["mask_build_seconds"],
            "training_attack_feature_ratio": trained.perturb_ratio[model_name],
            "training_attack_feature_count": trained.perturb_features[model_name],
            **measure_inference_cost(model, eval_set.x_eval, config.batch_size, device),
        })

        model_df = evaluate_attack_suite(
            model, eval_set.x_eval, eval_set.y_eval, eval_set.attack_mask,
            eval_set.numeric_mins, eval_set.numeric_maxs, config, device)
        model_df.insert(0, "seed", seed)
        model_df.insert(1, "model", model_name)
        results_rows.append(model_df)

        full_probs = predict_proba(model, eval_set.x_test, batch_size=config.batch_size, device=device)
        full_clean_rows.append({
            "seed": seed,
            "model": model_name,
            "subset": "full_test",
            **classification_metrics(eval_set.y_test, full_probs),
        })

        for attack, epsilon in config.validity_attack_settings:
            attacked_inputs, attacked_probs = generate_adversarial_examples(
                model, eval_set.x_eval, eval_set.y_eval, attack, epsilon, config,
                eval_set.attack_mask, eval_set.numeric_mins, eval_set.numeric_maxs,
                device, config.batch_size, return_inputs=True,
            )
            validity_rows.append({
                "seed": seed,
                "model": model_name,
                "attack": attack,
                "epsilon": epsilon,
                **classification_metrics(eval_set.y_eval, attacked_probs, clean_pred=clean_pred),
                **compute_attack_validity_metrics(
                    x_clean=eval_set.x_eval,
                    x_adv=attacked_inputs,
                    numeric_mask=eval_set.attack_mask,
                    numeric_mins=eval_set.numeric_mins,
                    numeric_maxs=eval_set.numeric_maxs,
                ),
            })

        if include_categories and eval_set.attack_categories is not None and eval_set.top_categories:
            category_df = evaluate_category_recall(
                model_name, model, eval_set.x_eval, eval_set.y_eval,
                eval_set.attack_categories, list(eval_set.top_categories),
                config.category_attack, config.category_epsilon,
                eval_set.attack_mask, eval_set.numeric_mins, eval_set.numeric_maxs,
                config, device,
            )
            category_df.insert(0, "seed", seed)
            category_rows.append(category_df)

        for attack, epsilon in full_test_attack_settings:
            full_clean_probs = predict_proba(model, full_x, batch_size=config.batch_size, device=device)
            full_clean_pred = (full_clean_probs >= 0.5).astype(np.int32)
            _, attacked_probs = generate_adversarial_examples(
                model, full_x, full_y, attack, epsilon, config, eval_set.attack_mask,
                eval_set.numeric_mins, eval_set.numeric_maxs, device, config.batch_size,
                return_inputs=False,
            )
            full_attack_rows.append({
                "seed": seed,
                "model": model_name,
                "subset": full_subset,
                "attack": attack,
                "epsilon": epsilon,
                "test_rows": len(full_y),
                **classification_metrics(full_y, attacked_probs, clean_pred=full_clean_pred),
            })

    empty_full_attack = pd.DataFrame(columns=[
        "seed", "model", "subset", "attack", "epsilon", "test_rows",
        "accuracy", "precision", "recall", "f1", "auc", "attack_success_rate"])
    return EvaluationResults(
        metrics=pd.concat(results_rows, ignore_index=True),
        efficiency=pd.DataFrame(efficiency_rows),
        validity=pd.DataFrame(validity_rows),
        categories=(pd.concat(category_rows, ignore_index=True) if category_rows
                    else pd.DataFrame()),
        full_test_clean=pd.DataFrame(full_clean_rows),
        full_test_attack=(pd.DataFrame(full_attack_rows) if full_attack_rows else empty_full_attack),
    )
