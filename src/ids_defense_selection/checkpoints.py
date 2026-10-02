"""Train once, evaluate many times: checkpoint bundles for trained defenses.

A bundle stores the state dicts of the shared baseline and of every trained
defense, the feature masks (the constrained defenses need them at evaluation
time), the training-cost table and the flat :class:`ExperimentConfig` that
produced the models.  The models have tens of thousands of parameters, so a
bundle is a few hundred KB; evaluation variants (larger epsilon, more PGD steps,
adaptive attacks, phi4, ...) can then run without retraining.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .backbones import CNN1DBackbone, FTTransformerBackbone, MLPBackbone
from .defenses import TrainedDefenses

BUNDLE_NAME = "trained_defenses.pt"
METADATA_NAME = "metadata.json"
FORMAT_VERSION = 1
BACKBONE_NAMES = ("mlp", "cnn1d", "ft_transformer")


def _flat_config(config) -> dict:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def _model_factory(backbone: str, input_dim: int, config: dict):
    """Return a callable that builds a fresh model with the trained shape."""
    if backbone == "mlp":
        hidden = tuple(config["hidden_dims"])
        return lambda: MLPBackbone(input_dim, hidden, config["dropout"])
    if backbone == "cnn1d":
        return lambda: CNN1DBackbone(input_dim, config["dropout"])
    if backbone == "ft_transformer":
        return lambda: FTTransformerBackbone(
            input_dim,
            dropout=config["dropout"],
            d_token=config["ft_d_token"],
            n_heads=config["ft_n_heads"],
            n_layers=config["ft_n_layers"],
            d_ffn=config["ft_d_ffn"],
        )
    raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONE_NAMES}")


def save_checkpoint(output_dir, trained: TrainedDefenses, config, *, seed: int,
                    backbone: str, input_dim: int,
                    dataset_split: dict | None = None) -> Path:
    """Save every trained defense of one seed; returns the checkpoint directory."""
    if backbone not in BACKBONE_NAMES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONE_NAMES}")
    directory = Path(output_dir) / "checkpoints" / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format_version": FORMAT_VERSION,
        "backbone": backbone,
        "seed": int(seed),
        "input_dim": int(input_dim),
        "config": _flat_config(config),
        "dataset_split": dataset_split or {},
        "state_dicts": {
            name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for name, model in trained.models.items()
        },
        "baseline_state_dict": {
            key: value.detach().cpu()
            for key, value in trained.baseline_model.state_dict().items()
        },
        "selected_mask": np.asarray(trained.selected_mask),
        "class_aware_mask": np.asarray(trained.class_aware_mask),
        "train_cost": trained.train_cost,
        "perturb_ratio": trained.perturb_ratio,
        "perturb_features": trained.perturb_features,
        "sensitivity_table": trained.sensitivity_table.to_dict(orient="records"),
    }
    torch.save(bundle, directory / BUNDLE_NAME)

    metadata = {
        "format_version": FORMAT_VERSION,
        "backbone": backbone,
        "seed": int(seed),
        "input_dim": int(input_dim),
        "defenses": sorted(bundle["state_dicts"]),
        "dataset_split": bundle["dataset_split"],
        "train_path": bundle["config"].get("train_path"),
        "test_path": bundle["config"].get("test_path"),
        "ft_capacity": {
            key: bundle["config"].get(key)
            for key in ("ft_d_token", "ft_n_heads", "ft_n_layers", "ft_d_ffn")
        },
    }
    (directory / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return directory


def load_checkpoint(path) -> dict:
    """Load a bundle from a seed directory or from the ``.pt`` file itself."""
    bundle_path = Path(path)
    if bundle_path.is_dir():
        bundle_path = bundle_path / BUNDLE_NAME
    if not bundle_path.is_file():
        raise FileNotFoundError(f"no checkpoint bundle at {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {bundle.get('format_version')} != {FORMAT_VERSION}; "
            "retrain with the current code")
    return bundle


def rebuild_trained_defenses(bundle: dict, device) -> TrainedDefenses:
    """Recreate the :class:`TrainedDefenses` object from a loaded bundle."""
    torch_device = torch.device(device)
    factory = _model_factory(bundle["backbone"], int(bundle["input_dim"]), bundle["config"])
    models = {}
    for name, state_dict in bundle["state_dicts"].items():
        model = factory()
        model.load_state_dict(state_dict)
        model.to(torch_device).eval()
        models[name] = model

    baseline = factory()
    baseline.load_state_dict(bundle["baseline_state_dict"])
    baseline.to(torch_device).eval()

    return TrainedDefenses(
        baseline_model=baseline,
        models=models,
        train_cost=bundle["train_cost"],
        perturb_ratio=bundle["perturb_ratio"],
        perturb_features=bundle["perturb_features"],
        selected_mask=np.asarray(bundle["selected_mask"]),
        class_aware_mask=np.asarray(bundle["class_aware_mask"]),
        sensitivity_table=pd.DataFrame(bundle["sensitivity_table"]),
    )
