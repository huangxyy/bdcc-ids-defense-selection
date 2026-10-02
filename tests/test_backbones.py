"""FT-Transformer capacity is configurable for large-GPU ablations."""
from __future__ import annotations

from ids_defense_selection import (
    FT_TRANSFORMER_KWARGS,
    FTTransformerBackbone,
    ft_transformer_kwargs,
)
from ids_defense_selection.config import ExperimentConfig


def _config(**overrides: object) -> ExperimentConfig:
    return ExperimentConfig(train_path="train.csv", test_path="test.csv", **overrides)


def test_default_capacity_matches_the_paper_protocol() -> None:
    assert ft_transformer_kwargs() == FT_TRANSFORMER_KWARGS
    assert ft_transformer_kwargs(_config()) == FT_TRANSFORMER_KWARGS


def test_config_drives_the_ft_transformer_shape() -> None:
    config = _config(ft_d_token=64, ft_n_heads=4, ft_n_layers=3, ft_d_ffn=128)
    kwargs = ft_transformer_kwargs(config)
    assert kwargs == {"d_token": 64, "n_heads": 4, "n_layers": 3, "d_ffn": 128}

    model = FTTransformerBackbone(10, **kwargs)
    assert model.feature_weight.shape == (10, 64)
    assert len(model.transformer.layers) == 3
