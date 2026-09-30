"""Tests for the generated command-line interface."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from ids_defense_selection.config import (
    DEFAULT_EVAL_SUBSET_SEED,
    ExperimentConfig,
    build_parser,
    config_from_args,
    parse_tuple_value,
)
from ids_defense_selection.paths import BACKBONE_OUTPUT_SUBDIRS, default_output_dir, resolve_path


def test_parse_tuple_value_scalar_kinds() -> None:
    assert parse_tuple_value("7,13,21", "int") == (7, 13, 21)
    assert parse_tuple_value("0.02,0.05", "float") == (0.02, 0.05)
    assert parse_tuple_value("log_reg,hist_gbdt", "str") == ("log_reg", "hist_gbdt")


def test_parse_tuple_value_attack_pairs() -> None:
    assert parse_tuple_value("fgsm:0.05,pgd:0.1", "pair") == (("fgsm", 0.05), ("pgd", 0.1))
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tuple_value("fgsm", "pair")


def test_cli_exposes_every_config_field() -> None:
    parser = build_parser(require_paths=False)
    args = parser.parse_args([])
    for name in ExperimentConfig.__dataclass_fields__:
        assert hasattr(args, name), f"missing CLI flag for {name}"


def test_config_from_args_parses_tuples_and_keeps_defaults() -> None:
    parser = build_parser(require_paths=False)
    args = parser.parse_args(["--seeds", "1,2", "--epsilon-list", "0.05", "--device", "cpu"])
    config = config_from_args(args)
    assert config.seeds == (1, 2)
    assert config.epsilon_list == (0.05,)
    assert config.device == "cpu"
    # untouched fields keep the dataclass default
    assert config.eval_pgd_steps == 20
    assert config.batch_size == 1024


def test_training_and_evaluation_pgd_steps_are_independent() -> None:
    parser = build_parser(require_paths=False)
    args = parser.parse_args(["--adv-steps", "7", "--eval-pgd-steps", "20"])
    config = config_from_args(args)
    assert config.adv_steps == 7
    assert config.eval_pgd_steps == 20


def test_train_adv_steps_alias_still_works() -> None:
    parser = build_parser(require_paths=False)
    args = parser.parse_args(["--train-adv-steps", "7"])
    config = config_from_args(args)
    assert config.adv_steps == 7


def test_resolve_path_keeps_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "train.csv"
    assert resolve_path(absolute, base=tmp_path) == absolute
    assert resolve_path("data/train.csv", base=tmp_path) == tmp_path / "data/train.csv"


def test_default_output_dirs_follow_the_shared_layout(tmp_path: Path) -> None:
    assert set(BACKBONE_OUTPUT_SUBDIRS) == {"mlp", "cnn", "ft"}
    assert default_output_dir("mlp", root=tmp_path) == tmp_path / "mlp"
    assert default_output_dir("cnn", root=tmp_path) == tmp_path / "cnn1d"
    assert default_output_dir("ft", root=tmp_path) == tmp_path / "ft_transformer"


def test_evaluation_subset_and_full_test_defaults_are_configurable() -> None:
    config = ExperimentConfig(train_path="train.csv", test_path="test.csv")
    assert config.eval_subset_seed == DEFAULT_EVAL_SUBSET_SEED
    assert config.eval_subset_seed == 2026
    assert config.full_test_attack_rows == 0
    assert config.full_test_attack_settings == (("pgd", 0.10),)

    parser = build_parser(require_paths=False)
    args = parser.parse_args([
        "--eval-subset-seed", "7",
        "--full-test-attack-rows", "500",
        "--full-test-attack-settings", "fgsm:0.05",
    ])
    overridden = config_from_args(args)
    assert overridden.eval_subset_seed == 7
    assert overridden.full_test_attack_rows == 500
    assert overridden.full_test_attack_settings == (("fgsm", 0.05),)
