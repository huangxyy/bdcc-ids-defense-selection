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
    field_help,
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


def test_every_field_has_help_text_and_a_flag() -> None:
    parser = build_parser(require_paths=False)
    for name in ExperimentConfig.__dataclass_fields__:
        assert field_help(name), f"{name} has no help text in its metadata"
    # every field except the two dataset paths is exposed as a config flag
    args = parser.parse_args([])
    for name in ExperimentConfig.__dataclass_fields__:
        assert hasattr(args, name), f"missing CLI flag for {name}"


def test_values_are_normalised_on_construction() -> None:
    config = ExperimentConfig(
        train_path="train.csv",
        test_path="test.csv",
        device="  CUDA  ",
        training_budget_mode="MATCHED_CONTINUATION",
        seeds=[1, 2],
        epsilon_list=[0.05],
        transfer_attack_settings=[["fgsm", 0.05]],
    )
    assert config.device == "cuda"
    assert config.training_budget_mode == "matched_continuation"
    assert config.seeds == (1, 2)
    assert config.epsilon_list == (0.05,)
    assert config.transfer_attack_settings == (("fgsm", 0.05),)


@pytest.mark.parametrize("overrides, message", [
    ({"batch_size": 0}, "batch_size"),
    ({"dropout": 1.0}, "dropout"),
    ({"hidden_dims": (128, 64)}, "hidden_dims"),
    ({"adv_steps": 0}, "adv_steps"),
    ({"epsilon_list": ()}, "epsilon_list"),
    ({"seeds": ()}, "seeds"),
    ({"sensitivity_top_ratio": 0.0}, "sensitivity_top_ratio"),
    ({"training_budget_mode": "turbo"}, "training_budget_mode"),
    ({"category_attack": "quantum"}, "category_attack"),
    ({"extra_methods": ("unknown_method",)}, "extra_methods"),
    ({"transfer_attack_settings": (("fgsm", 0.0),)}, "transfer_attack_settings"),
])
def test_invalid_values_fail_fast(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(train_path="train.csv", test_path="test.csv", **overrides)


def test_validation_reports_every_problem_at_once() -> None:
    with pytest.raises(ValueError) as excinfo:
        ExperimentConfig(train_path="", test_path="test.csv", batch_size=0, adv_steps=0)
    text = str(excinfo.value)
    assert "train_path" in text and "batch_size" in text and "adv_steps" in text


def test_cli_tuple_typos_are_reported_as_argument_errors() -> None:
    parser = build_parser(require_paths=False)
    with pytest.raises(SystemExit):
        parser.parse_args(["--seeds", "1,abc"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--transfer-attack-settings", "fgsm"])


def test_config_from_args_reports_validation_errors_cleanly() -> None:
    parser = build_parser(require_paths=False)
    args = parser.parse_args(["--batch-size", "0"])
    with pytest.raises(SystemExit) as excinfo:
        config_from_args(args)
    assert "batch_size" in str(excinfo.value)
