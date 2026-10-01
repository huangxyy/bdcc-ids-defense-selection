"""Tests for the generated command-line interface."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import fields

import pytest

from ids_defense_selection.config import (
    DEFAULT_EVAL_SUBSET_SEED,
    GROUP_CLASSES,
    GROUP_ORDER,
    AttackConfig,
    EvaluationConfig,
    ExperimentConfig,
    MethodsConfig,
    PathsConfig,
    RuntimeConfig,
    SensitivityConfig,
    TrainingConfig,
    build_parser,
    config_from_args,
    default_instance,
    default_rows,
    emit_config,
    field_group,
    field_help,
    format_default_config,
    parse_tuple_value,
)
from ids_defense_selection.paths import BACKBONE_OUTPUT_SUBDIRS, default_output_dir, resolve_path
from ids_defense_selection.paths import PROJECT_ROOT


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


def test_adaptive_evaluation_defaults_and_validation() -> None:
    config = ExperimentConfig(train_path="train.csv", test_path="test.csv")
    assert config.adaptive_eval is False          # expensive: opt-in
    assert config.adaptive_epsilon == 0.10
    assert config.adaptive_steps == 40
    assert config.adaptive_restarts == 5

    parser = build_parser(require_paths=False)
    args = parser.parse_args(["--adaptive-eval", "--adaptive-steps", "10"])
    overridden = config_from_args(args)
    assert overridden.adaptive_eval is True
    assert overridden.adaptive_steps == 10

    with pytest.raises(ValueError, match="adaptive_steps"):
        ExperimentConfig(train_path="train.csv", test_path="test.csv", adaptive_steps=0)


def test_every_field_has_help_text_and_a_flag() -> None:
    parser = build_parser(require_paths=False)
    for name in ExperimentConfig.__dataclass_fields__:
        assert field_help(name), f"{name} has no help text in its metadata"
        assert field_group(name) in GROUP_ORDER, f"{name} has no config group"
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


def test_grouped_view_is_nested_and_covers_every_field() -> None:
    config = ExperimentConfig(train_path="train.csv", test_path="test.csv")
    grouped = config.grouped()
    assert list(grouped) == list(GROUP_ORDER)

    flat = set(ExperimentConfig.__dataclass_fields__)
    seen = [name for group in grouped.values() for name in group]
    assert sorted(seen) == sorted(flat)
    assert len(seen) == len(set(seen)), "a field must belong to exactly one group"

    assert config.training.batch_size == 1024
    assert config.attack.adv_epsilon == 0.06
    assert config.evaluation.epsilon_list == (0.02, 0.05, 0.10)
    assert config.methods.trades_beta == 6.0
    assert config.sensitivity.sensitivity_top_ratio == 0.3
    assert config.paths.output_dir == "outputs/default"
    assert config.runtime.device == "auto"


def test_field_metadata_groups_match_group_classes() -> None:
    for group_name, group_class in GROUP_CLASSES.items():
        declared = {f.name for f in fields(group_class)}
        tagged = {name for name in ExperimentConfig.__dataclass_fields__
                  if field_group(name) == group_name}
        assert declared == tagged, f"group {group_name!r} does not match its metadata"


def test_group_dataclasses_validate_standalone() -> None:
    assert PathsConfig(train_path="a", test_path="b", output_dir="c").problems() == []
    assert RuntimeConfig(device="cpu").problems() == []
    assert RuntimeConfig(device="cuda:1").problems() == []
    assert any("device must be" in issue for issue in RuntimeConfig(device="quantum").problems())
    assert TrainingConfig(batch_size=0).problems() == ["batch_size must be >= 1"]
    assert "adv_steps must be >= 1" in AttackConfig(adv_steps=0).problems()
    assert "seeds must contain at least one seed" in EvaluationConfig(seeds=()).problems()
    assert any("extra_methods" in issue for issue in MethodsConfig(extra_methods=("nope",)).problems())
    assert "sensitivity_top_ratio must be in (0, 1]" in SensitivityConfig(
        sensitivity_top_ratio=0.0).problems()


def test_problems_are_prefixed_with_their_group() -> None:
    with pytest.raises(ValueError) as excinfo:
        ExperimentConfig(train_path="", test_path="test.csv", adv_steps=0)
    text = str(excinfo.value)
    assert "[paths] train_path" in text
    assert "[attack] adv_steps" in text


def test_print_config_emits_grouped_json(capsys: pytest.CaptureFixture) -> None:
    parser = build_parser(require_paths=False)
    config = config_from_args(parser.parse_args([]))
    with pytest.raises(SystemExit):
        emit_config(config)
    payload = json.loads(capsys.readouterr().out)
    assert payload["training"]["batch_size"] == 1024
    assert payload["runtime"]["device"] == "auto"
    assert set(payload) == set(GROUP_ORDER)


def test_default_dataset_paths_are_repo_root_based() -> None:
    parser = build_parser(require_paths=False)
    args = parser.parse_args([])
    assert Path(args.train_path).is_absolute()
    assert Path(args.test_path).is_absolute()
    assert Path(args.train_path).name == "train.csv"
    assert Path(args.test_path).name == "test.csv"


def test_default_config_report_covers_every_flag() -> None:
    """`python -m ids_defense_selection.config` must list every flag exactly once."""
    rows = default_rows()
    flags = [flag for _, flag, _, _ in rows]
    expected = {"--" + name.replace("_", "-") for name in ExperimentConfig.__dataclass_fields__}
    assert set(flags) == expected
    assert len(flags) == len(set(flags))
    assert [group for group, _, _, _ in rows] == sorted(
        (group for group, _, _, _ in rows), key=GROUP_ORDER.index)

    report = format_default_config()
    assert "MISSING" not in report
    assert "--train-path" in report and "<required>" in report
    for _, flag, _, help_text in rows:
        assert flag in report
        assert help_text in report
    report = format_default_config()
    assert "MISSING" not in report
    assert "--train-path" in report and "<required>" in report
    for _, flag, _, help_text in rows:
        assert flag in report
        assert help_text in report


def test_config_module_runs_as_a_plain_script(tmp_path: Path) -> None:
    """`python code/ids_defense_selection/config.py` must not break on relative imports."""
    script = PROJECT_ROOT / "code" / "ids_defense_selection" / "config.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=tmp_path, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ExperimentConfig defaults" in completed.stdout
    assert "Resolved defaults for this repository" in completed.stdout
    assert "attempted relative import" not in completed.stderr


def test_default_instance_points_at_the_repository_dataset() -> None:
    config = default_instance()
    assert Path(config.train_path) == PROJECT_ROOT / "data" / "train.csv"
    assert Path(config.test_path) == PROJECT_ROOT / "data" / "test.csv"
    assert config.grouped()["runtime"]["device"] == "auto"
