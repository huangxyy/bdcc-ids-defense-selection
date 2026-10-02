"""Configuration: grouped sub-configs, validation and the generated CLI.

The configuration is organised in two layers:

* :class:`ExperimentConfig` is the **flat CLI schema**: every field is one flag,
  one line in ``--print-config`` and one entry of ``run_summary.json``.  Code
  keeps using ``config.batch_size``-style access, so nothing else had to change.
* The same fields are exposed as **seven domain groups** -- ``paths``,
  ``training``, ``attack``, ``evaluation``, ``methods``, ``sensitivity`` and
  ``runtime`` -- as immutable dataclasses (``config.training.batch_size``).
  Each group owns the validation rules for its own fields, so a rule lives next
  to the fields it protects, and ``config.grouped()`` /
  ``--print-config`` / ``run_summary.json`` show the experiment settings in the
  same structured way.

The CLI is generated from the dataclass metadata (help text, flag aliases and
the group each field belongs to), which keeps the schema, the help and the
grouped views from drifting apart.  This module imports only the standard
library plus the stdlib-only :mod:`ids_defense_selection.paths` module, which
keeps ``--dry-run`` and other tooling usable without torch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Allow `python src/ids_defense_selection/config.py`: a module executed by file
# path has no parent package, so it cannot resolve the relative imports below.
# Give it the package it belongs to before those imports run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ids_defense_selection"

from .paths import DEFAULT_DATA_DIR  # noqa: E402 - needs the guard above

# --------------------------------------------------------------------------- #
# Shared defaults and vocabularies
# --------------------------------------------------------------------------- #
#: Random seeds used throughout the study.
DEFAULT_SEEDS: tuple[int, ...] = (7, 13, 21, 42, 100)
#: Evaluation perturbation budgets.
DEFAULT_EPSILON_LIST: tuple[float, ...] = (0.02, 0.05, 0.10)
#: Seed of the fixed stratified evaluation subset.
DEFAULT_EVAL_SUBSET_SEED: int = 2026
#: Attack applied to the full test partition after training.
DEFAULT_FULL_TEST_ATTACK_SETTINGS: tuple[tuple[str, float], ...] = (("pgd", 0.10),)

#: Attack names accepted by the attack suite.
ATTACK_NAMES: tuple[str, ...] = ("clean", "fgsm", "pgd", "cw", "apgd")
#: Optional defense methods that can be enabled through ``--extra-methods``.
OPTIONAL_DEFENSE_METHODS: tuple[str, ...] = ("progressive", "sa_trades", "dst_sa_trades")
#: Non-neural reference classifiers.
REFERENCE_MODEL_NAMES: tuple[str, ...] = ("log_reg", "hist_gbdt")
#: Supported training protocols.
TRAINING_BUDGET_MODES: tuple[str, ...] = ("legacy", "matched_continuation")

AttackSetting = tuple[str, float]

#: Order in which the groups are printed and stored.
GROUP_ORDER: tuple[str, ...] = (
    "paths",
    "training",
    "attack",
    "evaluation",
    "methods",
    "sensitivity",
    "runtime",
)


def option(default: Any, help_text: str, *, group: str, aliases: tuple[str, ...] = ()) -> Any:
    """Declare a dataclass field carrying its CLI help, aliases and group."""
    metadata: dict[str, Any] = {"help": help_text, "group": group}
    if aliases:
        metadata["aliases"] = aliases
    return field(default=default, metadata=metadata)


def required_field(help_text: str, *, group: str) -> Any:
    """Declare a field without a default (must be provided by the caller)."""
    return field(metadata={"help": help_text, "group": group})


# =============================================================================
# Domain groups
#
# Each group is an immutable snapshot of the corresponding fields with its own
# ``problems()`` validation.  The flat ExperimentConfig below builds them in
# __post_init__, so ``config.training.batch_size`` and friends are always
# available and always consistent with the flat values.
# =============================================================================


@dataclass(frozen=True)
class PathsConfig:
    """Dataset locations and the output directory."""

    train_path: str = ""
    test_path: str = ""
    output_dir: str = "outputs/default"

    def problems(self) -> list[str]:
        issues = []
        if not self.train_path:
            issues.append("train_path must be a non-empty path")
        if not self.test_path:
            issues.append("test_path must be a non-empty path")
        if not self.output_dir:
            issues.append("output_dir must be a non-empty path")
        return issues


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation protocol and model shape."""

    training_budget_mode: str = "legacy"
    batch_size: int = 1024
    baseline_epochs: int = 10
    adv_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.15
    ft_d_token: int = 32
    ft_n_heads: int = 2
    ft_n_layers: int = 2
    ft_d_ffn: int = 64

    def problems(self) -> list[str]:
        issues = []
        if self.training_budget_mode not in TRAINING_BUDGET_MODES:
            issues.append(
                f"training_budget_mode must be one of {TRAINING_BUDGET_MODES}, got {self.training_budget_mode!r}"
            )
        if self.batch_size < 1:
            issues.append("batch_size must be >= 1")
        if self.baseline_epochs < 0:
            issues.append("baseline_epochs must be >= 0")
        if self.adv_epochs < 0:
            issues.append("adv_epochs must be >= 0")
        if self.learning_rate <= 0:
            issues.append("learning_rate must be > 0")
        if self.weight_decay < 0:
            issues.append("weight_decay must be >= 0")
        if len(self.hidden_dims) != 3 or any(h < 1 for h in self.hidden_dims):
            issues.append(f"hidden_dims must be three positive integers, got {tuple(self.hidden_dims)!r}")
        if not 0.0 <= self.dropout < 1.0:
            issues.append("dropout must be in [0, 1)")
        if self.ft_d_token < 1:
            issues.append("ft_d_token must be >= 1")
        if self.ft_n_heads < 1:
            issues.append("ft_n_heads must be >= 1")
        if self.ft_n_layers < 1:
            issues.append("ft_n_layers must be >= 1")
        if self.ft_d_ffn < 1:
            issues.append("ft_d_ffn must be >= 1")
        if self.ft_n_heads >= 1 and self.ft_d_token >= 1 and self.ft_d_token % self.ft_n_heads:
            issues.append(
                f"ft_d_token ({self.ft_d_token}) must be divisible by ft_n_heads ({self.ft_n_heads})"
            )
        return issues


@dataclass(frozen=True)
class AttackConfig:
    """Adversarial training budget and the attack optimisers."""

    adv_epsilon: float = 0.06
    adv_alpha: float = 0.015
    adv_steps: int = 20
    cw_steps: int = 30
    cw_lr: float = 0.01
    cw_c: float = 1.0
    apgd_steps: int = 50
    apgd_rho: float = 0.75

    def problems(self) -> list[str]:
        issues = []
        if self.adv_epsilon <= 0:
            issues.append("adv_epsilon must be > 0")
        if self.adv_alpha <= 0:
            issues.append("adv_alpha must be > 0")
        if self.adv_steps < 1:
            issues.append("adv_steps must be >= 1")
        if self.cw_steps < 1:
            issues.append("cw_steps must be >= 1")
        if self.cw_lr <= 0:
            issues.append("cw_lr must be > 0")
        if self.cw_c < 0:
            issues.append("cw_c must be >= 0")
        if self.apgd_steps < 1:
            issues.append("apgd_steps must be >= 1")
        if not 0.0 <= self.apgd_rho < 1.0:
            issues.append("apgd_rho must be in [0, 1)")
        return issues


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation subsets, budgets and the auxiliary attack matrices."""

    eval_attack_rows: int = 20000
    eval_subset_seed: int = DEFAULT_EVAL_SUBSET_SEED
    epsilon_list: tuple[float, ...] = DEFAULT_EPSILON_LIST
    eval_pgd_steps: int = 20
    eval_pgd_alpha_ratio: float = 0.05
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    top_attack_categories: int = 6
    transfer_attack_settings: tuple[AttackSetting, ...] = (("fgsm", 0.05), ("pgd", 0.1))
    validity_attack_settings: tuple[AttackSetting, ...] = (("fgsm", 0.05), ("pgd", 0.1))
    full_test_attack_settings: tuple[AttackSetting, ...] = DEFAULT_FULL_TEST_ATTACK_SETTINGS
    full_test_attack_rows: int = 0
    category_attack: str = "pgd"
    category_epsilon: float = 0.1
    ratio_attack: str = "pgd"
    ratio_attack_epsilon: float = 0.1
    # Adaptive attack suite (restarts + gradient-free cross-check + complement
    # attack); off by default because it multiplies evaluation cost.
    adaptive_eval: bool = False
    adaptive_epsilon: float = 0.10
    adaptive_steps: int = 40
    adaptive_restarts: int = 5

    def problems(self) -> list[str]:
        issues = []
        if self.eval_attack_rows < 1:
            issues.append("eval_attack_rows must be >= 1")
        if not self.epsilon_list or any(e <= 0 for e in self.epsilon_list):
            issues.append("epsilon_list must contain at least one positive epsilon")
        if self.eval_pgd_steps < 1:
            issues.append("eval_pgd_steps must be >= 1")
        if self.eval_pgd_alpha_ratio <= 0:
            issues.append("eval_pgd_alpha_ratio must be > 0")
        if not self.seeds:
            issues.append("seeds must contain at least one seed")
        if self.top_attack_categories < 1:
            issues.append("top_attack_categories must be >= 1")
        if self.full_test_attack_rows < 0:
            issues.append("full_test_attack_rows must be >= 0")
        if self.category_attack not in ATTACK_NAMES:
            issues.append(f"category_attack must be one of {ATTACK_NAMES}, got {self.category_attack!r}")
        if self.ratio_attack not in ATTACK_NAMES:
            issues.append(f"ratio_attack must be one of {ATTACK_NAMES}, got {self.ratio_attack!r}")
        if self.category_epsilon <= 0:
            issues.append("category_epsilon must be > 0")
        if self.ratio_attack_epsilon <= 0:
            issues.append("ratio_attack_epsilon must be > 0")
        if self.adaptive_epsilon <= 0:
            issues.append("adaptive_epsilon must be > 0")
        if self.adaptive_steps < 1:
            issues.append("adaptive_steps must be >= 1")
        if self.adaptive_restarts < 1:
            issues.append("adaptive_restarts must be >= 1")
        for name in ("transfer_attack_settings", "validity_attack_settings", "full_test_attack_settings"):
            for attack, epsilon in getattr(self, name):
                if attack not in ATTACK_NAMES:
                    issues.append(f"{name}: unknown attack {attack!r}; choose from {ATTACK_NAMES}")
                if epsilon <= 0:
                    issues.append(f"{name}: epsilon must be > 0, got {epsilon!r}")
        return issues


@dataclass(frozen=True)
class MethodsConfig:
    """Defense-specific coefficients and the optional extra methods."""

    trades_beta: float = 6.0
    free_at_replay: int = 4
    class_aware_minority_weight: float = 3.0
    extra_methods: tuple[str, ...] = ()
    progressive_ratios: tuple[float, ...] = (0.20, 0.30, 0.45, 0.60)
    sa_trades_gamma: float = 1.0
    dst_update_interval: int = 2
    dst_ema_alpha: float = 0.7
    reference_models: tuple[str, ...] = REFERENCE_MODEL_NAMES

    def problems(self) -> list[str]:
        issues = []
        if self.trades_beta <= 0:
            issues.append("trades_beta must be > 0")
        if self.free_at_replay < 1:
            issues.append("free_at_replay must be >= 1")
        if self.class_aware_minority_weight <= 0:
            issues.append("class_aware_minority_weight must be > 0")
        if not set(self.extra_methods) <= set(OPTIONAL_DEFENSE_METHODS):
            issues.append(
                f"extra_methods must be a subset of {OPTIONAL_DEFENSE_METHODS}, got {tuple(self.extra_methods)!r}"
            )
        if not self.progressive_ratios or any(not 0.0 < r <= 1.0 for r in self.progressive_ratios):
            issues.append("progressive_ratios must contain ratios in (0, 1]")
        if self.sa_trades_gamma <= 0:
            issues.append("sa_trades_gamma must be > 0")
        if self.dst_update_interval < 1:
            issues.append("dst_update_interval must be >= 1")
        if not 0.0 <= self.dst_ema_alpha <= 1.0:
            issues.append("dst_ema_alpha must be in [0, 1]")
        if not set(self.reference_models) <= set(REFERENCE_MODEL_NAMES):
            issues.append(
                f"reference_models must be a subset of {REFERENCE_MODEL_NAMES}, got {tuple(self.reference_models)!r}"
            )
        return issues


@dataclass(frozen=True)
class SensitivityConfig:
    """Feature-sensitivity mask estimation."""

    sensitivity_top_ratio: float = 0.3
    sensitivity_ratio_list: tuple[float, ...] = (0.2, 0.3, 0.4)
    sensitivity_batches: int = 16

    def problems(self) -> list[str]:
        issues = []
        if not 0.0 < self.sensitivity_top_ratio <= 1.0:
            issues.append("sensitivity_top_ratio must be in (0, 1]")
        if not self.sensitivity_ratio_list or any(not 0.0 < r <= 1.0 for r in self.sensitivity_ratio_list):
            issues.append("sensitivity_ratio_list must contain ratios in (0, 1]")
        if self.sensitivity_batches < 1:
            issues.append("sensitivity_batches must be >= 1")
        return issues


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution environment."""

    device: str = "auto"

    def problems(self) -> list[str]:
        if not self.device:
            return ["device must be a non-empty string"]
        if self.device not in ("auto", "cpu", "mps") and not re.fullmatch(r"cuda(:\d+)?", self.device):
            return [f"device must be auto, cpu, cuda, cuda:N or mps, got {self.device!r}"]
        return []


#: Group name -> dataclass holding that group's fields and validation.
GROUP_CLASSES: dict[str, type] = {
    "paths": PathsConfig,
    "training": TrainingConfig,
    "attack": AttackConfig,
    "evaluation": EvaluationConfig,
    "methods": MethodsConfig,
    "sensitivity": SensitivityConfig,
    "runtime": RuntimeConfig,
}


# =============================================================================
# The flat CLI schema
#
# Field order defines the flag order in --help and the order inside each group.
# The ``group=`` metadata is the only place that maps a field to a group, and
# tests assert that this mapping matches GROUP_CLASSES exactly.
# =============================================================================


@dataclass
class ExperimentConfig:
    """All hyperparameters of one backbone experiment in a single place.

    Flat access (``config.batch_size``) is the CLI schema and stays stable;
    grouped access (``config.training.batch_size``) is the maintenance view.
    Values are normalised and validated on construction; a failure lists every
    problem found across all groups.
    """

    # --- paths -------------------------------------------------------------
    train_path: str = required_field("UNSW-NB15 training partition (175,341 records)", group="paths")
    test_path: str = required_field("UNSW-NB15 testing partition (82,332 records)", group="paths")
    output_dir: str = option("outputs/default", "directory for all result files", group="paths")

    # --- training protocol --------------------------------------------------
    training_budget_mode: str = option(
        "legacy",
        "legacy = each defense trained independently; matched_continuation = "
        "shared clean pre-training plus an equal continuation budget",
        group="training",
    )
    batch_size: int = option(1024, "mini-batch size", group="training")
    baseline_epochs: int = option(10, "clean pre-training epochs", group="training")
    adv_epochs: int = option(8, "adversarial (continuation) epochs per defense", group="training")
    learning_rate: float = option(1e-3, "Adam learning rate", group="training")
    weight_decay: float = option(1e-5, "Adam weight decay", group="training")
    hidden_dims: tuple[int, ...] = option((128, 64, 32), "MLP hidden layer widths", group="training")
    dropout: float = option(0.15, "dropout probability", group="training")
    ft_d_token: int = option(32, "FT-Transformer token width d_token", group="training")
    ft_n_heads: int = option(2, "FT-Transformer attention heads (must divide ft_d_token)", group="training")
    ft_n_layers: int = option(2, "FT-Transformer encoder layers", group="training")
    ft_d_ffn: int = option(64, "FT-Transformer feed-forward width", group="training")

    # --- adversarial training budget ----------------------------------------
    adv_epsilon: float = option(
        0.06, "L-inf perturbation budget used DURING TRAINING (differs from the evaluation budgets)", group="attack"
    )
    adv_alpha: float = option(0.015, "PGD step size used during training", group="attack")
    adv_steps: int = option(20, "PGD steps used during training", group="attack", aliases=("--train-adv-steps",))

    # --- attack optimisers ---------------------------------------------------
    cw_steps: int = option(30, "C&W L2 optimisation steps", group="attack")
    cw_lr: float = option(0.01, "C&W L2 learning rate", group="attack")
    cw_c: float = option(1.0, "C&W L2 confidence constant", group="attack")
    apgd_steps: int = option(50, "APGD-CE steps", group="attack")
    apgd_rho: float = option(0.75, "APGD-CE step-size schedule parameter", group="attack")

    # --- evaluation protocol ------------------------------------------------
    eval_attack_rows: int = option(
        20000,
        "number of test rows used for the attack evaluation (stratified, shared by all defenses)",
        group="evaluation",
    )
    eval_subset_seed: int = option(
        DEFAULT_EVAL_SUBSET_SEED,
        "seed of the fixed stratified evaluation subset (shared by all defenses)",
        group="evaluation",
    )
    epsilon_list: tuple[float, ...] = option(
        DEFAULT_EPSILON_LIST, "evaluation perturbation budgets", group="evaluation"
    )
    eval_pgd_steps: int = option(
        20, "PGD steps used at evaluation time (independent of the training steps)", group="evaluation"
    )
    eval_pgd_alpha_ratio: float = option(
        0.05,
        "PGD evaluation step size as a fraction of epsilon "
        "(0.05 = eps/20, the value behind every reported result; 0.10 = eps/10)",
        group="evaluation",
    )
    seeds: tuple[int, ...] = option(DEFAULT_SEEDS, "random seeds; results are aggregated over them", group="evaluation")
    top_attack_categories: int = option(
        6, "how many attack categories enter the worst-class objective", group="evaluation"
    )

    # --- auxiliary attack matrices -------------------------------------------
    transfer_attack_settings: tuple[AttackSetting, ...] = option(
        (("fgsm", 0.05), ("pgd", 0.1)), "attack:epsilon pairs used for the transferability matrix", group="evaluation"
    )
    validity_attack_settings: tuple[AttackSetting, ...] = option(
        (("fgsm", 0.05), ("pgd", 0.1)), "attack:epsilon pairs used for the attack-validity check", group="evaluation"
    )
    full_test_attack_settings: tuple[AttackSetting, ...] = option(
        DEFAULT_FULL_TEST_ATTACK_SETTINGS,
        "attack:epsilon pairs applied to the full test partition after training",
        group="evaluation",
    )
    full_test_attack_rows: int = option(
        0, "rows of the test partition attacked at the end (0 = the whole partition)", group="evaluation"
    )

    # --- per-category and ratio analysis --------------------------------------
    category_attack: str = option(
        "pgd", "attack used for the per-category (worst-class) evaluation", group="evaluation"
    )
    category_epsilon: float = option(0.1, "epsilon for the per-category evaluation", group="evaluation")
    ratio_attack: str = option("pgd", "attack used for the perturbation-ratio analysis", group="evaluation")
    ratio_attack_epsilon: float = option(0.1, "epsilon for the perturbation-ratio analysis", group="evaluation")

    # --- adaptive attack evaluation (off by default: it is expensive) ---------
    adaptive_eval: bool = option(
        False, "also run the adaptive attack suite (restarts, gradient-free NES, complement attack)", group="evaluation"
    )
    adaptive_epsilon: float = option(0.10, "epsilon used by the adaptive attack suite", group="evaluation")
    adaptive_steps: int = option(40, "steps per restart in the adaptive attack suite", group="evaluation")
    adaptive_restarts: int = option(5, "random restarts in the adaptive attack suite", group="evaluation")

    # --- defense-specific coefficients ----------------------------------------
    trades_beta: float = option(6.0, "TRADES trade-off coefficient", group="methods")
    free_at_replay: int = option(4, "Free AT replay multiplier m", group="methods")
    class_aware_minority_weight: float = option(
        3.0, "extra loss weight for minority attack categories", group="methods", aliases=("--class-aware-weight",)
    )
    reference_models: tuple[str, ...] = option(
        REFERENCE_MODEL_NAMES, "non-neural reference classifiers (log_reg, hist_gbdt)", group="methods"
    )

    # --- optional extra defense methods (off by default) ----------------------
    extra_methods: tuple[str, ...] = option(
        (), "optional additional defenses to train as well: progressive, sa_trades, dst_sa_trades", group="methods"
    )
    progressive_ratios: tuple[float, ...] = option(
        (0.20, 0.30, 0.45, 0.60), "mask ratios of the progressive class-aware schedule", group="methods"
    )
    sa_trades_gamma: float = option(1.0, "sensitivity-weighting exponent of SA-TRADES", group="methods")
    dst_update_interval: int = option(2, "epochs between sensitivity re-estimates in DST-SA-TRADES", group="methods")
    dst_ema_alpha: float = option(0.7, "EMA smoothing factor for the DST sensitivity estimate", group="methods")

    # --- sensitivity analysis and masks --------------------------------------
    sensitivity_top_ratio: float = option(0.3, "fraction of features kept by the sensitivity mask", group="sensitivity")
    sensitivity_ratio_list: tuple[float, ...] = option(
        (0.2, 0.3, 0.4), "mask ratios swept during the sensitivity analysis", group="sensitivity"
    )
    sensitivity_batches: int = option(16, "mini-batches used to estimate feature sensitivity", group="sensitivity")

    # --- runtime ----------------------------------------------------------------
    device: str = option(
        "auto", "torch device: auto (CUDA if available, else MPS/CPU), cpu, cuda, cuda:N or mps", group="runtime"
    )

    # ------------------------------------------------------------------ #
    # normalisation, grouped views and validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self._normalise()
        self._build_groups()

    def _normalise(self) -> None:
        """Accept lists where tuples are declared and trim string fields."""
        for name, f in type(self).__dataclass_fields__.items():
            if not isinstance(f.default, tuple):
                continue
            value = getattr(self, name)
            if isinstance(value, list):
                value = tuple(value)
            if value and isinstance(value[0], list):
                value = tuple(tuple(item) for item in value)
            setattr(self, name, value)
        if isinstance(self.device, str):
            self.device = self.device.strip().lower()
        if isinstance(self.training_budget_mode, str):
            self.training_budget_mode = self.training_budget_mode.strip().lower()

    def _build_groups(self) -> None:
        """Materialise the grouped views and collect every validation problem."""
        problems: list[str] = []
        for name, group_class in GROUP_CLASSES.items():
            group = group_class(**{f.name: getattr(self, f.name) for f in fields(group_class)})
            setattr(self, name, group)
            problems.extend(f"[{name}] {issue}" for issue in group.problems())
        if problems:
            raise ValueError("invalid ExperimentConfig:\n  - " + "\n  - ".join(problems))

    # ------------------------------------------------------------------ #
    # public helpers for inspection
    # ------------------------------------------------------------------ #
    def group(self, name: str) -> Any:
        """Return one of the grouped views by name (see :data:`GROUP_ORDER`)."""
        if name not in GROUP_CLASSES:
            raise KeyError(f"unknown config group {name!r}; choose from {GROUP_ORDER}")
        return getattr(self, name)

    def grouped(self) -> dict[str, dict[str, Any]]:
        """Nested ``group -> {field: value}`` mapping for printing and storing."""
        return {name: asdict(self.group(name)) for name in GROUP_ORDER}

    def problems(self) -> list[str]:
        """All validation problems, prefixed with their group. Empty when valid."""
        issues: list[str] = []
        for name in GROUP_ORDER:
            issues.extend(f"[{name}] {issue}" for issue in self.group(name).problems())
        return issues


def field_help(name: str) -> str:
    """Help text declared in the dataclass metadata for *name*."""
    return str(ExperimentConfig.__dataclass_fields__[name].metadata.get("help", ""))


def field_aliases(name: str) -> tuple[str, ...]:
    """Extra flag spellings declared in the dataclass metadata for *name*."""
    return tuple(ExperimentConfig.__dataclass_fields__[name].metadata.get("aliases", ()))


def field_group(name: str) -> str:
    """Group declared in the dataclass metadata for *name*."""
    return str(ExperimentConfig.__dataclass_fields__[name].metadata.get("group", ""))


# =============================================================================
# Command-line interface
#
# Every field of ExperimentConfig has a matching command-line flag, generated
# from the dataclass itself, so the CLI can never drift out of sync:
#
#   scalar fields       --adv-epsilon 0.05
#   tuple[int]          --hidden-dims 256,128,64
#   tuple[float]        --epsilon-list 0.02,0.05,0.10
#   tuple[str]          --reference-models log_reg,hist_gbdt
#   tuple[(str,float)]  --transfer-attack-settings fgsm:0.05,pgd:0.10
#
# Run with --help to see the full list, or --print-config to dump the effective
# configuration as grouped JSON.
# =============================================================================


def parse_bool(raw: str | bool) -> bool:
    """Parse the boolean spellings accepted on the command line."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def tuple_value_kind(value: tuple) -> str:
    """Classify a tuple default so the CLI knows how to parse it."""
    if not value:
        return "str"
    first = value[0]
    if isinstance(first, tuple):
        return "pair"
    if isinstance(first, bool):
        return "str"
    if isinstance(first, int):
        return "int"
    if isinstance(first, float):
        return "float"
    return "str"


def format_tuple_default(value: object) -> str:
    """Render a tuple default for --help output."""
    if isinstance(value, tuple):
        if value and isinstance(value[0], tuple):
            return ",".join(f"{k}:{v}" for k, v in value)
        return ",".join(str(v) for v in value)
    return str(value)


def parse_tuple_value(raw: str, kind: str) -> tuple:
    """Turn a comma-separated string into the tuple type the config expects.

    int / float / str :  "7,13,21"            -> (7, 13, 21)
    pair              :  "fgsm:0.05,pgd:0.1"  -> (("fgsm", 0.05), ("pgd", 0.1))

    Raises ``argparse.ArgumentTypeError`` so argparse prints a clean message
    instead of a traceback.
    """
    items = [s.strip() for s in str(raw).split(",") if s.strip()]
    if kind == "pair":
        pairs: list[AttackSetting] = []
        for item in items:
            key, sep, value = item.partition(":")
            if not sep or not key.strip() or not value.strip():
                raise argparse.ArgumentTypeError(f"expected key:value pairs such as 'fgsm:0.05,pgd:0.10', got {item!r}")
            try:
                pairs.append((key.strip(), float(value)))
            except ValueError:
                raise argparse.ArgumentTypeError(f"{value!r} is not a number (in {item!r})") from None
        return tuple(pairs)

    converters = {"int": int, "float": float, "str": str}
    if kind not in converters:
        raise ValueError(f"unknown tuple kind {kind!r}")
    try:
        return tuple(converters[kind](item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse {raw!r} as comma-separated {kind} values ({exc})") from None


def add_config_arguments(
    parser: argparse.ArgumentParser, skip: tuple[str, ...] = (), defaults: dict[str, object] | None = None
) -> None:
    """Attach one flag per ExperimentConfig field, generated from the dataclass."""
    provided = defaults or {}
    for name, f in ExperimentConfig.__dataclass_fields__.items():
        if name in skip:
            continue
        flag = "--" + name.replace("_", "-")
        aliases = list(field_aliases(name))
        declared = f.default
        default = provided.get(name, declared)
        if isinstance(declared, tuple) and isinstance(default, list):
            default = tuple(default)
        help_text = field_help(name)

        if isinstance(declared, tuple):
            kind = tuple_value_kind(declared)
            example = format_tuple_default(default)
            parser.add_argument(
                flag,
                *aliases,
                default=default,
                metavar="LIST",
                type=lambda raw, kind=kind: parse_tuple_value(raw, kind),
                help=f"{help_text}  [comma-separated; default: {example}]",
            )
        elif isinstance(declared, bool):
            parser.add_argument(
                flag,
                *aliases,
                default=default,
                type=parse_bool,
                nargs="?",
                const=True,
                help=f"{help_text}  [default: {default}]",
            )
        elif isinstance(declared, int):
            parser.add_argument(flag, *aliases, default=default, type=int, help=f"{help_text}  [default: {default}]")
        elif isinstance(declared, float):
            parser.add_argument(flag, *aliases, default=default, type=float, help=f"{help_text}  [default: {default}]")
        else:
            parser.add_argument(flag, *aliases, default=default, type=str, help=f"{help_text}  [default: {default}]")


def config_from_args(args: argparse.Namespace, **overrides: object) -> ExperimentConfig:
    """Build a validated ExperimentConfig from parsed arguments.

    Anything the user did not pass keeps the dataclass default; tuple fields
    arrive as strings and are converted according to their declared type.  A
    validation failure is reported as a clean ``error: ...`` message instead of
    a traceback.
    """
    kwargs: dict[str, object] = {}
    for name, f in ExperimentConfig.__dataclass_fields__.items():
        raw = getattr(args, name, None)
        if raw is None:
            continue
        if isinstance(f.default, tuple) and isinstance(raw, str):
            raw = parse_tuple_value(raw, tuple_value_kind(f.default))
        kwargs[name] = raw
    kwargs.update(overrides)
    try:
        return ExperimentConfig(**kwargs)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


def build_parser(
    description: str | None = None,
    skip: tuple[str, ...] = (),
    defaults: dict[str, object] | None = None,
    require_paths: bool = True,
) -> argparse.ArgumentParser:
    """Standard parser: dataset paths plus every config field.

    defaults       per-script overrides for individual config fields, so each
                   backbone can keep its own default output directory, batch
                   size and so on while still exposing every parameter.
    require_paths  True for stand-alone scripts; False for the backbone scripts,
                   which fall back to data/train.csv and data/test.csv.
    """
    d = defaults or {}
    parser = argparse.ArgumentParser(
        description=description or "Adversarial robustness experiments for UNSW-NB15 intrusion detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every ExperimentConfig field can be set from the command line; see above.",
    )
    parser.add_argument(
        "--train-path",
        required=require_paths,
        default=None if require_paths else d.get("train_path", str(DEFAULT_DATA_DIR / "train.csv")),
        help=field_help("train_path"),
    )
    parser.add_argument(
        "--test-path",
        required=require_paths,
        default=None if require_paths else d.get("test_path", str(DEFAULT_DATA_DIR / "test.csv")),
        help=field_help("test_path"),
    )
    add_config_arguments(parser, skip=("train_path", "test_path") + tuple(skip), defaults=d)
    parser.add_argument(
        "--print-config", action="store_true", help="print the effective configuration as grouped JSON and exit"
    )
    return parser


def emit_config(config: ExperimentConfig) -> None:
    """Print the effective configuration as grouped JSON and exit (--print-config)."""
    print(json.dumps(config.grouped(), indent=2, default=str, ensure_ascii=False))
    raise SystemExit(0)


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse a bare ExperimentConfig from the standard parser."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.print_config:
        emit_config(config)
    return config


def _default_text(f: Any) -> str:
    """Render one field default the way it is typed on the command line."""
    if f.default is not MISSING:
        value = f.default
    elif f.default_factory is not MISSING:
        value = f.default_factory()
    else:
        return "<required>"
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def default_rows() -> list[tuple[str, str, str, str]]:
    """``(group, flag, default, help)`` for every field, in :data:`GROUP_ORDER`."""
    rows: list[tuple[str, str, str, str]] = []
    for group in GROUP_ORDER:
        for name, f in ExperimentConfig.__dataclass_fields__.items():
            if f.metadata.get("group") != group:
                continue
            rows.append((group, "--" + name.replace("_", "-"), _default_text(f), field_help(name)))
    return rows


def format_default_config() -> str:
    """Readable listing of every flag with its default and help text.

    Used by ``python -m ids_defense_selection.config``; for the *effective*
    values of one run use ``--print-config`` on any runner instead.
    """
    lines = ["ExperimentConfig defaults (group order, one flag per field)", ""]
    current_group = None
    for group, flag, default, help_text in default_rows():
        if group != current_group:
            current_group = group
            lines.append(f"[{group}]")
        lines.append(f"  {flag:<32} = {default}")
        if help_text:
            lines.append(f"      {help_text}")
    return "\n".join(lines)


def default_instance() -> ExperimentConfig:
    """A valid ``ExperimentConfig`` pointing at this repository's own dataset.

    ``train_path`` / ``test_path`` have no dataclass default (the runners fill
    them in), so a bare ``ExperimentConfig()`` raises; this helper is the
    documented way to get a ready-to-inspect instance.
    """
    return ExperimentConfig(
        train_path=str(DEFAULT_DATA_DIR / "train.csv"),
        test_path=str(DEFAULT_DATA_DIR / "test.csv"),
    )


if __name__ == "__main__":
    print(format_default_config())
    print()
    print("Resolved defaults for this repository (same shape as --print-config):")
    print(json.dumps(default_instance().grouped(), indent=2, default=str, ensure_ascii=False))
