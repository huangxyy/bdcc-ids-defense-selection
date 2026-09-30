"""Configuration dataclass, validation and the command-line interface around it.

Design rules:

* **One source of truth.**  Every field of :class:`ExperimentConfig` carries its
  own help text (and optional flag aliases) in dataclass metadata, and the CLI is
  generated from the dataclass.  Adding a field automatically adds a flag, its
  help text and its JSON entry in ``--print-config``/``run_summary.json``.
* **Fail fast.**  ``__post_init__`` normalises the values (lists -> tuples,
  lower-cased device/mode) and validates them, so a typo such as
  ``--adv-steps 0`` raises a readable ``ValueError`` before any training starts.
* **Dependency-free.**  This module imports only the standard library, which is
  what keeps ``--dry-run`` tooling working without torch installed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

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


def option(default: Any, help_text: str, *, aliases: tuple[str, ...] = ()) -> Any:
    """Declare a dataclass field that also carries its CLI help and aliases."""
    metadata: dict[str, Any] = {"help": help_text}
    if aliases:
        metadata["aliases"] = aliases
    return field(default=default, metadata=metadata)


@dataclass
class ExperimentConfig:
    """All hyperparameters of one backbone experiment in a single place.

    Values are normalised and validated on construction; see
    :meth:`ExperimentConfig._validate` for the accepted ranges.
    """

    # --- paths -------------------------------------------------------------
    train_path: str = field(metadata={
        "help": "UNSW-NB15 training partition (175,341 records)"})
    test_path: str = field(metadata={
        "help": "UNSW-NB15 testing partition (82,332 records)"})
    output_dir: str = option("outputs/default", "directory for all result files")

    # --- training protocol --------------------------------------------------
    training_budget_mode: str = option(
        "legacy",
        "legacy = each defense trained independently; matched_continuation = "
        "shared clean pre-training plus an equal continuation budget")
    batch_size: int = option(1024, "mini-batch size")
    baseline_epochs: int = option(10, "clean pre-training epochs")
    adv_epochs: int = option(8, "adversarial (continuation) epochs per defense")
    learning_rate: float = option(1e-3, "Adam learning rate")
    weight_decay: float = option(1e-5, "Adam weight decay")
    hidden_dims: tuple[int, ...] = option((128, 64, 32), "MLP hidden layer widths")
    dropout: float = option(0.15, "dropout probability")

    # --- evaluation protocol ------------------------------------------------
    eval_attack_rows: int = option(
        20000,
        "number of test rows used for the attack evaluation (stratified, shared by all defenses)")
    eval_subset_seed: int = option(
        DEFAULT_EVAL_SUBSET_SEED,
        "seed of the fixed stratified evaluation subset (shared by all defenses)")

    # --- adversarial training budget ----------------------------------------
    adv_epsilon: float = option(
        0.06, "L-inf perturbation budget used DURING TRAINING (differs from the evaluation budgets)")
    adv_alpha: float = option(0.015, "PGD step size used during training")
    adv_steps: int = option(
        20, "PGD steps used during training", aliases=("--train-adv-steps",))

    # --- evaluation budget and attacks --------------------------------------
    epsilon_list: tuple[float, ...] = option(
        DEFAULT_EPSILON_LIST, "evaluation perturbation budgets")
    eval_pgd_steps: int = option(
        20, "PGD steps used at evaluation time (independent of the training steps)")
    eval_pgd_alpha_ratio: float = option(
        0.05,
        "PGD evaluation step size as a fraction of epsilon "
        "(0.05 = eps/20, the value behind every reported result; 0.10 = eps/10)")
    seeds: tuple[int, ...] = option(
        DEFAULT_SEEDS, "random seeds; results are aggregated over them")

    # --- sensitivity analysis and masks --------------------------------------
    sensitivity_top_ratio: float = option(
        0.3, "fraction of features kept by the sensitivity mask")
    sensitivity_ratio_list: tuple[float, ...] = option(
        (0.2, 0.3, 0.4), "mask ratios swept during the sensitivity analysis")
    sensitivity_batches: int = option(
        16, "mini-batches used to estimate feature sensitivity")

    # --- auxiliary attack matrices -------------------------------------------
    transfer_attack_settings: tuple[AttackSetting, ...] = option(
        (("fgsm", 0.05), ("pgd", 0.1)),
        "attack:epsilon pairs used for the transferability matrix")
    validity_attack_settings: tuple[AttackSetting, ...] = option(
        (("fgsm", 0.05), ("pgd", 0.1)),
        "attack:epsilon pairs used for the attack-validity check")
    full_test_attack_settings: tuple[AttackSetting, ...] = option(
        DEFAULT_FULL_TEST_ATTACK_SETTINGS,
        "attack:epsilon pairs applied to the full test partition after training")
    full_test_attack_rows: int = option(
        0, "rows of the test partition attacked at the end (0 = the whole partition)")

    # --- per-category and ratio analysis --------------------------------------
    category_attack: str = option(
        "pgd", "attack used for the per-category (worst-class) evaluation")
    category_epsilon: float = option(0.1, "epsilon for the per-category evaluation")
    ratio_attack: str = option(
        "pgd", "attack used for the perturbation-ratio analysis")
    ratio_attack_epsilon: float = option(
        0.1, "epsilon for the perturbation-ratio analysis")
    top_attack_categories: int = option(
        6, "how many attack categories enter the worst-class objective")
    reference_models: tuple[str, ...] = option(
        REFERENCE_MODEL_NAMES,
        "non-neural reference classifiers (log_reg, hist_gbdt)")

    # --- defense-specific coefficients ----------------------------------------
    trades_beta: float = option(6.0, "TRADES trade-off coefficient")
    free_at_replay: int = option(4, "Free AT replay multiplier m")
    class_aware_minority_weight: float = option(
        3.0, "extra loss weight for minority attack categories",
        aliases=("--class-aware-weight",))

    # --- optional extra defense methods (off by default) ----------------------
    extra_methods: tuple[str, ...] = option(
        (), "optional additional defenses to train as well: progressive, sa_trades, dst_sa_trades")
    progressive_ratios: tuple[float, ...] = option(
        (0.20, 0.30, 0.45, 0.60), "mask ratios of the progressive class-aware schedule")
    sa_trades_gamma: float = option(1.0, "sensitivity-weighting exponent of SA-TRADES")
    dst_update_interval: int = option(
        2, "epochs between sensitivity re-estimates in DST-SA-TRADES")
    dst_ema_alpha: float = option(
        0.7, "EMA smoothing factor for the DST sensitivity estimate")

    # --- attack hyperparameters ------------------------------------------------
    cw_steps: int = option(30, "C&W L2 optimisation steps")
    cw_lr: float = option(0.01, "C&W L2 learning rate")
    cw_c: float = option(1.0, "C&W L2 confidence constant")
    apgd_steps: int = option(50, "APGD-CE steps")
    apgd_rho: float = option(0.75, "APGD-CE step-size schedule parameter")

    # --- runtime ----------------------------------------------------------------
    device: str = option(
        "auto", "torch device: auto (CUDA if available, else MPS/CPU), cpu, cuda, cuda:N or mps")

    # ------------------------------------------------------------------ #
    # normalisation and validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self._normalise()
        self._validate()

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

    def _validate(self) -> None:
        """Collect every problem and report them together in one ValueError."""
        problems: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        # paths and protocol
        check(bool(self.train_path), "train_path must be a non-empty path")
        check(bool(self.test_path), "test_path must be a non-empty path")
        check(bool(self.output_dir), "output_dir must be a non-empty path")
        check(self.training_budget_mode in TRAINING_BUDGET_MODES,
              f"training_budget_mode must be one of {TRAINING_BUDGET_MODES}, "
              f"got {self.training_budget_mode!r}")
        check(self.batch_size >= 1, "batch_size must be >= 1")
        check(self.baseline_epochs >= 0, "baseline_epochs must be >= 0")
        check(self.adv_epochs >= 0, "adv_epochs must be >= 0")
        check(self.learning_rate > 0, "learning_rate must be > 0")
        check(self.weight_decay >= 0, "weight_decay must be >= 0")
        check(len(self.hidden_dims) == 3 and all(h >= 1 for h in self.hidden_dims),
              f"hidden_dims must be three positive integers, got {tuple(self.hidden_dims)!r}")
        check(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")

        # evaluation protocol
        check(self.eval_attack_rows >= 1, "eval_attack_rows must be >= 1")
        check(bool(self.epsilon_list) and all(e > 0 for e in self.epsilon_list),
              "epsilon_list must contain at least one positive epsilon")
        check(self.eval_pgd_steps >= 1, "eval_pgd_steps must be >= 1")
        check(self.eval_pgd_alpha_ratio > 0, "eval_pgd_alpha_ratio must be > 0")
        check(bool(self.seeds), "seeds must contain at least one seed")

        # adversarial training budget
        check(self.adv_epsilon > 0, "adv_epsilon must be > 0")
        check(self.adv_alpha > 0, "adv_alpha must be > 0")
        check(self.adv_steps >= 1, "adv_steps must be >= 1")

        # sensitivity analysis, ratio sweep and extra methods
        check(0.0 < self.sensitivity_top_ratio <= 1.0,
              "sensitivity_top_ratio must be in (0, 1]")
        check(bool(self.sensitivity_ratio_list)
              and all(0.0 < r <= 1.0 for r in self.sensitivity_ratio_list),
              "sensitivity_ratio_list must contain ratios in (0, 1]")
        check(self.sensitivity_batches >= 1, "sensitivity_batches must be >= 1")
        check(self.full_test_attack_rows >= 0, "full_test_attack_rows must be >= 0")
        check(self.top_attack_categories >= 1, "top_attack_categories must be >= 1")
        check(set(self.extra_methods) <= set(OPTIONAL_DEFENSE_METHODS),
              f"extra_methods must be a subset of {OPTIONAL_DEFENSE_METHODS}, "
              f"got {tuple(self.extra_methods)!r}")
        check(set(self.reference_models) <= set(REFERENCE_MODEL_NAMES),
              f"reference_models must be a subset of {REFERENCE_MODEL_NAMES}, "
              f"got {tuple(self.reference_models)!r}")

        # defense coefficients
        check(self.trades_beta > 0, "trades_beta must be > 0")
        check(self.free_at_replay >= 1, "free_at_replay must be >= 1")
        check(self.class_aware_minority_weight > 0, "class_aware_minority_weight must be > 0")
        check(bool(self.progressive_ratios)
              and all(0.0 < r <= 1.0 for r in self.progressive_ratios),
              "progressive_ratios must contain ratios in (0, 1]")
        check(self.sa_trades_gamma > 0, "sa_trades_gamma must be > 0")
        check(self.dst_update_interval >= 1, "dst_update_interval must be >= 1")
        check(0.0 <= self.dst_ema_alpha <= 1.0, "dst_ema_alpha must be in [0, 1]")

        # attack hyperparameters
        check(self.category_attack in ATTACK_NAMES,
              f"category_attack must be one of {ATTACK_NAMES}, got {self.category_attack!r}")
        check(self.ratio_attack in ATTACK_NAMES,
              f"ratio_attack must be one of {ATTACK_NAMES}, got {self.ratio_attack!r}")
        check(self.category_epsilon > 0, "category_epsilon must be > 0")
        check(self.ratio_attack_epsilon > 0, "ratio_attack_epsilon must be > 0")
        for name in ("transfer_attack_settings", "validity_attack_settings",
                     "full_test_attack_settings"):
            for attack, epsilon in getattr(self, name):
                check(attack in ATTACK_NAMES,
                      f"{name}: unknown attack {attack!r}; choose from {ATTACK_NAMES}")
                check(epsilon > 0, f"{name}: epsilon must be > 0, got {epsilon!r}")
        check(self.cw_steps >= 1, "cw_steps must be >= 1")
        check(self.cw_lr > 0, "cw_lr must be > 0")
        check(self.cw_c >= 0, "cw_c must be >= 0")
        check(self.apgd_steps >= 1, "apgd_steps must be >= 1")
        check(0.0 <= self.apgd_rho < 1.0, "apgd_rho must be in [0, 1)")

        # runtime
        check(bool(self.device), "device must be a non-empty string")

        if problems:
            details = "\n  - ".join(problems)
            raise ValueError(f"invalid ExperimentConfig:\n  - {details}")


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
# configuration as JSON.
# =============================================================================


def field_help(name: str) -> str:
    """Help text declared in the dataclass metadata for *name*."""
    return str(ExperimentConfig.__dataclass_fields__[name].metadata.get("help", ""))


def field_aliases(name: str) -> tuple[str, ...]:
    """Extra flag spellings declared in the dataclass metadata for *name*."""
    return tuple(ExperimentConfig.__dataclass_fields__[name].metadata.get("aliases", ()))


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
                raise argparse.ArgumentTypeError(
                    f"expected key:value pairs such as 'fgsm:0.05,pgd:0.10', got {item!r}")
            try:
                pairs.append((key.strip(), float(value)))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"{value!r} is not a number (in {item!r})") from None
        return tuple(pairs)

    converters = {"int": int, "float": float, "str": str}
    if kind not in converters:
        raise ValueError(f"unknown tuple kind {kind!r}")
    try:
        return tuple(converters[kind](item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"cannot parse {raw!r} as comma-separated {kind} values ({exc})") from None


def add_config_arguments(parser: argparse.ArgumentParser, skip: tuple[str, ...] = (),
                         defaults: dict[str, object] | None = None) -> None:
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
                flag, *aliases, default=default, metavar="LIST",
                type=lambda raw, kind=kind: parse_tuple_value(raw, kind),
                help=f"{help_text}  [comma-separated; default: {example}]")
        elif isinstance(declared, bool):
            parser.add_argument(flag, *aliases, default=default, type=parse_bool, nargs="?",
                                const=True, help=f"{help_text}  [default: {default}]")
        elif isinstance(declared, int):
            parser.add_argument(flag, *aliases, default=default, type=int,
                                help=f"{help_text}  [default: {default}]")
        elif isinstance(declared, float):
            parser.add_argument(flag, *aliases, default=default, type=float,
                                help=f"{help_text}  [default: {default}]")
        else:
            parser.add_argument(flag, *aliases, default=default, type=str,
                                help=f"{help_text}  [default: {default}]")


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


def build_parser(description: str | None = None,
                 skip: tuple[str, ...] = (),
                 defaults: dict[str, object] | None = None,
                 require_paths: bool = True) -> argparse.ArgumentParser:
    """Standard parser: dataset paths plus every config field.

    defaults       per-script overrides for individual config fields, so each
                   backbone can keep its own default output directory, batch
                   size and so on while still exposing every parameter.
    require_paths  True for stand-alone scripts; False for the backbone scripts,
                   which fall back to data/train.csv and data/test.csv.
    """
    d = defaults or {}
    parser = argparse.ArgumentParser(
        description=description
        or "Adversarial robustness experiments for UNSW-NB15 intrusion detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every ExperimentConfig field can be set from the command line; see above.")
    parser.add_argument("--train-path", required=require_paths,
                        default=None if require_paths else d.get("train_path", "data/train.csv"),
                        help=field_help("train_path"))
    parser.add_argument("--test-path", required=require_paths,
                        default=None if require_paths else d.get("test_path", "data/test.csv"),
                        help=field_help("test_path"))
    add_config_arguments(parser, skip=("train_path", "test_path") + tuple(skip), defaults=d)
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective configuration as JSON and exit")
    return parser


def emit_config(config: ExperimentConfig) -> None:
    """Print the effective configuration as JSON and exit. Used by --print-config."""
    print(json.dumps(asdict(config), indent=2, default=str, ensure_ascii=False))
    raise SystemExit(0)


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse a bare ExperimentConfig from the standard parser."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.print_config:
        emit_config(config)
    return config
