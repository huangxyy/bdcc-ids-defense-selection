"""Configuration dataclass and the command-line interface generated from it.

Every field of :class:`ExperimentConfig` gets a matching command-line flag, so
the CLI can never drift out of sync with the configuration.  The configuration
is always passed around explicitly; there is no module-level mutable state.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

#: Defaults shared by the CLI, the Pareto analysis and the auxiliary scripts, so
#: the study protocol is defined in exactly one place.
DEFAULT_SEEDS: tuple[int, ...] = (7, 13, 21, 42, 100)
DEFAULT_EPSILON_LIST: tuple[float, ...] = (0.02, 0.05, 0.10)
DEFAULT_EVAL_SUBSET_SEED: int = 2026
DEFAULT_FULL_TEST_ATTACK_SETTINGS: tuple[tuple[str, float], ...] = (("pgd", 0.10),)


@dataclass
class ExperimentConfig:
    """All hyperparameters of one backbone experiment in a single place."""

    train_path: str
    test_path: str
    output_dir: str = "outputs/default"
    training_budget_mode: str = "legacy"
    batch_size: int = 1024
    baseline_epochs: int = 10
    adv_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.15
    eval_attack_rows: int = 20000
    #: Seed of the fixed stratified evaluation subset, shared by every defense
    #: and every backbone so the reported numbers are comparable.
    eval_subset_seed: int = DEFAULT_EVAL_SUBSET_SEED
    # Adversarial training budget (PGD used to build the training examples).
    adv_epsilon: float = 0.06
    adv_alpha: float = 0.015
    adv_steps: int = 20
    # Evaluation budget.
    epsilon_list: tuple[float, ...] = DEFAULT_EPSILON_LIST
    eval_pgd_steps: int = 20
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    sensitivity_top_ratio: float = 0.3
    sensitivity_ratio_list: tuple[float, ...] = (0.2, 0.3, 0.4)
    sensitivity_batches: int = 16
    transfer_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    validity_attack_settings: tuple[tuple[str, float], ...] = (("fgsm", 0.05), ("pgd", 0.1))
    #: Attack applied to the (full) test partition after training; `rows` = 0
    #: attacks the complete partition.
    full_test_attack_settings: tuple[tuple[str, float], ...] = DEFAULT_FULL_TEST_ATTACK_SETTINGS
    full_test_attack_rows: int = 0
    category_attack: str = "pgd"
    category_epsilon: float = 0.1
    ratio_attack: str = "pgd"
    ratio_attack_epsilon: float = 0.1
    top_attack_categories: int = 6
    reference_models: tuple[str, ...] = ("log_reg", "hist_gbdt")
    # TRADES
    trades_beta: float = 6.0
    # --- optional extra methods (see the Optional extra defense methods section) ---
    extra_methods: tuple[str, ...] = ()
    progressive_ratios: tuple[float, ...] = (0.20, 0.30, 0.45, 0.60)
    sa_trades_gamma: float = 1.0
    dst_update_interval: int = 2
    dst_ema_alpha: float = 0.7
    # Free AT
    free_at_replay: int = 4
    # Class-aware constrained AT
    class_aware_minority_weight: float = 3.0
    # C&W attack
    # PGD evaluation step size as a fraction of epsilon.
    # 0.05 = eps/20, the value used for every reported result.
    # Set to 0.10 to match the "alpha = eps/10" wording of the manuscript.
    eval_pgd_alpha_ratio: float = 0.05
    cw_steps: int = 30
    cw_lr: float = 0.01
    cw_c: float = 1.0
    # APGD attack
    apgd_steps: int = 50
    apgd_rho: float = 0.75
    device: str = "cpu"


# =============================================================================
# Command-line interface
#
# Every field of ExperimentConfig has a matching command-line flag, generated
# from the dataclass itself.  Adding a field to ExperimentConfig automatically
# adds a flag here, so the CLI can never drift out of sync with the config.
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

CONFIG_HELP: dict[str, str] = {
    "output_dir": "directory for all result files",
    "training_budget_mode": "legacy = each defense trained independently; "
                            "matched_continuation = shared clean pre-training plus an equal continuation budget",
    "batch_size": "mini-batch size",
    "baseline_epochs": "clean pre-training epochs",
    "adv_epochs": "adversarial (continuation) epochs per defense",
    "learning_rate": "Adam learning rate",
    "weight_decay": "Adam weight decay",
    "hidden_dims": "MLP hidden layer widths",
    "dropout": "dropout probability",
    "eval_attack_rows": "number of test rows used for the attack evaluation (stratified, shared by all defenses)",
    "eval_subset_seed": "seed of the fixed stratified evaluation subset (shared by all defenses)",
    "adv_epsilon": "L-inf perturbation budget used DURING TRAINING (differs from the evaluation budgets)",
    "adv_alpha": "PGD step size used during training",
    "adv_steps": "PGD steps used during training",
    "epsilon_list": "evaluation perturbation budgets",
    "eval_pgd_steps": "PGD steps used at evaluation time (independent of the training steps)",
    "seeds": "random seeds; results are aggregated over them",
    "sensitivity_top_ratio": "fraction of features kept by the sensitivity mask",
    "sensitivity_ratio_list": "mask ratios swept during the sensitivity analysis",
    "sensitivity_batches": "mini-batches used to estimate feature sensitivity",
    "transfer_attack_settings": "attack:epsilon pairs used for the transferability matrix",
    "validity_attack_settings": "attack:epsilon pairs used for the attack-validity check",
    "full_test_attack_settings": "attack:epsilon pairs applied to the full test partition after training",
    "full_test_attack_rows": "rows of the test partition attacked at the end (0 = the whole partition)",
    "category_attack": "attack used for the per-category (worst-class) evaluation",
    "category_epsilon": "epsilon for the per-category evaluation",
    "ratio_attack": "attack used for the perturbation-ratio analysis",
    "ratio_attack_epsilon": "epsilon for the perturbation-ratio analysis",
    "top_attack_categories": "how many attack categories enter the worst-class objective",
    "reference_models": "non-neural reference classifiers (log_reg, hist_gbdt)",
    "trades_beta": "TRADES trade-off coefficient",
    "extra_methods": "optional additional defenses to train as well: progressive, sa_trades, dst_sa_trades",
    "progressive_ratios": "mask ratios of the progressive class-aware schedule",
    "sa_trades_gamma": "sensitivity-weighting exponent of SA-TRADES",
    "dst_update_interval": "epochs between sensitivity re-estimates in DST-SA-TRADES",
    "dst_ema_alpha": "EMA smoothing factor for the DST sensitivity estimate",
    "free_at_replay": "Free AT replay multiplier m",
    "class_aware_minority_weight": "extra loss weight for minority attack categories",
    "cw_steps": "C&W L2 optimisation steps",
    "cw_lr": "C&W L2 learning rate",
    "cw_c": "C&W L2 confidence constant",
    "apgd_steps": "APGD-CE steps",
    "apgd_rho": "APGD-CE step-size schedule parameter",
    "device": "torch device, e.g. cpu or cuda",
    "eval_pgd_alpha_ratio": "PGD evaluation step size as a fraction of epsilon "
                            "(0.05 = eps/20, the value behind every reported result; 0.10 = eps/10)",
}

# Shorter flag names used by older scripts, kept working.
CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "class_aware_minority_weight": ("--class-aware-weight",),
    "adv_steps": ("--train-adv-steps",),
}


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
    """
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if kind == "pair":
        out = []
        for it in items:
            key, sep, val = it.partition(":")
            if not sep:
                raise argparse.ArgumentTypeError(
                    f"expected key:value pairs such as 'fgsm:0.05,pgd:0.10', got {it!r}")
            out.append((key.strip(), float(val)))
        return tuple(out)
    conv = {"int": int, "float": float, "str": str}[kind]
    return tuple(conv(x) for x in items)


def add_config_arguments(parser: argparse.ArgumentParser, skip: tuple[str, ...] = (),
                         defaults: dict[str, object] | None = None) -> None:
    """Attach one flag per ExperimentConfig field. Generated, never hand-maintained."""
    for name, field in ExperimentConfig.__dataclass_fields__.items():
        if name in skip:
            continue
        flag = "--" + name.replace("_", "-")
        aliases = list(CONFIG_ALIASES.get(name, ()))
        default = (defaults or {}).get(name, field.default)
        help_text = CONFIG_HELP.get(name, "")

        if isinstance(default, tuple):
            example = format_tuple_default(default)
            parser.add_argument(flag, *aliases, default=default, metavar="LIST",
                                help=f"{help_text}  [comma-separated; default: {example}]")
        elif isinstance(default, bool):
            parser.add_argument(flag, *aliases, default=default, type=parse_bool, nargs="?",
                                const=True, help=f"{help_text}  [default: {default}]")
        elif isinstance(default, int):
            parser.add_argument(flag, *aliases, default=default, type=int,
                                help=f"{help_text}  [default: {default}]")
        elif isinstance(default, float):
            parser.add_argument(flag, *aliases, default=default, type=float,
                                help=f"{help_text}  [default: {default}]")
        else:
            parser.add_argument(flag, *aliases, default=default, type=str,
                                help=f"{help_text}  [default: {default}]")


def config_from_args(args: argparse.Namespace, **overrides: object) -> ExperimentConfig:
    """Build an ExperimentConfig from parsed arguments.

    Anything the user did not pass keeps the dataclass default. Tuple fields
    arrive as strings and are converted according to their default's element type.
    """
    kwargs: dict[str, object] = {}
    for name, field in ExperimentConfig.__dataclass_fields__.items():
        raw = getattr(args, name, None)
        if raw is None:
            continue
        default = field.default
        if isinstance(default, tuple) and isinstance(raw, str):
            raw = parse_tuple_value(raw, tuple_value_kind(default))
        kwargs[name] = raw
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


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
        description=description or "Adversarial robustness experiments for UNSW-NB15 intrusion detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every ExperimentConfig field can be set from the command line; see above.")
    parser.add_argument("--train-path", required=require_paths,
                        default=None if require_paths else d.get("train_path", "data/train.csv"),
                        help="UNSW-NB15 training partition (175,341 records)")
    parser.add_argument("--test-path", required=require_paths,
                        default=None if require_paths else d.get("test_path", "data/test.csv"),
                        help="UNSW-NB15 testing partition (82,332 records)")
    add_config_arguments(parser, skip=("train_path", "test_path") + tuple(skip), defaults=d)
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective configuration as JSON and exit")
    return parser


def emit_config(config: ExperimentConfig) -> None:
    """Print the effective configuration as JSON and exit. Used by --print-config."""
    print(json.dumps(asdict(config), indent=2, default=str))
    raise SystemExit(0)


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse a bare ExperimentConfig from the standard parser."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.print_config:
        emit_config(config)
    return config
