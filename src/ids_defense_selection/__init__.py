"""Backbone-conditioned Pareto analysis for preference-aware IDS defense selection.

Public API of the package.  Experiment runners import from here, which keeps
``scripts/run_*.py`` and the analysis scripts thin and consistent::

    from ids_defense_selection import ExperimentConfig, MLPBackbone, train_all_defenses

Attributes are resolved lazily (PEP 562), so importing a light submodule such as
``ids_defense_selection.paths`` does not pull in torch, pandas or matplotlib.
Submodules are exposed as well, so ``idsds.style`` / ``idsds.phi4`` work without
an explicit ``import ids_defense_selection.style``::

    import ids_defense_selection as idsds

    idsds.ExperimentConfig       # -> ids_defense_selection.config.ExperimentConfig
    idsds.paths.PROJECT_ROOT     # -> the submodule itself
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from . import (
        adaptive as adaptive,
        attacks as attacks,
        backbones as backbones,
        checkpoints as checkpoints,
        config as config,
        data as data,
        defenses as defenses,
        device as device,
        evaluation as evaluation,
        experiment as experiment,
        mcda as mcda,
        paths as paths,
        phi4 as phi4,
        reporting as reporting,
        selection as selection,
        spec as spec,
        stats as stats,
        style as style,
        supportedness as supportedness,
    )

    # Flat exports, mirroring ``_EXPORTS`` below.  These re-exports are what
    # let type checkers and IDEs resolve ``idsds.run_mlp_experiment`` to the
    # real function; without them the module ``__getattr__`` fallback types
    # every name as ``Any`` and go-to-definition cannot follow it.  They never
    # execute at runtime - the whole block is skipped because
    # ``TYPE_CHECKING`` is False there.  ``test_type_checking_reexports...``
    # keeps this list in sync with ``_EXPORTS``.
    from .config import (
        DEFAULT_EPSILON_LIST as DEFAULT_EPSILON_LIST,
        DEFAULT_EVAL_SUBSET_SEED as DEFAULT_EVAL_SUBSET_SEED,
        DEFAULT_FULL_TEST_ATTACK_SETTINGS as DEFAULT_FULL_TEST_ATTACK_SETTINGS,
        DEFAULT_SEEDS as DEFAULT_SEEDS,
        ExperimentConfig as ExperimentConfig,
        GROUP_CLASSES as GROUP_CLASSES,
        GROUP_ORDER as GROUP_ORDER,
        AttackConfig as AttackConfig,
        EvaluationConfig as EvaluationConfig,
        MethodsConfig as MethodsConfig,
        OPTIONAL_DEFENSE_METHODS as OPTIONAL_DEFENSE_METHODS,
        PathsConfig as PathsConfig,
        REFERENCE_MODEL_NAMES as REFERENCE_MODEL_NAMES,
        RuntimeConfig as RuntimeConfig,
        SensitivityConfig as SensitivityConfig,
        TRAINING_BUDGET_MODES as TRAINING_BUDGET_MODES,
        TrainingConfig as TrainingConfig,
        add_config_arguments as add_config_arguments,
        build_parser as build_parser,
        config_from_args as config_from_args,
        default_instance as default_instance,
        default_rows as default_rows,
        emit_config as emit_config,
        field_aliases as field_aliases,
        field_group as field_group,
        field_help as field_help,
        format_default_config as format_default_config,
        parse_args as parse_args,
        parse_tuple_value as parse_tuple_value,
    )
    from .paths import (
        BACKBONE_OUTPUT_SUBDIRS as BACKBONE_OUTPUT_SUBDIRS,
        BACKBONE_RUNNERS as BACKBONE_RUNNERS,
        DEFAULT_DATA_DIR as DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT_ROOT as DEFAULT_OUTPUT_ROOT,
        PHI4_OUTPUT_SUBDIRS as PHI4_OUTPUT_SUBDIRS,
        PROJECT_ROOT as PROJECT_ROOT,
        default_output_dir as default_output_dir,
        default_phi4_output_dir as default_phi4_output_dir,
        resolve_path as resolve_path,
    )
    from .attacks import (
        ATTACK_NAMES as ATTACK_NAMES,
        apgd_attack as apgd_attack,
        clamp_numeric as clamp_numeric,
        compute_attack_validity_metrics as compute_attack_validity_metrics,
        count_out_of_range as count_out_of_range,
        cw_attack as cw_attack,
        fgsm_attack as fgsm_attack,
        generate_adversarial_examples as generate_adversarial_examples,
        max_violation as max_violation,
        pgd_attack as pgd_attack,
    )
    from .backbones import (
        CNN1DBackbone as CNN1DBackbone,
        FTTransformerBackbone as FTTransformerBackbone,
        FT_TRANSFORMER_KWARGS as FT_TRANSFORMER_KWARGS,
        MLPBackbone as MLPBackbone,
        build_ft_transformer as build_ft_transformer,
        ft_transformer_kwargs as ft_transformer_kwargs,
    )
    from .data import (
        build_features as build_features,
        load_unsw_nb15 as load_unsw_nb15,
        make_dataloader as make_dataloader,
        numpy_to_torch as numpy_to_torch,
        set_seed as set_seed,
        stratified_subset_indices as stratified_subset_indices,
    )
    from .spec import (
        split_summary as split_summary,
    )
    from .device import (
        describe_device as describe_device,
        device_report as device_report,
        log_device as log_device,
        resolve_device as resolve_device,
    )
    from .defenses import (
        DEFENSE_ORDER as DEFENSE_ORDER,
        OPTIONAL_DEFENSES as OPTIONAL_DEFENSES,
        TrainedDefenses as TrainedDefenses,
        compute_class_aware_sensitivity_mask as compute_class_aware_sensitivity_mask,
        compute_progressive_masks as compute_progressive_masks,
        compute_sensitivity_mask as compute_sensitivity_mask,
        compute_sensitivity_weights as compute_sensitivity_weights,
        fit_class_aware_constrained as fit_class_aware_constrained,
        fit_dst_sa_trades as fit_dst_sa_trades,
        fit_free_at as fit_free_at,
        fit_progressive_class_aware as fit_progressive_class_aware,
        fit_reference_models as fit_reference_models,
        fit_sa_trades as fit_sa_trades,
        fit_supervised as fit_supervised,
        fit_trades as fit_trades,
        summarize_training_budget as summarize_training_budget,
        train_all_defenses as train_all_defenses,
    )
    from .evaluation import (
        EvaluationResults as EvaluationResults,
        EvaluationSet as EvaluationSet,
        classification_metrics as classification_metrics,
        count_parameters as count_parameters,
        evaluate_attack_suite as evaluate_attack_suite,
        evaluate_category_recall as evaluate_category_recall,
        evaluate_defenses as evaluate_defenses,
        evaluate_model_on_attack as evaluate_model_on_attack,
        evaluate_transfer_attack as evaluate_transfer_attack,
        measure_inference_cost as measure_inference_cost,
        predict_proba as predict_proba,
    )
    from .experiment import (
        prepare_attack_categories as prepare_attack_categories,
        run_mlp_experiment as run_mlp_experiment,
    )
    from .checkpoints import (
        load_checkpoint as load_checkpoint,
        rebuild_trained_defenses as rebuild_trained_defenses,
        save_checkpoint as save_checkpoint,
    )
    from .reporting import (
        BackboneRunFrames as BackboneRunFrames,
        compute_significance_tests as compute_significance_tests,
        plot_clean_f1_bar as plot_clean_f1_bar,
        plot_efficiency_tradeoff as plot_efficiency_tradeoff,
        plot_metric_curve as plot_metric_curve,
        plot_ratio_ablation as plot_ratio_ablation,
        plot_transfer_heatmap as plot_transfer_heatmap,
        run_paired_test as run_paired_test,
        summarize_results as summarize_results,
        write_backbone_outputs as write_backbone_outputs,
    )

#: public name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # config
    "DEFAULT_EPSILON_LIST": ".config",
    "DEFAULT_EVAL_SUBSET_SEED": ".config",
    "DEFAULT_FULL_TEST_ATTACK_SETTINGS": ".config",
    "DEFAULT_SEEDS": ".config",
    "ExperimentConfig": ".config",
    "GROUP_CLASSES": ".config",
    "GROUP_ORDER": ".config",
    "AttackConfig": ".config",
    "EvaluationConfig": ".config",
    "MethodsConfig": ".config",
    "OPTIONAL_DEFENSE_METHODS": ".config",
    "PathsConfig": ".config",
    "REFERENCE_MODEL_NAMES": ".config",
    "RuntimeConfig": ".config",
    "SensitivityConfig": ".config",
    "TRAINING_BUDGET_MODES": ".config",
    "TrainingConfig": ".config",
    "add_config_arguments": ".config",
    "build_parser": ".config",
    "config_from_args": ".config",
    "default_instance": ".config",
    "default_rows": ".config",
    "emit_config": ".config",
    "field_aliases": ".config",
    "field_group": ".config",
    "field_help": ".config",
    "format_default_config": ".config",
    "parse_args": ".config",
    "parse_tuple_value": ".config",
    # paths
    "BACKBONE_OUTPUT_SUBDIRS": ".paths",
    "BACKBONE_RUNNERS": ".paths",
    "DEFAULT_DATA_DIR": ".paths",
    "DEFAULT_OUTPUT_ROOT": ".paths",
    "PHI4_OUTPUT_SUBDIRS": ".paths",
    "PROJECT_ROOT": ".paths",
    "default_output_dir": ".paths",
    "default_phi4_output_dir": ".paths",
    "resolve_path": ".paths",
    # attacks
    "ATTACK_NAMES": ".attacks",
    "apgd_attack": ".attacks",
    "clamp_numeric": ".attacks",
    "compute_attack_validity_metrics": ".attacks",
    "count_out_of_range": ".attacks",
    "cw_attack": ".attacks",
    "fgsm_attack": ".attacks",
    "generate_adversarial_examples": ".attacks",
    "max_violation": ".attacks",
    "pgd_attack": ".attacks",
    # backbones
    "CNN1DBackbone": ".backbones",
    "FTTransformerBackbone": ".backbones",
    "FT_TRANSFORMER_KWARGS": ".backbones",
    "MLPBackbone": ".backbones",
    "build_ft_transformer": ".backbones",
    "ft_transformer_kwargs": ".backbones",
    # checkpoints
    "load_checkpoint": ".checkpoints",
    "rebuild_trained_defenses": ".checkpoints",
    "save_checkpoint": ".checkpoints",
    # data
    "build_features": ".data",
    "load_unsw_nb15": ".data",
    "make_dataloader": ".data",
    "numpy_to_torch": ".data",
    "set_seed": ".data",
    "stratified_subset_indices": ".data",
    # spec
    "split_summary": ".spec",
    # device
    "describe_device": ".device",
    "device_report": ".device",
    "log_device": ".device",
    "resolve_device": ".device",
    # defenses
    "DEFENSE_ORDER": ".defenses",
    "OPTIONAL_DEFENSES": ".defenses",
    "TrainedDefenses": ".defenses",
    "compute_class_aware_sensitivity_mask": ".defenses",
    "compute_progressive_masks": ".defenses",
    "compute_sensitivity_mask": ".defenses",
    "compute_sensitivity_weights": ".defenses",
    "fit_class_aware_constrained": ".defenses",
    "fit_dst_sa_trades": ".defenses",
    "fit_free_at": ".defenses",
    "fit_progressive_class_aware": ".defenses",
    "fit_reference_models": ".defenses",
    "fit_sa_trades": ".defenses",
    "fit_supervised": ".defenses",
    "fit_trades": ".defenses",
    "summarize_training_budget": ".defenses",
    "train_all_defenses": ".defenses",
    # evaluation
    "EvaluationResults": ".evaluation",
    "EvaluationSet": ".evaluation",
    "classification_metrics": ".evaluation",
    "count_parameters": ".evaluation",
    "evaluate_attack_suite": ".evaluation",
    "evaluate_category_recall": ".evaluation",
    "evaluate_defenses": ".evaluation",
    "evaluate_model_on_attack": ".evaluation",
    "evaluate_transfer_attack": ".evaluation",
    "measure_inference_cost": ".evaluation",
    "predict_proba": ".evaluation",
    # experiment
    "prepare_attack_categories": ".experiment",
    "run_mlp_experiment": ".experiment",
    # reporting
    "BackboneRunFrames": ".reporting",
    "compute_significance_tests": ".reporting",
    "plot_clean_f1_bar": ".reporting",
    "plot_efficiency_tradeoff": ".reporting",
    "plot_metric_curve": ".reporting",
    "plot_ratio_ablation": ".reporting",
    "plot_transfer_heatmap": ".reporting",
    "run_paired_test": ".reporting",
    "summarize_results": ".reporting",
    "write_backbone_outputs": ".reporting",
}

#: Submodules reachable as attributes: ``idsds.paths``, ``idsds.style``, ...
_SUBMODULES: tuple[str, ...] = (
    "adaptive",
    "attacks",
    "backbones",
    "checkpoints",
    "config",
    "data",
    "defenses",
    "device",
    "evaluation",
    "experiment",
    "mcda",
    "paths",
    "phi4",
    "reporting",
    "selection",
    "spec",
    "stats",
    "style",
    "supportedness",
)

__all__ = sorted(_EXPORTS)

#: Every name reachable as an attribute: the flat exports plus the submodules.
#: Sorted once here so both typo hints and ``dir()`` are deterministic.
_EXPOSED_NAMES: tuple[str, ...] = tuple(sorted({*_EXPORTS, *_SUBMODULES}))


def _typo_hint(name: str) -> str:
    """Return ``"; did you mean ...?"`` for a near miss, else ``""``.

    ``difflib`` is imported here rather than at module level: it costs a couple
    of milliseconds and only a failed attribute lookup ever needs it.
    """
    if name.startswith("__") and name.endswith("__"):
        # Tooling probes modules for dunders (pickle, copy, inspect); don't scan.
        return ""
    import difflib  # deferred on purpose - see docstring

    close = difflib.get_close_matches(name, _EXPOSED_NAMES, n=3, cutoff=0.6)
    if not close:
        return ""
    return f"; did you mean {', '.join(map(repr, close))}?"


def __getattr__(name: str) -> Any:
    """Import what *name* needs on first access and cache the result.

    A public name pulls in the submodule that defines it (``idsds.MLPBackbone``
    imports ``.backbones``); a submodule name returns the module itself
    (``idsds.paths``).  Unknown names raise ``AttributeError``; near misses get
    the closest matches as a hint, which turns typos into an actionable message.
    """
    if name in _SUBMODULES:
        value: Any = importlib.import_module(f"{__name__}.{name}")
    else:
        module_name = _EXPORTS.get(name)
        if module_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}{_typo_hint(name)}")
        value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPOSED_NAMES))
