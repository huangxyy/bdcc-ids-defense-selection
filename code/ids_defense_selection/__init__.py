"""Backbone-conditioned Pareto analysis for preference-aware IDS defense selection.

Public API of the package.  Experiment runners import from here, which keeps
``code/run_*.py`` and the analysis scripts thin and consistent::

    from ids_defense_selection import ExperimentConfig, MLPBackbone, train_all_defenses

Attributes are resolved lazily (PEP 562), so importing a light submodule such as
``ids_defense_selection.paths`` does not pull in torch, pandas or matplotlib.
"""
from __future__ import annotations

import importlib
from typing import Any

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
    "emit_config": ".config",
    "field_aliases": ".config",
    "field_group": ".config",
    "field_help": ".config",
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
    # data
    "build_features": ".data",
    "load_unsw_nb15": ".data",
    "make_dataloader": ".data",
    "numpy_to_torch": ".data",
    "set_seed": ".data",
    "split_summary": ".spec",
    "stratified_subset_indices": ".data",
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
    "summarize_results": ".reporting",
    "write_backbone_outputs": ".reporting",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the defining submodule on first attribute access and cache the result."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
