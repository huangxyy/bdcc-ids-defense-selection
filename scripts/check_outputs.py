#!/usr/bin/env python
"""Validate a completed ``outputs/`` tree before using the numbers.

Checks, per backbone directory:

* the standard artefacts exist (raw/mean/std, efficiency, significance,
  hyperparameters, run summary);
* the files still carry the expected columns and the canonical six defenses;
* the run used the official UNSW-NB15 split direction;
* run-to-run dispersion is actually present (more than one seed).

Use ``--require-analysis`` after ``scripts/pareto_selection.py`` to also check
the decision artefacts (``risk_profile_4d.csv``, ``decision_settings.json``).

    uv run python scripts/check_outputs.py
    uv run python scripts/check_outputs.py --backbones mlp,cnn1d
    uv run python scripts/check_outputs.py --require-analysis
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BACKBONE_DIRS = {
    "mlp": "mlp",
    "cnn1d": "cnn1d",
    "ft": "ft_transformer",
}

REQUIRED_FILES = (
    "raw_results.csv",
    "mean_results.csv",
    "std_results.csv",
    "efficiency_raw.csv",
    "efficiency_mean.csv",
    "significance_tests.csv",
    "hyperparameters.csv",
    "run_summary.json",
)

REQUIRED_COLUMNS = {
    "raw_results.csv": {"seed", "model", "attack", "epsilon", "f1", "attack_success_rate"},
    "mean_results.csv": {"model", "attack", "epsilon", "f1", "attack_success_rate"},
    "std_results.csv": {"model", "attack", "epsilon", "f1", "attack_success_rate"},
    "efficiency_mean.csv": {"model", "train_seconds", "inference_ms_per_sample"},
    "significance_tests.csv": {"comparison", "attack", "epsilon", "metric", "n_seeds",
                               "t_pvalue", "wilcoxon_pvalue"},
    "hyperparameters.csv": {"group", "field", "value", "help"},
}

def check_backbone(label: str, directory: Path, require_analysis: bool,
                   require_checkpoints: bool,
                   errors: list[str], warnings: list[str]) -> None:
    """Append every problem found in one backbone directory."""
    if not directory.is_dir():
        errors.append(f"{label}: missing directory {directory}")
        return

    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            errors.append(f"{label}: missing {name}")

    for name, columns in REQUIRED_COLUMNS.items():
        path = directory / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        missing = columns - set(frame.columns)
        if missing:
            errors.append(f"{label}: {name} is missing columns {sorted(missing)}")
        if name == "raw_results.csv" and "seed" in frame.columns:
            seeds = frame["seed"].nunique()
            if seeds < 2:
                warnings.append(f"{label}: raw_results.csv has {seeds} seed; "
                                "Pareto decisions cannot be checked for stability")
            elif seeds < 5:
                warnings.append(f"{label}: raw_results.csv has {seeds} seeds; "
                                "the manuscript reports five")
        if name in ("mean_results.csv", "std_results.csv") and "model" in frame.columns:
            models = frame["model"].nunique()
            if models < 6:
                errors.append(f"{label}: {name} covers {models} defenses, expected 6")
        if name == "std_results.csv" and "f1" in frame.columns:
            if frame["f1"].fillna(0.0).abs().max() == 0.0:
                warnings.append(f"{label}: std_results.csv is all zero; "
                                "dispersion was not aggregated over seeds")

    summary_path = directory / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        split = summary.get("dataset_split", {})
        if split and not split.get("official_direction", False):
            errors.append(f"{label}: run_summary.json says the UNSW-NB15 split is "
                          "not in the official direction")
        if "resolved_device" not in summary:
            warnings.append(f"{label}: run_summary.json has no resolved_device")

    if not (directory / "attack_generalization.csv").is_file():
        warnings.append(f"{label}: no attack_generalization.csv "
                        "(unseen-attack evidence missing)")

    if require_analysis:
        analysis_files = {
            "risk_profile_4d.csv": {"phi1_clean_f1", "phi2_resilience",
                                    "phi3_cost_eff", "phi4_fairness"},
            "decision_comparators.csv": {"theta_name", "pareto_weighted",
                                         "weighted_no_pareto", "weighted_fixed01",
                                         "topsis_no_pareto", "topsis_pareto"},
            "candidate_dependence.csv": {"theta_name", "removed", "selection",
                                         "selection_without", "changed"},
            "theta_sweep.csv": {"theta1", "theta2", "theta3", "theta4", "selected"},
            "theta_summary.csv": {"defense", "n_regions", "share"},
            "admissibility_sweep.csv": {"tau2", "tau4", "n_admissible",
                                        "pareto_size", "no_candidate"},
        }
        for name, columns in analysis_files.items():
            path = directory / name
            if not path.is_file():
                errors.append(f"{label}: missing analysis artefact {name}")
                continue
            missing = columns - set(pd.read_csv(path).columns)
            if missing:
                errors.append(f"{label}: {name} is missing columns {sorted(missing)}")
            if name == "risk_profile_4d.csv":
                frame = pd.read_csv(path)
                for column in ("is_pareto_optimal", "is_pareto_optimal_deterministic"):
                    if column in frame.columns and frame[column].isna().any():
                        errors.append(f"{label}: {name} has NaN values in {column}")
                if "is_pareto_optimal" in frame.columns and not frame[
                        "is_pareto_optimal"].fillna(False).any():
                    errors.append(f"{label}: {name} marks no Pareto-optimal candidate")
        if not (directory / "switching_regions.json").is_file():
            errors.append(f"{label}: missing analysis artefact switching_regions.json")

    if require_checkpoints:
        bundles = sorted((directory / "checkpoints").glob("seed*/trained_defenses.pt"))
        if not bundles:
            errors.append(f"{label}: --require-checkpoints was set but no "
                          "checkpoints/seed*/trained_defenses.pt exists")
        for bundle in bundles:
            if not (bundle.parent / "metadata.json").is_file():
                errors.append(f"{label}: {bundle.parent.name} has no metadata.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT / "outputs"),
                        help="outputs root (default: <repo>/outputs)")
    parser.add_argument("--backbones", default="mlp,cnn1d,ft",
                        help="comma-separated subset of: mlp, cnn1d, ft")
    parser.add_argument("--require-analysis", action="store_true",
                        help="also require the Pareto/decision artefacts")
    parser.add_argument("--require-checkpoints", action="store_true",
                        help="require at least one saved checkpoint bundle per backbone")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    keys = [key.strip().lower() for key in args.backbones.split(",") if key.strip()]
    unknown = [key for key in keys if key not in BACKBONE_DIRS]
    if unknown:
        print(f"unknown backbone(s): {unknown}; choose from {sorted(BACKBONE_DIRS)}")
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    print(f"Checking {root}")
    for key in keys:
        label = BACKBONE_DIRS[key]
        before = len(errors)
        check_backbone(label, root / label, args.require_analysis,
                       args.require_checkpoints, errors, warnings)
        print(f"  [{'ok' if len(errors) == before else 'FAIL'}] {label}")

    if args.require_analysis:
        settings = root / "decision_settings.json"
        if not settings.is_file():
            errors.append(f"analysis: missing {settings}")
        else:
            payload = json.loads(settings.read_text(encoding="utf-8"))
            for field in ("confidence_margin", "min_phi2", "min_phi4", "backbones"):
                if field not in payload:
                    errors.append(f"analysis: decision_settings.json has no {field!r}")

    print()
    for message in warnings:
        print(f"  [warn] {message}")
    for message in errors:
        print(f"  [FAIL] {message}")
    if errors:
        print(f"\n{len(errors)} problem(s) found.")
        return 1
    print(f"\nAll checks passed ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
