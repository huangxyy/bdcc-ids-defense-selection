#!/usr/bin/env python
"""Evaluate saved checkpoints without retraining.

Training remains the expensive step; this script loads a checkpoint bundle
(``<run>/checkpoints/seed<seed>/trained_defenses.pt``), rebuilds the six
defenses and reruns the evaluation with new settings — larger epsilon, more PGD
steps, adaptive attacks, a different evaluation subset, and so on:

    uv run python scripts/evaluate_checkpoints.py \
        --checkpoint outputs/ft_transformer/checkpoints/seed7 \
        --output-dir outputs/ft_eval_probe \
        --eval-pgd-alpha-ratio 0.10 --eval-pgd-steps 50 \
        --epsilon-list 0.05,0.10,0.20 --adaptive-eval
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
import subprocess
from dataclasses import fields
from pathlib import Path

from ids_defense_selection import (
    DEFAULT_DATA_DIR,
    EvaluationSet,
    ExperimentConfig,
    build_features,
    evaluate_defenses,
    load_checkpoint,
    load_unsw_nb15,
    log_device,
    rebuild_trained_defenses,
    resolve_device,
    stratified_subset_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split(",") if part.strip())


def _parse_attack_settings(raw: str) -> tuple[tuple[str, float], ...]:
    settings = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        attack, _, epsilon = part.partition(":")
        settings.append((attack.strip(), float(epsilon)))
    return tuple(settings)


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _is_readable_file(value) -> bool:
    """``Path.is_file`` raises PermissionError for unreachable prefixes."""
    try:
        return Path(str(value)).is_file()
    except OSError:
        return False


def build_eval_config(bundle: dict, args) -> ExperimentConfig:
    """Training config from the bundle, with only the requested eval overrides."""
    known = {field.name for field in fields(ExperimentConfig)}
    flat = {key: value for key, value in bundle["config"].items() if key in known}
    # Checkpoints record the absolute data paths of the machine that trained them.
    # Fall back to the repository copy when those paths do not exist here, so a
    # checkpoint stays usable after the run directory is synced to another host.
    for key, fallback in (("train_path", DEFAULT_DATA_DIR / "train.csv"),
                          ("test_path", DEFAULT_DATA_DIR / "test.csv")):
        value = flat.get(key)
        if not value or not _is_readable_file(value):
            if value:
                print(f"[eval-only] {key}={value} is not available; using {fallback}",
                      flush=True)
            flat[key] = str(fallback)
    overrides = {
        "output_dir": args.output_dir,
        "device": args.device,
        "eval_attack_rows": args.eval_attack_rows,
        "batch_size": args.batch_size,
        "eval_subset_seed": args.eval_subset_seed,
        "epsilon_list": args.epsilon_list,
        "eval_pgd_steps": args.eval_pgd_steps,
        "eval_pgd_alpha_ratio": args.eval_pgd_alpha_ratio,
        "adaptive_eval": args.adaptive_eval,
        "adaptive_steps": args.adaptive_steps,
        "adaptive_restarts": args.adaptive_restarts,
        "adaptive_epsilon": args.adaptive_epsilon,
        "full_test_attack_settings": args.full_test_attack_settings,
        "full_test_attack_rows": args.full_test_attack_rows,
        "train_path": args.train_path,
        "test_path": args.test_path,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return ExperimentConfig(**{**flat, **overrides})


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="seed directory or trained_defenses.pt produced with --save-checkpoints")
    parser.add_argument("--output-dir", required=True,
                        help="directory for the evaluation-only outputs")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N or mps")
    parser.add_argument("--eval-attack-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="evaluation batch size (does not change per-sample attacks)")
    parser.add_argument("--eval-subset-seed", type=int, default=None)
    parser.add_argument("--epsilon-list", type=_parse_floats, default=None)
    parser.add_argument("--eval-pgd-steps", type=int, default=None)
    parser.add_argument("--eval-pgd-alpha-ratio", type=float, default=None)
    parser.add_argument("--adaptive-eval", action="store_true", default=None)
    parser.add_argument("--adaptive-steps", type=int, default=None)
    parser.add_argument("--adaptive-restarts", type=int, default=None)
    parser.add_argument("--adaptive-epsilon", type=float, default=None)
    parser.add_argument("--full-test-attack-settings", type=_parse_attack_settings, default=None)
    parser.add_argument("--full-test-attack-rows", type=int, default=None)
    parser.add_argument("--train-path", default=None,
                        help="override the data path recorded in the checkpoint")
    parser.add_argument("--test-path", default=None,
                        help="override the data path recorded in the checkpoint")
    args = parser.parse_args()

    bundle = load_checkpoint(args.checkpoint)
    config = build_eval_config(bundle, args)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    log_device(config.device, device)
    print(f"[eval-only] checkpoint={args.checkpoint} backbone={bundle['backbone']} "
          f"seed={bundle['seed']} device={device}", flush=True)

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    if x_train.shape[1] != int(bundle["input_dim"]):
        print(f"error: checkpoint expects {bundle['input_dim']} features but the data "
              f"yields {x_train.shape[1]}; the split or preprocessing changed", flush=True)
        return 2

    indices = stratified_subset_indices(y_test, config.eval_attack_rows,
                                        seed=config.eval_subset_seed)
    eval_set = EvaluationSet(
        x_eval=x_test[indices], y_eval=y_test[indices],
        x_test=x_test, y_test=y_test,
        attack_mask=metadata["numeric_mask"],
        numeric_mins=metadata["numeric_mins"],
        numeric_maxs=metadata["numeric_maxs"],
    )

    trained = rebuild_trained_defenses(bundle, device)
    evaluation = evaluate_defenses(
        trained, config, eval_set, device, seed=bundle["seed"],
        include_categories=True,
        full_test_attack_settings=config.full_test_attack_settings,
        full_test_attack_rows=config.full_test_attack_rows,
    )

    evaluation.metrics.to_csv(out_dir / "eval_raw_results.csv", index=False)
    group = ["model", "attack", "epsilon"]
    evaluation.metrics.groupby(group, as_index=False).mean(numeric_only=True) \
        .to_csv(out_dir / "eval_mean_results.csv", index=False)
    written = ["eval_raw_results.csv", "eval_mean_results.csv"]
    if not evaluation.adaptive.empty:
        evaluation.adaptive.to_csv(out_dir / "adaptive_attack_raw.csv", index=False)
        evaluation.adaptive.drop(columns=["seed"], errors="ignore") \
            .groupby(["model", "epsilon"], as_index=False).mean(numeric_only=True) \
            .to_csv(out_dir / "adaptive_attack_mean.csv", index=False)
        written += ["adaptive_attack_raw.csv", "adaptive_attack_mean.csv"]
    if not evaluation.full_test_attack.empty:
        evaluation.full_test_attack.to_csv(out_dir / "full_test_attack_raw.csv", index=False)
        written.append("full_test_attack_raw.csv")

    summary = {
        "mode": "eval-only",
        "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "backbone": bundle["backbone"],
        "seed": bundle["seed"],
        "code_revision": _git_revision(),
        "dataset_split": bundle.get("dataset_split", {}),
        "eval_config": {field.name: getattr(config, field.name)
                        for field in fields(config)},
        "files": written,
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[eval-only] saved {len(written)} files to {out_dir}", flush=True)

    eps = max(config.epsilon_list)
    subset = evaluation.metrics[(evaluation.metrics["attack"] == "pgd") &
                                ((evaluation.metrics["epsilon"] - eps).abs() < 1e-9)]
    if not subset.empty:
        print(f"\n=== PGD eps={eps:g} (eval-only) ===")
        print(subset[["model", "f1", "attack_success_rate"]]
              .sort_values("attack_success_rate").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
