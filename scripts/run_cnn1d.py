#!/usr/bin/env python
"""Run the 1D-CNN backbone experiment.

    uv run python scripts/run_cnn1d.py --device cuda

Trains the six defenses on the 1D-CNN backbone, evaluates the attack suite,
the attack-validity checks and a full-test PGD attack, and writes the standard
backbone outputs into outputs/cnn1d/ by default.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

from pathlib import Path

import numpy as np
import pandas as pd

from ids_defense_selection import (
    BackboneRunFrames,
    CNN1DBackbone,
    DEFAULT_DATA_DIR,
    EvaluationSet,
    build_features,
    build_parser,
    config_from_args,
    default_output_dir,
    emit_config,
    evaluate_defenses,
    load_unsw_nb15,
    log_device,
    make_dataloader,
    resolve_device,
    resolve_path,
    save_checkpoint,
    set_seed,
    split_summary,
    stratified_subset_indices,
    train_all_defenses,
    write_backbone_outputs,
)


def main() -> None:
    parser = build_parser(
        "Run the 1D-CNN matched-budget IDS experiment.",
        defaults={
            "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
            "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
            "output_dir": str(default_output_dir("cnn")),
            "training_budget_mode": "matched_continuation",
        },
        require_paths=False,
    )
    args = parser.parse_args()

    config = config_from_args(
        args,
        train_path=str(resolve_path(args.train_path)),
        test_path=str(resolve_path(args.test_path)),
        output_dir=str(resolve_path(args.output_dir)),
    )
    if args.print_config:
        emit_config(config)

    out_dir = Path(config.output_dir)
    device = resolve_device(config.device)
    log_device(config.device, device)
    print(
        f"[cnn1d] output_dir={out_dir} device={device} "
        f"training_budget_mode={config.training_budget_mode} seeds={config.seeds}",
        flush=True,
    )

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    eval_indices = stratified_subset_indices(y_test, config.eval_attack_rows,
                                             seed=config.eval_subset_seed)
    eval_set = EvaluationSet(
        x_eval=x_test[eval_indices],
        y_eval=y_test[eval_indices],
        x_test=x_test,
        y_test=y_test,
        attack_mask=metadata["numeric_mask"],
        numeric_mins=metadata["numeric_mins"],
        numeric_maxs=metadata["numeric_maxs"],
    )

    results: list[pd.DataFrame] = []
    sensitivity: list[pd.DataFrame] = []
    efficiency: list[pd.DataFrame] = []
    validity: list[pd.DataFrame] = []
    full_test_clean: list[pd.DataFrame] = []
    full_test_attack: list[pd.DataFrame] = []
    adaptive_frames: list[pd.DataFrame] = []

    for seed in config.seeds:
        set_seed(seed)
        train_loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
        print(f"[cnn1d][seed {seed}] training started", flush=True)
        trained = train_all_defenses(
            model_factory=lambda: CNN1DBackbone(x_train.shape[1], config.dropout),
            config=config,
            train_loader=train_loader,
            device=device,
            metadata=metadata,
            seed=seed,
        )
        if config.save_checkpoints:
            checkpoint_dir = save_checkpoint(
                out_dir, trained, config, seed=seed, backbone="cnn1d",
                input_dim=x_train.shape[1],
                dataset_split=split_summary(len(train_df), len(test_df)))
            print(f"[cnn1d][seed {seed}] checkpoint saved: {checkpoint_dir}", flush=True)
        print(f"[cnn1d][seed {seed}] evaluation started", flush=True)
        evaluation = evaluate_defenses(
            trained, config, eval_set, device, seed,
            include_categories=False,
            full_test_attack_settings=config.full_test_attack_settings,
            full_test_attack_rows=config.full_test_attack_rows,
        )
        results.append(evaluation.metrics)
        sensitivity.append(trained.sensitivity_table)
        efficiency.append(evaluation.efficiency)
        validity.append(evaluation.validity)
        full_test_clean.append(evaluation.full_test_clean)
        full_test_attack.append(evaluation.full_test_attack)
        if not evaluation.adaptive.empty:
            adaptive_frames.append(evaluation.adaptive)
        print(f"[cnn1d][seed {seed}] completed", flush=True)

    frames = BackboneRunFrames(
        results=pd.concat(results, ignore_index=True),
        sensitivity=pd.concat(sensitivity, ignore_index=True),
        efficiency=pd.concat(efficiency, ignore_index=True),
        validity=pd.concat(validity, ignore_index=True),
        full_test_clean=pd.concat(full_test_clean, ignore_index=True),
        full_test_attack=pd.concat(full_test_attack, ignore_index=True),
        adaptive=(pd.concat(adaptive_frames, ignore_index=True) if adaptive_frames
                  else pd.DataFrame()),
    )
    summary = {
        "architecture": "1D-CNN (Conv1d(1,32,3) -> Conv1d(32,64,3) -> AvgPool -> FC(64,32) -> FC(32,1))",
        "resolved_device": str(device),
        "dataset_split": split_summary(len(train_df), len(test_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(eval_set.y_eval)),
        "full_test_attack_rows": int(frames.full_test_attack["test_rows"].iloc[0])
        if not frames.full_test_attack.empty else 0,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "transformed_feature_count": int(len(metadata["feature_names"])),
    }
    aggregated = write_backbone_outputs(out_dir, config, frames, summary)

    key = aggregated["mean_results"]
    reference_epsilon = max(config.epsilon_list)
    key = key[(key["attack"] == "pgd") & np.isclose(key["epsilon"], reference_epsilon)]
    print(f"\n=== 1D-CNN key results (PGD eps={reference_epsilon:.2f} on eval subset) ===", flush=True)
    print(key[["model", "f1", "attack_success_rate"]].to_string(index=False), flush=True)
    print(f"\nAll saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
