#!/usr/bin/env python
"""Run the MLP backbone experiment (the reference pipeline).

    uv run python code/run_mlp.py --device cuda

Trains the six defenses, evaluates the full attack suite plus the transfer
matrix, the sensitivity-ratio ablation and the reference models, and writes all
tables and figures into outputs/mlp/ by default.
"""
from __future__ import annotations

from ids_defense_selection import (
    DEFAULT_DATA_DIR,
    build_parser,
    config_from_args,
    default_output_dir,
    emit_config,
    resolve_path,
    run_mlp_experiment,
)


def main() -> None:
    parser = build_parser(
        "Run the MLP matched-budget IDS experiment.",
        defaults={
            "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
            "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
            "output_dir": str(default_output_dir("mlp")),
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
    run_mlp_experiment(config)


if __name__ == "__main__":
    main()
