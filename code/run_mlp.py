#!/usr/bin/env python
"""Run the MLP backbone experiment (the reference pipeline).

    uv run python code/run_mlp.py --device cuda

Trains the six defenses, evaluates the full attack suite plus the transfer
matrix, the sensitivity-ratio ablation and the reference models, and writes all
tables and figures into outputs/mlp/ by default.
"""
from __future__ import annotations

from pathlib import Path

from ids_defense_selection import build_parser, config_from_args, emit_config, resolve_path
from ids_defense_selection import run_mlp_experiment


def main() -> None:
    parser = build_parser(
        "Run the MLP matched-budget IDS experiment.",
        defaults={"output_dir": "outputs/mlp", "training_budget_mode": "matched_continuation"},
        require_paths=False,
    )
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    config = config_from_args(
        args,
        train_path=resolve_path(base_dir, args.train_path),
        test_path=resolve_path(base_dir, args.test_path),
        output_dir=resolve_path(base_dir, args.output_dir),
    )
    if args.print_config:
        emit_config(config)
    run_mlp_experiment(config)


if __name__ == "__main__":
    main()
