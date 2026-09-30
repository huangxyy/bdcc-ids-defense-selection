#!/usr/bin/env python
"""Worst-class adversarial recall (phi4) for the 1D-CNN backbone.

Trains the six defenses once per seed and records per-attack-category recall
under PGD at epsilon = 0.10, then aggregates mean and standard deviation across
seeds.  Pass ``--seeds`` to change the seed set (default: the five seeds used
throughout the study).

    uv run python code/evaluate_phi4_cnn.py --device cuda --output-dir outputs/phi4_cnn
"""
from __future__ import annotations

from pathlib import Path

import torch

from ids_defense_selection import (
    CNN1DBackbone,
    build_parser,
    config_from_args,
    emit_config,
    resolve_path,
)
from ids_defense_selection.phi4 import evaluate_phi4


def main() -> None:
    parser = build_parser(
        "Evaluate phi4 (worst-class recall) for the 1D-CNN backbone.",
        defaults={"output_dir": "outputs/phi4_cnn", "batch_size": 512},
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

    device = torch.device(config.device)
    print(f"[phi4-cnn] output_dir={config.output_dir} device={config.device} seeds={config.seeds}",
          flush=True)
    evaluate_phi4(config, lambda input_dim: CNN1DBackbone(input_dim, config.dropout), device)


if __name__ == "__main__":
    main()
