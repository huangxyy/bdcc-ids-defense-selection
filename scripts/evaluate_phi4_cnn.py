#!/usr/bin/env python
"""Worst-class adversarial recall (phi4) for the 1D-CNN backbone.

Trains the six defenses once per seed and records per-attack-category recall
under PGD at epsilon = 0.10, then aggregates mean and standard deviation across
seeds.  Pass ``--seeds`` to change the seed set (default: the five seeds used
throughout the study).

    uv run python scripts/evaluate_phi4_cnn.py --device cuda --output-dir outputs/phi4_cnn
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

from ids_defense_selection import (
    CNN1DBackbone,
    DEFAULT_DATA_DIR,
    build_parser,
    config_from_args,
    default_phi4_output_dir,
    emit_config,
    log_device,
    resolve_device,
    resolve_path,
)
from ids_defense_selection.phi4 import evaluate_phi4


def main() -> None:
    parser = build_parser(
        "Evaluate phi4 (worst-class recall) for the 1D-CNN backbone.",
        defaults={
            "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
            "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
            "output_dir": str(default_phi4_output_dir("cnn")),
            "batch_size": 512,
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

    device = resolve_device(config.device)
    log_device(config.device, device)
    print(f"[phi4-cnn] output_dir={config.output_dir} device={device} seeds={config.seeds}",
          flush=True)
    evaluate_phi4(config, lambda input_dim: CNN1DBackbone(input_dim, config.dropout), device)


if __name__ == "__main__":
    main()
