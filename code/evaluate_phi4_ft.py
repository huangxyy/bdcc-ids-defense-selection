#!/usr/bin/env python
"""Worst-class adversarial recall (phi4) for the FT-Transformer backbone.

Trains the six defenses once per seed and records per-attack-category recall
under PGD at epsilon = 0.10, then aggregates mean and standard deviation across
seeds.  Training uses the reduced PGD budget (``--adv-steps``, default 7) while
evaluation always uses ``--eval-pgd-steps`` (default 20); the two budgets are
kept separate on purpose.

    uv run python code/evaluate_phi4_ft.py --device cuda --output-dir outputs/phi4_ft
"""
from __future__ import annotations

import torch

from ids_defense_selection import (
    DEFAULT_DATA_DIR,
    FT_TRANSFORMER_KWARGS,
    FTTransformerBackbone,
    build_parser,
    config_from_args,
    default_phi4_output_dir,
    emit_config,
    resolve_path,
)
from ids_defense_selection.phi4 import evaluate_phi4


def main() -> None:
    parser = build_parser(
        "Evaluate phi4 (worst-class recall) for the FT-Transformer backbone.",
        defaults={
            "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
            "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
            "output_dir": str(default_phi4_output_dir("ft")),
            "batch_size": 256,
            "adv_steps": 7,  # cheaper PGD used during training only
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

    device = torch.device(config.device)
    print(f"[phi4-ft] output_dir={config.output_dir} device={config.device} seeds={config.seeds}",
          flush=True)
    evaluate_phi4(
        config,
        lambda input_dim: FTTransformerBackbone(input_dim, **FT_TRANSFORMER_KWARGS),
        device,
    )


if __name__ == "__main__":
    main()
