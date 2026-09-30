#!/usr/bin/env python
"""Cross-backbone transfer-attack experiment.

Trains a standard model and a PGD-AT model for MLP / 1D-CNN / FT-Transformer,
then measures how adversarial examples generated on one architecture transfer
to the other two.  The attacks obey the same feature constraints as the main
experiment (continuous features only, inside the training [min, max] box).

    uv run python code/evaluate_cross_backbone_transfer.py --device cuda

Outputs (appendix tables): outputs/cross_backbone_transfer_{raw,mean}.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ids_defense_selection import (
    CNN1DBackbone,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    FT_TRANSFORMER_KWARGS,
    FTTransformerBackbone,
    MLPBackbone,
    build_features,
    build_parser,
    config_from_args,
    emit_config,
    fit_supervised,
    generate_adversarial_examples,
    load_unsw_nb15,
    make_dataloader,
    pgd_attack,
    predict_proba,
    resolve_path,
    set_seed,
)


def train_standard(model, x_train, y_train, config, device):
    """Clean training with the shared baseline budget."""
    loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    return fit_supervised(model, loader, config.baseline_epochs, device, config)


def train_pgd_at(model, x_train, y_train, config, metadata, device):
    """PGD adversarial training under the shared feature constraints."""
    mask_t = torch.from_numpy(metadata["numeric_mask"].astype(np.float32)).to(device)
    mins_t = torch.from_numpy(metadata["numeric_mins"].astype(np.float32)).to(device)
    maxs_t = torch.from_numpy(metadata["numeric_maxs"].astype(np.float32)).to(device)
    loader = make_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)

    def attack_builder(module, xb, yb):
        return pgd_attack(module, xb, yb, config.adv_epsilon, config.adv_alpha,
                          config.adv_steps, mask_t, mins_t, maxs_t)

    return fit_supervised(model, loader, config.adv_epochs, device, config,
                          adversarial=True, attack_builder=attack_builder)


def compute_transfer_asr(source_model, target_model, x_eval, y_eval, config, metadata, device,
                         *, attack: str, epsilon: float):
    """Attack the source model and measure the attack-success rate on the target."""
    attacked_inputs, _ = generate_adversarial_examples(
        source_model, x_eval, y_eval, attack, epsilon, config,
        metadata["numeric_mask"], metadata["numeric_mins"], metadata["numeric_maxs"],
        device, config.batch_size, return_inputs=True,
    )

    clean_pred = (predict_proba(target_model, x_eval, config.batch_size, device) >= 0.5).astype(int)
    adv_pred = (predict_proba(target_model, attacked_inputs, config.batch_size, device) >= 0.5).astype(int)
    y_int = y_eval.astype(int)
    correct = clean_pred == y_int
    if correct.sum() == 0:
        return 0.0
    return float((adv_pred[correct] != y_int[correct]).mean())


def main() -> None:
    parser = build_parser(
        "Cross-backbone transfer attack experiment.",
        defaults={
            "train_path": str(DEFAULT_DATA_DIR / "train.csv"),
            "test_path": str(DEFAULT_DATA_DIR / "test.csv"),
            "output_dir": str(DEFAULT_OUTPUT_ROOT),
        },
        require_paths=False,
    )
    parser.add_argument("--eval-rows", type=int, default=20000,
                        help="number of test rows used for the transfer matrix")
    parser.add_argument("--transfer-attack", default="pgd",
                        help="attack used to generate the transferred examples")
    parser.add_argument("--transfer-epsilon", type=float, default=0.10,
                        help="perturbation budget of the transferred attack")
    args = parser.parse_args()

    config = config_from_args(
        args,
        train_path=str(resolve_path(args.train_path)),
        test_path=str(resolve_path(args.test_path)),
        output_dir=str(resolve_path(args.output_dir)),
    )
    if args.print_config:
        emit_config(config)
    seeds = list(config.seeds)
    device = torch.device(config.device)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[transfer] device={device} seeds={seeds} "
          f"attack={args.transfer_attack} eps={args.transfer_epsilon}", flush=True)

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    n_features = x_train.shape[1]

    rng = np.random.RandomState(config.eval_subset_seed)
    idx = rng.choice(len(x_test), min(args.eval_rows, len(x_test)), replace=False)
    x_eval, y_eval = x_test[idx], y_test[idx]

    backbone_factories = {
        "MLP": lambda: MLPBackbone(n_features, config.hidden_dims, config.dropout),
        "1D-CNN": lambda: CNN1DBackbone(n_features, config.dropout),
        "FT-Transformer": lambda: FTTransformerBackbone(n_features, **FT_TRANSFORMER_KWARGS),
    }
    backbone_names = list(backbone_factories)
    results = []

    for seed in seeds:
        print(f"\n{'=' * 60}\n[transfer] seed {seed}\n{'=' * 60}", flush=True)
        set_seed(seed)
        models = {}
        for name, factory in backbone_factories.items():
            models[(name, "clean")] = train_standard(
                factory(), x_train, y_train, config, device)
            models[(name, "pgd_at")] = train_pgd_at(
                factory(), x_train, y_train, config, metadata, device)
            print(f"  {name:16s} trained", flush=True)

        for condition in ("clean", "pgd_at"):
            for source in backbone_names:
                for target in backbone_names:
                    if source == target:
                        continue
                    asr = compute_transfer_asr(
                        models[(source, condition)], models[(target, condition)],
                        x_eval, y_eval, config, metadata, device,
                        attack=args.transfer_attack, epsilon=args.transfer_epsilon,
                    )
                    print(f"  [{condition:6s}] {source:16s} -> {target:16s}: ASR={asr * 100:.2f}%",
                          flush=True)
                    results.append({
                        "seed": seed, "condition": condition,
                        "source": source, "target": target,
                        "transfer_asr": asr,
                    })

    df = pd.DataFrame(results)
    raw_path = out_dir / "cross_backbone_transfer_raw.csv"
    df.to_csv(raw_path, index=False)
    mean_df = (df.groupby(["condition", "source", "target"])["transfer_asr"]
               .mean().reset_index())
    mean_path = out_dir / "cross_backbone_transfer_mean.csv"
    mean_df.to_csv(mean_path, index=False)

    for condition in ("clean", "pgd_at"):
        label = "Standard (Clean)" if condition == "clean" else "PGD-AT"
        print(f"\n=== Transfer ASR matrix: {label} ===", flush=True)
        subset = mean_df[mean_df.condition == condition]
        for source in backbone_names:
            row = f"{source:16s}"
            for target in backbone_names:
                if source == target:
                    row += f" | {'---':>16s}"
                    continue
                value = subset[(subset.source == source) & (subset.target == target)]
                row += (f" | {value.iloc[0]['transfer_asr'] * 100:>15.2f}%"
                        if not value.empty else f" | {'N/A':>16s}")
            print(row, flush=True)

    print(f"\nResults saved to:\n  {raw_path}\n  {mean_path}", flush=True)


if __name__ == "__main__":
    main()
