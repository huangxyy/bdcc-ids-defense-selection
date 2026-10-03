#!/usr/bin/env python
"""Cross-backbone transfer attacks from saved checkpoints (no retraining).

The training runs already store the clean and PGD-AT weights of every backbone
in ``outputs/<backbone>/checkpoints/seed<seed>/``.  This script loads them and
measures the attack-success rate of adversarial examples generated on a source
backbone when they are applied to a different target backbone:

    uv run python scripts/transfer_from_checkpoints.py --device cpu \
        --seeds 7,13 --eval-rows 1500 --epsilon 0.10
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import json
from pathlib import Path

import pandas as pd

from ids_defense_selection import (
    ExperimentConfig,
    build_features,
    generate_adversarial_examples,
    load_checkpoint,
    load_unsw_nb15,
    log_device,
    predict_proba,
    rebuild_trained_defenses,
    resolve_device,
    stratified_subset_indices,
)

BACKBONES = {
    "mlp": ("mlp", "MLP"),
    "cnn1d": ("cnn1d", "1D-CNN"),
    "ft": ("ft_transformer", "FT-Transformer"),
}
CONDITIONS = {"clean": "standard", "pgd_at": "adv_training"}


def transfer_asr(source, target, x_eval, y_eval, config, metadata, device,
                 attack: str, epsilon: float) -> float:
    attacked, _ = generate_adversarial_examples(
        source, x_eval, y_eval, attack, epsilon, config,
        metadata["numeric_mask"], metadata["numeric_mins"], metadata["numeric_maxs"],
        device, config.batch_size, return_inputs=True)
    clean = (predict_proba(target, x_eval, config.batch_size, device) >= 0.5).astype(int)
    adv = (predict_proba(target, attacked, config.batch_size, device) >= 0.5).astype(int)
    labels = y_eval.astype(int)
    correct = clean == labels
    if correct.sum() == 0:
        return float("nan")
    return float((adv[correct] != labels[correct]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--backbones", default="mlp,cnn1d,ft")
    parser.add_argument("--seeds", default="7,13")
    parser.add_argument("--eval-rows", type=int, default=1500)
    parser.add_argument("--attack", default="pgd")
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--alpha-ratio", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    keys = [key.strip().lower() for key in args.backbones.split(",") if key.strip()]
    unknown = [key for key in keys if key not in BACKBONES]
    if unknown:
        print(f"unknown backbone(s): {unknown}; choose from {sorted(BACKBONES)}")
        return 2
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]

    config = ExperimentConfig(
        train_path="data/train.csv", test_path="data/test.csv",
        output_dir=str(Path(args.outputs_root) / "transfer_checkpoint"),
        device=args.device, eval_pgd_steps=args.eval_steps,
        eval_pgd_alpha_ratio=args.alpha_ratio, eval_attack_rows=args.eval_rows,
    )
    device = resolve_device(config.device)
    log_device(config.device, device)
    print(f"[transfer] device={device} seeds={seeds} rows={args.eval_rows} "
          f"attack={args.attack} eps={args.epsilon} steps={args.eval_steps}",
          flush=True)

    train_df, test_df = load_unsw_nb15(config.train_path, config.test_path)
    x_train, y_train, x_test, y_test, metadata = build_features(train_df, test_df)
    indices = stratified_subset_indices(y_test, args.eval_rows,
                                        seed=config.eval_subset_seed)
    x_eval, y_eval = x_test[indices], y_test[indices]

    root = Path(args.outputs_root)
    rows = []
    for seed in seeds:
        models = {}
        for key in keys:
            subdir, label = BACKBONES[key]
            bundle = load_checkpoint(root / subdir / "checkpoints" / f"seed{seed}")
            trained = rebuild_trained_defenses(bundle, device)
            for condition, model_name in CONDITIONS.items():
                models[(key, condition)] = trained.models[model_name]
            print(f"[transfer] seed {seed} {label} loaded", flush=True)
        for condition in CONDITIONS:
            for source in keys:
                for target in keys:
                    if source == target:
                        continue
                    asr = transfer_asr(
                        models[(source, condition)], models[(target, condition)],
                        x_eval, y_eval, config, metadata, device,
                        args.attack, args.epsilon)
                    rows.append({
                        "seed": seed, "condition": condition,
                        "source": BACKBONES[source][1], "target": BACKBONES[target][1],
                        "attack": args.attack, "epsilon": args.epsilon,
                        "asr": asr,
                    })
                    print(f"[transfer] seed {seed} {condition} "
                          f"{BACKBONES[source][1]} -> {BACKBONES[target][1]}: "
                          f"{asr:.4f}", flush=True)

    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["condition", "source", "target"], as_index=False)
               .agg(asr_mean=("asr", "mean"), asr_std=("asr", "std"))
               .fillna(0.0))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "transfer_raw.csv", index=False)
    summary.to_csv(out_dir / "transfer_mean.csv", index=False)
    (out_dir / "transfer_summary.json").write_text(json.dumps({
        "seeds": seeds, "eval_rows": args.eval_rows, "attack": args.attack,
        "epsilon": args.epsilon, "eval_steps": args.eval_steps,
        "alpha_ratio": args.alpha_ratio, "device": str(device),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for condition in CONDITIONS:
        table = summary[summary["condition"] == condition].pivot_table(
            index="source", columns="target", values="asr_mean")
        print(f"\n=== transfer ASR ({condition}, {args.attack} "
              f"eps={args.epsilon:g}) ===")
        print(table.round(4).to_string())
    print(f"\nsaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
