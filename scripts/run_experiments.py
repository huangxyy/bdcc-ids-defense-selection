#!/usr/bin/env python
"""Reproduce the full experiment suite, one backbone after another.

Runs the three backbone experiments in sequence (running them in parallel would
make them compete for the GPU and invalidate the training-cost measurements,
which are part of objective phi3).

    uv run python scripts/run_experiments.py --device cuda
    uv run python scripts/run_experiments.py --backbones mlp,cnn --device cuda
    uv run python scripts/run_experiments.py --dry-run
    uv run python scripts/run_experiments.py --device cuda --seeds 7,13,21,42,100,11,23,37,59,89 --eval-attack-rows 82332 --epsilon-list 0.02,0.05,0.10,0.15 --adaptive-eval
    uv run python scripts/run_experiments.py --backbones ft --ft-capacity medium --device cuda
    uv run python scripts/run_experiments.py --backbones ft --device cuda --set baseline_epochs=20 --set ft_d_token=96 --set ft_n_heads=6

Outputs land in outputs/<backbone>/:
    mean_results.csv, std_results.csv, raw_results.csv,
    significance_tests.csv, efficiency_mean.csv, run_summary.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path when run by path)

import argparse
import difflib
import subprocess
import sys
import time
from pathlib import Path

#: Directory that holds this script and every other entry point.
SCRIPTS = Path(__file__).resolve().parent
from ids_defense_selection.config import ExperimentConfig  # noqa: E402

from ids_defense_selection.paths import (  # noqa: E402
    BACKBONE_OUTPUT_SUBDIRS,
    BACKBONE_RUNNERS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    resolve_path,
)

#: Backbone CLI key -> runner script, canonical output directory and description.
BACKBONES = {
    "mlp": {
        "script": BACKBONE_RUNNERS["mlp"],
        "out": DEFAULT_OUTPUT_ROOT / BACKBONE_OUTPUT_SUBDIRS["mlp"],
        "extra": [],
        "note": "MLP backbone (128-64-32)",
    },
    "cnn": {
        "script": BACKBONE_RUNNERS["cnn"],
        "out": DEFAULT_OUTPUT_ROOT / BACKBONE_OUTPUT_SUBDIRS["cnn"],
        "extra": [],
        "note": "1D-CNN backbone",
    },
    "ft": {
        "script": BACKBONE_RUNNERS["ft"],
        "out": DEFAULT_OUTPUT_ROOT / BACKBONE_OUTPUT_SUBDIRS["ft"],
        "extra": [],
        "note": "FT-Transformer backbone (slowest: allow several hours per seed)",
    },
}


#: FT-Transformer capacity presets.  "paper" is the submitted configuration and
#: the default for every main-table run; medium/large are the E7 capacity
#: ablation (the command is only extended for the ft backbone).
FT_CAPACITY_PRESETS: dict[str, dict[str, int] | None] = {
    "paper": None,
    "medium": {"ft_d_token": 64, "ft_n_heads": 4, "ft_n_layers": 3, "ft_d_ffn": 128},
    "large": {"ft_d_token": 128, "ft_n_heads": 8, "ft_n_layers": 4, "ft_d_ffn": 256},
}

#: Fields the orchestrator owns; --set must not silently clobber them.
RESERVED_SET_FIELDS = frozenset({"train_path", "test_path", "output_dir", "device"})
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def resolve_scale(args) -> dict[str, object]:
    """Collect the explicit scaling flags; unset values use the backbone config."""
    return {
        "seeds": args.seeds,
        "epsilon_list": args.epsilon_list,
        "eval_attack_rows": args.eval_attack_rows,
        "adaptive_eval": bool(args.adaptive_eval),
        "adaptive_steps": args.adaptive_steps,
        "adaptive_restarts": args.adaptive_restarts,
    }


def parse_overrides(items: list[str]) -> tuple[dict[str, str], list[str]]:
    """Parse ``--set name=value`` into config fields; return (overrides, errors).

    Any field of :class:`ExperimentConfig` is accepted, using either its field
    name (``baseline_epochs``) or its flag spelling (``baseline-epochs``).
    Values are forwarded to the backbone parsers, which own the type conversion,
    so ``--set hidden_dims=512,256,128`` and ``--set epsilon_list=0.02,0.05``
    behave exactly like the dedicated flags.
    """
    valid = set(ExperimentConfig.__dataclass_fields__)
    overrides: dict[str, str] = {}
    errors: list[str] = []
    for item in items:
        if "=" not in item:
            errors.append(f"--set expects name=value, got {item!r}")
            continue
        name, value = item.split("=", 1)
        name = name.strip().replace("-", "_")
        value = value.strip()
        if name not in valid:
            close = difflib.get_close_matches(name, sorted(valid), n=3, cutoff=0.6)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            errors.append(f"unknown config field {name!r}{hint}")
            continue
        if name in RESERVED_SET_FIELDS:
            errors.append(f"--set {name} is not allowed; use --data-dir / --device instead")
            continue
        if not value:
            errors.append(f"--set {name} needs a value, got {item!r}")
            continue
        if name == "adaptive_eval" and value.lower() not in TRUE_VALUES | FALSE_VALUES:
            errors.append(
                f"--set adaptive_eval expects one of {sorted(TRUE_VALUES | FALSE_VALUES)}, got {value!r}"
            )
            continue
        overrides[name] = value
    return overrides, errors


def build_cmd(key: str, data_dir: Path, device: str, scale: dict[str, object],
              ft_capacity: str, overrides: dict[str, str]) -> list[str]:
    spec = BACKBONES[key]
    overridden = set(overrides)
    cmd = [sys.executable, "-u", str(SCRIPTS / spec["script"]),
           "--train-path", str(data_dir / "train.csv"),
           "--test-path", str(data_dir / "test.csv"),
           "--output-dir", str(spec["out"]),
           "--device", device,
           "--training-budget-mode", "matched_continuation"]
    if scale["seeds"] and "seeds" not in overridden:
        cmd += ["--seeds", str(scale["seeds"])]
    if scale["epsilon_list"] and "epsilon_list" not in overridden:
        cmd += ["--epsilon-list", str(scale["epsilon_list"])]
    if scale["eval_attack_rows"] is not None and "eval_attack_rows" not in overridden:
        cmd += ["--eval-attack-rows", str(scale["eval_attack_rows"])]
    if scale["adaptive_eval"] and "adaptive_eval" not in overridden:
        cmd.append("--adaptive-eval")
    if scale["adaptive_steps"] is not None and "adaptive_steps" not in overridden:
        cmd += ["--adaptive-steps", str(scale["adaptive_steps"])]
    if scale["adaptive_restarts"] is not None and "adaptive_restarts" not in overridden:
        cmd += ["--adaptive-restarts", str(scale["adaptive_restarts"])]
    if key == "ft":
        preset = FT_CAPACITY_PRESETS[ft_capacity]
        if preset:
            for field, value in preset.items():
                if field not in overridden:
                    cmd += ["--" + field.replace("_", "-"), str(value)]
    for name, value in overrides.items():
        flag = "--" + name.replace("_", "-")
        if name == "adaptive_eval":
            if value.lower() in TRUE_VALUES:
                cmd.append(flag)
        else:
            cmd += [flag, value]
    cmd += spec["extra"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", default="mlp,cnn,ft",
                    help="comma-separated subset of: mlp, cnn, ft")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--device", default="auto",
                    help="torch device passed to every backbone: auto, cpu, cuda, cuda:N or mps")
    ap.add_argument("--ft-capacity", choices=tuple(FT_CAPACITY_PRESETS), default="paper",
                    help="FT-Transformer capacity: paper = submitted model, "
                         "medium/large = capacity-ablation presets")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds forwarded to every backbone (e.g. a 10-seed run)")
    ap.add_argument("--epsilon-list", default=None,
                    help="comma-separated evaluation budgets forwarded to every backbone")
    ap.add_argument("--eval-attack-rows", type=int, default=None,
                    help="evaluation subset size forwarded to every backbone "
                         "(use the test-split size for a full-test evaluation)")
    ap.add_argument("--adaptive-eval", action="store_true",
                    help="forward --adaptive-eval to every backbone (expensive)")
    ap.add_argument("--adaptive-steps", type=int, default=None,
                    help="forwarded adaptive-attack steps")
    ap.add_argument("--adaptive-restarts", type=int, default=None,
                    help="forwarded adaptive-attack restarts")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="FIELD=VALUE",
                    help="override any ExperimentConfig field (repeatable), e.g. "
                         "--set batch_size=2048 --set ft_d_token=96; wins over the other flags")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    data_dir = resolve_path(args.data_dir)
    args.data_dir = str(data_dir)
    scale = resolve_scale(args)
    overrides, override_errors = parse_overrides(args.overrides)
    if override_errors:
        for message in override_errors:
            print(f"error: {message}", file=sys.stderr)
        return 2
    if overrides.get("adaptive_eval", "").lower() in TRUE_VALUES:
        scale["adaptive_eval"] = True

    keys = [k.strip().lower() for k in args.backbones.split(",") if k.strip()]
    unknown = [k for k in keys if k not in BACKBONES]
    if unknown:
        print(f"unknown backbone(s): {unknown}. choose from {list(BACKBONES)}")
        return 2
    adaptive_tuning_given = any(
        name in overrides or getattr(args, name) is not None
        for name in ("adaptive_steps", "adaptive_restarts")
    )
    if adaptive_tuning_given and not scale["adaptive_eval"]:
        print("warning: --adaptive-steps/--adaptive-restarts have no effect "
              "without --adaptive-eval", file=sys.stderr)
    if args.ft_capacity != "paper" and "ft" not in keys:
        print("warning: --ft-capacity is ignored because the ft backbone is not selected",
              file=sys.stderr)

    print("=" * 78)
    print("Backbone-conditioned Pareto analysis -- full reproduction")
    print("=" * 78)
    print(f"  backbones : {', '.join(keys)}")
    print(f"  data dir  : {args.data_dir}")
    print(f"  device    : {args.device}")
    print(f"  ft capacity: {args.ft_capacity}")
    print(f"  seeds     : {scale['seeds'] or 'backbone default'}")
    print(f"  eval rows : {scale['eval_attack_rows'] or 'backbone default'}")
    print(f"  epsilons  : {scale['epsilon_list'] or 'backbone default'}")
    print(f"  adaptive  : {'on' if scale['adaptive_eval'] else 'off'}"
          + (f" ({scale['adaptive_steps']} steps x {scale['adaptive_restarts']} restarts)"
             if scale["adaptive_eval"] and scale["adaptive_steps"] and scale["adaptive_restarts"]
             else ""))
    if overrides:
        print("  overrides : " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
    print()

    # Step 0: dataset layout. Never start a multi-hour run on a reversed split.
    print("-- step 0: dataset check " + "-" * 46)
    if args.dry_run:
        print("   (dry run -- dataset check skipped)")
    else:
        rc = subprocess.call([sys.executable, str(SCRIPTS / "prepare_data.py"), "--data-dir", args.data_dir])
        if rc != 0:
            print("\nDataset is not ready. Fix it and re-run.")
            return rc
    print()

    t_all = time.perf_counter()
    done: list[tuple[str, float, str]] = []
    for i, key in enumerate(keys, 1):
        spec = BACKBONES[key]
        cmd = build_cmd(key, data_dir, args.device, scale, args.ft_capacity, overrides)
        print(f"-- step {i}/{len(keys)}: {key} -- {spec['note']} " + "-" * 20)
        print("   " + " ".join(cmd))
        if args.dry_run:
            print("   (dry run -- not executed)")
            continue
        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"   FAILED (exit {rc}) after {dt/60:.1f} min")
            return rc
        print(f"   done in {dt/60:.1f} min -> {spec['out']}/")
        done.append((key, dt, str(spec["out"])))
        print()

    if args.dry_run:
        print("dry run complete.")
        return 0

    print("=" * 78)
    print(f"All done in {(time.perf_counter()-t_all)/60:.1f} min")
    for key, dt, out in done:
        print(f"  {key:<5} {dt/60:>7.1f} min   {out}/")
    print()
    print("Next steps")
    print("  uv run python scripts/evaluate_phi4_cnn.py --device cuda --output-dir outputs/phi4_cnn")
    print("  uv run python scripts/evaluate_phi4_ft.py  --device cuda --output-dir outputs/phi4_ft")
    print("  uv run python scripts/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
