# A Backbone-Conditioned Pareto Analysis Framework
## for Preference-Aware IDS Defense Selection

**English** | [简体中文](README.zh-CN.md)

Reference implementation and evaluation code for the paper:

> **A Backbone-Conditioned Pareto Analysis Framework for Preference-Aware IDS Defense Selection**  
> Xiyuan Huang, Na Zhao, Yi Zheng, Xin Su, Shuang Zhao, Qingquan Liao, Yu Tong  
> *Big Data and Cognitive Computing*

The framework treats adversarial IDS defense selection as a **multi-objective decision problem**.
Each candidate defense is described by a four-dimensional profile, dominated candidates are removed
by Pareto filtering, and a deployment-specific recommendation follows from a preference vector.

---

## Quick start

The project is managed with [uv](https://docs.astral.sh/uv/). Install it first
(`curl -LsSf https://astral.sh/uv/install.sh | sh`, or any of the methods in the
uv documentation), then:

```bash
git clone https://github.com/huangxyy/bdcc-ids-defense-selection.git
cd bdcc-ids-defense-selection

# create the environment and install the locked dependencies from uv.lock
uv sync

# 1. the dataset ships with the repo (see "Dataset" below); validate it:
uv run python scripts/prepare_data.py

# 2. a few seconds: confirms the environment and the data are usable
uv run python scripts/smoke_test.py

# 3. the full reproduction
uv run python scripts/run_experiments.py

# 4. verify the checkout end to end (data, lint, tests, smoke test, dry run)
make verify
```

`uv run` executes the command inside the project environment, so no manual activation is
needed; every `python ...` example below is meant to be run the same way
(`uv run python ...`). The interpreter is pinned by `.python-version` and uv installs it
automatically when missing.

All paths are resolved against the repository root, so every documented command can be
started from any working directory.

Run `uv run python scripts/run_experiments.py --dry-run` to see the exact commands without executing them.

`make verify` is the single entry point for validation; `make quick` runs the fast subset
(data + lint + tests) in about ten seconds. The verification levels, the manuscript-to-command
map and the acceptance criteria are in [docs/verification.md](docs/verification.md); the layout
conventions are in [docs/structure.md](docs/structure.md).

---

## Running on a GPU server

The interpreter is pinned by `.python-version` (`3.12`) and the dependencies by `uv.lock`, so a
fresh server only needs uv — no root and no manual CUDA toolchain setup:

Package downloads default to the **Tsinghua PyPI mirror** (configured in `pyproject.toml`), so a
server in mainland China does not wait for `pypi.org`; outside China override it with
`UV_DEFAULT_INDEX=https://pypi.org/simple uv sync`. Both server workflows have a one-command
shortcut:

```bash
bash scripts/setup_server.sh          # uv sync (mirror-aware, CPython 3.12 + locked deps)
bash scripts/setup_server.sh --pip    # reuse an existing conda torch: pip-install the other deps
```

```bash
# once, no root required
curl -LsSf https://astral.sh/uv/install.sh | sh

cd bdcc-ids-defense-selection
uv sync                       # downloads CPython 3.12 + the locked CUDA build of torch
uv run python scripts/check_devices.py
```

`check_devices.py` prints the torch CUDA build, every visible GPU (name, memory, compute
capability) and the device `auto` resolves to; the same value is recorded in every
`run_summary.json` as `resolved_device`, so a silent CPU fallback cannot happen.

If `check_devices.py` reports no usable GPU while `nvidia-smi` shows one, the installed torch
wheel does not match the driver. Install the wheel for your driver's CUDA version:

```bash
# example: CUDA 13.0 wheels (RTX 50-series / Blackwell need a recent CUDA build)
uv pip install --python .venv/bin/python --reinstall torch \
    --index-url https://download.pytorch.org/whl/cu130
uv run python scripts/check_devices.py     # expect "available    : True"
```

Other indices (`cu128`, `cu126`, `cu118`, ...) live at <https://download.pytorch.org/whl/>.
A wheel installed this way is outside `uv.lock`; a later plain `uv sync` restores the locked
torch, so re-run the command if the GPU disappears again. As a sanity check, a machine with
driver 595 / CUDA 13.2 and an RTX 5090 works with the default `uv sync` install.

---

## Dataset

The two official UNSW-NB15 partitions **ship with this repository**, already stripped of the
byte-order mark and in the official split direction:

- **[data/train.csv](data/train.csv)** — 175,341 records (the official training partition)
- **[data/test.csv](data/test.csv)** — 82,332 records (the official testing partition)

`git clone` / `git pull` is therefore all a new machine (or GPU server) needs before running the
experiments. The upstream source is
<https://research.unsw.edu.au/projects/unsw-nb15-dataset>; if you download the partitions
yourself, place them under the same names:

```
data/train.csv    175,341 records    <- official UNSW_NB15_training-set.csv
data/test.csv      82,332 records    <- official UNSW_NB15_testing-set.csv
```

This is the **official split direction**: the larger partition is used for training.
A frequent mistake is to save the two downloaded files under the names train/test without
checking, which silently reverses the split. `scripts/prepare_data.py` detects that and
`--fix-swap` repairs it:

```bash
uv run python scripts/prepare_data.py --fix-swap
```

CIC-IDS2017 is **not** shipped (it is optional and only needed by the cross-dataset scripts);
see `data/README.md` (also available in [简体中文](data/README.zh-CN.md)).

---

## Repository layout

```
bdcc-ids-defense-selection/
├── pyproject.toml                            # project metadata + dependencies (uv)
├── uv.lock                                   # fully pinned dependency lock
├── .python-version                           # interpreter version used by uv
├── Makefile                                  # make verify / test / lint / smoke / dry-run
├── docs/
│   ├── structure.md                          # repository layout and conventions
│   └── verification.md                       # verification levels, commands, artefact map
├── src/
│   └── ids_defense_selection/                # the library (everything importable)
│       ├── config.py                         # ExperimentConfig + generated CLI
│       ├── data.py                           # feature pipeline, loaders, seeding
│       ├── attacks.py                        # FGSM / PGD / C&W / APGD + constraints
│       ├── defenses.py                       # six defenses, optional extras, masks
│       ├── evaluation.py                     # metrics, prediction, shared eval loop
│       ├── reporting.py                      # aggregation, significance tests, figures
│       ├── selection.py                      # Pareto filtering + preference-weighted selection
│       ├── backbones.py                      # MLP / 1D-CNN / FT-Transformer models
│       ├── experiment.py                     # the MLP end-to-end pipeline
│       ├── phi4.py                           # worst-class recall evaluation
│       ├── adaptive.py                       # adaptive attack suite
│       └── style.py                          # shared matplotlib style + palette
├── scripts/                                  # entry points (CLI + experiment orchestration)
│   ├── _bootstrap.py                         # puts src/ on sys.path for every script
│   ├── run_experiments.py                    # one-command reproduction of all three backbones
│   ├── run_mlp.py                            # MLP experiment
│   ├── run_cnn1d.py                          # 1D-CNN experiment
│   ├── run_ft_transformer.py                 # FT-Transformer experiment
│   ├── evaluate_phi4_cnn.py                  # phi4 (worst-class recall), 1D-CNN
│   ├── evaluate_phi4_ft.py                   # phi4 (worst-class recall), FT-Transformer
│   ├── pareto_selection.py                   # CLI for ids_defense_selection.selection
│   ├── evaluate_cross_backbone_transfer.py   # transfer-attack matrix
│   ├── run_cicids2017.py                     # CIC-IDS2017 cross-dataset run
│   ├── run_cicids2017_source_disjoint.py     # source-disjoint CIC-IDS2017 variant
│   ├── sweep_hyperparameters.py              # hyperparameter sensitivity
│   ├── analyze_extended.py                   # epsilon sweep, ROC, gradient masking
│   ├── prepare_data.py                       # validate / repair the dataset layout
│   ├── smoke_test.py                         # fast end-to-end sanity check
│   ├── check_devices.py                      # CPU / CUDA / MPS availability report
│   ├── verify.py                             # one-command verification (make verify)
│   ├── check_outputs.py                      # validate a completed outputs/ tree
│   ├── setup_server.sh                       # one-command server setup (mirror-aware)
│   └── make_figures_matlab.m                 # MATLAB: Figure 2 (Pareto) and related panels
├── tests/                                    # unit, integration and structure tests
├── data/                                     # UNSW-NB15 partitions (shipped)
└── outputs/                                  # experiment outputs (ignored; see outputs/README.md)
```

### Which script produces which reported result

| Reported item | Script |
|---|---|
| Tables 3-5, four-dimensional profiles | `run_mlp.py`, `run_cnn1d.py`, `run_ft_transformer.py` |
| Tables 6-7, Pareto and preference selection | `pareto_selection.py` |
| phi4 (worst-class recall) multi-seed values | `evaluate_phi4_cnn.py`, `evaluate_phi4_ft.py` |
| Figure 2 | `make_figures_matlab.m` (MATLAB) |

The remaining scripts (`run_cicids2017*.py`, `evaluate_cross_backbone_transfer.py`,
`sweep_hyperparameters.py`, `analyze_extended.py`) implement analyses that are **not** part of the
reported tables. The adaptive attack suite lives in
`src/ids_defense_selection/adaptive.py`; it runs as part of the evaluation
(`evaluate_defenses`) whenever `--adaptive-eval` is passed, and `smoke_test.py` exercises it too.

---

## Running the experiments

### Device selection

`--device` accepts `auto` (the default), `cpu`, `cuda`, `cuda:N` and `mps`. `auto` picks CUDA
when a GPU is visible, then Apple MPS, then CPU; an explicit `cuda` request on a CPU-only
machine fails immediately with a clear message instead of a deep torch error. The resolved
device is printed at startup and recorded as `resolved_device` in `run_summary.json`.

```bash
uv run python scripts/check_devices.py                  # inspect CPU / CUDA / MPS
uv run python scripts/run_experiments.py                     # auto-selects the device
uv run python scripts/run_mlp.py --device cpu           # pin CPU
uv run python scripts/run_mlp.py --device cuda:1        # pin a specific GPU
```

### Full pipeline

```bash
uv run python scripts/run_experiments.py --device cuda                     # MLP, 1D-CNN, FT-Transformer
uv run python scripts/run_experiments.py --backbones mlp,cnn --device cuda # a subset
```

### Large-GPU runs (RTX 5090 and similar)

```bash
uv run python scripts/run_experiments.py --backbones ft --ft-capacity medium --device cuda
uv run python scripts/run_experiments.py --device cuda \
  --seeds 7,13,21,42,100,11,23,37,59,89 --eval-attack-rows 82332 \
  --epsilon-list 0.02,0.05,0.10,0.15 --adaptive-eval \
  --adaptive-steps 100 --adaptive-restarts 10
```

The example scales the *evidence* while keeping the manuscript's training protocol
(10+8 epochs, eps_train = 0.06, batch size): ten seeds, the full 82,332-row test partition,
four budgets, and 100-step x 10-restart adaptive attacks. `--ft-capacity medium|large`
switches the FT-Transformer to the capacity-ablation presets (d_token 64/4 heads/3
layers/d_ffn 128, or 128/8/4/256); main runs keep the submitted `paper` capacity.

Every one of the 55 configuration fields can be overridden in the orchestrator with a
repeatable `--set field=value` (it wins over the other flags and presets):

```bash
uv run python scripts/run_experiments.py --device cuda \
  --seeds 7,13,21,42,100,11,23,37,59,89 --eval-attack-rows 82332 \
  --set baseline_epochs=20 --set adv_epochs=16 --set batch_size=2048 \
  --set ft_d_token=96 --set ft_n_heads=6 --set ft_n_layers=4 --set ft_d_ffn=384 \
  --set learning_rate=2e-3 --set adv_epsilon=0.09
```

### Train once, evaluate many times

`--save-checkpoints` stores the shared baseline and all six defenses (plus the
feature masks, cost table and the exact config) under
`<output-dir>/checkpoints/seed<seed>/`. Evaluation variants then rerun without
retraining:

```bash
uv run python scripts/run_ft_transformer.py --device cuda --save-checkpoints \
  --output-dir outputs/ft_transformer

uv run python scripts/evaluate_checkpoints.py \
  --checkpoint outputs/ft_transformer/checkpoints/seed7 \
  --output-dir outputs/ft_eval_probe --device cuda \
  --eval-pgd-alpha-ratio 0.10 --eval-pgd-steps 50 \
  --epsilon-list 0.05,0.10,0.20 --adaptive-eval
```

Backbones run **sequentially on purpose**: running them in parallel makes them compete for the
GPU, which would corrupt the training-cost measurements feeding objective `phi3`.

Each run writes into `outputs/<backbone>/`:

```
mean_results.csv          per (defense, attack, epsilon): mean over seeds
std_results.csv           same cells, standard deviation over seeds
raw_results.csv           one row per seed
significance_tests.csv    paired t-test and Wilcoxon between defenses
efficiency_mean.csv       parameter count, training seconds, inference latency
attack_generalization.csv robustness to attacks that were never used in training
hyperparameters.csv       every ExperimentConfig field (group, value, help)
run_summary.json          the complete configuration actually used
risk_profile_4d.csv       the four-dimensional profile and the Pareto flag
```

The attack applied to the test partition after training is configurable as well:
`--full-test-attack-settings pgd:0.10` selects the attack and its budget, and
`--full-test-attack-rows 0` (the default) attacks the complete partition.

The adaptive attack suite (restart PGD, gradient-free NES and the complement attack against
masked defenses) is opt-in because it is expensive: add `--adaptive-eval` (optionally
`--adaptive-steps`, `--adaptive-restarts`, `--adaptive-epsilon`) and the run also writes
`adaptive_attack_raw.csv` / `adaptive_attack_mean.csv`.

### Worst-class recall (phi4), five seeds

```bash
uv run python scripts/evaluate_phi4_cnn.py --device cuda --output-dir outputs/phi4_cnn
uv run python scripts/evaluate_phi4_ft.py  --device cuda --output-dir outputs/phi4_ft
```

Both scripts default to the five study seeds; pass `--seeds 42` for the single-seed
estimate, or `--seeds 7,13` for a subset. The FT-Transformer trains with `--adv-steps 7`
and evaluates with `--eval-pgd-steps 20`; the two budgets are recorded separately in
`run_summary.json`.

### Pareto filtering and selection

```bash
uv run python scripts/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10
```

The decision step now accounts for run-to-run variation and can enforce minimum standards:

```bash
# uncertainty-aware Pareto (mean +/- 1 std), drop candidates below the thresholds
uv run python scripts/pareto_selection.py --confidence-margin 1.0 --min-phi2 0.80 --min-phi4 0.10
```

`risk_profile_4d.csv` carries the per-objective standard deviations and both the
uncertainty-aware and the point-estimate Pareto flags; `outputs/decision_settings.json`
records the thresholds and margin that produced the recommendations.

### Expected runtime

Single-GPU timings measured on an RTX 4060 Laptop (8 GB), full official split:

| Backbone | one seed, six defenses | five seeds |
|---|---|---|
| MLP | ~8 min | ~40 min |
| 1D-CNN | ~15 min | ~1.3 h |
| FT-Transformer | ~2.8 h | ~14 h |

The FT-Transformer dominates the total cost. All backbones also run on CPU, substantially slower.

---

## Hyperparameters

| Setting | Value |
|---|---|
| Protocol | matched continuation: 10 clean epochs, then 8 epochs |
| Optimizer | Adam, lr = 1e-3, weight decay = 1e-5 |
| Batch size | 1024 (MLP, 1D-CNN); 512 (FT-Transformer) |
| Dropout | 0.15 |
| FT-Transformer capacity | d_token = 32, heads = 2, layers = 2, d_ffn = 64 (`--ft-d-token`, `--ft-n-heads`, `--ft-n-layers`, `--ft-d-ffn`) |
| Checkpoints | off by default; `--save-checkpoints` writes `<output-dir>/checkpoints/seed<seed>/` (all defenses + metadata) for evaluation-only reruns |
| Random seeds | 7, 13, 21, 42, 100 (set with --seeds) |
| Evaluation subset | 20,000 stratified test samples (`--eval-attack-rows`), fixed seed 2026 (`--eval-subset-seed`), shared by all defenses |
| **Adversarial training budget** | **epsilon = 0.06, alpha = 0.015, 20 steps** (FT-Transformer: 7 training steps) |
| Evaluation budgets | epsilon in {0.02, 0.05, 0.10} |
| Evaluation PGD steps | 20 (`--eval-pgd-steps`), independent of the training budget |
| Device | `auto` (CUDA → MPS → CPU); pin with `--device cpu`, `cuda` or `cuda:N` |

| Defense | Training-time attack | Mask | Extra |
|---|---|---|---|
| Standard | none | - | - |
| PGD-AT | PGD | all 39 continuous features | - |
| Constrained | PGD | top-30% most sensitive features | - |
| TRADES | PGD on KL | all continuous | beta = 6.0 |
| Free AT | replayed PGD | all continuous | m = 4 replays |
| Class-Aware | PGD | class-wise top-30% | minority weight = 3.0 |

Evaluation attacks: FGSM; PGD (20 steps; step size set by `eval_pgd_alpha_ratio`, default 0.05 = eps/20);
C&W L2 (30 steps, lr = 0.01, c = 1.0); APGD-CE (50 steps, rho = 0.75).

> **The training and evaluation budgets differ.** Adversarial training uses epsilon = 0.06 while
> evaluation sweeps 0.02 / 0.05 / 0.10. This is stated explicitly because it is a frequent source
> of confusion when comparing numbers. The two PGD step counts are separate fields
> (`--adv-steps` for training, `--eval-pgd-steps` for evaluation), so a backbone can never
> silently evaluate with its reduced training budget.

---

## Full parameterization

**Every field of `ExperimentConfig` is a command-line flag.** The flags are generated from the
dataclass itself, so the CLI cannot drift out of sync with the configuration: adding a field to
`ExperimentConfig` automatically adds a flag. All 56 fields are exposed identically by all three
backbone scripts.

Each field also carries its own help text and legacy flag aliases as dataclass metadata, and the
values are validated as soon as the configuration is built: a typo such as `--batch-size 0`, an
unknown attack name or a malformed tuple fails immediately with one readable message that lists
every problem found.

For maintenance and for reading an experiment at a glance, the same fields are exposed as seven
immutable groups — `paths`, `training`, `attack`, `evaluation`, `methods`, `sensitivity` and
`runtime` — and each group validates its own fields. `--print-config` and `run_summary.json` print
the configuration in that structure, and the same views are available in Python
(`config.training.batch_size`, `config.evaluation.epsilon_list`, `config.runtime.device`, ...)
while the flat access used by the code (`config.batch_size`) keeps working unchanged.

```bash
uv run python scripts/run_cnn1d.py --help              # the full list, with defaults
uv run python scripts/run_mlp.py --train-path data/train.csv --test-path data/test.csv --print-config
```

Complex types are parsed from comma-separated strings:

| Field type | Flag example |
|---|---|
| `int` / `float` / `str` | `--adv-epsilon 0.05` |
| `tuple[int, ...]` | `--hidden-dims 256,128,64` |
| `tuple[float, ...]` | `--epsilon-list 0.02,0.05,0.10` |
| `tuple[str, ...]` | `--reference-models log_reg,hist_gbdt` |
| `tuple[tuple[str, float], ...]` | `--transfer-attack-settings fgsm:0.05,pgd:0.10` |

Anything not passed keeps its dataclass default, and `--print-config` dumps the effective
configuration as JSON before any work starts:

```bash
uv run python scripts/run_ft_transformer.py --seeds 1,2,3 --adv-epsilon 0.09 --print-config
```

A few representative settings:

```bash
# a different MLP architecture and a different evaluation budget list
uv run python scripts/run_mlp.py --train-path data/train.csv --test-path data/test.csv \
    --hidden-dims 256,128,64 --dropout 0.3 --epsilon-list 0.02,0.05,0.10

# three seeds instead of five
uv run python scripts/run_cnn1d.py --seeds 7,42,100

# the epsilon/10 PGD step size written in the manuscript methods section
uv run python scripts/run_mlp.py --eval-pgd-alpha-ratio 0.10
```

### Optional extra defense methods

This repository also contains three defense implementations that are **not** part of the six
defenses the paper evaluates: Progressive Class-Aware Constrained AT, SA-TRADES and
DST-SA-TRADES. They are kept in full so the implementations stay available and reproducible,
but they are only trained when explicitly requested:

```bash
uv run python scripts/run_mlp.py --extra-methods progressive,sa_trades,dst_sa_trades
```

With the default (empty) setting the code trains exactly the six defenses the paper reports and
nothing else. When enabled, each extra method joins the same registries as the six and flows
through the identical evaluation, cost and selection code paths.

---

## Reproducibility

`ids_defense_selection.set_seed()` seeds Python, NumPy and PyTorch **and** disables cuDNN non-determinism:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

With these settings two runs with the same seed produce bit-identical results. Without them the
GPU reduction order varies between runs and results differ in the third decimal place, which
matters because several Pareto decisions here hinge on margins of that size.

---

## Development

The library lives in `src/ids_defense_selection/`; the scripts in `scripts/` are thin entry points.
Anything that should be unit-tested belongs in the library (for example, the Pareto decision logic
is `ids_defense_selection/selection.py`, and `scripts/pareto_selection.py` is only its CLI).
Install the development tools and run the checks from the repository root:

```bash
uv sync --group dev
make quick             # dataset check + lint + tests (about ten seconds)
make verify            # the same plus the real-data smoke test and the dry run
uv run pytest          # pytest directly
uv run ruff check      # lint / undefined-name checks
```

The tests build a small synthetic UNSW-NB15-shaped dataset, so they need neither the real data
nor a GPU. `tests/test_structure.py` enforces the layout conventions, so a new script that cannot
run from an arbitrary working directory is caught by the suite.

---

## Notes on the feature space

The one-hot width depends on how many categorical levels the **training** partition contains, so it
is a property of the split rather than a fixed constant. With the official split this code produces
**194 transformed features (39 continuous + 155 one-hot)**. The manuscript reports 190, which
corresponds to the reversed split used in the submitted experiments. `smoke_test.py` prints the
value it actually obtains, so the two can never be silently confused.

---

## Citation

```bibtex
@article{huang2026backbone,
  title   = {A Backbone-Conditioned Pareto Analysis Framework for
             Preference-Aware IDS Defense Selection},
  author  = {Huang, Xiyuan and Zhao, Na and Zheng, Yi and Su, Xin and
             Zhao, Shuang and Liao, Qingquan and Tong, Yu},
  journal = {Big Data and Cognitive Computing},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.

## Contact

Xiyuan Huang - huangxiyuan2020@qq.com - [@huangxyy](https://github.com/huangxyy)

Questions, bug reports and pull requests are welcome through the issue tracker at
<https://github.com/huangxyy/bdcc-ids-defense-selection/issues>.
