# A Backbone-Conditioned Pareto Analysis Framework
## for Preference-Aware IDS Defense Selection

Reference implementation and evaluation code for the paper:

> **A Backbone-Conditioned Pareto Analysis Framework for Preference-Aware IDS Defense Selection**  
> Xiyuan Huang, Na Zhao, Yi Zheng, Xin Su, Shuang Zhao, Qingquan Liao, Yu Tong  
> *Big Data and Cognitive Computing*

The framework treats adversarial IDS defense selection as a **multi-objective decision problem**.
Each candidate defense is described by a four-dimensional profile, dominated candidates are removed
by Pareto filtering, and a deployment-specific recommendation follows from a preference vector.

---

## Quick start

```bash
git clone https://github.com/huangxyy/bdcc-ids-defense-selection.git
cd bdcc-ids-defense-selection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. put the dataset in data/ (see "Dataset" below), then validate it:
python code/prepare_data.py

# 2. a few seconds: confirms the environment and the data are usable
python code/smoke_test.py

# 3. the full reproduction
python run_experiments.py --device cuda
```

Run `python run_experiments.py --dry-run` to see the exact commands without executing them.

---

## Dataset

The data is **not** shipped with this repository. Download UNSW-NB15 from
<https://research.unsw.edu.au/projects/unsw-nb15-dataset> and place the two official partitions as:

```
data/train.csv    175,341 records    <- official UNSW_NB15_training-set.csv
data/test.csv      82,332 records    <- official UNSW_NB15_testing-set.csv
```

This is the **official split direction**: the larger partition is used for training.
A frequent mistake is to save the two downloaded files under the names train/test without
checking, which silently reverses the split. `code/prepare_data.py` detects that and
`--fix-swap` repairs it:

```bash
python code/prepare_data.py --fix-swap
```

CIC-IDS2017 is optional and only needed by the cross-dataset scripts; see `data/README.md`.

---

## Repository layout

```
bdcc-ids-defense-selection/
├── run_experiments.py                        # one-command reproduction of all three backbones
├── requirements.txt
├── code/
│   ├── prepare_data.py                       # validate / repair the dataset layout
│   ├── smoke_test.py                         # fast end-to-end sanity check
│   ├── ids_core.py                           # shared library, CLI generator, MLP experiment
│   ├── backbone_cnn1d.py                     # 1D-CNN experiment
│   ├── backbone_ft_transformer.py            # FT-Transformer experiment
│   ├── pareto_selection.py                   # Pareto filtering + preference-weighted selection
│   ├── adaptive_attacks.py                   # adaptive evaluation suite
│   ├── eval_worst_class_recall_cnn.py        # phi4, single seed, 1D-CNN
│   ├── eval_worst_class_recall_cnn_5seed.py  # phi4, five seeds, 1D-CNN
│   ├── eval_worst_class_recall_ft.py         # phi4, single seed, FT-Transformer
│   ├── eval_worst_class_recall_ft_5seed.py   # phi4, five seeds, FT-Transformer
│   ├── eval_worst_class_recall_all.py        # phi4 batch runner
│   ├── eval_cross_backbone_transfer.py       # transfer-attack matrix
│   ├── dataset_cicids2017.py                 # CIC-IDS2017 cross-dataset run
│   ├── dataset_cicids2017_source_disjoint.py # source-disjoint CIC-IDS2017 variant
│   ├── sweep_hyperparameters.py              # hyperparameter sensitivity
│   ├── analysis_extended.py                  # effect sizes, ROC, epsilon sweep
│   ├── make_figures_matlab.m                 # MATLAB: Figure 2 (Pareto) and related panels
│   └── plot_style.py                         # shared matplotlib style
├── data/            # place the datasets here (not tracked)
└── outputs/         # experiment outputs (not tracked)
```

### Which script produces which reported result

| Reported item | Script |
|---|---|
| Tables 3-5, four-dimensional profiles | `ids_core.py` (MLP), `backbone_cnn1d.py`, `backbone_ft_transformer.py` |
| Tables 6-7, Pareto and preference selection | `pareto_selection.py` |
| phi4 (worst-class recall) multi-seed values | `eval_worst_class_recall_*_5seed.py` |
| Figure 2 | `make_figures_matlab.m` (MATLAB) |

The remaining scripts (`adaptive_attacks.py`, `dataset_cicids2017*.py`, `eval_cross_backbone_transfer.py`,
`sweep_hyperparameters.py`, `analysis_extended.py`) implement analyses that are **not** part of the
reported tables. They are included for completeness and for follow-up work.

---

## Running the experiments

### Full pipeline

```bash
python run_experiments.py --device cuda                     # MLP, 1D-CNN, FT-Transformer
python run_experiments.py --backbones mlp,cnn --device cuda # a subset
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
run_summary.json          the complete configuration actually used
risk_profile_4d.csv       the four-dimensional profile and the Pareto flag
```

### Worst-class recall (phi4), five seeds

```bash
python code/eval_worst_class_recall_cnn_5seed.py --device cuda --out-dir outputs/phi4_cnn
python code/eval_worst_class_recall_ft_5seed.py  --device cuda --out-dir outputs/phi4_ft
```

Always pass `--out-dir`. Without it the results go to a fixed directory and a second run
overwrites the first.

### Pareto filtering and selection

```bash
python code/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10
```

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
| Random seeds | 7, 13, 21, 42, 100 (set with --seeds) |
| Evaluation subset | 20,000 stratified test samples, fixed seed 2026, shared by all defenses |
| **Adversarial training budget** | **epsilon = 0.06, alpha = 0.015, 20 steps** (FT-Transformer: 7 training steps) |
| Evaluation budgets | epsilon in {0.02, 0.05, 0.10} |

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
> of confusion when comparing numbers.

---

## Full parameterization

**Every field of `ExperimentConfig` is a command-line flag.** The flags are generated from the
dataclass itself, so the CLI cannot drift out of sync with the configuration: adding a field to
`ExperimentConfig` automatically adds a flag. All 43 fields are exposed identically by all three
backbone scripts.

```bash
python code/backbone_cnn1d.py --help              # the full list, with defaults
python code/ids_core.py --train-path data/train.csv --test-path data/test.csv --print-config
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
python code/backbone_ft_transformer.py --seeds 1,2,3 --adv-epsilon 0.09 --print-config
```

A few representative settings:

```bash
# a different MLP architecture and a different evaluation budget list
python code/ids_core.py --train-path data/train.csv --test-path data/test.csv \
    --hidden-dims 256,128,64 --dropout 0.3 --epsilon-list 0.02,0.05,0.10

# three seeds instead of five
python code/backbone_cnn1d.py --seeds 7,42,100

# the epsilon/10 PGD step size written in the manuscript methods section
python code/ids_core.py --eval-pgd-alpha-ratio 0.10
```

### Optional extra defense methods

This repository also contains three defense implementations that are **not** part of the six
defenses the paper evaluates: Progressive Class-Aware Constrained AT, SA-TRADES and
DST-SA-TRADES. They are kept in full so the implementations stay available and reproducible,
but they are only trained when explicitly requested:

```bash
python code/ids_core.py --extra-methods progressive,sa_trades,dst_sa_trades
```

With the default (empty) setting the code trains exactly the six defenses the paper reports and
nothing else. When enabled, each extra method joins the same registries as the six and flows
through the identical evaluation, cost and selection code paths.

---

## Reproducibility

`ids_core.set_seed()` seeds Python, NumPy and PyTorch **and** disables cuDNN non-determinism:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

With these settings two runs with the same seed produce bit-identical results. Without them the
GPU reduction order varies between runs and results differ in the third decimal place, which
matters because several Pareto decisions here hinge on margins of that size.

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