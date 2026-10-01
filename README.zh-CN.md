# 面向偏好感知 IDS 防御选择的骨干条件化帕累托分析框架

[English](README.md) | **简体中文**

论文的参考实现与评测代码：

> **A Backbone-Conditioned Pareto Analysis Framework for Preference-Aware IDS Defense Selection**
> （面向偏好感知 IDS 防御选择的骨干条件化帕累托分析框架）
> Xiyuan Huang, Na Zhao, Yi Zheng, Xin Su, Shuang Zhao, Qingquan Liao, Yu Tong
> *Big Data and Cognitive Computing*

本框架把对抗场景下的 IDS 防御选择建模为**多目标决策问题**：每种候选防御用四维画像描述，
先通过帕累托过滤剔除被支配的候选，再根据部署偏好向量给出推荐。

---

## 快速开始

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖。请先安装 uv
（`curl -LsSf https://astral.sh/uv/install.sh | sh`，或参考 uv 文档中的其它安装方式），然后：

```bash
git clone https://github.com/huangxyy/bdcc-ids-defense-selection.git
cd bdcc-ids-defense-selection

# 创建环境，并按 uv.lock 安装锁定版本的依赖
uv sync

# 1. 数据集已随仓库提供（见下文"数据集"），先校验：
uv run python code/prepare_data.py

# 2. 几秒钟：确认环境与数据可用
uv run python code/smoke_test.py

# 3. 完整复现
uv run python run_experiments.py
```

`uv run` 会在项目环境中执行命令，无需手动激活环境；下文所有 `python ...` 示例都应以同样方式运行
（即 `uv run python ...`）。解释器版本由 `.python-version` 固定，缺失时 uv 会自动安装。

所有路径都相对仓库根目录解析，因此下文的每条命令都可以在任意工作目录下执行。

运行 `uv run python run_experiments.py --dry-run` 可以只查看将要执行的命令而不真正运行。

---

## 在 GPU 服务器上运行

解释器由 `.python-version`（`3.12`）固定、依赖由 `uv.lock` 固定，所以一台新服务器只需要装
uv，不需要 root，也不用自己配 CUDA 工具链：

```bash
# 只需要执行一次，无需 root
curl -LsSf https://astral.sh/uv/install.sh | sh

cd bdcc-ids-defense-selection
uv sync                       # 自动下载 CPython 3.12 与锁定的 CUDA 版 torch
uv run python code/check_devices.py
```

`check_devices.py` 会打印 torch 的 CUDA 构建、每块可见 GPU（型号/显存/算力）以及 `auto`
解析到的设备；同样的值会写进每次运行的 `run_summary.json`（`resolved_device`），因此不会
出现悄悄退回 CPU 的情况。

如果 `check_devices.py` 显示没有可用 GPU、而 `nvidia-smi` 能看到显卡，说明 torch 的 CUDA
构建与驱动不匹配，改装对应 CUDA 版本的轮子：

```bash
# 例：CUDA 13.0 版本（RTX 50 系 / Blackwell 需要较新的 CUDA 构建）
uv pip install --python .venv/bin/python --reinstall torch \
    --index-url https://download.pytorch.org/whl/cu130
uv run python code/check_devices.py     # 期望 "available    : True"
```

其它版本（`cu128`、`cu126`、`cu118` 等）见 <https://download.pytorch.org/whl/>。
这种安装方式不受 `uv.lock` 约束，之后再执行普通的 `uv sync` 会恢复锁定的 torch；
如果 GPU 又不可用，重跑上面的命令即可。实测参考：驱动 595 / CUDA 13.2 + RTX 5090
可以直接使用默认的 `uv sync` 安装。

---

## 数据集

UNSW-NB15 的两个官方划分**已随仓库提供**（已去掉 BOM，方向为官方方向）：

- **[data/train.csv](data/train.csv)** —— 175,341 条（官方训练划分）
- **[data/test.csv](data/test.csv)** —— 82,332 条（官方测试划分）

因此新机器（或 GPU 服务器）只需 `git clone` / `git pull` 就能开跑，无需再单独下载。
上游来源：<https://research.unsw.edu.au/projects/unsw-nb15-dataset>；如果你自行下载，
请保持同样的文件名：

```
data/train.csv    175,341 条    <- 官方 UNSW_NB15_training-set.csv
data/test.csv      82,332 条    <- 官方 UNSW_NB15_testing-set.csv
```

这是**官方的划分方向**：较大的划分用于训练。一个常见错误是把下载的两个文件直接命名为
train/test 而没有确认哪个是哪个，这会悄悄把划分方向反过来。`code/prepare_data.py`
能够检测出这种情况，并用 `--fix-swap` 修复：

```bash
uv run python code/prepare_data.py --fix-swap
```

CIC-IDS2017 **不在仓库中**（可选，只有跨数据集脚本需要）；详见
[data/README.zh-CN.md](data/README.zh-CN.md)。

---

## 仓库结构

```
bdcc-ids-defense-selection/
├── pyproject.toml                            # 项目元数据与依赖（uv）
├── uv.lock                                   # 完整锁定的依赖版本
├── .python-version                           # uv 使用的解释器版本
├── run_experiments.py                        # 一条命令复现三个骨干网络
├── code/
│   ├── ids_defense_selection/                # 库（所有可导入的实现）
│   │   ├── config.py                         # ExperimentConfig 与自动生成的 CLI
│   │   ├── data.py                           # 特征工程、数据加载、随机种子
│   │   ├── attacks.py                        # FGSM / PGD / C&W / APGD 与约束
│   │   ├── defenses.py                       # 六种防御、可选扩展、敏感度掩码
│   │   ├── evaluation.py                     # 指标、预测、共享评测循环
│   │   ├── reporting.py                      # 聚合、显著性检验、图表
│   │   ├── backbones.py                      # MLP / 1D-CNN / FT-Transformer 模型
│   │   ├── experiment.py                     # MLP 端到端流程
│   │   ├── phi4.py                           # 最差类别召回评测
│   │   ├── adaptive.py                       # 自适应攻击套件
│   │   └── style.py                          # 共享 matplotlib 样式与配色
│   ├── prepare_data.py                       # 校验 / 修复数据集目录
│   ├── smoke_test.py                         # 快速端到端自检
│   ├── check_devices.py                      # CPU / CUDA / MPS 可用性报告
│   ├── run_mlp.py                            # MLP 实验
│   ├── run_cnn1d.py                          # 1D-CNN 实验
│   ├── run_ft_transformer.py                 # FT-Transformer 实验
│   ├── evaluate_phi4_cnn.py                  # phi4（最差类别召回），1D-CNN
│   ├── evaluate_phi4_ft.py                   # phi4（最差类别召回），FT-Transformer
│   ├── pareto_selection.py                   # 帕累托过滤 + 偏好加权选择
│   ├── evaluate_cross_backbone_transfer.py   # 迁移攻击矩阵
│   ├── run_cicids2017.py                     # CIC-IDS2017 跨数据集运行
│   ├── run_cicids2017_source_disjoint.py     # 源域不重叠的 CIC-IDS2017 变体
│   ├── sweep_hyperparameters.py              # 超参数敏感性
│   ├── analyze_extended.py                   # epsilon 扫描、ROC、梯度遮蔽
│   └── make_figures_matlab.m                 # MATLAB：图 2（帕累托）及相关面板
├── data/            # 数据集放置于此（不入版本库）
└── outputs/         # 实验输出（不入版本库）
```

### 各脚本与论文结果的对应关系

| 论文内容 | 对应脚本 |
|---|---|
| 表 3-5，四维画像 | `run_mlp.py`、`run_cnn1d.py`、`run_ft_transformer.py` |
| 表 6-7，帕累托与偏好选择 | `pareto_selection.py` |
| phi4（最差类别召回）多随机种子数值 | `evaluate_phi4_cnn.py`、`evaluate_phi4_ft.py` |
| 图 2 | `make_figures_matlab.m`（MATLAB） |

其余脚本（`run_cicids2017*.py`、`evaluate_cross_backbone_transfer.py`、
`sweep_hyperparameters.py`、`analyze_extended.py`）实现的是**不属于**论文表格的分析，保留在此是为了
完整性与后续研究。自适应攻击套件位于 `code/ids_defense_selection/adaptive.py`，由 `smoke_test.py` 使用。

---

## 运行实验

### 设备选择

`--device` 支持 `auto`（默认）、`cpu`、`cuda`、`cuda:N` 和 `mps`。`auto` 会优先使用可见的
CUDA GPU，其次 Apple MPS，最后回退到 CPU；在没有 GPU 的机器上显式指定 `cuda` 会立即报出明确
错误，而不是抛出一段难以理解的 torch 异常。解析结果会在启动时打印，并作为 `resolved_device`
记录在 `run_summary.json` 中。

```bash
uv run python code/check_devices.py                  # 查看 CPU / CUDA / MPS
uv run python run_experiments.py                     # 自动选择设备
uv run python code/run_mlp.py --device cpu           # 固定用 CPU
uv run python code/run_mlp.py --device cuda:1        # 固定用某块 GPU
```

### 完整流程

```bash
uv run python run_experiments.py --device cuda                     # MLP、1D-CNN、FT-Transformer
uv run python run_experiments.py --backbones mlp,cnn --device cuda # 只跑子集
```

各个骨干网络**故意串行运行**：并行运行会让它们争抢 GPU，从而污染用于目标 phi3 的训练成本测量。

每次运行写入 `outputs/<backbone>/`：

```
mean_results.csv          按 (防御, 攻击, epsilon) 聚合的种子均值
std_results.csv           同一批单元格的种子标准差
raw_results.csv           每个种子一行
significance_tests.csv    防御两两之间的配对 t 检验与 Wilcoxon 检验
efficiency_mean.csv       参数量、训练耗时、推理延迟
attack_generalization.csv 对训练从未见过的攻击族的鲁棒性
hyperparameters.csv       全部配置字段（分组、取值、说明）
run_summary.json          实际使用的完整配置
risk_profile_4d.csv       四维画像与帕累托标记
```

训练结束后对整个测试集施加的攻击同样可配置：`--full-test-attack-settings pgd:0.10`
指定攻击方式与预算，`--full-test-attack-rows 0`（默认）表示攻击完整测试集。

自适应攻击套件（重启 PGD、无梯度 NES、针对掩码防御的 complement attack）默认关闭（开销大），
加上 `--adaptive-eval`（可选 `--adaptive-steps`、`--adaptive-restarts`、`--adaptive-epsilon`）后，
运行会额外写出 `adaptive_attack_raw.csv` / `adaptive_attack_mean.csv`。

### 最差类别召回（phi4），五个随机种子

```bash
uv run python code/evaluate_phi4_cnn.py --device cuda --output-dir outputs/phi4_cnn
uv run python code/evaluate_phi4_ft.py  --device cuda --output-dir outputs/phi4_ft
```

两个脚本默认使用论文的五个随机种子；单种子估计可传 `--seeds 42`，子集可传 `--seeds 7,13`。
FT-Transformer 训练时使用 `--adv-steps 7`、评测时使用 `--eval-pgd-steps 20`，两个预算会分别记录在
`run_summary.json` 中。

### 帕累托过滤与选择

```bash
uv run python code/pareto_selection.py --ref-attack pgd --ref-epsilon 0.10
```

决策步骤现在会考虑种子间的波动，并支持设置最低标准：

```bash
# 不确定性感知帕累托（均值 ± 1 倍标准差），并淘汰低于门槛的候选
uv run python code/pareto_selection.py --confidence-margin 1.0 --min-phi2 0.80 --min-phi4 0.10
```

`risk_profile_4d.csv` 会带上各目标的标准差，以及"统计判定"和"点估计判定"两种帕累托标记；
`outputs/decision_settings.json` 记录产生推荐所用的门槛与置信度余量。

### 预期运行时间

单 GPU 计时，测试环境为 RTX 4060 Laptop（8 GB），使用完整官方划分：

| 骨干网络 | 单种子、六种防御 | 五个种子 |
|---|---|---|
| MLP | 约 8 分钟 | 约 40 分钟 |
| 1D-CNN | 约 15 分钟 | 约 1.3 小时 |
| FT-Transformer | 约 2.8 小时 | 约 14 小时 |

FT-Transformer 占总成本的绝大部分。所有骨干网络也能在 CPU 上运行，但会慢得多。

---

## 超参数

| 设置 | 取值 |
|---|---|
| 协议 | 匹配延续（matched continuation）：先 10 个干净 epoch，再 8 个 epoch |
| 优化器 | Adam，lr = 1e-3，weight decay = 1e-5 |
| 批大小 | 1024（MLP、1D-CNN）；512（FT-Transformer） |
| Dropout | 0.15 |
| 随机种子 | 7、13、21、42、100（用 `--seeds` 设置） |
| 评测子集 | 20,000 条分层测试样本（`--eval-attack-rows`），固定种子 2026（`--eval-subset-seed`），所有防御共享 |
| **对抗训练预算** | **epsilon = 0.06，alpha = 0.015，20 步**（FT-Transformer：训练 7 步） |
| 评测扰动预算 | epsilon ∈ {0.02, 0.05, 0.10} |
| 评测 PGD 步数 | 20（`--eval-pgd-steps`），与训练预算相互独立 |
| 设备 | `auto`（CUDA → MPS → CPU）；用 `--device cpu`、`cuda`、`cuda:N` 固定 |

| 防御 | 训练时攻击 | 掩码 | 额外设置 |
|---|---|---|---|
| Standard | 无 | - | - |
| PGD-AT | PGD | 全部 39 个连续特征 | - |
| Constrained | PGD | 最敏感的 top-30% 特征 | - |
| TRADES | KL 上的 PGD | 全部连续特征 | beta = 6.0 |
| Free AT | 重放式 PGD | 全部连续特征 | 重放 m = 4 |
| Class-Aware | PGD | 按类别取 top-30% | 少数类权重 = 3.0 |

评测攻击：FGSM；PGD（20 步；步长由 `eval_pgd_alpha_ratio` 控制，默认 0.05 = ε/20）；
C&W L2（30 步，lr = 0.01，c = 1.0）；APGD-CE（50 步，rho = 0.75）。

> **训练预算与评测预算不同。** 对抗训练使用 epsilon = 0.06，而评测扫描 0.02 / 0.05 / 0.10。
> 这里明确说明，是因为在对比数值时这一点经常引起混淆。两个 PGD 步数是相互独立的字段
> （训练用 `--adv-steps`，评测用 `--eval-pgd-steps`），因此骨干网络不可能再用缩减后的训练步数
> 悄悄完成评测。

---

## 完整参数化

**`ExperimentConfig` 的每个字段都是一个命令行参数。** 参数由 dataclass 自动生成，因此 CLI
不可能与配置脱节：给 `ExperimentConfig` 增加一个字段，就会自动增加一个参数。全部 51 个字段在三个
骨干脚本中完全一致。

每个字段的帮助文本和旧参数别名都写在 dataclass 元数据里；配置在构造时就会做校验：像
`--batch-size 0`、未知的攻击名或格式错误的元组，都会立即报出一条可读的错误，并一次性列出所有问题。

为了方便维护和一眼看清实验设置，同样的字段还按域拆成七个不可变分组——`paths`、`training`、
`attack`、`evaluation`、`methods`、`sensitivity`、`runtime`，每组自己校验自己的字段。
`--print-config` 和 `run_summary.json` 都按这个结构输出；在 Python 里也可以用
`config.training.batch_size`、`config.evaluation.epsilon_list`、`config.runtime.device` 访问，
而代码里原有的扁平访问（`config.batch_size`）保持不变。

```bash
uv run python code/run_cnn1d.py --help              # 查看完整参数列表与默认值
uv run python code/run_mlp.py --train-path data/train.csv --test-path data/test.csv --print-config
```

复杂类型用逗号分隔的字符串解析：

| 字段类型 | 参数示例 |
|---|---|
| `int` / `float` / `str` | `--adv-epsilon 0.05` |
| `tuple[int, ...]` | `--hidden-dims 256,128,64` |
| `tuple[float, ...]` | `--epsilon-list 0.02,0.05,0.10` |
| `tuple[str, ...]` | `--reference-models log_reg,hist_gbdt` |
| `tuple[tuple[str, float], ...]` | `--transfer-attack-settings fgsm:0.05,pgd:0.10` |

未传入的字段保持 dataclass 默认值；`--print-config` 会在开始任何工作之前以 JSON 打印生效配置：

```bash
uv run python code/run_ft_transformer.py --seeds 1,2,3 --adv-epsilon 0.09 --print-config
```

一些有代表性的用法：

```bash
# 换一种 MLP 结构，并使用不同的评测预算列表
uv run python code/run_mlp.py --train-path data/train.csv --test-path data/test.csv \
    --hidden-dims 256,128,64 --dropout 0.3 --epsilon-list 0.02,0.05,0.10

# 用三个种子代替五个
uv run python code/run_cnn1d.py --seeds 7,42,100

# 论文方法部分写到的 eps/10 的 PGD 步长
uv run python code/run_mlp.py --eval-pgd-alpha-ratio 0.10
```

### 可选的额外防御方法

本仓库还包含三种**不属于**论文所评测的六种防御的实现：Progressive Class-Aware
Constrained AT、SA-TRADES 和 DST-SA-TRADES。它们被完整保留，以便实现可查、可复现，但只有在显式
请求时才会训练：

```bash
uv run python code/run_mlp.py --extra-methods progressive,sa_trades,dst_sa_trades
```

默认（空）设置下，代码只训练论文报告的六种防御，不会训练其它任何东西。启用后，每种额外方法会加入与
六种防御相同的注册表，并流经完全一致的评测、成本与选择代码路径。

---

## 可复现性

`ids_defense_selection.set_seed()` 会为 Python、NumPy 和 PyTorch 设置随机种子，**并**关闭 cuDNN 的非确定性：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

这些设置使得相同种子的两次运行产生逐位一致的结果。若不设置，GPU 的归约顺序会在运行之间变化，结果会在
小数点后第三位出现差异——这在这里很重要，因为若干帕累托判定恰恰依赖这一量级的差值。

---

## 开发

库代码位于 `code/ids_defense_selection/`，`code/` 下的脚本只是导入该库的薄入口。安装开发工具并在仓库
根目录运行检查：

```bash
uv sync --group dev
uv run pytest          # 合成数据上的端到端流程测试（CPU，数秒）
uv run ruff check      # 静态检查（含未定义名称）
```

测试会构造一个小型、形如 UNSW-NB15 的合成数据集，因此既不需要真实数据，也不需要 GPU。

---

## 关于特征空间的说明

one-hot 的宽度取决于**训练**划分中包含多少类别取值，因此它是划分本身的属性，而非固定常数。使用官方
划分时，本代码会产生 **194 个变换后特征（39 个连续 + 155 个 one-hot）**。论文中报告的是 190，对应投稿
实验所使用的反向划分。`smoke_test.py` 会打印实际得到的数值，因此两者不会被悄悄混淆。

---

## 引用

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

## 许可证

MIT，详见 `LICENSE`。

## 联系方式

Xiyuan Huang - huangxiyuan2020@qq.com - [@huangxyy](https://github.com/huangxyy)

欢迎通过 issue 跟踪器提问、报告问题或提交 pull request：
<https://github.com/huangxyy/bdcc-ids-defense-selection/issues>。
