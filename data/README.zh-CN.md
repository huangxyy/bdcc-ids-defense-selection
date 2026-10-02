# 数据目录

[English](README.md) | **简体中文**

## UNSW-NB15（必需）

两个官方划分**已随仓库提供**：

- [`train.csv`](train.csv) —— 175,341 条（官方训练划分，UTF-8 无 BOM）
- [`test.csv`](test.csv) —— 82,332 条（官方测试划分）

`git clone` / `git pull` 即可，主实验不需要再下载任何数据。

来源：<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

如果你要替换它们，请保持代码期望的文件名（以及官方方向——较大的划分是训练集）：

```
data/train.csv    175,341 条    <- 官方 UNSW_NB15_training-set.csv
data/test.csv      82,332 条    <- 官方 UNSW_NB15_testing-set.csv
```

两个文件都必须包含代码使用的列，尤其是 `attack_cat`（多分类攻击标签）和 `label`（二分类标签）。

### 校验目录

```bash
uv run python scripts/prepare_data.py
```

该命令会统计每个文件的记录数并与官方划分大小比较，同时检查必需的列是否存在。

校验脚本和训练代码都用 ``utf-8-sig`` 读取 CSV，因此官方文件自带的 BOM 会被自动去掉
（否则第一列会变成 ``"\\ufeffid"``，导致 ``id`` 列没有被正确剔除）。

### 如果两个文件被弄反了

很容易出现这样的情况：下载了 `UNSW_NB15_training-set.csv` 和 `UNSW_NB15_testing-set.csv`，
**没有确认哪个是哪个**就保存成 `train.csv` / `test.csv`。这会悄悄变成在小划分上训练、在大划分上评测。
`prepare_data.py` 能检测出这种情况并修复：

```bash
uv run python scripts/prepare_data.py --fix-swap
```

交换之后，`train.csv` 有 175,341 条记录，`test.csv` 有 82,332 条——即官方方向。

## CIC-IDS2017（可选）

**不随仓库提供**；只有 `scripts/run_cicids2017.py` 和 `scripts/run_cicids2017_source_disjoint.py` 需要。

来源：<https://www.unb.ca/cic/datasets/ids-2017.html>

```
data/cicids2017/monday.csv
data/cicids2017/tuesday.csv
data/cicids2017/wednesday.csv
data/cicids2017/thursday.csv
data/cicids2017/friday.csv
```
