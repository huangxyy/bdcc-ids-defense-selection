# Data

**English** | [简体中文](README.zh-CN.md)

## UNSW-NB15 (required)

The two official partitions **ship with this repository**:

- [`train.csv`](train.csv) — 175,341 records (official training partition, UTF-8 without BOM)
- [`test.csv`](test.csv) — 82,332 records (official testing partition)

`git clone` / `git pull` is all you need; there is nothing to download for the main experiments.

Source: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>

If you replace them, keep the names this code expects (and the official direction —
the larger partition is the training one):

```
data/train.csv    175,341 records    <- official UNSW_NB15_training-set.csv
data/test.csv      82,332 records    <- official UNSW_NB15_testing-set.csv
```

Both files must contain the columns used by the code, in particular
`attack_cat` (multi-class attack label) and `label` (binary label).

### Verify the layout

```bash
uv run python code/prepare_data.py
```

This counts the records in each file and compares them against the official partition sizes.
It also checks that the required columns are present.

Both the validator and the training code read the CSVs with ``utf-8-sig``, so the byte-order
mark shipped with the distributed files is stripped automatically (otherwise the first column
would be ``"\\ufeffid"`` and the ``id`` drop would miss it).

### If the two files are reversed

It is easy to download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` and save them as
`train.csv` / `test.csv` **without noticing which is which**. That silently makes the experiment train on
the small partition and evaluate on the large one. `prepare_data.py` detects this and can repair it:

```bash
uv run python code/prepare_data.py --fix-swap
```

After the swap, `train.csv` holds 175,341 records and `test.csv` holds 82,332 -- the official direction.

## CIC-IDS2017 (optional)

**Not** shipped with the repository; only needed by `code/run_cicids2017.py` and
`code/run_cicids2017_source_disjoint.py`.

Source: <https://www.unb.ca/cic/datasets/ids-2017.html>

```
data/cicids2017/monday.csv
data/cicids2017/tuesday.csv
data/cicids2017/wednesday.csv
data/cicids2017/thursday.csv
data/cicids2017/friday.csv
```
