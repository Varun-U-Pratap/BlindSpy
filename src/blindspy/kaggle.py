"""E16 data: the Kaggle CSV used by Isnan et al. (2026), as its own data version (kaggle-k1).

E00 showed the CSV is the eight A2 files (VPN and NonVPN x 4 timeouts) stacked; E16 setup showed
it keeps their row order, in FILE_ORDER, so each row's source file, timeout and row number are
recovered by position (and checked against the ARFF files at 12 significant digits).

Two splits, both stratified 80/20 with seed 42:
- "isnan": Isnan et al.'s setup, exact duplicates (23 features + label) dropped over the whole
  CSV (timeouts mixed), then one split;
- "per_timeout": the corrected setup, each timeout on its own: duplicates dropped within the
  timeout, then a split per timeout, so a flow and its copies from other timeouts never sit on
  both sides.
Data v1 is not touched.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from blindspy.data import HEADERS, ROOT, TIMEOUTS, load_raw

CSV = ROOT / "datasets" / "kaggle" / "consolidated_traffic_data.csv"
FEATURES = HEADERS["T"]
# (task, timeout) of each block of rows in the CSV, top to bottom (found and checked in E16)
FILE_ORDER = [("A2-NonVPN", 120), ("A2-VPN", 120), ("A2-NonVPN", 15), ("A2-VPN", 15),
              ("A2-NonVPN", 30), ("A2-VPN", 30), ("A2-NonVPN", 60), ("A2-VPN", 60)]  # fmt: skip
SEED = 42
TEST_SIZE = 0.2


def _keys(df: pd.DataFrame) -> pd.Series:
    return df[FEATURES].map(lambda v: f"{v:.12g}").agg("|".join, axis=1) + "#" + df["label"]


def load_csv(path: Path = CSV, check: bool = True) -> pd.DataFrame:
    """The CSV with label, source task, timeout and row number in its source file."""
    df = pd.read_csv(path).rename(columns={"traffic_type": "label"})
    if list(df.columns[:-1]) != FEATURES:
        raise ValueError("Kaggle CSV columns differ from the A2 header")
    blocks, start = [], 0
    for task, timeout in FILE_ORDER:
        raw = load_raw(task, timeout)
        block = df.iloc[start : start + len(raw)].copy()
        if check and not (_keys(block).to_numpy() == _keys(raw).to_numpy()).all():
            raise ValueError(f"rows {start}-{start + len(raw) - 1} are not {task} {timeout} s")
        block["source_task"], block["timeout"] = task, timeout
        block["source_row"] = range(len(raw))
        blocks.append(block)
        start += len(raw)
    if start != len(df):
        raise ValueError(f"{len(df) - start} CSV rows left after the 8 A2 files")
    out = pd.concat(blocks)
    out.index.name = "kaggle_row"
    return out


def _split(rows: pd.DataFrame) -> dict:
    """Rows in the order train_test_split returns them (not sorted): a seeded forest's bootstrap
    and an unshuffled 5-fold CV depend on row order, and Isnan et al. used this order."""
    train, test = train_test_split(rows.index.to_numpy(), test_size=TEST_SIZE,
                                   stratify=rows["label"], random_state=SEED)  # fmt: skip
    return {"train": [int(i) for i in train], "test": [int(i) for i in test]}


def isnan_split(df: pd.DataFrame) -> dict:
    kept = df.drop_duplicates(subset=[*FEATURES, "label"], keep="first")
    return {"protocol": "isnan", "rows": len(df), "after_dedup": len(kept), **_split(kept)}


def timeout_splits(df: pd.DataFrame) -> dict:
    out = {}
    for t in TIMEOUTS:
        part = df[df["timeout"] == t]
        kept = part.drop_duplicates(subset=[*FEATURES, "label"], keep="first")
        out[str(t)] = {"protocol": "per_timeout", "timeout": t, "rows": len(part),
                       "after_dedup": len(kept), **_split(kept)}  # fmt: skip
    return out


def cross_timeout_counts(df: pd.DataFrame) -> dict:
    """Distinct rows (features + label) by the number of timeouts they appear in."""
    n = df.assign(key=_keys(df)).groupby("key")["timeout"].nunique()
    return {str(k): int((n == k).sum()) for k in range(1, 5)}


def processed_dir(version: str, root: Path = ROOT) -> Path:
    return root / "data" / "processed" / "kaggle" / version


def build(version: str, root: Path = ROOT, path: Path = CSV) -> dict:
    """Write data/processed/kaggle/<version>/ (rows.parquet, splits, manifest.json); refuses to
    overwrite an existing version (written to a .partial folder first)."""
    import hashlib

    out = processed_dir(version, root)
    if out.exists():
        raise FileExistsError(f"{out} exists; a data version is never overwritten")
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    df = load_csv(path)
    df.to_parquet(tmp / "rows.parquet")
    isnan, per_t = isnan_split(df), timeout_splits(df)
    (tmp / "splits").mkdir()
    (tmp / "splits" / "isnan.json").write_text(json.dumps(isnan, indent=1) + "\n")
    for t, sp in per_t.items():
        (tmp / "splits" / f"per_timeout_{t}s.json").write_text(json.dumps(sp, indent=1) + "\n")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    sizes = df.groupby(["source_task", "timeout"], sort=False).size()
    manifest = {
        "data_version": f"kaggle-{version}",
        "source_csv_sha256": sha(path),
        "rows": len(df),
        "labels": sorted(df["label"].unique()),
        "features": FEATURES,
        "file_order": [f"{task} {t} s ({n} rows)" for (task, t), n in sizes.items()],
        "distinct_rows_by_number_of_timeouts": cross_timeout_counts(df),
        "isnan": {k: (len(v) if isinstance(v, list) else v) for k, v in isnan.items()},
        "per_timeout": {t: {k: (len(v) if isinstance(v, list) else v) for k, v in sp.items()}
                        for t, sp in per_t.items()},  # fmt: skip
        "seed": SEED, "test_size": TEST_SIZE,
        "files_sha256": {str(p.relative_to(tmp)): sha(p)
                         for p in sorted(tmp.rglob("*")) if p.is_file()},  # fmt: skip
    }  # fmt: skip
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    tmp.rename(out)
    return manifest


def subsample(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """E16 (d): a stratified sample of n training rows, in train_test_split's order."""
    if n >= len(train):
        return train
    keep, _ = train_test_split(train.index.to_numpy(), train_size=n, stratify=train["label"],
                               random_state=seed)  # fmt: skip
    return train.loc[keep]


def load_split(version: str, name: str, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) rows for split `name`: "isnan" or "per_timeout_<t>s"."""
    folder = processed_dir(version, root)
    rows = pd.read_parquet(folder / "rows.parquet")
    sp = json.loads((folder / "splits" / f"{name}.json").read_text())
    return rows.loc[sp["train"]], rows.loc[sp["test"]]
