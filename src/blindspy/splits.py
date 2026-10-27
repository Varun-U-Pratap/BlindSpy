"""Fixed stratified train / validation / test split per task and timeout."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(df: pd.DataFrame, train: float, val: float, test: float, seed: int) -> dict:
    """Stratified split of df's index (raw_row values) into train / val / test."""
    if abs(train + val + test - 1) > 1e-9:
        raise ValueError("split fractions must add up to 1")
    rest, test_idx = train_test_split(
        df.index.to_numpy(), test_size=test, stratify=df["label"], random_state=seed
    )
    train_idx, val_idx = train_test_split(
        rest, test_size=val / (train + val), stratify=df.loc[rest, "label"], random_state=seed
    )
    return {
        "seed": seed,
        "fractions": {"train": train, "val": val, "test": test},
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def split_path(folder: Path, task: str, timeout: int) -> Path:
    return folder / "splits" / f"{task}_{timeout}s.json"


def save_split(split: dict, folder: Path, task: str, timeout: int) -> Path:
    path = split_path(folder, task, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + "\n")
    return path


def load_split(folder: Path, task: str, timeout: int) -> dict:
    return json.loads(split_path(folder, task, timeout).read_text())
