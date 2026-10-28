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


def subsample_train(parts: tuple, fraction: float, seed: int) -> tuple:
    """E13: a stratified `fraction` of the training rows (seeded); validation and test rows
    unchanged. fraction 1 returns the parts as they are."""
    train, val, test = parts
    if fraction >= 1:
        return parts
    sub, _ = train_test_split(train, train_size=fraction, stratify=train["label"],
                              random_state=seed)  # fmt: skip
    return sub, val, test


def resplit_train_val(parts: tuple, seed: int) -> tuple:
    """E14: train + validation rows split again, stratified with `seed`, keeping both sizes;
    the test rows are passed through untouched (never resplit or looked at)."""
    train, val, test = parts
    both = pd.concat([train, val])
    new_train, new_val = train_test_split(both, test_size=len(val), stratify=both["label"],
                                          random_state=seed)  # fmt: skip
    return new_train, new_val, test


def resample(parts: tuple, cfg: dict) -> tuple:
    """Apply a config's E13 / E14 resampling, if any: train_fraction + subsample_seed, or
    split_seed. Other configs get the parts unchanged."""
    if cfg.get("split_seed") is not None:
        parts = resplit_train_val(parts, cfg["split_seed"])
    if cfg.get("train_fraction") is not None:
        parts = subsample_train(parts, cfg["train_fraction"], cfg["subsample_seed"])
    return parts
