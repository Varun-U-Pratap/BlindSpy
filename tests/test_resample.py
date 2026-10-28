"""E13 / E14 resampling of the training rows (splits.subsample_train, splits.resplit_train_val)."""

import numpy as np
import pandas as pd

from blindspy import splits


def parts(seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.array(["A"] * 600 + ["B"] * 300 + ["C"] * 100))
    df = pd.DataFrame({"x": rng.normal(size=1000), "label": y}, index=np.arange(1000) + 7)
    return df.iloc[:700], df.iloc[700:850], df.iloc[850:]


def test_subsample_train_is_stratified_deterministic_and_leaves_val_test_alone():
    tr, va, te = parts()
    sub, va2, te2 = splits.subsample_train((tr, va, te), 0.2, seed=43)
    assert len(sub) == 140 and set(sub.index) <= set(tr.index)
    assert va2 is va and te2 is te
    share = tr["label"].value_counts(normalize=True)
    assert (sub["label"].value_counts(normalize=True) - share).abs().max() < 0.01
    again = splits.subsample_train((tr, va, te), 0.2, seed=43)[0]
    assert list(again.index) == list(sub.index)
    assert list(splits.subsample_train((tr, va, te), 0.2, seed=44)[0].index) != list(sub.index)
    assert splits.subsample_train((tr, va, te), 1.0, seed=43)[0] is tr


def test_resplit_keeps_the_test_rows_and_the_sizes():
    tr, va, te = parts()
    tr2, va2, te2 = splits.resplit_train_val((tr, va, te), seed=45)
    assert te2 is te
    assert len(tr2) == len(tr) and len(va2) == len(va)
    assert set(tr2.index) | set(va2.index) == set(tr.index) | set(va.index)
    assert not set(tr2.index) & set(va2.index) and not set(va2.index) & set(te.index)
    assert set(va2.index) != set(va.index)
    both = pd.concat([tr, va])["label"].value_counts(normalize=True) * len(va)
    got = va2["label"].value_counts()
    assert all(abs(got[c] - both[c]) <= 1 for c in both.index)  # stratified on train + val


def test_resample_from_config():
    p = parts()
    assert splits.resample(p, {}) == p
    sub = splits.resample(p, {"train_fraction": 0.5, "subsample_seed": 42})[0]
    assert len(sub) == 350
    tr2, va2, te2 = splits.resample(p, {"split_seed": 46})
    assert te2 is p[2] and len(va2) == len(p[1])
