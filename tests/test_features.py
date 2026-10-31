"""Preprocessing steps (src/blindspy/features.py)."""

import numpy as np
import pandas as pd
import pytest

from blindspy import features, models


def test_signed_log_values():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1e8])
    out = features.signed_log(x)
    assert np.allclose(out, [-np.log1p(1000), -np.log(2), 0.0, np.log(2), np.log1p(1e8)])
    assert np.all(np.isfinite(out)) and np.all(np.diff(out) > 0)  # defined and order kept


def test_signed_log_keeps_minus_one_apart_from_zero():
    out = features.signed_log(np.array([-1.0, 0.0]))
    assert out[0] < out[1] == 0.0


def test_signed_log_step_keeps_dataframe_columns():
    df = pd.DataFrame({"a": [-1.0, 0.0, 10.0], "b": [5.0, -3.0, 1e7]})
    step = features.signed_log_step().fit(df)
    assert list(step.get_feature_names_out()) == ["a", "b"]
    assert np.allclose(step.transform(df), features.signed_log(df.to_numpy()))


def test_knn_log_pipelines():
    for name, scaler in (
        ("knn_log_standard", "StandardScaler"),
        ("knn_log_robust", "RobustScaler"),
    ):
        pipe = models.build_model(name, {"n_neighbors": 5}, seed=42)
        assert [s for s, _ in pipe.steps] == ["log", "scale", "model"]
        assert type(pipe["scale"]).__name__ == scaler and pipe[-1].n_neighbors == 5


def test_preprocessing_param_names_the_function():
    from blindspy import train

    pipe = models.build_model("knn_log_robust", {"n_neighbors": 5}, seed=42)
    X = pd.DataFrame({"a": [1.0, 2.0]})
    p = train._base_params({"task": "t", "timeout": 15, "model": "knn_log_robust"}, pipe, X, 42, 3)
    assert p["preprocessing"] == "log: FunctionTransformer(signed_log); scale: RobustScaler()"
    tree = models.build_model("tree", {}, seed=42)
    assert (
        train._base_params({"task": "t", "timeout": 15, "model": "tree"}, tree, X, 42, 3)[
            "preprocessing"
        ]
        == "none"
    )


A1_COLS = ["duration", "min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat",
           "std_fiat", "std_biat", "flowPktsPerSecond", "flowBytesPerSecond", "min_flowiat",
           "max_flowiat", "mean_flowiat", "std_flowiat", "min_active", "mean_active",
           "max_active", "std_active", "min_idle", "mean_idle", "max_idle", "std_idle"]  # fmt: skip
A2_COLS = [c.replace("std_fiat", "total_fiat").replace("std_biat", "total_biat") for c in A1_COLS]


@pytest.mark.parametrize("cols", [A1_COLS, A2_COLS])
def test_feature_groups_split_all_23_features(cols):
    groups = features.feature_groups(cols)
    assert list(groups) == ["fiat", "biat", "flowiat", "active", "idle", "rates_duration"]
    assert sorted(sum(groups.values(), [])) == sorted(cols)  # every feature in exactly one group
    assert all(len(g) == 4 for k, g in groups.items() if k != "rates_duration")
    assert groups["rates_duration"] == ["duration", "flowPktsPerSecond", "flowBytesPerSecond"]


def test_feature_sets():
    sel = features.select_features
    assert sel(A1_COLS, "all") == A1_COLS
    assert set(sel(A2_COLS, "only_fiat")) == {"total_fiat", "min_fiat", "max_fiat", "mean_fiat"}
    assert "min_flowiat" not in sel(A1_COLS, "only_fiat")  # "fiat" is not a suffix of "flowiat"
    assert len(sel(A1_COLS, "no_idle")) == 19 and "min_idle" not in sel(A1_COLS, "no_idle")
    assert sel(A1_COLS, "timing_only") == [c for c in A1_COLS if c != "flowBytesPerSecond"]
    assert features.FEATURE_SETS[0] == "all" and len(features.FEATURE_SETS) == 14
    with pytest.raises(ValueError, match="feature set"):
        sel(A1_COLS, "only_size")
