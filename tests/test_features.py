"""Preprocessing steps (src/blindspy/features.py)."""

import numpy as np
import pandas as pd

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
