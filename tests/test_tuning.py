"""Optuna search (src/blindspy/tuning.py) on toy data."""

import optuna
import pytest

from blindspy import tuning

SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "max_depth": {"type": "int_or_none", "low": 5, "high": 40},
    "max_features": {"type": "choice_or_float", "choices": ["sqrt", "log2"], "low": 0.3,
                     "high": 1.0},
    "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
}  # fmt: skip


def test_sample_respects_every_range():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    for _ in range(30):
        p = tuning.sample(study.ask(), SPACE)
        assert 100 <= p["n_estimators"] <= 1000 and 0.01 <= p["learning_rate"] <= 0.3
        assert p["max_depth"] is None or 5 <= p["max_depth"] <= 40
        assert p["max_features"] in ("sqrt", "log2") or 0.3 <= p["max_features"] <= 1.0
        assert p["weights"] in ("uniform", "distance")


def test_unknown_type_is_an_error():
    study = optuna.create_study()
    with pytest.raises(ValueError, match="unknown"):
        tuning.sample(study.ask(), {"x": {"type": "nope"}})


def toy(n: int = 300, seed: int = 0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = pd.DataFrame(rng.normal(size=(n, 4)) + y[:, None], columns=list("abcd"))
    return X, pd.Series(np.array(["A", "B", "C"])[y], name="label")


def test_search_is_deterministic_and_counts_fits():
    X, y = toy()
    space = {"n_neighbors": {"type": "int", "low": 1, "high": 30},
             "weights": {"type": "categorical", "choices": ["uniform", "distance"]}}  # fmt: skip
    runs = [tuning.search("knn_log_standard", {}, space, X, y, n_trials=6, seed=42, folds=5)
            for _ in range(2)]  # fmt: skip
    a, b = (r["trials"] for r in runs)
    assert len(a) == 6 and a[["params", "cv_f1_macro_mean"]].equals(
        b[["params", "cv_f1_macro_mean"]]
    )
    s = runs[0]["summary"]
    assert s["search_n_trials"] == 6 and s["search_total_fits"] == 30
    best = a.loc[a["cv_f1_macro_mean"].idxmax()]
    assert s["search_best_cv_f1_macro_mean"] == best["cv_f1_macro_mean"]
    assert runs[0]["best_params"] == best["params"]


def test_fixed_params_reach_every_trial():
    X, y = toy()
    seen = []
    space = {"n_estimators": {"type": "int", "low": 5, "high": 10}}
    r = tuning.search("lgbm", {"subsample_freq": 1}, space, X, y, n_trials=2, seed=42, folds=3,
                      on_trial=lambda t, params, res: seen.append(params))  # fmt: skip
    assert all(p["subsample_freq"] == 1 for p in seen) and r["best_params"]["subsample_freq"] == 1


def test_optuna_kind_logs_trials_and_the_retrained_best(tmp_path, monkeypatch):
    import importlib.util

    import mlflow

    from blindspy import tracking

    spec = importlib.util.spec_from_file_location(
        "run_experiment", tracking.ROOT / "scripts" / "run_experiment.py"
    )
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)  # fmt: skip
    X, y = toy(400)
    df = X.assign(label=y.to_numpy())
    parts = (df.iloc[:280], df.iloc[280:340], df.iloc[340:])
    monkeypatch.setattr(rx.data, "load_task", lambda *a, **k: parts)
    monkeypatch.setattr(rx.tracking, "data_tags", lambda *a, **k: {"data_version": "toy"})
    monkeypatch.setattr(rx, "_log_data_rules", lambda version: None)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment(experiment_id=mlflow.create_experiment(
        "t", artifact_location=(tmp_path / "artifacts").as_uri()))  # fmt: skip
    cfg = {"experiment_id": "E07", "run_name": "E07-knn-toy-15s", "kind": "optuna",
           "task": "toy", "timeout": 15, "model": "knn_log_standard", "trials": 4,
           "search_space": {"n_neighbors": {"type": "int", "low": 1, "high": 20}},
           "save_model": True}  # fmt: skip
    base = {"data_version": "toy", "seed": 42, "cv_folds": 5, "model_compress": 3}
    with mlflow.start_run(run_name=cfg["run_name"], tags={"runner": "x"}) as parent:
        rx.run_optuna(cfg, base)
    runs = mlflow.search_runs(experiment_names=["t"])
    trials = runs[runs["tags.trial"] == "True"]
    assert len(trials) == 4 and (trials["tags.runner"] == "x").all()
    assert (trials["tags.mlflow.parentRunId"] == parent.info.run_id).all()
    best = runs[runs["run_id"] == parent.info.run_id].iloc[0]
    assert best["metrics.search_total_fits"] == 20 and best["tags.tuned"] == "True"
    # the retrained best scores exactly as its trial did in the search (same folds and seed)
    assert best["metrics.cv_f1_macro_mean"] == best["metrics.search_best_cv_f1_macro_mean"]
    assert best["metrics.cv_f1_macro_mean"] == trials["metrics.cv_f1_macro_mean"].max()
    assert (tmp_path / "artifacts").exists()
    assert best["params.model.n_neighbors"] == str(
        int(trials.loc[trials["metrics.cv_f1_macro_mean"].idxmax(), "params.model.n_neighbors"])
    )
