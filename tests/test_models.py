"""Model registry, evaluation metrics and the training run (on toy data, no dataset needed)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from blindspy import evaluate, models, train


def toy(n_classes: int, n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    X = pd.DataFrame(rng.normal(size=(n, 4)) + y[:, None] * 2.0, columns=list("abcd"))
    names = np.array([f"C{i}" for i in range(n_classes)])
    return X, pd.Series(names[y], name="label")


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_every_registry_entry_is_a_fitted_pipeline_with_seed(name):
    X, y = toy(3)
    pipe = models.build_model(name, {}, seed=42)
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)
    est = pipe[-1]
    if "random_state" in est.get_params():
        assert est.get_params()["random_state"] == 42
    scores = models.class_scores(pipe, X)
    assert scores.shape == (len(X), 3)


def test_scaling_sits_inside_the_pipeline():
    for name in ("knn", "logreg", "svm_rbf", "knn_minmax"):
        steps = [s for s, _ in models.build_model(name, {}, seed=42).steps]
        assert steps[0] == "scale" and steps[-1] == "model"


def test_params_reach_the_estimator():
    pipe = models.build_model("tree", {"criterion": "entropy", "max_depth": 3}, seed=42)
    assert pipe[-1].criterion == "entropy" and pipe[-1].max_depth == 3


def test_binary_scores_have_two_columns_for_decision_function_models():
    X, y = toy(2)
    pipe = models.build_model("svm_rbf", {}, seed=42).fit(X, y)
    s = models.class_scores(pipe, X)
    assert s.shape == (len(X), 2) and np.allclose(s[:, 0], -s[:, 1])


def test_complexity():
    X, y = toy(3)
    tree = models.build_model("tree", {}, seed=42).fit(X, y)
    c = models.complexity(tree)
    assert c["tree_depth"] == tree[-1].get_depth() and c["tree_nodes"] == tree[-1].tree_.node_count
    svm = models.build_model("svm_rbf", {}, seed=42).fit(X, y)
    assert models.complexity(svm)["support_vectors"] == int(svm[-1].n_support_.sum())
    assert models.complexity(models.build_model("knn", {}, seed=42).fit(X, y)) == {}


def test_evaluate_metrics_by_hand(tmp_path):
    y = pd.Series(["A", "A", "A", "B", "B", "C"])
    p = np.array(["A", "A", "B", "B", "A", "C"])
    s = np.eye(3)[[0, 0, 1, 1, 0, 2]]
    m, info = evaluate.evaluate(y, p, s, ["A", "B", "C"], tmp_path)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision.A"] == pytest.approx(2 / 3) and m["recall.A"] == pytest.approx(2 / 3)
    assert m["recall.B"] == pytest.approx(1 / 2) and m["f1.C"] == 1.0
    assert m["f1_macro"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert m["f1_gap_weighted_minus_macro"] == pytest.approx(m["f1_weighted"] - m["f1_macro"])
    assert info["best_class_f1"] == "C" and info["worst_class_recall"] == "B"
    pairs = info["top_confusion_pairs"]
    assert {(q["true"], q["predicted"]) for q in pairs} == {("A", "B"), ("B", "A")}
    assert all(q["count"] == 1 for q in pairs)
    for f in ("confusion_matrix.png", "confusion_matrix.csv", "per_class_report.csv",
              "pr_curves.png", "confusion_pairs.csv"):  # fmt: skip
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "roc_curve.png").exists()
    cm = pd.read_csv(tmp_path / "confusion_matrix.csv", index_col=0)
    assert cm.loc["B", "A"] == 1 and cm.to_numpy().sum() == 6


def test_evaluate_binary_has_roc(tmp_path):
    y = pd.Series(["Non-VPN", "VPN", "VPN", "Non-VPN"])
    s = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    m, _ = evaluate.evaluate(y, np.array(["Non-VPN", "VPN", "Non-VPN", "Non-VPN"]), s,
                             ["Non-VPN", "VPN"], tmp_path)  # fmt: skip
    assert m["roc_auc"] == 1.0 and (tmp_path / "roc_curve.png").exists()


def test_train_run_logs_everything(tmp_path, monkeypatch):
    X, y = toy(3, n=400)
    df = X.assign(label=y.to_numpy())
    parts = (df.iloc[:280], df.iloc[280:340], df.iloc[340:])
    cfg = {"experiment_id": "E_test", "task": "toy", "timeout": 15, "model": "tree",
           "params": {"criterion": "entropy"}, "save_model": True}  # fmt: skip
    logged = train.train_and_score(cfg, parts, seed=42, cv_folds=5, out_dir=tmp_path)
    m = logged["metrics"]
    for key in ("cv_f1_macro_mean", "cv_f1_macro_std", "val_f1_macro", "val_accuracy",
                "gen_gap_f1_macro", "train_time_s", "predict_ms_per_1000", "model_size_bytes",
                "peak_memory_mb", "val_precision.C0", "model_tree_depth"):  # fmt: skip
        assert key in m, key
    assert m["gen_gap_f1_macro"] == pytest.approx(m["cv_f1_macro_mean"] - m["val_f1_macro"])
    assert logged["params"]["n_train"] == 280 and logged["params"]["n_val"] == 60
    assert (tmp_path / "model" / "model.joblib").stat().st_size == m["model_size_bytes"]
    assert json.loads((tmp_path / "data" / "class_counts.json").read_text())["val"]
    assert json.loads(logged["params"]["class_counts.train"]) == y[:280].value_counts().to_dict()
    assert (tmp_path / "eval" / "feature_importance.csv").exists()
    assert logged["tags"]["feature_importance"] == "impurity"
    # the test part is never used: changing it changes nothing
    parts2 = (parts[0], parts[1], parts[2].assign(label="C0"))
    again = train.train_and_score(cfg, parts2, seed=42, cv_folds=5, out_dir=tmp_path / "b")
    for k in ("val_f1_macro", "cv_f1_macro_mean"):
        assert again["metrics"][k] == m[k]


def test_paper_benchmark_pools_10_fold_predictions(tmp_path):
    X, y = toy(2, n=200)
    df = X.assign(label=y.to_numpy())
    cfg = {"experiment_id": "E_test", "task": "toy", "timeout": 15, "model": "knn_minmax",
           "params": {"n_neighbors": 1}, "save_model": False}  # fmt: skip
    logged = train.paper_benchmark(cfg, df, seed=42, folds=10, out_dir=tmp_path)
    m = logged["metrics"]
    assert "bench_precision.C0" in m and "bench_recall.C1" in m and "fold_f1_macro_std" in m
    assert logged["params"]["n_rows"] == 200 and logged["params"]["cv_folds"] == 10


def test_training_time_is_measured_without_tracemalloc():
    """The timed fit runs with tracemalloc off (it slows fitting); peak memory comes from a
    second fit of a clone with tracemalloc on, and the first fit is the model kept."""
    import tracemalloc

    seen = []

    class Probe(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            seen.append(tracemalloc.is_tracing())
            self.classes_ = np.unique(y)
            return self

    X, y = toy(2)
    pipe = Pipeline([("model", Probe())])
    seconds, peak = train._fit_measured(pipe, X, y)
    assert seen == [False, True]
    assert hasattr(pipe[-1], "classes_") and seconds >= 0 and peak >= 0


class Allocates(ClassifierMixin, BaseEstimator):
    """Holds about 80 MB during fit (module level, so a spawned child can unpickle it)."""

    def fit(self, X, y):
        block = np.ones(10_000_000)
        self.classes_ = np.unique(y)
        self.total_ = float(block.sum())
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_child_process_rss_sees_memory_of_the_fit():
    X, y = toy(2)
    mb = train._child_peak_rss_mb(Pipeline([("model", Allocates())]), X, y)
    assert 70 <= mb <= 200


def test_compressed_model_reloads_and_predicts_identically(tmp_path):
    X, y = toy(3, n=400)
    df = X.assign(label=y.to_numpy())
    parts = (df.iloc[:280], df.iloc[280:340], df.iloc[340:])
    cfg = {"experiment_id": "E_test", "task": "toy", "timeout": 15, "model": "tree",
           "params": {}, "save_model": True}  # fmt: skip
    logged = train.train_and_score(cfg, parts, seed=42, cv_folds=5, out_dir=tmp_path, compress=3)
    m, p = logged["metrics"], logged["params"]
    path = tmp_path / "model" / "model.joblib"
    assert path.stat().st_size == m["model_file_bytes"] < m["model_size_bytes"]
    assert p["model_compress"] == 3 and p["cv_n_jobs"] == -1 and "model_n_jobs" in p
    assert m["child_peak_rss_mb"] >= 0
    reference = models.build_model("tree", {}, seed=42).fit(*train._xy(parts[0]))
    reloaded = joblib.load(path)
    X_val = parts[1].drop(columns="label")
    assert (reloaded.predict(X_val) == reference.predict(X_val)).all()
    assert np.array_equal(reloaded.predict_proba(X_val), reference.predict_proba(X_val))


def test_random_forest_entry_uses_one_thread_and_reports_complexity():
    X, y = toy(3)
    pipe = models.build_model("rf", {"n_estimators": 20}, seed=42).fit(X, y)
    est = pipe[-1]
    assert est.n_jobs == 1 and est.random_state == 42 and est.class_weight is None
    c = models.complexity(pipe)
    depths = [t.get_depth() for t in est.estimators_]
    assert c["n_trees"] == 20 and c["tree_depth_mean"] == pytest.approx(np.mean(depths))
    assert c["tree_depth_max"] == max(depths)
    assert c["tree_nodes_total"] == sum(t.tree_.node_count for t in est.estimators_)


@pytest.mark.parametrize("name", ["xgb", "lgbm"])
def test_boosting_entries_take_string_labels_one_thread_and_report_complexity(name):
    X, y = toy(3)
    pipe = models.build_model(name, {"n_estimators": 10}, seed=42).fit(X, y)
    est = pipe[-1]
    assert est.get_params()["n_jobs"] == 1 and est.get_params()["random_state"] == 42
    assert list(pipe.classes_) == ["C0", "C1", "C2"]
    assert set(pipe.predict(X)) <= {"C0", "C1", "C2"}
    assert models.class_scores(pipe, X).shape == (len(X), 3)
    c = models.complexity(pipe)
    assert c["n_trees"] == 30  # 10 rounds x 3 classes
    assert 1 <= c["tree_depth_mean"] <= c["tree_depth_max"] and c["tree_nodes_total"] >= 30


def test_xgb_binary_and_balanced_weights():
    X, y = toy(2)
    plain = models.build_model("xgb", {"n_estimators": 5}, seed=42).fit(X, y)
    assert list(plain.classes_) == ["C0", "C1"] and models.complexity(plain)["n_trees"] == 5
    y_skew = y.where(y.index % 4 != 0, "C0")  # make C0 the larger class
    a = models.build_model("xgb", {"n_estimators": 5}, seed=42).fit(X, y_skew)
    b = models.build_model("xgb", {"n_estimators": 5, "class_weight": "balanced"}, seed=42)
    b.fit(X, y_skew)
    assert not np.allclose(a.predict_proba(X), b.predict_proba(X))
    assert "class_weight" not in b[-1].get_xgb_params()
