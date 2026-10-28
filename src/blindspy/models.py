"""Model registry: name -> function returning a full sklearn Pipeline (preprocessing + model).

Every builder takes the config's hyperparameters and the seed. Scaling is a pipeline step, so
it is fitted on training rows (or training folds) only.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from blindspy.features import signed_log_step


def _majority(params: dict, seed: int) -> Pipeline:
    params = {"strategy": "most_frequent", "random_state": seed, **params}
    return Pipeline([("model", DummyClassifier(**params))])


def _tree(params: dict, seed: int) -> Pipeline:
    return Pipeline([("model", DecisionTreeClassifier(random_state=seed, **params))])


def _knn(params: dict, seed: int) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(**params))])


def _knn_minmax(params: dict, seed: int) -> Pipeline:
    # Weka IBk's default distance (EuclideanDistance) rescales each attribute to its
    # training range; MinMaxScaler does the same. Paper benchmark only.
    return Pipeline([("scale", MinMaxScaler()), ("model", KNeighborsClassifier(**params))])


def _knn_log_standard(params: dict, seed: int) -> Pipeline:
    # E04: signed log, then standard scaling (features.py explains the signed log)
    return Pipeline([("log", signed_log_step()), ("scale", StandardScaler()),
                     ("model", KNeighborsClassifier(**params))])  # fmt: skip


def _knn_log_robust(params: dict, seed: int) -> Pipeline:
    # E04: signed log, then median / interquartile-range scaling
    return Pipeline([("log", signed_log_step()), ("scale", RobustScaler()),
                     ("model", KNeighborsClassifier(**params))])  # fmt: skip


def _rf(params: dict, seed: int) -> Pipeline:
    # one thread per model; the CV folds run in parallel (train.CV_N_JOBS)
    params = {"n_jobs": 1, **params}
    return Pipeline([("model", RandomForestClassifier(random_state=seed, **params))])


def _logreg(params: dict, seed: int) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(random_state=seed, **params))]
    )


def _svm_rbf(params: dict, seed: int) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("model", SVC(kernel="rbf", random_state=seed, **params))]
    )


MODELS = {
    "majority": _majority,
    "tree": _tree,
    "knn": _knn,
    "knn_minmax": _knn_minmax,
    "knn_log_standard": _knn_log_standard,
    "knn_log_robust": _knn_log_robust,
    "logreg": _logreg,
    "svm_rbf": _svm_rbf,
    "rf": _rf,
}


def build_model(name: str, params: dict, seed: int) -> Pipeline:
    return MODELS[name](dict(params or {}), seed)


def class_scores(pipe: Pipeline, X) -> np.ndarray:
    """One score column per class (in pipe.classes_ order), for ROC and PR curves.

    Probabilities where the model has them; otherwise decision_function, with a binary
    model's single column expanded to (-s, s).
    """
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    s = pipe.decision_function(X)
    return np.column_stack([-s, s]) if s.ndim == 1 else s


def _ensemble(depths: list[int], nodes: list[int]) -> dict:
    return {
        "n_trees": len(depths),
        "tree_depth_mean": float(np.mean(depths)),
        "tree_depth_max": int(max(depths)),
        "tree_nodes_total": int(sum(nodes)),
        "tree_nodes_mean": float(np.mean(nodes)),
    }


def complexity(pipe: Pipeline) -> dict:
    """Tree depth / nodes / leaves, ensemble size and depth, or SVM support-vector count;
    empty for other models."""
    est = pipe[-1]
    if isinstance(est, RandomForestClassifier):
        trees = est.estimators_
        return _ensemble([t.get_depth() for t in trees], [t.tree_.node_count for t in trees])
    if isinstance(est, DecisionTreeClassifier):
        return {
            "tree_depth": est.get_depth(),
            "tree_nodes": est.tree_.node_count,
            "tree_leaves": est.get_n_leaves(),
        }
    if isinstance(est, SVC):
        return {"support_vectors": int(est.n_support_.sum())}
    return {}
