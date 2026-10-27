"""Model registry: name -> function returning a full sklearn Pipeline (preprocessing + model).

Every builder takes the config's hyperparameters and the seed. Scaling is a pipeline step, so
it is fitted on training rows (or training folds) only.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
    "logreg": _logreg,
    "svm_rbf": _svm_rbf,
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


def complexity(pipe: Pipeline) -> dict:
    """Tree depth / nodes / leaves, or SVM support-vector count; empty for other models."""
    est = pipe[-1]
    if isinstance(est, DecisionTreeClassifier):
        return {
            "tree_depth": est.get_depth(),
            "tree_nodes": est.tree_.node_count,
            "tree_leaves": est.get_n_leaves(),
        }
    if isinstance(est, SVC):
        return {"support_vectors": int(est.n_support_.sum())}
    return {}
