"""Model registry: name -> function returning a full sklearn Pipeline (preprocessing + model).

Every builder takes the config's hyperparameters and the seed. Scaling is a pipeline step, so
it is fitted on training rows (or training folds) only.
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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


class XGBLabelled(XGBClassifier):
    """XGBClassifier that takes and returns class names (XGBoost wants labels 0..k-1), and
    class_weight="balanced" (XGBoost has no class_weight): sample weights from
    compute_sample_weight("balanced") on the rows it is fitted on."""

    def __init__(self, *, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params.pop("class_weight", None)  # a wrapper setting, not a booster parameter
        return params

    def fit(self, X, y, sample_weight=None, **kwargs):
        labels = LabelEncoder().fit(y)
        # XGBoost's own fit checks classes_ against 0..k-1, so the names are set afterwards
        self.__dict__.pop("labels_", None)
        if self.class_weight == "balanced" and sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        super().fit(X, labels.transform(y), sample_weight=sample_weight, **kwargs)
        self.labels_ = labels
        return self

    @property
    def classes_(self):
        if "labels_" in self.__dict__:
            return self.labels_.classes_
        return super().classes_

    def predict(self, X, **kwargs):
        return self.labels_.inverse_transform(super().predict(X, **kwargs).astype(int))


def _xgb(params: dict, seed: int) -> Pipeline:
    params = {"n_jobs": 1, **params}
    return Pipeline([("model", XGBLabelled(random_state=seed, **params))])


def _lgbm(params: dict, seed: int) -> Pipeline:
    params = {"n_jobs": 1, "verbose": -1, **params}
    return Pipeline([("model", LGBMClassifier(random_state=seed, **params))])


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
    "xgb": _xgb,
    "lgbm": _lgbm,
}


BALANCE = ["none", "class_weight", "oversample", "smote"]  # E08 ladder, simplest first


def build_model(name: str, params: dict, seed: int, balance: str | None = None) -> Pipeline:
    """balance (E08): "class_weight" sets class_weight="balanced" on the model; "oversample"
    (RandomOverSampler) and "smote" (SMOTE, k_neighbors=5) add a resampling step right
    before the model in an imblearn Pipeline, which resamples only the rows it is fitted on
    (each CV training fold, the training set), never validation or test rows."""
    balance = balance or "none"
    if balance not in BALANCE:
        raise ValueError(f"unknown balance {balance!r}; one of {BALANCE}")
    params = dict(params or {})
    if balance == "class_weight":
        if "class_weight" not in MODELS[name]({}, seed)[-1].get_params():
            raise ValueError(f"model {name!r} has no class_weight")
        params["class_weight"] = "balanced"
    pipe = MODELS[name](params, seed)
    if balance in ("oversample", "smote"):
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline

        sampler = (RandomOverSampler(random_state=seed) if balance == "oversample"
                   else SMOTE(k_neighbors=5, random_state=seed))  # fmt: skip
        pipe = ImbPipeline([*pipe.steps[:-1], ("balance", sampler), pipe.steps[-1]])
    return pipe


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


def _lgbm_depth(node: dict) -> int:
    if "leaf_index" in node or "left_child" not in node:
        return 0
    return 1 + max(_lgbm_depth(node["left_child"]), _lgbm_depth(node["right_child"]))


def complexity(pipe: Pipeline) -> dict:
    """Tree depth / nodes / leaves, ensemble size and depth, or SVM support-vector count;
    empty for other models."""
    est = pipe[-1]
    if isinstance(est, RandomForestClassifier):
        trees = est.estimators_
        return _ensemble([t.get_depth() for t in trees], [t.tree_.node_count for t in trees])
    if isinstance(est, XGBClassifier):
        # text dump: one line per node, indented by one tab per level
        dumps = est.get_booster().get_dump()
        depths = [max(len(n) - len(n.lstrip("\t")) for n in d.splitlines()) for d in dumps]
        return _ensemble(depths, [len(d.splitlines()) for d in dumps])
    if isinstance(est, LGBMClassifier):
        trees = est.booster_.dump_model()["tree_info"]
        return _ensemble([_lgbm_depth(t["tree_structure"]) for t in trees],
                         [2 * t["num_leaves"] - 1 for t in trees])  # fmt: skip
    if isinstance(est, DecisionTreeClassifier):
        return {
            "tree_depth": est.get_depth(),
            "tree_nodes": est.tree_.node_count,
            "tree_leaves": est.get_n_leaves(),
        }
    if isinstance(est, SVC):
        return {"support_vectors": int(est.n_support_.sum())}
    return {}
