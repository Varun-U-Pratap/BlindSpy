"""SHAP explanations of the final models (models/final), for notebook 05 and the report.

TreeExplainer on the tree model at the end of each pipeline, fed the input that model sees at
prediction time (resampling steps act only in fit). Values are in the model's raw output units
(log-odds; softmax margins for the multi-class models) and add up to the margin. Because many of
the 23 features are correlated, the values describe how the model uses them, not what causes the
traffic patterns.
"""

import numpy as np
import pandas as pd

from blindspy import calibration
from blindspy.features import feature_groups
from blindspy.predict import _model_input

EXAMPLES = ["most confident correct", "least confident correct", "most confident error",
            "weakest class, median confidence"]  # fmt: skip


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(rows, features, classes) SHAP values of the base model and its class names. A binary
    model gives one column towards classes[1]; it is returned as [-v, v] so every task has one
    column per class."""
    import shap

    pipe = calibration.base_model(model)
    est = pipe[-1]
    v = shap.TreeExplainer(est)(_model_input(pipe, X)).values
    if v.ndim == 2:
        v = np.stack([-v, v], axis=2)
    return v, [str(c) for c in est.classes_]


def importance(values: np.ndarray, features) -> pd.Series:
    """Mean |SHAP| per feature over rows, then averaged over classes; largest first."""
    return pd.Series(np.abs(values).mean(axis=0).mean(axis=1), index=list(features)).sort_values(
        ascending=False, kind="mergesort"
    )


def group_table(imp: pd.Series) -> pd.DataFrame:
    """Mean |SHAP| summed per E09 feature group, with each group's share of the total."""
    groups = feature_groups(list(imp.index))
    total = imp.sum()
    return pd.DataFrame([{"group": g, "features": len(cols),
                          "mean |SHAP| (sum)": imp[cols].sum(), "share": imp[cols].sum() / total}
                         for g, cols in groups.items()])  # fmt: skip


def pick_examples(y, label, conf, weakest: str) -> dict[str, int]:
    """Row positions of the 4 worked examples, by a rule fixed before looking at them (ties go to
    the first row in raw_row order): the most and the least confident correct prediction, the most
    confident wrong prediction and the correct prediction of the weakest class (lowest test F1)
    with the median confidence of that class's correct predictions."""
    y, label, conf = np.asarray(y).astype(str), np.asarray(label).astype(str), np.asarray(conf)
    ok = np.flatnonzero(y == label)
    bad = np.flatnonzero(y != label)
    weak = ok[y[ok] == weakest]
    order = weak[np.argsort(conf[weak], kind="mergesort")]
    picks = {EXAMPLES[0]: ok[np.argmax(conf[ok])], EXAMPLES[1]: ok[np.argmin(conf[ok])],
             EXAMPLES[3]: order[(len(order) - 1) // 2]}  # fmt: skip
    if len(bad):
        picks[EXAMPLES[2]] = bad[np.argmax(conf[bad])]
    return {k: int(picks[k]) for k in EXAMPLES if k in picks}


def example_rows(values, classes, X, pos: int, label: str, k: int = 5) -> pd.DataFrame:
    """The k features with the largest SHAP value towards the predicted label, for one row."""
    v = values[pos, :, classes.index(str(label))]
    top = np.argsort(-v, kind="mergesort")[:k]
    return pd.DataFrame({"rank": np.arange(1, k + 1), "feature": X.columns[top],
                         "value": X.iloc[pos, top].to_numpy(), "SHAP": v[top]})  # fmt: skip
