"""E15 confidence calibration (E12 addendum item 3, decided 31 Oct before the test set).

ECE: confidence = the probability given to the predicted class; 15 equal-width bins on [0, 1];
ECE = sum over bins of (rows in bin / all rows) x |accuracy in bin - mean confidence in bin|.

The calibrator is CalibratedClassifierCV(method="isotonic", cv=5, ensemble=False): it fits the
base model once on all the rows it is given (calibrated_classifiers_[0].estimator) and learns
the isotonic mapping from 5-fold out-of-fold predictions. Calibration changes only the reported
confidence: the predicted class is always the base model's own prediction, and the confidence
is the calibrated probability of THAT class. CalibratedClassifierCV.predict returns the
calibrated argmax, which can differ, so it is never used.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

BINS = 15
ADOPT_ABOVE = 0.05  # adopt only if the uncalibrated validation ECE is above this (E12 addendum)


def reliability_bins(confidence, correct, bins: int = BINS) -> pd.DataFrame:
    """One row per equal-width bin: rows, mean confidence, accuracy (empty bins have 0 rows).
    A confidence of exactly 1.0 falls in the last bin."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": b, "low": edges[b], "high": edges[b + 1], "rows": int(m.sum()),
                     "mean_confidence": float(confidence[m].mean()) if m.any() else np.nan,
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def ece(confidence, correct, bins: int = BINS) -> float:
    t = reliability_bins(confidence, correct, bins)
    t = t[t["rows"] > 0]
    gap = (t["accuracy"] - t["mean_confidence"]).abs()
    return float((t["rows"] * gap).sum() / t["rows"].sum())


def fit_calibrated(pipe, X, y) -> CalibratedClassifierCV:
    """Isotonic calibrator around an unfitted clone of pipe, fitted on (X, y) only."""
    cal = CalibratedClassifierCV(clone(pipe), method="isotonic", cv=5, ensemble=False)
    return cal.fit(X, y)


def base_model(model):
    """The model that predicts the class: the calibrator's base model, or the model itself."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def predict_with_confidence(model, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(predicted label, confidence, probability matrix in model.classes_ order).

    The label is the base model's prediction; the probabilities are calibrated when model is a
    CalibratedClassifierCV, and the confidence is the probability of the predicted label."""
    label = np.asarray(base_model(model).predict(X))
    proba = model.predict_proba(X)
    cols = np.searchsorted(np.asarray(model.classes_), label)
    return label, proba[np.arange(len(label)), cols], proba


def plot_reliability(tables: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    """Reliability diagram: accuracy against mean confidence per bin, one line per table."""
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for name, t in tables.items():
        t = t[t["rows"] > 0]
        ax.plot(t["mean_confidence"], t["accuracy"], "o-", ms=3, lw=1.2, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
