"""Scores and plots for one set of predictions: every metric in the plan's Metrics table."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def top_confusion_pairs(cm: pd.DataFrame, n: int = 5) -> list[dict]:
    """The n largest off-diagonal cells: true class, predicted class, count (count > 0 only)."""
    pairs = [
        {"true": t, "predicted": p, "count": int(cm.loc[t, p])}
        for t in cm.index
        for p in cm.columns
        if t != p and cm.loc[t, p] > 0
    ]
    return sorted(pairs, key=lambda q: -q["count"])[:n]


def _plot_confusion(cm: pd.DataFrame, path: Path) -> None:
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    k = len(cm)
    fig, ax = plt.subplots(figsize=(1.1 + 0.55 * k, 0.9 + 0.5 * k))
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm.iat[i, j]}", ha="center", va="center", fontsize=7,
                    color="white" if norm.iat[i, j] > 0.5 else "black")  # fmt: skip
    ax.set_xticks(range(k), cm.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(k), cm.index, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (counts; colour = row share)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_pr(y: np.ndarray, scores: np.ndarray, classes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y == c, scores[:, i])
        ax.plot(rec, prec, lw=1, label=c)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall per class (one vs rest)", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_roc(y_pos: np.ndarray, score: np.ndarray, auc: float, positive: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_pos, score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC (positive class: {positive})", fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(
    y_true, y_pred, scores: np.ndarray, classes: list[str], out_dir: Path
) -> tuple[dict, dict]:
    """Metrics (name -> float) and a summary of named results; writes the files to out_dir.

    scores holds one column per class in `classes` order. Files: confusion_matrix.png/.csv,
    per_class_report.csv, confusion_pairs.csv, pr_curves.png, and roc_curve.png when there
    are two classes (positive class = classes[1]).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=classes, zero_division=0)
    m = {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float(np.average(f1, weights=support)),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
    }
    m["f1_gap_weighted_minus_macro"] = m["f1_weighted"] - m["f1_macro"]
    for i, c in enumerate(classes):
        m[f"precision.{c}"] = float(prec[i])
        m[f"recall.{c}"] = float(rec[i])
        m[f"f1.{c}"] = float(f1[i])
        m[f"avg_precision.{c}"] = float(average_precision_score(y == c, scores[:, i]))
    if len(classes) == 2:
        m["roc_auc"] = float(roc_auc_score(y == classes[1], scores[:, 1]))
        _plot_roc(
            y == classes[1], scores[:, 1], m["roc_auc"], classes[1], out_dir / "roc_curve.png"
        )

    cm = pd.DataFrame(confusion_matrix(y, p, labels=classes), index=classes, columns=classes)
    cm.index.name = "true \\ predicted"
    cm.to_csv(out_dir / "confusion_matrix.csv")
    _plot_confusion(cm, out_dir / "confusion_matrix.png")
    report = pd.DataFrame(
        {"class": classes, "precision": prec, "recall": rec, "f1": f1, "support": support}
    )
    report.to_csv(out_dir / "per_class_report.csv", index=False)
    pairs = top_confusion_pairs(cm)
    pd.DataFrame(pairs, columns=["true", "predicted", "count"]).to_csv(
        out_dir / "confusion_pairs.csv", index=False
    )
    _plot_pr(y, scores, classes, out_dir / "pr_curves.png")

    info = {
        "best_class_f1": classes[int(np.argmax(f1))],
        "worst_class_f1": classes[int(np.argmin(f1))],
        "best_class_recall": classes[int(np.argmax(rec))],
        "worst_class_recall": classes[int(np.argmin(rec))],
        "top_confusion_pairs": pairs,
    }
    return {k: float(v) for k, v in m.items()}, info
