"""Train and score one config. Returns params / metrics / tags and writes the run's files.

MLflow logging is done by scripts/run_experiment.py, so these functions can be tested
without a tracking store. The test set is never used here.
"""

import io
import json
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from blindspy import evaluate, models

CV_SCORES = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns="label"), df["label"]


def _fit_measured(pipe, X, y) -> tuple[float, float]:
    """Fit pipe; return (seconds, peak MB of Python-tracked memory during the fit)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    pipe.fit(X, y)
    seconds = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return seconds, peak


def _predict_ms_per_1000(pipe, X, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipe.predict(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) / len(X) * 1000 * 1000


def _model_size(pipe, out_dir: Path, save: bool) -> int:
    """Bytes of the joblib file; written to out_dir/model/model.joblib when save is true."""
    if not save:
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        return buf.getbuffer().nbytes
    path = out_dir / "model" / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path.stat().st_size


def _base_params(cfg: dict, pipe, X: pd.DataFrame, seed: int) -> dict:
    return {
        "task": cfg["task"],
        "timeout": cfg["timeout"],
        "model": cfg["model"],
        **{f"model.{k}": v for k, v in (cfg.get("params") or {}).items()},
        "estimator": repr(pipe[-1]),
        "pipeline_steps": " -> ".join(type(s).__name__ for _, s in pipe.steps),
        "seed": seed,
        "class_balancing": "none",
        "features": ",".join(X.columns),
        "n_features": X.shape[1],
        "save_model": bool(cfg.get("save_model", True)),
    }


def train_and_score(
    cfg: dict, parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], seed: int,
    cv_folds: int, out_dir: Path,
) -> dict:  # fmt: skip
    """5-fold CV on train, fit on train, score on validation. parts = (train, val, test)."""
    out_dir = Path(out_dir)
    train, val, test = parts
    X_tr, y_tr = _xy(train)
    X_val, y_val = _xy(val)
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed)

    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    cv_res = cross_validate(clone(pipe), X_tr, y_tr, cv=cv, scoring=CV_SCORES, n_jobs=-1)
    cv_time = time.perf_counter() - t0

    train_time, peak_mb = _fit_measured(pipe, X_tr, y_tr)
    classes = [str(c) for c in pipe.classes_]
    pred = pipe.predict(X_val)
    m, info = evaluate.evaluate(y_val, pred, models.class_scores(pipe, X_val), classes,
                                out_dir / "eval")  # fmt: skip

    metrics = {f"val_{k}": v for k, v in m.items()}
    for name in CV_SCORES:
        metrics[f"cv_{name}_mean"] = float(cv_res[f"test_{name}"].mean())
        metrics[f"cv_{name}_std"] = float(cv_res[f"test_{name}"].std())
    metrics["gen_gap_f1_macro"] = metrics["cv_f1_macro_mean"] - metrics["val_f1_macro"]
    metrics["cv_time_s"] = cv_time
    metrics["train_time_s"] = train_time
    metrics["peak_memory_mb"] = peak_mb
    metrics["predict_ms_per_1000"] = _predict_ms_per_1000(pipe, X_val)
    metrics["model_size_bytes"] = _model_size(pipe, out_dir, cfg.get("save_model", True))
    metrics.update({f"model_{k}": v for k, v in models.complexity(pipe).items()})

    counts = {name: part["label"].value_counts().sort_index().to_dict()
              for name, part in (("train", train), ("val", val), ("test", test))}  # fmt: skip
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "class_counts.json").write_text(json.dumps(counts, indent=1))
    params = {
        **_base_params(cfg, pipe, X_tr, seed),
        "protocol": "holdout: 5-fold stratified CV on train, fit on train, score on validation",
        "cv_folds": cv_folds,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "classes": ",".join(classes),
    }
    tags = {
        "protocol": "holdout",
        **{k: v for k, v in info.items() if k.endswith(("f1", "recall"))},
    }
    tags["top_confusion_pairs"] = "; ".join(
        f"{q['true']} -> {q['predicted']}: {q['count']}" for q in info["top_confusion_pairs"]
    )
    return {"params": params, "metrics": metrics, "tags": tags, "info": info}


def paper_benchmark(cfg: dict, df: pd.DataFrame, seed: int, folds: int, out_dir: Path) -> dict:
    """The paper's protocol: stratified k-fold CV on the whole file, predictions pooled
    over the folds (as Weka reports them). Never used for model selection."""
    out_dir = Path(out_dir)
    X, y = _xy(df)
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed)
    classes = sorted(str(c) for c in y.unique())
    pred = np.empty(len(y), dtype=object)
    scores = np.zeros((len(y), len(classes)))
    fold_f1 = []
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    for tr, te in cv.split(X, y):
        p = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        pred[te] = p.predict(X.iloc[te])
        cols = [classes.index(str(c)) for c in p.classes_]
        scores[np.ix_(te, cols)] = models.class_scores(p, X.iloc[te])
        fold_f1.append(f1_score(y.iloc[te], pred[te].astype(str), labels=classes,
                                average="macro", zero_division=0))  # fmt: skip
    cv_time = time.perf_counter() - t0
    m, info = evaluate.evaluate(y, pred.astype(str), scores, classes, out_dir / "eval")

    metrics = {f"bench_{k}": v for k, v in m.items()}
    metrics["fold_f1_macro_mean"] = float(np.mean(fold_f1))
    metrics["fold_f1_macro_std"] = float(np.std(fold_f1))
    metrics["cv_time_s"] = cv_time
    train_time, peak_mb = _fit_measured(pipe, X, y)
    metrics["train_time_s"] = train_time
    metrics["peak_memory_mb"] = peak_mb
    metrics["predict_ms_per_1000"] = _predict_ms_per_1000(pipe, X)
    metrics["model_size_bytes"] = _model_size(pipe, out_dir, cfg.get("save_model", True))
    metrics.update({f"model_{k}": v for k, v in models.complexity(pipe).items()})

    params = {
        **_base_params(cfg, pipe, X, seed),
        "protocol": f"paper: {folds}-fold stratified CV on the whole file, pooled predictions",
        "cv_folds": folds,
        "n_rows": len(df),
        "classes": ",".join(classes),
        "rows_source": cfg.get("rows", "v1"),
    }
    tags = {"protocol": "paper_10fold", "rows_source": cfg.get("rows", "v1")}
    tags.update({k: v for k, v in info.items() if k.endswith(("f1", "recall"))})
    tags["top_confusion_pairs"] = "; ".join(
        f"{q['true']} -> {q['predicted']}: {q['count']}" for q in info["top_confusion_pairs"]
    )
    return {"params": params, "metrics": metrics, "tags": tags, "info": info}
