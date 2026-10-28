"""Train and score one config. Returns params / metrics / tags and writes the run's files.

MLflow logging is done by scripts/run_experiment.py, so these functions can be tested
without a tracking store. The test set is never used here.
"""

import io
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from blindspy import evaluate, models

CV_N_JOBS = -1  # CV folds run in parallel; each model itself uses one thread (n_jobs=1)

CV_SCORES = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns="label"), df["label"]


def _fit_measured(pipe, X, y) -> tuple[float, float]:
    """Fit pipe; return (seconds, peak MB of Python-tracked memory during a fit).

    tracemalloc slows fitting (up to 2.3x for KNN), so the timed fit runs without it and
    peak memory comes from a second fit of a clone. pipe keeps the timed fit.
    """
    t0 = time.perf_counter()
    pipe.fit(X, y)
    seconds = time.perf_counter() - t0
    tracemalloc.start()
    clone(pipe).fit(X, y)
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


def _fit_rss(pipe, X, y) -> int:
    """Runs in a fresh child process: bytes the process's peak RSS rose above its RSS just
    before the fit. Unlike tracemalloc this sees C/C++ memory (XGBoost, LightGBM, libsvm)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True, check=True).stdout  # fmt: skip
    before = int(out) * 1024  # ps reports KB
    pipe.fit(X, y)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KB on Linux
    return peak - before


def _child_peak_rss_mb(pipe, X, y) -> float:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx) as pool:
        return pool.submit(_fit_rss, clone(pipe), X, y).result() / 2**20


def _model_sizes(pipe, out_dir: Path, save: bool, compress: int) -> tuple[int, int]:
    """(uncompressed joblib bytes, bytes of the file as saved with joblib compress=compress).

    The file is written to out_dir/model/model.joblib when save is true; compress=0 saves it
    uncompressed (Phase 2), so both numbers are then equal."""
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    raw = buf.getbuffer().nbytes
    if not save:
        if not compress:
            return raw, raw
        buf = io.BytesIO()
        joblib.dump(pipe, buf, compress=compress)
        return raw, buf.getbuffer().nbytes
    path = out_dir / "model" / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path, compress=compress)
    return raw, path.stat().st_size


def _cost(pipe, X_fit, y_fit, X_pred, out_dir: Path, save: bool, compress: int) -> dict:
    """Training time, memory (tracemalloc and child-process RSS), prediction time and model
    sizes; pipe is left fitted on (X_fit, y_fit)."""
    train_time, peak_mb = _fit_measured(pipe, X_fit, y_fit)
    raw, file_bytes = _model_sizes(pipe, out_dir, save, compress)
    return {
        "train_time_s": train_time,
        "peak_memory_mb": peak_mb,
        "child_peak_rss_mb": _child_peak_rss_mb(pipe, X_fit, y_fit),
        "predict_ms_per_1000": _predict_ms_per_1000(pipe, X_pred),
        "model_size_bytes": raw,
        "model_file_bytes": file_bytes,
    }


def _describe_step(step) -> str:
    """repr of a pipeline step; a FunctionTransformer is named by its function."""
    func = getattr(step, "func", None)
    return f"{type(step).__name__}({func.__name__})" if func is not None else repr(step)


def _base_params(cfg: dict, pipe, X: pd.DataFrame, seed: int, compress: int) -> dict:
    return {
        "task": cfg["task"],
        "timeout": cfg["timeout"],
        "model": cfg["model"],
        **{f"model.{k}": v for k, v in (cfg.get("params") or {}).items()},
        "estimator": repr(pipe[-1]),
        "pipeline_steps": " -> ".join(type(s).__name__ for _, s in pipe.steps),
        "preprocessing": "; ".join(f"{n}: {_describe_step(s)}" for n, s in pipe.steps[:-1])
        or "none",
        "seed": seed,
        "class_balancing": "none",
        "features": ",".join(X.columns),
        "n_features": X.shape[1],
        "save_model": bool(cfg.get("save_model", True)),
        "model_compress": compress,
        "model_n_jobs": str(pipe[-1].get_params().get("n_jobs", "n/a")),
        "cv_n_jobs": CV_N_JOBS,
    }


def train_and_score(
    cfg: dict, parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], seed: int,
    cv_folds: int, out_dir: Path, compress: int = 0,
) -> dict:  # fmt: skip
    """5-fold CV on train, fit on train, score on validation. parts = (train, val, test)."""
    out_dir = Path(out_dir)
    train, val, test = parts
    X_tr, y_tr = _xy(train)
    X_val, y_val = _xy(val)
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed)

    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    cv_res = cross_validate(clone(pipe), X_tr, y_tr, cv=cv, scoring=CV_SCORES, n_jobs=CV_N_JOBS)
    cv_time = time.perf_counter() - t0

    save = cfg.get("save_model", True)
    cost = _cost(pipe, X_tr, y_tr, X_val, out_dir, save, compress)
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
    metrics.update(cost)
    metrics.update({f"model_{k}": v for k, v in models.complexity(pipe).items()})
    has_imp = evaluate.feature_importance(pipe, list(X_tr.columns), out_dir / "eval")

    counts = {name: part["label"].value_counts().sort_index().to_dict()
              for name, part in (("train", train), ("val", val), ("test", test))}  # fmt: skip
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "class_counts.json").write_text(json.dumps(counts, indent=1))
    params = {
        **_base_params(cfg, pipe, X_tr, seed, compress),
        "protocol": "holdout: 5-fold stratified CV on train, fit on train, score on validation",
        "cv_folds": cv_folds,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "classes": ",".join(classes),
        **{f"class_counts.{k}": json.dumps(v) for k, v in counts.items()},
    }
    tags = {
        "protocol": "holdout",
        "feature_importance": "impurity" if has_imp else "n/a",
        **{k: v for k, v in info.items() if k.endswith(("f1", "recall"))},
    }
    tags["top_confusion_pairs"] = "; ".join(
        f"{q['true']} -> {q['predicted']}: {q['count']}" for q in info["top_confusion_pairs"]
    )
    return {"params": params, "metrics": metrics, "tags": tags, "info": info}


def paper_benchmark(
    cfg: dict, df: pd.DataFrame, seed: int, folds: int, out_dir: Path, compress: int = 0
) -> dict:
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
    metrics.update(_cost(pipe, X, y, X, out_dir, cfg.get("save_model", True), compress))
    metrics.update({f"model_{k}": v for k, v in models.complexity(pipe).items()})
    evaluate.feature_importance(pipe, list(X.columns), out_dir / "eval")

    params = {
        **_base_params(cfg, pipe, X, seed, compress),
        "protocol": f"paper: {folds}-fold stratified CV on the whole file, pooled predictions",
        "cv_folds": folds,
        "cv_n_jobs": 1,  # the folds run one after another here
        "n_rows": len(df),
        "classes": ",".join(classes),
        "rows_source": cfg.get("rows", "v1"),
        "class_counts.all": json.dumps(y.value_counts().sort_index().to_dict()),
    }
    tags = {"protocol": "paper_10fold", "rows_source": cfg.get("rows", "v1")}
    tags.update({k: v for k, v in info.items() if k.endswith(("f1", "recall"))})
    tags["top_confusion_pairs"] = "; ".join(
        f"{q['true']} -> {q['predicted']}: {q['count']}" for q in info["top_confusion_pairs"]
    )
    return {"params": params, "metrics": metrics, "tags": tags, "info": info}
