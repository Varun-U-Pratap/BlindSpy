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

from blindspy import calibration, evaluate, features, models

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


def _save_booster_config(pipe, out_dir: Path) -> None:
    """XGBoost: the resolved booster settings (defaults filled in) -> model/xgboost_config.json"""
    est = pipe[-1]
    if hasattr(est, "get_booster"):
        (out_dir / "model").mkdir(parents=True, exist_ok=True)
        (out_dir / "model" / "xgboost_config.json").write_text(est.get_booster().save_config())


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


def _estimator_params(est) -> str:
    """Every setting of the final estimator that is not None, as JSON. The repr is cut at 700
    characters and XGBoost shows its defaults as None; its resolved settings are saved in
    model/xgboost_config.json."""
    params = {k: v for k, v in est.get_params(deep=False).items() if v is not None}
    return json.dumps(params, sort_keys=True, default=str)


def _base_params(cfg: dict, pipe, X: pd.DataFrame, seed: int, compress: int) -> dict:
    return {
        "task": cfg["task"],
        "timeout": cfg["timeout"],
        "model": cfg["model"],
        **{f"model.{k}": v for k, v in (cfg.get("params") or {}).items()},
        "estimator": repr(pipe[-1]),
        "estimator_params": _estimator_params(pipe[-1]),
        "pipeline_steps": " -> ".join(type(s).__name__ for _, s in pipe.steps),
        "preprocessing": "; ".join(f"{n}: {_describe_step(s)}" for n, s in pipe.steps[:-1])
        or "none",
        "seed": seed,
        "class_balancing": cfg.get("balance") or "none",
        **({"builds_on": cfg["builds_on"]} if cfg.get("builds_on") else {}),
        **({"feature_set": cfg["feature_set"]} if cfg.get("feature_set") else {}),
        # E13 / E14 resampling (splits.resample), E16 (d) train_rows (kaggle.subsample);
        # n_train_rows = training rows actually used
        **{
            k: cfg[k]
            for k in ("train_fraction", "train_rows", "subsample_seed", "split_seed")
            if k in cfg
        },
        "n_train_rows": len(X),
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
    if cfg.get("feature_set"):  # E09: keep one feature set (names from the data's columns)
        keep = features.select_features(X_tr.columns, cfg["feature_set"])
        X_tr, X_val = X_tr[keep], X_val[keep]
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed, cfg.get("balance"))

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
    _save_booster_config(pipe, out_dir)

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


def calibration_run(
    cfg: dict, parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], seed: int,
    out_dir: Path, compress: int = 0,
) -> dict:  # fmt: skip
    """E15 on validation: the config's model fitted on train (as its source run), its validation
    ECE, then the isotonic calibrator fitted on train and its validation ECE, both with the
    uncalibrated model's predicted class. Adoption: uncalibrated ECE above 0.05 and calibrated
    ECE lower. The test set is never used."""
    out_dir = Path(out_dir)
    train, val, test = parts
    X_tr, y_tr = _xy(train)
    X_val, y_val = _xy(val)
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed, cfg.get("balance"))
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    classes = [str(c) for c in pipe.classes_]
    pred = pipe.predict(X_val)
    m, info = evaluate.evaluate(y_val, pred, models.class_scores(pipe, X_val), classes,
                                out_dir / "eval")  # fmt: skip
    t0 = time.perf_counter()
    cal = calibration.fit_calibrated(pipe, X_tr, y_tr, seed)
    cal_time = time.perf_counter() - t0
    cal_pred, cal_conf, cal_proba = calibration.predict_with_confidence(cal, X_val)
    raw_conf = pipe.predict_proba(X_val).max(axis=1)
    correct = pred == y_val.to_numpy()
    bins = {"uncalibrated": calibration.reliability_bins(raw_conf, correct),
            "isotonic": calibration.reliability_bins(cal_conf, correct)}  # fmt: skip
    ece_raw = calibration.ece(raw_conf, correct)
    ece_cal = calibration.ece(cal_conf, correct)
    adopt = ece_raw > calibration.ADOPT_ABOVE and ece_cal < ece_raw

    (out_dir / "calibration").mkdir(parents=True, exist_ok=True)
    pd.concat(bins, names=["probabilities"]).to_csv(
        out_dir / "calibration" / "reliability_bins.csv"
    )
    calibration.plot_reliability(
        bins, f"{cfg['task']} validation: ECE {ece_raw:.4f} uncalibrated, {ece_cal:.4f} isotonic",
        out_dir / "calibration" / "reliability.png")  # fmt: skip
    pd.DataFrame({"raw_row": val.index, "true": y_val.to_numpy(), "predicted": pred,
                  "confidence_uncalibrated": raw_conf, "confidence_isotonic": cal_conf,
                  **{f"isotonic_proba.{c}": cal_proba[:, i] for i, c in enumerate(classes)}}
                 ).to_csv(out_dir / "calibration" / "val_predictions.csv", index=False)  # fmt: skip
    raw, file_bytes = _model_sizes(cal, out_dir, cfg.get("save_model", True), compress)

    metrics = {f"val_{k}": v for k, v in m.items()}
    metrics.update({
        "val_ece_uncalibrated": ece_raw, "val_ece_isotonic": ece_cal,
        "val_mean_confidence_uncalibrated": float(raw_conf.mean()),
        "val_mean_confidence_isotonic": float(cal_conf.mean()),
        # the calibrator's base model must predict exactly as the plain fit (1.0 = every row)
        "val_label_agreement_base_vs_plain": float((cal_pred == pred).mean()),
        # how often the calibrated argmax would differ from the predicted class (never used)
        "val_calibrated_argmax_differs": float((cal_proba.argmax(axis=1)
                                                != np.searchsorted(classes, pred)).mean()),
        "train_time_s": fit_time, "calibration_fit_time_s": cal_time,
        "model_size_bytes": raw, "model_file_bytes": file_bytes,
    })  # fmt: skip
    params = {
        **_base_params(cfg, pipe, X_tr, seed, compress),
        "protocol": "calibration: fit on train, ECE on validation (uncalibrated and isotonic)",
        "calibration.method": "CalibratedClassifierCV(method='isotonic', cv=StratifiedKFold(5, "
                              f"shuffle=True, random_state={seed}), ensemble=False)",
        "calibration.ece_bins": calibration.BINS,
        "calibration.adopt_rule": f"uncalibrated ECE > {calibration.ADOPT_ABOVE} and "
                                  "isotonic ECE lower",
        "calibration.confidence": "probability of the uncalibrated model's predicted class",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "classes": ",".join(classes),
    }  # fmt: skip
    tags = {"protocol": "calibration", "calibration_adopted": str(adopt)}
    return {"params": params, "metrics": metrics, "tags": tags, "info": info}


def final_parts(
    parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], rehearsal: bool
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(rows to fit on, rows to score, name of the scored rows) for E12.

    Official: fit on train + validation, score the test rows once. Rehearsal (scratch store
    only): fit on train, score validation; the test rows are not returned at all."""
    train, val, test = parts
    if rehearsal:
        return train, val, "rehearsal"
    return pd.concat([train, val]), test, "test"


def final_run(
    cfg: dict, fit: pd.DataFrame, score: pd.DataFrame, name: str, seed: int, cv_folds: int,
    out_dir: Path, compress: int = 0,
) -> dict:  # fmt: skip
    """E12: fit the config's model on `fit`, 5-fold CV on the same rows, predict `score` once.

    Metrics are prefixed with `name` ("test", or "rehearsal" in the scratch rehearsal). Every
    row's prediction is saved (predictions/<name>_predictions.csv: raw_row, true, predicted,
    confidence, the model's class scores and, with cfg["calibrate"] == "isotonic", the
    calibrated probabilities), so intervals and tests are computed later without refitting.
    With calibration (E15) the predicted class stays the model's own; the confidence is the
    calibrated probability of that class, and the saved model is the calibrator (its base
    model is the plain fit)."""
    out_dir = Path(out_dir)
    X_fit, y_fit = _xy(fit)
    X_s, y_s = _xy(score)
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed, cfg.get("balance"))
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    cv_res = cross_validate(clone(pipe), X_fit, y_fit, cv=cv, scoring=CV_SCORES, n_jobs=CV_N_JOBS)
    cv_time = time.perf_counter() - t0

    cost = _cost(pipe, X_fit, y_fit, X_s, out_dir, False, compress)
    classes = [str(c) for c in pipe.classes_]
    pred = pipe.predict(X_s)
    scores = np.asarray(models.class_scores(pipe, X_s), dtype=float)
    m, info = evaluate.evaluate(y_s, pred, scores, classes, out_dir / "eval")
    has_proba = hasattr(pipe, "predict_proba")
    correct = pred == y_s.to_numpy()
    conf_raw = scores[np.arange(len(pred)), np.searchsorted(classes, pred)] if has_proba else None
    saved, conf, cal_proba = pipe, conf_raw, None
    calibrate = cfg.get("calibrate") or "none"
    if calibrate == "isotonic":
        t1 = time.perf_counter()
        saved = calibration.fit_calibrated(pipe, X_fit, y_fit, seed)
        cost["calibration_fit_time_s"] = time.perf_counter() - t1
        label, conf, cal_proba = calibration.predict_with_confidence(saved, X_s)
        if not (label == pred).all():
            raise RuntimeError("calibrator's base model predicts differently from the plain fit")
    elif calibrate != "none":
        raise ValueError(f"unknown calibrate {calibrate!r}")
    raw, file_bytes = _model_sizes(saved, out_dir, cfg.get("save_model", True), compress)
    cost.update({"model_size_bytes": raw, "model_file_bytes": file_bytes})

    metrics = {f"{name}_{k}": v for k, v in m.items()}
    for cv_name in CV_SCORES:
        metrics[f"cv_{cv_name}_mean"] = float(cv_res[f"test_{cv_name}"].mean())
        metrics[f"cv_{cv_name}_std"] = float(cv_res[f"test_{cv_name}"].std())
    metrics[f"gen_gap_f1_macro_cv_minus_{name}"] = metrics["cv_f1_macro_mean"] - m["f1_macro"]
    if has_proba:
        metrics[f"{name}_ece_uncalibrated"] = calibration.ece(conf_raw, correct)
        metrics[f"{name}_ece_used"] = calibration.ece(conf, correct)
        metrics[f"{name}_mean_confidence_used"] = float(np.mean(conf))
    metrics["cv_time_s"] = cv_time
    metrics.update(cost)
    metrics.update({f"model_{k}": v for k, v in models.complexity(pipe).items()})
    has_imp = evaluate.feature_importance(pipe, list(X_fit.columns), out_dir / "eval")
    _save_booster_config(pipe, out_dir)

    kind = "proba" if has_proba else "decision"
    cols = {f"{kind}.{c}": scores[:, i] for i, c in enumerate(classes)}
    preds = pd.DataFrame({"raw_row": score.index, "true": y_s.to_numpy(), "predicted": pred,
                          "confidence": conf if has_proba else np.nan, **cols})  # fmt: skip
    if cal_proba is not None:
        for i, c in enumerate(classes):
            preds[f"calibrated_proba.{c}"] = cal_proba[:, i]
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions" / f"{name}_predictions.csv", index=False)
    counts = {"fit": fit["label"].value_counts().sort_index().to_dict(),
              name: score["label"].value_counts().sort_index().to_dict()}  # fmt: skip
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "class_counts.json").write_text(json.dumps(counts, indent=1))

    fit_desc = "train" if name == "rehearsal" else "train + validation"
    score_desc = "validation (rehearsal)" if name == "rehearsal" else "test, once"
    params = {
        **_base_params(cfg, pipe, X_fit, seed, compress),
        "protocol": f"final: {cv_folds}-fold stratified CV and fit on {fit_desc}, "
                    f"score {score_desc}",
        "cv_folds": cv_folds, "n_fit": len(fit), f"n_{name}": len(score),
        "classes": ",".join(classes), "score_kind": kind, "calibrate": calibrate,
        **({"calibration.method": "CalibratedClassifierCV(method='isotonic', cv=StratifiedKFold("
                                  f"5, shuffle=True, random_state={seed}), ensemble=False)"}
           if calibrate == "isotonic" else {}),
        **{k: cfg[k] for k in ("source_run_id", "source_val_f1_macro", "role") if k in cfg},
        **{f"class_counts.{k}": json.dumps(v) for k, v in counts.items()},
    }  # fmt: skip
    tags = {
        "protocol": "final_test" if name == "test" else "final_rehearsal",
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


def kaggle_run(
    cfg: dict, train: pd.DataFrame, test: pd.DataFrame, seed: int, out_dir: Path,
    compress: int = 0,
) -> dict:  # fmt: skip
    """E16 on the Kaggle CSV (data version kaggle-k1): Isnan et al.'s 80/20 protocol.

    With cfg["grid"]: their GridSearchCV (5-fold, scikit-learn's default stratified folds
    without shuffling, accuracy as the score) over the grid on the training rows, the best
    settings refitted on all training rows; otherwise the config's fixed model. Then one score
    on the 80/20 split's test rows (Kaggle rows, not our v1 test set)."""
    from sklearn.model_selection import GridSearchCV

    out_dir = Path(out_dir)
    feats = [c for c in train.columns if c not in ("label", "source_task", "timeout",
                                                   "source_row")]  # fmt: skip
    X_tr, y_tr, X_te, y_te = train[feats], train["label"], test[feats], test["label"]
    pipe = models.build_model(cfg["model"], cfg.get("params"), seed)
    extra, grid_metrics = {}, {}
    if cfg.get("grid"):
        grid = {f"model__{k}": v for k, v in cfg["grid"].items()}
        t0 = time.perf_counter()
        gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=CV_N_JOBS, refit=False)
        gs.fit(X_tr, y_tr)
        grid_metrics = {"grid_time_s": time.perf_counter() - t0,
                        "grid_best_cv_accuracy": float(gs.best_score_),
                        "grid_best_cv_accuracy_std":
                            float(gs.cv_results_["std_test_score"][gs.best_index_]),
                        "grid_configs": len(gs.cv_results_["params"]),
                        "grid_total_fits": len(gs.cv_results_["params"]) * 5}  # fmt: skip
        best = {k.removeprefix("model__"): v for k, v in gs.best_params_.items()}
        pipe = models.build_model(cfg["model"], {**(cfg.get("params") or {}), **best}, seed)
        extra = {
            "grid": json.dumps(cfg["grid"]),
            "grid_best_params": json.dumps(best),
            "grid_cv": "GridSearchCV cv=5 (stratified, no shuffle), scoring=accuracy",
        }
        (out_dir / "grid").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gs.cv_results_).drop(columns="params").to_csv(
            out_dir / "grid" / "cv_results.csv", index=False)  # fmt: skip
    cost = _cost(pipe, X_tr, y_tr, X_te, out_dir, cfg.get("save_model", True), compress)
    classes = [str(c) for c in pipe.classes_]
    pred = pipe.predict(X_te)
    m, info = evaluate.evaluate(y_te, pred, models.class_scores(pipe, X_te), classes,
                                out_dir / "eval")  # fmt: skip
    metrics = {f"kaggle_test_{k}": v for k, v in m.items()}
    if test["timeout"].nunique() > 1:  # test rows from several timeouts: accuracy per timeout
        for t, rows in test.groupby("timeout"):
            hit = pred[test["timeout"].to_numpy() == t] == rows["label"].to_numpy()
            metrics[f"kaggle_test_accuracy_{t}s"] = float(hit.mean())
    metrics.update(grid_metrics)
    metrics.update(cost)
    metrics.update({f"model_{k}": v for k, v in models.complexity(pipe).items()})
    _save_booster_config(pipe, out_dir)
    counts = {name: part["label"].value_counts().sort_index().to_dict()
              for name, part in (("train", train), ("test", test))}  # fmt: skip
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "class_counts.json").write_text(json.dumps(counts, indent=1))
    params = {
        **_base_params(cfg, pipe, X_tr, seed, compress), **extra,
        "protocol": f"kaggle {cfg['split']}: fit on the split's training rows, score once on its "
                    "test rows",
        "split": cfg["split"], "n_train": len(train), "n_test": len(test),
        "timeouts_in_train": ",".join(str(t) for t in sorted(train["timeout"].unique())),
        "classes": ",".join(classes),
    }  # fmt: skip
    tags = {"protocol": "kaggle_isnan" if cfg["split"] == "isnan" else "kaggle_per_timeout",
            **{k: v for k, v in info.items() if k.endswith(("f1", "recall"))}}  # fmt: skip
    return {"params": params, "metrics": metrics, "tags": tags, "info": info}
