"""Predict with a final model: models/final/<task>.joblib + <task>.json, written by
scripts/save_final_models.py.

The input must hold exactly the model's 23 features, in the JSON's order, all numeric and
finite (an optional `label` column at the end is ignored and echoed as true_label). For every
row: the predicted label (the model's own class), its confidence (the calibrated probability of
that class where E15 adopted calibration, else the model's probability) and the 3 features that
pushed the model most towards that label (SHAP TreeExplainer on the tree model's input; values
are in the model's raw output units, log-odds, so they add up to the margin, not to a probability).
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from blindspy import calibration
from blindspy.data import ROOT

FINAL_DIR = ROOT / "models" / "final"
TIME_GROUPS = ("_fiat", "_biat", "_flowiat", "_active", "_idle")


class InputError(ValueError):
    """The input file does not match the model's feature list, order or types."""


def feature_unit(name: str) -> str:
    """Units verified in Phase 1 (reports/data_checks_v1.json, E00)."""
    if name == "duration" or name.endswith(TIME_GROUPS):
        return "microseconds"
    return {"flowBytesPerSecond": "bytes per second",
            "flowPktsPerSecond": "packets per second"}[name]  # fmt: skip


def write_final(model, meta: dict, folder: Path = FINAL_DIR, compress: int = 3) -> dict:
    """Save <task>.joblib (joblib compression `compress`) and <task>.json: meta plus the feature
    list with units, the class labels and the file's size and SHA-256. Returns the JSON."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{meta['task']}.joblib"
    joblib.dump(model, path, compress=compress)
    base = calibration.base_model(model)
    meta = {
        **meta,
        "features": [{"name": c, "unit": feature_unit(c)} for c in base.feature_names_in_],
        "missing_value_code": "-1 = not measurable (fewer than 2 packets in that direction, or "
                              "no active/idle period); never replace it with 0 (E00)",
        "class_labels": [str(c) for c in model.classes_],
        "model_file_bytes": path.stat().st_size,
        "model_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }  # fmt: skip
    (folder / f"{meta['task']}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load_final(task: str, folder: Path = FINAL_DIR) -> tuple[object, dict]:
    """The model and its JSON. joblib files can run code when loaded, so only the project's own
    model files are loaded: the file's SHA-256 must equal the one recorded in the JSON."""
    meta = json.loads((Path(folder) / f"{task}.json").read_text())
    path = Path(folder) / f"{task}.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["model_file_sha256"]:
        raise InputError(f"{path} does not match the SHA-256 in {task}.json; not loading it")
    return joblib.load(path), meta


def validate(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """The feature columns of df, checked against the model's JSON; raises InputError."""
    want = [f["name"] for f in meta["features"]]
    cols = list(df.columns)
    if cols and cols[-1] == "label":
        cols = cols[:-1]
    if not len(df):
        raise InputError("the input has no rows")
    missing = [c for c in want if c not in cols]
    extra = [c for c in cols if c not in want]
    if missing or extra:
        raise InputError(f"columns do not match the {meta['task']} model: missing {missing}, "
                         f"unexpected {extra}; expected exactly: {', '.join(want)}")  # fmt: skip
    if cols != want:
        raise InputError(f"columns are in the wrong order; expected: {', '.join(want)}")
    X = df[want]
    bad = [c for c in want if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        rows = {c: [int(i) + 1
                    for i in np.flatnonzero(pd.to_numeric(X[c], errors="coerce").isna())][:5]
                for c in bad}  # fmt: skip
        raise InputError(f"non-numeric values in {bad} (first bad data rows: {rows})")
    finite = np.isfinite(X.to_numpy(dtype=float))
    if not finite.all():
        r, c = np.argwhere(~finite)[0]
        raise InputError(f"missing or infinite value in column {want[c]}, data row {r + 1}; "
                         "every feature must be a number (-1 marks 'not measurable')")  # fmt: skip
    return X.astype(float)


def _model_input(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """What the pipeline's last step sees at prediction time (resampling steps only act in fit)."""
    for _, step in pipe.steps[:-1]:
        if not hasattr(step, "fit_resample"):
            X = step.transform(X)
    return X


def top_features(pipe, X: pd.DataFrame, labels, k: int = 3) -> list[list[tuple[str, float]]]:
    """Per row, the k features with the largest SHAP value towards the predicted label."""
    import shap

    est = pipe[-1]
    values = shap.TreeExplainer(est)(_model_input(pipe, X)).values
    classes = [str(c) for c in est.classes_]
    out = []
    for i, lab in enumerate(labels):
        j = classes.index(str(lab))
        if values.ndim == 3:
            v = values[i, :, j]
        else:  # binary: one column, towards classes[1]
            v = values[i] if j == 1 else -values[i]
        order = np.argsort(-v)[:k]
        out.append([(X.columns[f], float(v[f])) for f in order])
    return out


def predict(task: str, df: pd.DataFrame, folder: Path = FINAL_DIR) -> pd.DataFrame:
    model, meta = load_final(task, folder)
    X = validate(df, meta)
    label, conf, _ = calibration.predict_with_confidence(model, X)
    tops = top_features(calibration.base_model(model), X, label)
    out = pd.DataFrame({"row": np.arange(1, len(X) + 1), "label": label, "confidence": conf})
    for n in range(3):
        out[f"top_feature_{n + 1}"] = [f"{t[n][0]} ({t[n][1]:+.3f})" for t in tops]
    if "label" in df.columns:
        out.insert(2, "true_label", df["label"].to_numpy())
    return out
