"""Independent recheck of E12: recomputes every test score, interval, paired test and verdict
from the MLflow database and the saved prediction files with plain scikit-learn / numpy (no
blindspy.results code), and compares each value with reports/phase4_tables.md.

Usage: uv run python scripts/check_e12_independent.py > reports/e12_independent_check.txt
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
B, SEED, ALPHA = 10_000, 42, 0.05


def resample_indices(y):
    """Within each true class (sorted), all B resamples at once: the order fixed in E12."""
    rng = np.random.default_rng(SEED)
    pos = [np.flatnonzero(y == c) for c in np.unique(y)]
    return np.concatenate([q[rng.integers(0, len(q), size=(B, len(q)))] for q in pos], axis=1)


def macro_f1_draws(y, yh, idx):
    cls = np.unique(np.concatenate([y, yh]))
    k = len(cls)
    yc, hc = np.searchsorted(cls, y), np.searchsorted(cls, yh)
    out = np.empty(len(idx))
    for b, ii in enumerate(idx):
        cm = np.bincount(yc[ii] * k + hc[ii], minlength=k * k).reshape(k, k)
        tp = np.diag(cm)
        den = cm.sum(0) + cm.sum(1)
        out[b] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)[den > 0].mean()
    return out


def main() -> int:
    db = sqlite3.connect(ROOT / "mlruns" / "utsav-s-r" / "mlflow.db")
    runs = pd.read_sql("select run_uuid, name from runs where name like 'E12-%'", db)
    par = pd.read_sql("select run_uuid, key, value from params", db)
    met = pd.read_sql("select run_uuid, key, value from latest_metrics", db)
    p_ = lambda r, k: par[(par.run_uuid == r) & (par.key == k)].value.iloc[0]  # noqa: E731
    m_ = lambda r, k: met[(met.run_uuid == r) & (met.key == k)].value.iloc[0]  # noqa: E731
    fails, rows, preds = [], [], {}
    print(f"{len(runs)} E12 runs in the store")
    for _, x in runs.iterrows():
        r, name, task = x["run_uuid"], x["name"], p_(x["run_uuid"], "task")
        art = ROOT / "mlruns" / "utsav-s-r" / "artifacts" / r / "artifacts" / "predictions"
        p = pd.read_csv(art / "test_predictions.csv").sort_values("raw_row", ignore_index=True)
        split = json.loads((ROOT / "data/processed/v1/splits" / f"{task}_15s.json").read_text())
        if list(p["raw_row"]) != sorted(split["test"]):
            fails.append(f"{name}: prediction rows != test rows")
        y, yh = p["true"].astype(str).to_numpy(), p["predicted"].astype(str).to_numpy()
        mf, acc = f1_score(y, yh, average="macro"), accuracy_score(y, yh)
        if abs(mf - m_(r, "test_f1_macro")) > 1e-12 or abs(acc - m_(r, "test_accuracy")) > 1e-12:
            fails.append(f"{name}: logged F1/accuracy differ")
        if task == "A1":
            s = p["proba.VPN"] if "proba.VPN" in p else p["decision.VPN"]
            if abs(roc_auc_score(y == "VPN", s) - m_(r, "test_roc_auc")) > 1e-9:
                fails.append(f"{name}: ROC-AUC differs")
        preds[name] = (y, yh)
        rows.append({"name": name, "task": task, "role": p_(r, "role"), "test": mf,
                     "val": float(p_(r, "source_val_f1_macro"))})  # fmt: skip
    t = pd.DataFrame(rows)
    draws, values = {}, {}
    for _, g in t.groupby("task"):
        idx = resample_indices(preds[g["name"].iloc[0]][0])
        for name in g["name"]:
            y, yh = preds[name]
            draws[name] = macro_f1_draws(y, yh, idx)
            a = (y[idx] == yh[idx]).mean(axis=1)
            f = draws[name]
            values[name] = [f"{np.percentile(v, 2.5):.4f}-{np.percentile(v, 97.5):.4f}"
                            for v in (f, a)]  # fmt: skip
    tests = []
    for task, g in t.groupby("task"):
        h = f"E12-final-{task}"
        for name in g[g["name"] != h]["name"]:
            d = draws[h] - draws[name]
            pv = min(1, 2 * min((1 + (d <= 0).sum()) / (B + 1), (1 + (d >= 0).sum()) / (B + 1)))
            tests.append(
                {
                    "h": h,
                    "o": name,
                    "p": pv,
                    "ci": f"{np.percentile(d, 2.5):.4f}-{np.percentile(d, 97.5):.4f}",
                }
            )
    x = pd.DataFrame(tests).sort_values("p", kind="mergesort", ignore_index=True)
    x["holm"] = np.minimum(np.maximum.accumulate((len(x) - x.index) * x["p"]), 1)
    x["rej"] = x["holm"] <= ALPHA
    md = (ROOT / "reports" / "phase4_tables.md").read_text().splitlines()
    for name, (ci, aci) in values.items():
        if not any(f"| {name} |" in line and ci in line and aci in line for line in md):
            fails.append(f"{name}: interval {ci} / {aci} not in phase4_tables.md")
    for _, r in x.iterrows():
        line = [ln for ln in md if f"| {r.h} | {r.o} |" in ln]
        want = [f"{r.p:.4f}", f"{r.holm:.4f}", r.ci, "| yes |" if r.rej else "| no |"]
        if not line or not all(w in line[0] for w in want):
            fails.append(f"test {r.h} vs {r.o}: {want} not in phase4_tables.md")
    head = t[t["role"] == "headline"].set_index("task")
    gap = (head["test"] - head["val"]).round(4).to_dict()
    old = t[t["role"] == "old_rule"].set_index("task")["test"]
    rho = {k: round(float(spearmanr(g["val"], g["test"]).statistic), 4)
           for k, g in t[t["role"] != "old_rule"].groupby("task")}  # fmt: skip
    print("test macro F1, headline:", head["test"].round(4).to_dict())
    print("(a) test - val:", gap, "-> supported" if all(abs(v) <= 0.03 for v in gap.values())
          else "-> NOT supported")  # fmt: skip
    print("old-rule picks (not comparison members):", old.round(4).to_dict())
    print("(c) Spearman:", rho)
    print(f"{len(x)} paired tests, {int(x['rej'].sum())} clear after Holm")
    print(f"{len(values)} intervals and {len(x)} tests compared with reports/phase4_tables.md")
    print("FAILED:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
