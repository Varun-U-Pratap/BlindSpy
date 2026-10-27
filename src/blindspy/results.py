"""Tables and threshold checks built from reports/history.csv (no number typed by hand)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from blindspy.data import CORE_TASKS, ROOT, TIMEOUTS

BASELINE_MODELS = ["majority", "tree", "knn", "logreg", "svm_rbf"]
MODEL_EXP = {"majority": "E01", "tree": "E02", "knn": "E03", "logreg": "E03", "svm_rbf": "E03"}
BENCH_ALGO = {"j48": "C4.5", "ibk": "KNN"}


def load_history(path: Path = ROOT / "reports" / "history.csv") -> pd.DataFrame:
    """Finished runs from committed + pushed code, the newest run per run name."""
    h = pd.read_csv(path, low_memory=False)
    h = h[(h["status"] == "FINISHED") & (h["tags.git_dirty"].astype(str) == "False")]
    h = h[h["tags.git_pushed"].astype(str) == "True"]
    h = h.sort_values("start_time").drop_duplicates("run_name", keep="last")
    h["params.timeout"] = pd.to_numeric(h["params.timeout"], errors="coerce")
    return h


def baselines(h: pd.DataFrame) -> pd.DataFrame:
    """The 80 holdout baseline runs (E01-E03), one row each."""
    b = h[h["tags.experiment_id"].isin(["E01", "E02", "E03"]) & (h["tags.protocol"] == "holdout")]
    return b[b["params.model"].isin(BASELINE_MODELS)].copy()


def benchmark(h: pd.DataFrame) -> pd.DataFrame:
    b = h[h["tags.protocol"] == "paper_10fold"].copy()
    b["algo"] = b["run_name"].str.split("-").str[2]
    return b


def _cell(b: pd.DataFrame, model: str, task: str, t: int) -> pd.Series:
    row = b[(b["params.model"] == model) & (b["params.task"] == task) & (b["params.timeout"] == t)]
    if len(row) != 1:
        raise ValueError(f"expected 1 run for {model} {task} {t}s, found {len(row)}")
    return row.iloc[0]


def score_grid(b: pd.DataFrame, metric: str, with_std: str | None = None) -> pd.DataFrame:
    """Rows = task x timeout, columns = model; values = metric (optionally 'mean +- std')."""
    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            row = {"task": task, "timeout": f"{t} s"}
            for m in BASELINE_MODELS:
                r = _cell(b, m, task, t)
                v = f"{r[metric]:.3f}"
                if with_std:
                    v += f" +- {r[with_std]:.3f}"
                row[m] = v
            rows.append(row)
    return pd.DataFrame(rows)


def majority_check(b: pd.DataFrame) -> pd.DataFrame:
    """E01: validation accuracy vs p and macro F1 vs 2p / ((1 + p) k), per config."""
    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            r = _cell(b, "majority", task, t)
            counts = json.loads(r["params.class_counts.val"])
            k = len(counts)
            p = max(counts.values()) / sum(counts.values())
            f1 = 2 * p / ((1 + p) * k)
            rows.append({
                "task": task, "timeout": f"{t} s", "classes": k,
                "largest_class": max(counts, key=counts.get), "p": round(p, 4),
                "val_accuracy": round(r["metrics.val_accuracy"], 4),
                "expected_macro_f1": round(f1, 4),
                "val_macro_f1": round(r["metrics.val_f1_macro"], 4),
                "ok": bool(abs(r["metrics.val_accuracy"] - p) <= 0.001
                           and abs(r["metrics.val_f1_macro"] - f1) <= 0.001),
            })  # fmt: skip
    return pd.DataFrame(rows)


def verdicts(b: pd.DataFrame, bench: pd.DataFrame, paper: pd.DataFrame) -> dict:
    """Every Phase 2 threshold, computed from the runs."""
    f1 = {(m, task, t): _cell(b, m, task, t)["metrics.val_f1_macro"]
          for m in BASELINE_MODELS for task in CORE_TASKS for t in TIMEOUTS}  # fmt: skip
    cfgs = [(task, t) for task in CORE_TASKS for t in TIMEOUTS]
    out = {}
    out["E01"] = {"configs_ok": int(majority_check(b)["ok"].sum()), "needed": 16}
    out["E01"]["supported"] = out["E01"]["configs_ok"] == 16
    tree_gain = [f1["tree", c[0], c[1]] - f1["majority", c[0], c[1]] for c in cfgs]
    a1_tree = [f1["tree", "A1", t] for t in TIMEOUTS]
    out["E02a"] = {"configs_with_gain_ge_0.30": sum(g >= 0.30 for g in tree_gain),
                   "min_gain": round(min(tree_gain), 4), "needed": 16}  # fmt: skip
    out["E02a"]["supported"] = out["E02a"]["configs_with_gain_ge_0.30"] == 16
    out["E02b"] = {"A1_val_macro_f1": [round(x, 4) for x in a1_tree], "needed": ">= 0.80 x 4"}
    out["E02b"]["supported"] = all(x >= 0.80 for x in a1_tree)
    for m in ("knn", "logreg", "svm_rbf"):
        gains = [f1[m, c[0], c[1]] - f1["majority", c[0], c[1]] for c in cfgs]
        out[f"E03a_{m}"] = {"configs_with_gain_ge_0.10": sum(g >= 0.10 for g in gains),
                            "min_gain": round(min(gains), 4), "needed": 16,
                            "supported": all(g >= 0.10 for g in gains)}  # fmt: skip
    close = [abs(f1["knn", c[0], c[1]] - f1["tree", c[0], c[1]]) <= 0.10 for c in cfgs]
    out["E03b"] = {"configs_knn_within_0.10_of_tree": sum(close), "needed": 12,
                   "supported": sum(close) >= 12}  # fmt: skip
    for m in ("logreg", "svm_rbf"):
        below = [f1[m, c[0], c[1]] < f1["tree", c[0], c[1]] for c in cfgs]
        out[f"E03c_{m}"] = {"configs_below_tree": sum(below), "needed": 12,
                            "supported": sum(below) >= 12}  # fmt: skip
    comp = paper_a1(bench, paper)
    for exp, algo in (("E02c", "C4.5"), ("E03d", "KNN")):
        c = comp[comp["algorithm"] == algo]
        n = int((c["diff_published"].abs() <= 0.05).sum())
        out[exp] = {"A1_values_within_0.05": n, "of": len(c), "needed": 12, "supported": n >= 12}
    for exp, algo in (("E02d", "j48"), ("E03e", "ibk")):
        d = dedup_effect(bench)
        d = d[d["algo"] == algo]
        n = int((d["v1_minus_published"] < 0).sum())
        out[exp] = {"files_v1_lower": n, "of": len(d), "needed": 12, "supported": n >= 12,
                    "mean_change": round(float(d["v1_minus_published"].mean()), 4)}  # fmt: skip
    return out


def load_paper(path: Path = ROOT / "reports" / "paper_baseline.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _bench_cell(bench: pd.DataFrame, algo: str, rows: str, task: str, t: int) -> pd.Series:
    r = bench[(bench["algo"] == algo) & (bench["tags.rows_source"] == rows)
              & (bench["params.task"] == task) & (bench["params.timeout"] == t)]  # fmt: skip
    if len(r) != 1:
        raise ValueError(f"expected 1 benchmark run for {algo} {rows} {task} {t}s, found {len(r)}")
    return r.iloc[0]


def paper_a1(bench: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """A1: every chart value of the paper next to ours (published rows and v1)."""
    p = paper[(paper["task"] == "A1") & (paper["source"] == "read_from_chart")]
    rows = []
    for _, q in p.iterrows():
        algo = "j48" if q["algorithm"] == "C4.5" else "ibk"
        t = int(q["timeout_s"])
        key = f"metrics.bench_{q['metric']}.{q['class']}"
        pub = _bench_cell(bench, algo, "published", "A1", t)[key]
        v1 = _bench_cell(bench, algo, "v1", "A1", t)[key]
        rows.append({
            "algorithm": q["algorithm"], "class": q["class"], "timeout": t,
            "metric": q["metric"], "paper": q["value"], "paper_source": "chart +-0.01",
            "ours_published": round(pub, 3), "ours_v1": round(v1, 3),
            "diff_published": round(pub - q["value"], 3), "diff_v1": round(v1 - q["value"], 3),
        })  # fmt: skip
    return pd.DataFrame(rows).sort_values(["algorithm", "class", "metric", "timeout"])


def _weighted_precision(r: pd.Series) -> float:
    counts = json.loads(r["params.class_counts.all"])
    total = sum(counts.values())
    return sum(r[f"metrics.bench_precision.{c}"] * n / total for c, n in counts.items())


def paper_text_values(bench: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """A2 and B: the values the paper states in its text, next to ours."""
    p = paper[(paper["source"] == "stated_in_text") & (paper["task"] != "A1")]
    rows = []
    for _, q in p.iterrows():
        algo = "j48" if q["algorithm"] == "C4.5" else "ibk"
        for rows_src in ("published", "v1"):
            if q["class"] == "AVERAGE":
                r = _bench_cell(bench, algo, rows_src, q["task"], int(q["timeout_s"]))
                ours = {"macro": r["metrics.bench_precision_macro"],
                        "weighted": _weighted_precision(r)}  # fmt: skip
            elif q["class"] == "AVERAGE_BEST_TIMEOUT":
                rs = [_bench_cell(bench, algo, rows_src, q["task"], t) for t in TIMEOUTS]
                ours = {"macro": max(r["metrics.bench_precision_macro"] for r in rs),
                        "weighted": max(_weighted_precision(r) for r in rs)}  # fmt: skip
            else:
                r = _bench_cell(bench, algo, rows_src, q["task"], int(q["timeout_s"]))
                ours = {"class": r[f"metrics.bench_precision.{q['class']}"]}
            for kind, v in ours.items():
                rows.append({
                    "task": q["task"], "algorithm": q["algorithm"], "class": q["class"],
                    "timeout": q["timeout_s"], "metric": "precision", "paper": q["value"],
                    "rows": rows_src, "ours": round(v, 3), "ours_kind": kind,
                    "diff": round(v - q["value"], 3),
                })  # fmt: skip
    return pd.DataFrame(rows)


def dedup_effect(bench: pd.DataFrame) -> pd.DataFrame:
    """10-fold macro F1 on published rows vs data v1, per algorithm and file."""
    rows = []
    for algo in BENCH_ALGO:
        for task in CORE_TASKS:
            for t in TIMEOUTS:
                pub = _bench_cell(bench, algo, "published", task, t)
                v1 = _bench_cell(bench, algo, "v1", task, t)
                rows.append({
                    "algo": algo, "task": task, "timeout": t,
                    "rows_published": int(pub["params.n_rows"]),
                    "rows_v1": int(v1["params.n_rows"]),
                    "f1_published": round(pub["metrics.bench_f1_macro"], 4),
                    "f1_v1": round(v1["metrics.bench_f1_macro"], 4),
                    "v1_minus_published": round(
                        v1["metrics.bench_f1_macro"] - pub["metrics.bench_f1_macro"], 4),
                })  # fmt: skip
    return pd.DataFrame(rows)


def cost_table(b: pd.DataFrame) -> pd.DataFrame:
    """Per model: median and max over the 16 configs of time, size and memory."""
    cols = {
        "metrics.train_time_s": "train time (s)",
        "metrics.cv_time_s": "5-fold CV time (s)",
        "metrics.predict_ms_per_1000": "predict ms / 1,000 flows",
        "metrics.model_size_bytes": "model file (KB)",
        "metrics.peak_memory_mb": "peak fit memory (MB)",
    }
    rows = []
    for m in BASELINE_MODELS:
        s = b[b["params.model"] == m]
        row = {"model": m}
        for c, name in cols.items():
            v = s[c] / 1024 if c == "metrics.model_size_bytes" else s[c]
            row[f"{name} median"] = round(float(v.median()), 3)
            row[f"{name} max"] = round(float(v.max()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _v(r: pd.Series, kind: str, cls: str) -> float:
    return r[f"metrics.val_{kind}.{cls}"]


def class_extremes(b: pd.DataFrame, model: str) -> pd.DataFrame:
    """Best / worst class by validation F1 and recall, macro vs weighted F1 and the gap,
    the generalisation gap and the top confusion pairs, per config, for one model."""
    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            r = _cell(b, model, task, t)

            rows.append({
                "task": task, "timeout": f"{t} s",
                "macro F1": round(r["metrics.val_f1_macro"], 3),
                "weighted F1": round(r["metrics.val_f1_weighted"], 3),
                "gap": round(r["metrics.val_f1_gap_weighted_minus_macro"], 3),
                "CV - val": round(r["metrics.gen_gap_f1_macro"], 3),
                "best F1": f"{r['tags.best_class_f1']} {_v(r, 'f1', r['tags.best_class_f1']):.2f}",
                "worst F1": f"{r['tags.worst_class_f1']} "
                            f"{_v(r, 'f1', r['tags.worst_class_f1']):.2f}",
                "best recall": f"{r['tags.best_class_recall']} "
                               f"{_v(r, 'recall', r['tags.best_class_recall']):.2f}",
                "worst recall": f"{r['tags.worst_class_recall']} "
                                f"{_v(r, 'recall', r['tags.worst_class_recall']):.2f}",
            })  # fmt: skip
    return pd.DataFrame(rows)


def confusion_pairs(b: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            r = _cell(b, model, task, t)
            rows.append({"task": task, "timeout": f"{t} s",
                         "top 5 confusion pairs (true -> predicted: flows)":
                         r["tags.top_confusion_pairs"]})  # fmt: skip
    return pd.DataFrame(rows)


def complexity(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            tr = _cell(b, "tree", task, t)
            sv = _cell(b, "svm_rbf", task, t)
            rows.append({
                "task": task, "timeout": f"{t} s", "n_train": int(tr["params.n_train"]),
                "tree depth": int(tr["metrics.model_tree_depth"]),
                "tree nodes": int(tr["metrics.model_tree_nodes"]),
                "tree leaves": int(tr["metrics.model_tree_leaves"]),
                "SVM support vectors": int(sv["metrics.model_support_vectors"]),
            })  # fmt: skip
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, bool | np.bool_):
                cells.append("yes" if v else "no")
            elif isinstance(v, int | np.integer):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def duplicate_profile() -> pd.DataFrame:
    """What the exact duplicates removed in v1 look like (reads the raw ARFF files).

    Per file: rows removed, share of them whose 23 features also occur with another label in
    the file, the same share over all rows, and median packets per flow (flowPktsPerSecond x
    duration / 1e6; A1 120 s uses its published, corrupt flowPktsPerSecond, so its packet
    numbers are not meaningful)."""
    from blindspy.data import load_raw

    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            df = load_raw(task, t)
            feats = list(df.columns[:-1])
            dup = df.duplicated(keep="first")
            key = df[feats].astype(str).agg("|".join, axis=1)
            n_labels = df.groupby(key)["label"].transform("nunique")
            pkts = (df["flowPktsPerSecond"] * df["duration"] / 1e6).round()
            rows.append({
                "task": task, "timeout": t, "duplicates removed": int(dup.sum()),
                "removed with other-label twin": round(float((n_labels[dup] > 1).mean()), 3),
                "all rows with other-label twin": round(float((n_labels > 1).mean()), 3),
                "median packets, removed": float(pkts[dup].median()),
                "median packets, all": float(pkts.median()),
            })  # fmt: skip
    return pd.DataFrame(rows)
