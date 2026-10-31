"""Tables and threshold checks built from reports/history.csv (no number typed by hand)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
    text = paper[(paper["task"] == "A1") & (paper["source"] == "stated_in_text")]
    stated = set(zip(text["algorithm"], text["class"], text["timeout_s"].astype(int),
                     text["metric"], strict=True))  # fmt: skip
    rows = []
    for _, q in p.iterrows():
        algo = "j48" if q["algorithm"] == "C4.5" else "ibk"
        t = int(q["timeout_s"])
        key = f"metrics.bench_{q['metric']}.{q['class']}"
        pub = _bench_cell(bench, algo, "published", "A1", t)[key]
        v1 = _bench_cell(bench, algo, "v1", "A1", t)[key]
        rows.append({
            "algorithm": q["algorithm"], "class": q["class"], "timeout": t,
            "metric": q["metric"], "paper": q["value"],
            "paper_source": "chart +-0.01 and text" if (q["algorithm"], q["class"], t, q["metric"])
            in stated else "chart +-0.01",
            "ours_published": round(pub, 3), "ours_v1": round(v1, 3),
            "diff_published": round(pub - q["value"], 3), "diff_v1": round(v1 - q["value"], 3),
        })  # fmt: skip
    return pd.DataFrame(rows).sort_values(["algorithm", "class", "metric", "timeout"])


def paper_task_table(
    bench: pd.DataFrame, paper: pd.DataFrame, task: str, rows: str
) -> pd.DataFrame:
    """One task: our per-class precision and recall (10-fold, pooled) for both algorithms and
    every timeout, with the paper's value beside ours wherever the paper gives one
    ("C" = read from a chart, +-0.01; "T" = stated in the text; "C+T" = both)."""
    src = {}
    for _, q in paper[paper["task"] == task].iterrows():
        k = (q["algorithm"], q["class"], str(q["timeout_s"]), q["metric"])
        tag = "T" if q["source"] == "stated_in_text" else "C"
        src[k] = (q["value"], "C+T" if k in src and src[k][1] != tag else tag)
    classes = json.loads(_bench_cell(bench, "j48", rows, task, 15)["params.class_counts.all"])
    out = []
    for cls in [*sorted(classes), "AVERAGE (macro)"]:
        for metric in ("precision", "recall"):
            row = {"class": cls, "metric": metric}
            for algo, name in BENCH_ALGO.items():
                for t in TIMEOUTS:
                    r = _bench_cell(bench, algo, rows, task, t)
                    key = f"metrics.bench_{metric}_macro" if cls.startswith("AVERAGE") else (
                        f"metrics.bench_{metric}.{cls}")  # fmt: skip
                    cell = f"{r[key]:.3f}"
                    p = src.get((name, "AVERAGE" if cls.startswith("AVERAGE") else cls, str(t),
                                 metric))  # fmt: skip
                    if p:
                        cell += f" (paper {p[0]:.3f} {p[1]})"
                    row[f"{name} {t} s"] = cell
            out.append(row)
    return pd.DataFrame(out)


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
            elif isinstance(v, float) and np.isnan(v):  # e.g. no spread for a single run
                cells.append("")
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


# ---- Phase 3 -------------------------------------------------------------------------------

# Experiments whose holdout runs compete for "current best", in the order they ran.
SELECTION_EXPS = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E10", "E17"]
WINNER_MARGIN = 0.03  # decided 28 Oct: above the largest Phase 2 CV std (0.025)


def is_winner(gain: float) -> bool:
    """gain >= 0.03, rounded first so an exact 0.03 isn't lost to float error."""
    return bool(round(gain, 10) >= WINNER_MARGIN)


def model_label(r: pd.Series) -> str:
    """Run name without the task and timeout, e.g. 'E04-knn_log_standard'."""
    suffix = f"-{r['params.task']}-{int(r['params.timeout'])}s"
    name = r["run_name"]
    return name[: -len(suffix)] if name.endswith(suffix) else name


def selection_runs(h: pd.DataFrame) -> pd.DataFrame:
    """Holdout runs of the model-selection experiments; Optuna trials and pilots excluded."""
    s = h[(h["tags.protocol"] == "holdout") & h["tags.experiment_id"].isin(SELECTION_EXPS)]
    if "tags.trial" in s.columns:
        s = s[s["tags.trial"].astype(str) != "True"]
    s = s.copy()
    s["label"] = s.apply(model_label, axis=1)
    return s


def best_run(s: pd.DataFrame, task: str, t: int, upto: str | None = None) -> pd.Series:
    """Highest validation macro F1 for task and timeout among experiments up to `upto`
    (inclusive; all if None). Ties keep the earlier experiment."""
    exps = SELECTION_EXPS if upto is None else SELECTION_EXPS[: SELECTION_EXPS.index(upto) + 1]
    c = s[(s["params.task"] == task) & (s["params.timeout"] == t)
          & s["tags.experiment_id"].isin(exps)]  # fmt: skip
    if c.empty:
        raise ValueError(f"no run for {task} {t}s up to {upto}")
    c = c.assign(order=c["tags.experiment_id"].map(SELECTION_EXPS.index))
    return c.sort_values(["metrics.val_f1_macro", "order"], ascending=[False, True]).iloc[0]


def screen(s: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Every 15 s run of `exp` against the best run of the earlier experiments: gain and the
    winner flag (gain >= WINNER_MARGIN -> rerun at 30, 60 and 120 s)."""
    prev = SELECTION_EXPS[SELECTION_EXPS.index(exp) - 1]
    rows = []
    for _, r in s[(s["tags.experiment_id"] == exp) & (s["params.timeout"] == 15)].iterrows():
        b = best_run(s, r["params.task"], 15, prev)
        gain = r["metrics.val_f1_macro"] - b["metrics.val_f1_macro"]
        rows.append({
            "task": r["params.task"], "model": r["label"],
            "val macro F1": round(r["metrics.val_f1_macro"], 4),
            "CV macro F1": f"{r['metrics.cv_f1_macro_mean']:.4f} +- "
                           f"{r['metrics.cv_f1_macro_std']:.4f}",
            "best before": b["label"], "best before F1": round(b["metrics.val_f1_macro"], 4),
            "gain": round(gain, 4), "winner (gain >= 0.03)": is_winner(gain),
        })  # fmt: skip
    order = {t: i for i, t in enumerate(CORE_TASKS)}
    return (
        pd.DataFrame(rows)
        .sort_values(["task", "model"], key=lambda c: c.map(order) if c.name == "task" else c)
        .reset_index(drop=True)
    )


def timeout_table(s: pd.DataFrame, label: str, task: str) -> pd.DataFrame:
    """A model's validation macro F1 at every timeout it ran, next to the best other run of
    the experiments up to its own (so later experiments don't change the table)."""
    exp = label.split("-")[0]
    rows = []
    for t in TIMEOUTS:
        r = s[(s["label"] == label) & (s["params.task"] == task) & (s["params.timeout"] == t)]
        if r.empty:
            continue
        r = r.iloc[0]
        others = s[s["label"] != label]
        b = best_run(others, task, t, exp)
        rows.append({
            "task": task, "timeout": f"{t} s", "model": label,
            "val macro F1": round(r["metrics.val_f1_macro"], 4),
            "CV macro F1": f"{r['metrics.cv_f1_macro_mean']:.4f} +- "
                           f"{r['metrics.cv_f1_macro_std']:.4f}",
            "best other run": b["label"], "its F1": round(b["metrics.val_f1_macro"], 4),
            "difference": round(r["metrics.val_f1_macro"] - b["metrics.val_f1_macro"], 4),
        })  # fmt: skip
    return pd.DataFrame(rows)


def current_best(s: pd.DataFrame, upto: str | None = None) -> pd.DataFrame:
    """The current best per task at 15 s (after experiment `upto`)."""
    rows = []
    for task in CORE_TASKS:
        b = best_run(s, task, 15, upto)
        rows.append({"task": task, "current best at 15 s": b["label"],
                     "val macro F1": round(b["metrics.val_f1_macro"], 4),
                     "CV macro F1": f"{b['metrics.cv_f1_macro_mean']:.4f} +- "
                                    f"{b['metrics.cv_f1_macro_std']:.4f}"})  # fmt: skip
    return pd.DataFrame(rows)


def _complexity_text(r: pd.Series) -> str:
    def m(k):
        v = r.get(f"metrics.model_{k}")
        return None if v is None or v != v else v

    if m("n_trees") is not None:
        depth = f"depth {m('tree_depth_mean'):.1f} mean / {int(m('tree_depth_max'))} max"
        return f"{int(m('n_trees'))} trees, {depth}, {int(m('tree_nodes_total')):,} nodes"
    if m("tree_depth") is not None:
        return f"depth {int(m('tree_depth'))}, {int(m('tree_nodes')):,} nodes"
    if m("support_vectors") is not None:
        return f"{int(m('support_vectors')):,} support vectors"
    return "stores the training rows" if "knn" in r["params.model"] else ""


def run_details(s: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Per run of `exp`: scores, F1 gap, best/worst class, complexity and cost (time,
    memory, sizes; MiB = 2**20 bytes)."""
    rows = []
    for _, r in s[s["tags.experiment_id"] == exp].sort_values(
            ["params.task", "params.timeout", "run_name"]).iterrows():  # fmt: skip
        rows.append({
            "run": r["run_name"], "val macro F1": round(r["metrics.val_f1_macro"], 3),
            "weighted F1": round(r["metrics.val_f1_weighted"], 3),
            "accuracy": round(r["metrics.val_accuracy"], 3),
            "balanced acc.": round(r["metrics.val_balanced_accuracy"], 3),
            "CV - val": round(r["metrics.gen_gap_f1_macro"], 3),
            "worst F1": f"{r['tags.worst_class_f1']} "
                        f"{_v(r, 'f1', r['tags.worst_class_f1']):.2f}",
            "best F1": f"{r['tags.best_class_f1']} {_v(r, 'f1', r['tags.best_class_f1']):.2f}",
            "complexity": _complexity_text(r),
            "train s": round(r["metrics.train_time_s"], 3),
            "5-fold CV s": round(r["metrics.cv_time_s"], 2),
            "predict ms/1k": round(r["metrics.predict_ms_per_1000"], 2),
            "model MiB (raw / file)": f"{r['metrics.model_size_bytes'] / 2**20:.2f} / "
                                     f"{r['metrics.model_file_bytes'] / 2**20:.2f}",
            "memory MiB (tracemalloc / child RSS)": f"{r['metrics.peak_memory_mb']:.1f} / "
                                                   f"{r['metrics.child_peak_rss_mb']:.1f}",
        })  # fmt: skip
    return pd.DataFrame(rows)


def phase3_verdicts(s: pd.DataFrame, runner: str | None = None) -> dict:
    """Every pre-stated Phase 3 threshold that can be computed so far. E10 also reads the
    confusion matrices from mlruns/<runner> (skipped without a runner)."""
    out = {}
    f1 = {(r["label"], r["params.task"], r["params.timeout"]): r["metrics.val_f1_macro"]
          for _, r in s.iterrows()}  # fmt: skip
    if any(k[0].startswith("E04-") for k in f1):
        std = [f1["E04-knn_log_standard", t, 15] for t in CORE_TASKS]
        rob = [f1["E04-knn_log_robust", t, 15] for t in CORE_TASKS]
        knn = [f1["E03-knn", t, 15] for t in CORE_TASKS]
        gain = [max(a, b) - k for a, b, k in zip(std, rob, knn, strict=True)]
        out["E04a"] = {
            "gains_vs_E03_knn": [round(g, 4) for g in gain],
            "tasks_with_gain_ge_0.02": sum(g >= 0.02 for g in gain),
            "needed": 3,
        }
        out["E04a"]["supported"] = out["E04a"]["tasks_with_gain_ge_0.02"] >= 3
        diff = [round(a - b, 4) for a, b in zip(std, rob, strict=True)]
        out["E04b"] = {
            "standard_minus_robust": diff,
            "tasks_within_0.02": sum(abs(d) <= 0.02 for d in diff),
            "needed": 3,
        }
        out["E04b"]["supported"] = out["E04b"]["tasks_within_0.02"] >= 3
    if any(k[0] == "E05-rf" for k in f1):
        rf = [f1["E05-rf", t, 15] for t in CORE_TASKS]
        tree = [rf_ - f1["E02-tree", t, 15] for rf_, t in zip(rf, CORE_TASKS, strict=True)]
        prev = [
            rf_ - best_run(s, t, 15, "E04")["metrics.val_f1_macro"]
            for rf_, t in zip(rf, CORE_TASKS, strict=True)
        ]
        out["E05a"] = {
            "gains_vs_E02_tree": [round(g, 4) for g in tree],
            "tasks_with_gain_ge_0.03": sum(is_winner(g) for g in tree),
            "needed": 3,
        }
        out["E05a"]["supported"] = out["E05a"]["tasks_with_gain_ge_0.03"] >= 3
        out["E05b"] = {
            "gains_vs_best_after_E04": [round(g, 4) for g in prev],
            "tasks_above": sum(g > 0 for g in prev),
            "needed": 2,
        }
        out["E05b"]["supported"] = out["E05b"]["tasks_above"] >= 2
    if any(k[0] == "E06-xgb" for k in f1):
        xgb = [f1["E06-xgb", t, 15] for t in CORE_TASKS]
        lgbm = [f1["E06-lgbm", t, 15] for t in CORE_TASKS]
        prev = [best_run(s, t, 15, "E05")["metrics.val_f1_macro"] for t in CORE_TASKS]
        gain = [max(a, b) - p for a, b, p in zip(xgb, lgbm, prev, strict=True)]
        out["E06a"] = {
            "gains_vs_best_after_E05": [round(g, 4) for g in gain],
            "tasks_above": sum(g > 0 for g in gain),
            "needed": 2,
        }
        out["E06a"]["supported"] = out["E06a"]["tasks_above"] >= 2
        diff = [round(a - b, 4) for a, b in zip(xgb, lgbm, strict=True)]
        out["E06b"] = {
            "xgb_minus_lgbm": diff,
            "tasks_within_0.02": sum(abs(d) <= 0.02 for d in diff),
            "needed": 3,
        }
        out["E06b"]["supported"] = out["E06b"]["tasks_within_0.02"] >= 3
    if any(k[0].startswith("E07-") for k in f1):
        e07 = [k for k in f1 if k[0].startswith("E07-") and k[2] == 15]
        better = {f"{lab} {t}": round(f1[lab, t, 15] - f1[UNTUNED[lab], t, 15], 4)
                  for lab, t, _ in sorted(e07)}  # fmt: skip
        out["E07a"] = {
            "tuned_minus_untuned": better,
            "searches_above": sum(g > 0 for g in better.values()),
            "of": len(better),
            "needed": 8,
        }
        out["E07a"]["supported"] = out["E07a"]["searches_above"] >= 8
        best7 = {t: max(f1[lab, t, 15] for lab, tt, _ in e07 if tt == t) for t in CORE_TASKS}
        gain = [best7[t] - best_run(s, t, 15, "E06")["metrics.val_f1_macro"] for t in CORE_TASKS]
        out["E07b"] = {
            "best_E07_minus_best_after_E06": [round(g, 4) for g in gain],
            "tasks_with_gain_ge_0.01": sum(round(g, 10) >= 0.01 for g in gain),
            "needed": 2,
        }
        out["E07b"]["supported"] = out["E07b"]["tasks_with_gain_ge_0.01"] >= 2
        diff = []
        for t in CORE_TASKS:
            ens = max(f1[lab, tt, 15] for lab, tt, _ in e07 if tt == t and "knn" not in lab)
            diff.append(round(f1["E07-knn_log_standard", t, 15] - ens, 4))
        out["E07c"] = {
            "knn_minus_best_tuned_ensemble": diff,
            "tasks_knn_below": sum(d < 0 for d in diff),
            "needed": 4,
        }
        out["E07c"]["supported"] = out["E07c"]["tasks_knn_below"] == 4
    if any(k[0].startswith("E08-") for k in f1):
        lad = e08_ladder(s, E08_BASE)
        ran = lad[(lad["step"] != "none") & (lad["run"] != "not run yet")]
        best = ran.loc[ran.groupby("task")["val macro F1"].idxmax()].set_index("task")
        imb = ["A2-VPN", "A2-NonVPN", "B"]
        gain = {t: best.loc[t, "change vs none"] for t in imb}
        out["E08a"] = {"best_step": {t: best.loc[t, "step"] for t in imb},
                       "best_step_minus_none": gain,
                       "tasks_with_gain_ge_0.01": sum(round(g, 10) >= 0.01 for g in gain.values()),
                       "needed": 2}  # fmt: skip
        out["E08a"]["supported"] = out["E08a"]["tasks_with_gain_ge_0.01"] >= 2
        a1 = ran[ran["task"] == "A1"]["change vs none"].tolist()
        out["E08b"] = {"A1_changes": a1, "all_within_0.01": all(abs(c) <= 0.01 for c in a1)}
        out["E08b"]["supported"] = out["E08b"]["all_within_0.01"]
        weak = {}
        for t in imb:
            none = lad[(lad["task"] == t) & (lad["step"] == "none")].iloc[0]
            col = "F1 of weakest class without balancing"
            weak[t] = round(float(best.loc[t, col].split()[-1])
                            - float(none[col].split()[-1]), 4)  # fmt: skip
        out["E08c"] = {"weakest_class_f1_change_with_best_step": weak,
                       "tasks_raised": sum(w > 0 for w in weak.values()), "needed": 2}  # fmt: skip
        out["E08c"]["supported"] = out["E08c"]["tasks_raised"] >= 2
    if runner and any(k[0] == "E10-xgb_two_step" for k in f1):
        d, t = e10_table(s, runner).to_dict("records")
        grp = "group errors (VPN / non-VPN wrong)"
        diff = round(t["val macro F1"] - d["val macro F1"], 4)
        out["E10a"] = {"two_step_minus_direct": diff, "supported": abs(diff) <= 0.01}
        out["E10b"] = {"two_step_group_errors": t[grp], "direct_group_errors": d[grp],
                       "supported": t[grp] < d[grp]}  # fmt: skip
        out["E10c"] = {"share_of_two_step_errors_from_step_1":
                       t["share of errors from the group"],
                       "supported": t["share of errors from the group"] > 0.5}  # fmt: skip
    if any(k[0].startswith("E17-") for k in f1):
        e17 = sorted(k for k in f1 if k[0].startswith("E17-") and k[2] == 15)
        better = {f"{lab} {t}": round(f1[lab, t, 15] - f1[UNTUNED[lab], t, 15], 4)
                  for lab, t, _ in e17}  # fmt: skip
        out["E17a"] = {"tuned_minus_untuned": better,
                       "searches_above": sum(g > 0 for g in better.values()),
                       "of": len(better), "needed": 12}  # fmt: skip
        out["E17a"]["supported"] = out["E17a"]["searches_above"] >= 12
        best = {t: best_run(s, t, 15, "E17")["label"] for t in CORE_TASKS}
        boosted = sum(lab.split("-", 1)[1].split("_")[0] in ("xgb", "lgbm")
                      for lab in best.values())  # fmt: skip
        out["E17b"] = {"best_after_E17": best, "tasks_won_by_xgb_or_lgbm": boosted,
                       "needed": 3, "supported": boosted >= 3}  # fmt: skip
        logs = {
            f"{r['label']} {r['params.task']}": str(r.get("params.model.signed_log"))
            for _, r in s[s["label"].isin(["E17-logreg", "E17-svm_rbf"])].iterrows()
        }
        n_log = sum(v == "True" for v in logs.values())
        out["E17c"] = {"signed_log_chosen": logs, "searches_with_log": n_log, "needed": 6,
                       "supported": n_log >= 6}  # fmt: skip
    return out  # fmt: skip


def vpn_group_errors(cm: pd.DataFrame) -> dict:
    """Task B errors from a 14-class confusion matrix (rows true, columns predicted), split
    into errors that cross the VPN / non-VPN line (for the E10 two-step: exactly the errors
    of a wrong first step) and errors inside the right group (wrong application type)."""
    vpn_t = cm.index.str.startswith("VPN-")
    vpn_p = cm.columns.str.startswith("VPN-")
    v = cm.to_numpy()
    total = int(v.sum())
    correct = int(np.trace(v))
    cross = int(v[np.ix_(vpn_t, ~vpn_p)].sum() + v[np.ix_(~vpn_t, vpn_p)].sum())
    return {"rows": total, "errors": total - correct, "group errors (VPN / non-VPN wrong)": cross,
            "type errors (right group, wrong type)": total - correct - cross,
            "group accuracy": round(1 - cross / total, 4),
            "share of errors from the group": round(cross / (total - correct), 4)}  # fmt: skip


def run_confusion(run_id: str, runner: str) -> pd.DataFrame:
    """A run's validation confusion matrix from its MLflow artifacts."""
    path = ROOT / "mlruns" / runner / "artifacts" / run_id / "artifacts" / "eval"
    return pd.read_csv(path / "confusion_matrix.csv", index_col=0)


E10_DIRECT = "E07-xgb"  # the current best B model after E08 (E10.md)


def e10_table(s: pd.DataFrame, runner: str) -> pd.DataFrame:
    """E10: the direct 14-class model and the two-step model on the same B validation rows."""
    rows = []
    for label in (E10_DIRECT, "E10-xgb_two_step"):
        r = s[(s["label"] == label) & (s["params.task"] == "B")
              & (s["params.timeout"] == 15)].iloc[0]  # fmt: skip
        rows.append({
            "structure": "two-step" if "two_step" in label else "direct 14-class",
            "run": label, "val macro F1": round(r["metrics.val_f1_macro"], 4),
            "CV macro F1": f"{r['metrics.cv_f1_macro_mean']:.4f} +- "
                           f"{r['metrics.cv_f1_macro_std']:.4f}",
            "val accuracy": round(r["metrics.val_accuracy"], 4),
            **vpn_group_errors(run_confusion(r["run_id"], runner)),
        })  # fmt: skip
    return pd.DataFrame(rows)


E08_ORDER = ["none", "class_weight", "oversample", "smote"]
# the current best per task after E07 (table "Current best at 15 s after E07"): E08's "none" step
E08_BASE = {"A1": "E06-xgb", "A2-VPN": "E07-lgbm", "A2-NonVPN": "E07-lgbm", "B": "E07-xgb"}


def e08_ladder(s: pd.DataFrame, base: dict[str, str]) -> pd.DataFrame:
    """E08 per task and step (15 s): validation macro F1, the F1 of the class weakest without
    balancing, whether the step helped (above every earlier step) and what happens next under
    the stop rule (a step runs only if every earlier one helped). base: task -> label of the
    current best after E07 (the "none" step)."""
    rows = []
    for task in CORE_TASKS:
        at = s[(s["params.task"] == task) & (s["params.timeout"] == 15)].set_index("label")
        none = at.loc[base[task]]
        weak = none["tags.worst_class_f1"]
        best = -1.0
        for step in E08_ORDER:
            label = base[task] if step == "none" else f"E08-{none['params.model']}_{step}"
            if label not in at.index:
                rows.append({"task": task, "step": step, "run": "not run yet",
                             "next": "run this step"})  # fmt: skip
                break
            f1 = at.loc[label, "metrics.val_f1_macro"]
            helped = step == "none" or f1 > best
            rows.append({
                "task": task, "step": step, "run": label, "val macro F1": round(f1, 4),
                "change vs none": round(f1 - none["metrics.val_f1_macro"], 4),
                "F1 of weakest class without balancing":
                    f"{weak} {_v(at.loc[label], 'f1', weak):.4f}",
                "helped": "" if step == "none" else ("yes" if helped else "no"),
                "next": "" if helped else "stop: this step did not help",
            })  # fmt: skip
            if not helped:
                break
            best = max(best, f1)
        else:
            rows[-1]["next"] = "ladder complete"
    return pd.DataFrame(rows).astype(object).fillna("")


def run_settings(run_name: str, path: Path = ROOT / "reports" / "history.csv") -> dict:
    """Model and hyperparameters of the newest selected run with this name, read from the
    logged strings (so tuned floats are copied exactly) for configs that build on it."""
    import yaml

    rid = load_history(path).set_index("run_name").loc[run_name, "run_id"]
    raw = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("run_id").loc[rid]
    # MLflow logs Python's None as the string "None", which YAML would read as text
    params = {c.removeprefix("params.model."): None if v == "None" else yaml.safe_load(v)
              for c, v in raw.items() if c.startswith("params.model.") and v != ""}  # fmt: skip
    return {"model": raw["params.model"], "params": params}


# E07: each tuned model's untuned version (same model type and preprocessing)
UNTUNED = {"E07-xgb": "E06-xgb", "E07-lgbm": "E06-lgbm", "E07-rf": "E05-rf",
           "E07-knn_log_standard": "E04-knn_log_standard",
           "E17-tree": "E02-tree", "E17-logreg": "E03-logreg", "E17-svm_rbf": "E03-svm_rbf",
           "E17-rf": "E05-rf", "E17-lgbm": "E06-lgbm"}  # fmt: skip


def e07_searches(s: pd.DataFrame, exp: str = "E07") -> pd.DataFrame:
    """Per Optuna search of `exp` (E07 or E17): size, time, the chosen settings, and the
    retrained model next to its untuned version (CV on train and validation)."""
    rows = []
    e07 = s[(s["tags.experiment_id"] == exp) & (s["params.timeout"] == 15)]
    for _, r in e07.sort_values(["params.task", "label"]).iterrows():
        u = s[(s["label"] == UNTUNED[r["label"]]) & (s["params.task"] == r["params.task"])
              & (s["params.timeout"] == 15)].iloc[0]  # fmt: skip
        space = json.loads(r["params.search.space"])
        est = json.loads(r["params.estimator_params"])
        chosen = []
        for k in space:
            # estimator_params leaves out None settings; pipeline options such as signed_log are
            # not estimator settings, so they come from the logged param
            v = est[k] if k in est else r.get(f"params.model.{k}")
            v = None if v is None or v != v else v
            chosen.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
        rows.append({
            "task": r["params.task"], "model": r["label"],
            "trials / fits": f"{int(r['metrics.search_n_trials'])} / "
                             f"{int(r['metrics.search_total_fits'])}",
            "best trial": int(r["metrics.search_best_trial"]),
            "search min": round(r["metrics.search_time_s"] / 60, 1),
            "chosen settings": ", ".join(chosen),
            "CV untuned -> tuned": f"{u['metrics.cv_f1_macro_mean']:.4f} -> "
                                   f"{r['metrics.cv_f1_macro_mean']:.4f}",
            "val untuned -> tuned": f"{u['metrics.val_f1_macro']:.4f} -> "
                                    f"{r['metrics.val_f1_macro']:.4f}",
            "val change": round(r["metrics.val_f1_macro"] - u["metrics.val_f1_macro"], 4),
            "CV best == retrained CV": bool(
                r["metrics.search_best_cv_f1_macro_mean"] == r["metrics.cv_f1_macro_mean"]),
        })  # fmt: skip
    order = {t: i for i, t in enumerate(CORE_TASKS)}
    return (pd.DataFrame(rows).sort_values("task", key=lambda c: c.map(order), kind="stable")
            .reset_index(drop=True))  # fmt: skip


def robust_scaling_check(timeout: int = 15) -> pd.DataFrame:
    """E04: after the signed log, which training columns have interquartile range 0 (so
    RobustScaler leaves them unscaled), and the largest |value| after each scaler."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    from blindspy.data import load_task
    from blindspy.features import signed_log

    rows = []
    for task in CORE_TASKS:
        train = load_task(task, timeout, data_version="v1")[0]
        X = signed_log(train.drop(columns="label"))
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        rows.append({
            "task": task, "timeout": f"{timeout} s",
            "columns with IQR 0": ", ".join(c for c, a, b in zip(X.columns, q75, q25,
                                                                  strict=True) if a == b),
            "max |x| RobustScaler": round(float(np.abs(RobustScaler().fit_transform(X)).max()), 1),
            "max |x| StandardScaler": round(
                float(np.abs(StandardScaler().fit_transform(X)).max()), 1),
        })  # fmt: skip
    return pd.DataFrame(rows)


def e07_candidates(s: pd.DataFrame) -> pd.DataFrame:
    """E07 scope (decided 28 Oct): per task, the best 2 of Random Forest (E05), XGBoost and
    LightGBM (E06) by 15 s validation macro F1, plus the KNN preprocessing with the higher
    15 s score (E03 vs E04)."""
    fam = {"E05-rf": "rf", "E06-xgb": "xgb", "E06-lgbm": "lgbm"}
    knn = {"E03-knn": "knn", "E04-knn_log_standard": "knn_log_standard",
           "E04-knn_log_robust": "knn_log_robust"}  # fmt: skip
    rows = []
    for task in CORE_TASKS:
        at = s[(s["params.task"] == task) & (s["params.timeout"] == 15)]
        ens = at[at["label"].isin(fam)].sort_values("metrics.val_f1_macro", ascending=False)
        kn = at[at["label"].isin(knn)].sort_values("metrics.val_f1_macro", ascending=False)
        rows.append({
            "task": task,
            "ranking (15 s val macro F1)": ", ".join(
                f"{fam[r['label']]} {r['metrics.val_f1_macro']:.4f}" for _, r in ens.iterrows()),
            "tuned in E07": ", ".join(fam[x] for x in ens["label"].iloc[:2]),
            "KNN preprocessing for E07": f"{knn[kn['label'].iloc[0]]} "
                                         f"{kn['metrics.val_f1_macro'].iloc[0]:.4f}",
        })  # fmt: skip
    return pd.DataFrame(rows)


# Final choice and comparison set (decided 28 Oct after E10)
METHOD_TYPE = {
    "tree": "decision tree",
    "knn": "KNN",
    "knn_log_standard": "KNN",
    "knn_log_robust": "KNN",
    "logreg": "logistic regression",
    "svm_rbf": "SVM-RBF",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "xgb_two_step": "XGBoost",
    "lgbm": "LightGBM",
}  # fmt: skip  (majority, E01, is a reference only)
FILE_LIMIT_BYTES = 50 * 10**6  # GitHub warns at 50 MB


def final_choice(s: pd.DataFrame) -> pd.DataFrame:
    """Per task the headline model: the highest validation macro F1 of all model-selection runs
    (any experiment, any timeout; ties keep the earlier experiment, then the shorter timeout).
    Only fallback: a saved model file over 50 MB is skipped for the next best."""
    rows = []
    for task in CORE_TASKS:
        c = s[(s["params.task"] == task) & s["params.model"].isin(METHOD_TYPE)]
        c = c.assign(order=c["tags.experiment_id"].map(SELECTION_EXPS.index))
        c = c.sort_values(["metrics.val_f1_macro", "order", "params.timeout"],
                          ascending=[False, True, True])  # fmt: skip
        skipped = c[c["metrics.model_file_bytes"] > FILE_LIMIT_BYTES]
        pick = c[c["metrics.model_file_bytes"] <= FILE_LIMIT_BYTES].iloc[0]
        above = skipped[skipped["metrics.val_f1_macro"] > pick["metrics.val_f1_macro"]]
        rows.append({"task": task, "run": pick["run_name"], "run_id": pick["run_id"],
                     "method type": METHOD_TYPE[pick["params.model"]],
                     "timeout": int(pick["params.timeout"]),
                     "val macro F1": pick["metrics.val_f1_macro"],
                     "next best": next(iter(c[c["run_name"] != pick["run_name"]]["run_name"]),
                                       "none"),
                     "next best F1": next(iter(c[c["run_name"] != pick["run_name"]]
                                               ["metrics.val_f1_macro"]), float("nan")),
                     "model file MiB": round(pick["metrics.model_file_bytes"] / 2**20, 2),
                     "skipped for file size": ", ".join(above["run_name"]) or "none"})  # fmt: skip
    return pd.DataFrame(rows)


def comparison_set(s: pd.DataFrame) -> pd.DataFrame:
    """Per task and method type, the run with the highest 15 s validation macro F1 (its
    default, tuned, balanced or two-step variants compete; ties keep the earlier experiment)."""
    rows = []
    at = s[(s["params.timeout"] == 15) & s["params.model"].isin(METHOD_TYPE)]
    at = at.assign(type=at["params.model"].map(METHOD_TYPE),
                   order=at["tags.experiment_id"].map(SELECTION_EXPS.index))  # fmt: skip
    for task in CORE_TASKS:
        for typ in dict.fromkeys(METHOD_TYPE.values()):
            c = at[(at["params.task"] == task) & (at["type"] == typ)]
            r = c.sort_values(["metrics.val_f1_macro", "order"], ascending=[False, True]).iloc[0]
            rows.append({"task": task, "method type": typ, "run": r["run_name"],
                         "run_id": r["run_id"], "val macro F1": r["metrics.val_f1_macro"],
                         "variants compared": len(c)})  # fmt: skip
    return pd.DataFrame(rows)


# The 28 Oct tie rule, kept only to show what it would have chosen (plan: its picks are scored
# on the test set too). Simpler first: type, then default < tuned, then balancing, then direct.
OLD_RULE_TYPE_RANK = {"decision tree": 0, "KNN": 1, "logistic regression": 2, "SVM-RBF": 3,
                      "Random Forest": 4, "LightGBM": 5, "XGBoost": 5}  # fmt: skip
OLD_RULE_BALANCE_RANK = {"none": 0, "class_weight": 1, "oversample": 2, "smote": 3}


def old_rule_choice(s: pd.DataFrame) -> pd.DataFrame:
    """What the superseded 28 Oct rule would pick per task: among runs within 0.03 of the best
    validation macro F1 (any experiment, any timeout), the simplest (type, default < tuned,
    balancing, direct < two-step), then the faster predictor, then 15 s."""
    rows = []
    for task in CORE_TASKS:
        c = s[(s["params.task"] == task) & s["params.model"].isin(METHOD_TYPE)].copy()
        best = c["metrics.val_f1_macro"].max()
        c = c[best - c["metrics.val_f1_macro"] < WINNER_MARGIN]
        builds = c["params.builds_on"] if "params.builds_on" in c else pd.Series("", c.index)
        tuned = (c["tags.tuned"].astype(str) == "True") | builds.astype(str).str.match(r"E(07|17)-")
        c = c.assign(
            rank_type=c["params.model"].map(METHOD_TYPE).map(OLD_RULE_TYPE_RANK),
            rank_tuned=tuned.astype(int),
            rank_balance=c["params.class_balancing"].fillna("none").map(OLD_RULE_BALANCE_RANK),
            rank_two_step=(c["params.model"] == "xgb_two_step").astype(int),
        )
        keys = ["rank_type", "rank_tuned", "rank_balance", "rank_two_step",
                "metrics.predict_ms_per_1000", "params.timeout"]  # fmt: skip
        pick = c.sort_values(keys).iloc[0]
        rows.append({"task": task, "run": pick["run_name"], "run_id": pick["run_id"],
                     "method type": METHOD_TYPE[pick["params.model"]],
                     "val macro F1": pick["metrics.val_f1_macro"],
                     "below the best by": round(best - pick["metrics.val_f1_macro"], 4),
                     "runs within 0.03": len(c)})  # fmt: skip
    return pd.DataFrame(rows)


def e09_table(h: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """E09 per task and feature set (15 s, validation): "all" is the headline run itself."""
    from blindspy.features import FEATURE_SETS

    head = final_choice(s).set_index("task")
    e = h[(h["tags.experiment_id"] == "E09") & (h["tags.protocol"] == "holdout")]
    rows = []
    for task in CORE_TASKS:
        base = h[h["run_name"] == head.loc[task, "run"]].iloc[0]
        ref = base["metrics.val_f1_macro"]
        for fs in FEATURE_SETS:
            r = base if fs == "all" else e[e["run_name"] == f"E09-{fs}-{task}-15s"]
            if fs != "all":
                if r.empty:
                    continue
                r = r.iloc[0]
            rows.append({
                "task": task, "feature set": fs, "features": int(r["params.n_features"]),
                "val macro F1": round(r["metrics.val_f1_macro"], 4),
                "change vs all": round(r["metrics.val_f1_macro"] - ref, 4),
                "CV macro F1": f"{r['metrics.cv_f1_macro_mean']:.4f} +- "
                               f"{r['metrics.cv_f1_macro_std']:.4f}",
            })  # fmt: skip
    return pd.DataFrame(rows)


def e09_verdicts(t: pd.DataFrame) -> dict:
    """E09 thresholds on e09_table: (a) no group alone reaches all 23 features (4 of 4 tasks);
    (b) dropping any one group costs at most 0.02 (3 of 4 tasks); (c) timing-only within 0.01
    (3 of 4 tasks). Also flags any set at least 0.03 above all 23 (stop and ask the user)."""
    by = {task: g.set_index("feature set")["change vs all"] for task, g in t.groupby("task")}
    only = {k: round(float(v[v.index.str.startswith("only_")].max()), 4) for k, v in by.items()}
    drop = {k: round(float(v[v.index.str.startswith("no_")].min()), 4) for k, v in by.items()}
    timing = {k: float(v["timing_only"]) for k, v in by.items()}
    above = {k: v[v >= WINNER_MARGIN].index.tolist() for k, v in by.items()}
    return {
        "E09a": {"best_single_group_minus_all": only,
                 "tasks_all_groups_below": sum(x < 0 for x in only.values()), "needed": 4,
                 "supported": all(x < 0 for x in only.values())},
        "E09b": {"worst_drop_one_group": drop,
                 "tasks_within_0.02": sum(x >= -0.02 for x in drop.values()), "needed": 3,
                 "supported": sum(x >= -0.02 for x in drop.values()) >= 3},
        "E09c": {"timing_only_minus_all": timing,
                 "tasks_within_0.01": sum(abs(x) <= 0.01 for x in timing.values()), "needed": 3,
                 "supported": sum(abs(x) <= 0.01 for x in timing.values()) >= 3},
        "E09_stop_and_ask": {"sets_at_least_0.03_above_all": above,
                             "triggered": any(above.values())},
    }  # fmt: skip


def _analysis_runs(h: pd.DataFrame, exp: str) -> pd.DataFrame:
    return h[(h["tags.experiment_id"] == exp) & (h["tags.protocol"] == "holdout")]


def e13_table(h: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """E13 learning curve per task: validation macro F1 mean +- std over the subsample seeds at
    each training fraction; 100% is the headline run (one run, the full training set)."""
    head = final_choice(s).set_index("task")
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())["tasks"]
    train_rows = {k: v["summary"]["train"] for k, v in manifest.items()}
    e = _analysis_runs(h, "E13")
    rows = []
    for task in CORE_TASKS:
        g = e[e["params.task"] == task].copy()
        g["f"] = g["params.train_fraction"].astype(float)
        for f, x in g.groupby("f"):
            rows.append(
                {
                    "task": task,
                    "train fraction": f,
                    "runs": len(x),
                    "training rows": int(x["params.n_train_rows"].astype(float).iloc[0]),
                    "val macro F1 mean": round(x["metrics.val_f1_macro"].mean(), 4),
                    "val macro F1 std": round(x["metrics.val_f1_macro"].std(ddof=1), 4),
                }
            )
        base = h[h["run_name"] == head.loc[task, "run"]].iloc[0]
        if not g.empty:
            rows.append({"task": task, "train fraction": 1.0, "runs": 1,
                         "training rows": train_rows[f"{task}_15s"],
                         "val macro F1 mean": round(base["metrics.val_f1_macro"], 4),
                         "val macro F1 std": float("nan")})  # fmt: skip
    return pd.DataFrame(rows)


def e13_verdicts(t: pd.DataFrame) -> dict:
    """(a) 100% above 10% in all 4 tasks; (b) gain 80% -> 100% below 0.01 in 3 of 4;
    (c) seed std at 10% above the std at 80% in 3 of 4."""
    m = {k: g.set_index("train fraction") for k, g in t.groupby("task")}
    up = {k: round(float(g.loc[1.0, "val macro F1 mean"] - g.loc[0.1, "val macro F1 mean"]), 4)
          for k, g in m.items()}  # fmt: skip
    last = {k: round(float(g.loc[1.0, "val macro F1 mean"] - g.loc[0.8, "val macro F1 mean"]), 4)
            for k, g in m.items()}  # fmt: skip
    spread = {k: [float(g.loc[0.1, "val macro F1 std"]), float(g.loc[0.8, "val macro F1 std"])]
              for k, g in m.items()}  # fmt: skip
    return {
        "E13a": {"gain_10_to_100": up, "supported": all(v > 0 for v in up.values())},
        "E13b": {"gain_80_to_100": last, "tasks_below_0.01": sum(v < 0.01 for v in last.values()),
                 "needed": 3, "supported": sum(v < 0.01 for v in last.values()) >= 3},
        "E13c": {"std_at_10_and_80": spread,
                 "tasks_10_above_80": sum(a > b for a, b in spread.values()), "needed": 3,
                 "supported": sum(a > b for a, b in spread.values()) >= 3},
    }  # fmt: skip


def e14_table(h: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """E14 per task over the split seeds: validation macro F1 and accuracy mean +- std, range,
    mean CV macro F1 and the mean CV - validation gap, against the official (seed 42) run."""
    head = final_choice(s).set_index("task")
    e = _analysis_runs(h, "E14")
    rows = []
    for task in CORE_TASKS:
        g = e[e["params.task"] == task]
        if g.empty:
            continue
        f1, cv = g["metrics.val_f1_macro"], g["metrics.cv_f1_macro_mean"]
        official = h[h["run_name"] == head.loc[task, "run"]].iloc[0]["metrics.val_f1_macro"]
        rows.append({
            "task": task, "seeds": len(g),
            "val macro F1 mean": round(f1.mean(), 4), "val macro F1 std": round(f1.std(ddof=1), 4),
            "val macro F1 min": round(f1.min(), 4), "val macro F1 max": round(f1.max(), 4),
            "val accuracy mean": round(g["metrics.val_accuracy"].mean(), 4),
            "val accuracy std": round(g["metrics.val_accuracy"].std(ddof=1), 4),
            "CV macro F1 mean": round(cv.mean(), 4),
            "CV - val mean": round((cv - f1).mean(), 4),
            "official split (seed 42) val macro F1": round(official, 4),
        })  # fmt: skip
    return pd.DataFrame(rows)


def e14_verdicts(t: pd.DataFrame) -> dict:
    """(a) std of validation macro F1 below 0.01 in 3 of 4 tasks; (b) |mean CV - mean val| below
    0.01 in 3 of 4; (c) the official split's score lies inside the 10 resplits' range in 4 of 4."""
    t = t.set_index("task")
    std = t["val macro F1 std"].to_dict()
    gap = t["CV - val mean"].to_dict()
    off = t["official split (seed 42) val macro F1"]
    inside = ((off >= t["val macro F1 min"]) & (off <= t["val macro F1 max"])).to_dict()
    return {
        "E14a": {"std": std, "tasks_below_0.01": sum(v < 0.01 for v in std.values()), "needed": 3,
                 "supported": sum(v < 0.01 for v in std.values()) >= 3},
        "E14b": {"cv_minus_val": gap, "tasks_below_0.01": sum(abs(v) < 0.01 for v in gap.values()),
                 "needed": 3, "supported": sum(abs(v) < 0.01 for v in gap.values()) >= 3},
        "E14c": {"official_inside_range": inside, "needed": 4,
                 "supported": all(inside.values())},
    }  # fmt: skip


def e14_mean_confusion(h: pd.DataFrame, task: str, runner: str) -> pd.DataFrame:
    """E14: the validation confusion matrix averaged over the split seeds (rows true)."""
    e = _analysis_runs(h, "E14")
    cms = [run_confusion(r, runner) for r in e[e["params.task"] == task]["run_id"]]
    return (sum(cms) / len(cms)).round(1)


ISNAN = {"accuracy": 0.8801, "f1_weighted": 0.8797}  # Isnan et al. 2026, test set of 8,198 rows


def e16_table(h: pd.DataFrame) -> pd.DataFrame:
    """E16 runs on the Kaggle data: protocol, rows, the grid's pick and the one test score."""
    e = h[(h["tags.experiment_id"] == "E16") & ~h["run_name"].str.startswith("E16-size-")]
    rows = []
    for _, r in e.sort_values("run_name").iterrows():
        rows.append({
            "run": r["run_name"], "split": r["params.split"],
            "train / test rows": f"{int(r['params.n_train']):,} / {int(r['params.n_test']):,}",
            "grid pick": pick if pd.notna(pick := r.get("params.grid_best_params"))
                         else "fixed (configs/final/B.yaml)",
            "grid best CV accuracy": round(r["metrics.grid_best_cv_accuracy"], 4)
                                     if pd.notna(r.get("metrics.grid_best_cv_accuracy")) else "",
            "test accuracy": round(r["metrics.kaggle_test_accuracy"], 4),
            "test weighted F1": round(r["metrics.kaggle_test_f1_weighted"], 4),
            "test macro F1": round(r["metrics.kaggle_test_f1_macro"], 4),
        })  # fmt: skip
    return pd.DataFrame(rows)


def _e16_size_runs(h: pd.DataFrame) -> pd.DataFrame:
    e = h[h["tags.experiment_id"] == "E16"]
    return e[e["run_name"].str.startswith("E16-size-") & (e["run_name"] != "E16-size-full")]


def e16_size_table(h: pd.DataFrame) -> pd.DataFrame:
    """E16 (d): (a)'s forest on the timeouts-mixed training rows cut to each per-timeout
    training size (5 subsample seeds), scored on (a)'s test rows (overall and per timeout),
    next to (b)'s forest trained on that many rows of one timeout (its own test rows)."""
    e = h[h["tags.experiment_id"] == "E16"].set_index("run_name")
    per_t = [c for c in e.columns if c.startswith("metrics.kaggle_test_accuracy_")]
    order = sorted(per_t, key=lambda c: int(c.rsplit("_", 1)[1].removesuffix("s")))
    b = e[e.index.str.startswith("E16-per_timeout-rf-")]
    rows = []
    groups = [("all", e.loc[["E16-size-full"]])] if "E16-size-full" in e.index else []
    size = _e16_size_runs(h).set_index("run_name")
    groups += [(n, g) for n, g in size.groupby(size["params.n_train"].astype(int))]
    for _, g in sorted(groups, key=lambda x: -int(x[1]["params.n_train"].iloc[0])):
        acc = g["metrics.kaggle_test_accuracy"]
        match = b[b["params.n_train"].astype(int) == int(g["params.n_train"].iloc[0])]
        rows.append({
            "training rows (timeouts mixed)": f"{int(g['params.n_train'].iloc[0]):,}",
            "runs": len(g),
            "test accuracy mean": round(acc.mean(), 4),
            "std": round(acc.std(), 4) if len(g) > 1 else "",
            "min - max": f"{acc.min():.4f} - {acc.max():.4f}" if len(g) > 1 else "",
            **{f"{c.rsplit('_', 1)[1]} test rows": round(g[c].mean(), 4) for c in order},
            "(b) same rows, one timeout": (f"{match.index[0].rsplit('-', 1)[1]}: "
                                           f"{match['metrics.kaggle_test_accuracy'].iloc[0]:.4f}"
                                           if len(match) else ""),
        })  # fmt: skip
    return pd.DataFrame(rows)


def e16_verdicts(h: pd.DataFrame) -> dict:
    """(a) accuracy and weighted F1 within 0.01 of Isnan et al.'s; (b) mean per-timeout forest
    accuracy at least 0.01 below (a)'s; (c) mean per-timeout XGBoost accuracy at least the
    forest's; (d), once its runs exist: with the training rows cut to the per-timeout sizes,
    mixing timeouts adds less than 0.01 accuracy over (b)."""
    e = h[h["tags.experiment_id"] == "E16"].set_index("run_name")
    a = e.loc["E16-isnan-rf"]
    acc, wf1 = a["metrics.kaggle_test_accuracy"], a["metrics.kaggle_test_f1_weighted"]
    rf = e[e.index.str.startswith("E16-per_timeout-rf-")]["metrics.kaggle_test_accuracy"]
    xgb = e[e.index.str.startswith("E16-per_timeout-xgb-")]["metrics.kaggle_test_accuracy"]
    out = {
        "E16a": {"accuracy": round(acc, 4), "weighted_f1": round(wf1, 4), "isnan": ISNAN,
                 "supported": bool(abs(acc - ISNAN["accuracy"]) <= 0.01
                                   and abs(wf1 - ISNAN["f1_weighted"]) <= 0.01)},
        "E16b": {"isnan_accuracy": round(acc, 4), "per_timeout_rf_mean": round(rf.mean(), 4),
                 "gap": round(acc - rf.mean(), 4), "timeouts": len(rf),
                 "supported": bool(round(acc - rf.mean(), 10) >= 0.01)},
        "E16c": {"per_timeout_xgb_mean": round(xgb.mean(), 4),
                 "per_timeout_rf_mean": round(rf.mean(), 4),
                 "supported": bool(xgb.mean() >= rf.mean())},
    }  # fmt: skip
    size = _e16_size_runs(h)
    if len(size):
        # (d): mean over the sizes of each size's mean over seeds, vs (b)'s mean
        mixed = size.groupby("params.n_train")["metrics.kaggle_test_accuracy"].mean().mean()
        full = e.loc["E16-size-full", "metrics.kaggle_test_accuracy"]
        out["E16d"] = {"isnan_accuracy": round(acc, 4), "size_matched_mixed_mean": round(mixed, 4),
                       "per_timeout_rf_mean": round(rf.mean(), 4),
                       "size_part": round(acc - mixed, 4), "rest": round(mixed - rf.mean(), 4),
                       "sizes": int(size["params.n_train"].nunique()),
                       "full_reproduces_a": bool(full == acc),
                       "supported": bool(round(mixed - rf.mean(), 10) < 0.01)}  # fmt: skip
    return out


# ---- Phase 4 -------------------------------------------------------------------------------


def _e15_runs(h: pd.DataFrame) -> pd.DataFrame:
    return h[(h["tags.experiment_id"] == "E15") & (h["tags.protocol"] == "calibration")]


def e15_table(h: pd.DataFrame) -> pd.DataFrame:
    """E15 per task (newest run per name): validation ECE uncalibrated and isotonic, the
    adoption decision (uncalibrated ECE > 0.05 and isotonic lower, as logged), and the checks
    that calibration leaves the predicted class and the validation macro F1 unchanged."""
    e = _e15_runs(h)
    rows = []
    for task in CORE_TASKS:
        g = e[e["params.task"] == task]
        if g.empty:
            continue
        r = g.iloc[0]
        final = yaml.safe_load((ROOT / "configs" / "final" / f"{task}.yaml").read_text())
        rows.append({
            "task": task, "run": r["run_name"], "headline": r["params.builds_on"],
            "val accuracy": round(r["metrics.val_accuracy"], 4),
            "mean confidence, uncalibrated":
                round(r["metrics.val_mean_confidence_uncalibrated"], 4),
            "mean confidence, isotonic": round(r["metrics.val_mean_confidence_isotonic"], 4),
            "ECE uncalibrated": round(r["metrics.val_ece_uncalibrated"], 4),
            "ECE isotonic": round(r["metrics.val_ece_isotonic"], 4),
            "adopted": str(r["tags.calibration_adopted"]) == "True",
            "same predicted class (share of rows)": r["metrics.val_label_agreement_base_vs_plain"],
            "calibrated argmax would differ (share, unused)":
                round(r["metrics.val_calibrated_argmax_differs"], 4),
            "val macro F1 = source run": r["metrics.val_f1_macro"] == final["source_val_f1_macro"],
        })  # fmt: skip
    return pd.DataFrame(rows)


def e15_verdicts(h: pd.DataFrame) -> dict:
    """(a) uncalibrated validation ECE above 0.05 in at least 2 of 4 tasks; (b) isotonic ECE
    lower in all 4; (c) same predicted class on every row and validation macro F1 equal to the
    source run in all 4. Judged on the unrounded logged values; reported rounded to 4 places."""
    t = e15_table(h).set_index("task")
    e = _e15_runs(h).set_index("params.task")
    raw_full = e["metrics.val_ece_uncalibrated"].to_dict()
    iso_full = e["metrics.val_ece_isotonic"].to_dict()
    raw, iso = t["ECE uncalibrated"].to_dict(), t["ECE isotonic"].to_dict()
    above = sum(raw_full[k] > 0.05 for k in raw)
    lower = {k: iso_full[k] < raw_full[k] for k in raw}
    same = ((t["same predicted class (share of rows)"] == 1.0)
            & t["val macro F1 = source run"]).to_dict()  # fmt: skip
    return {
        "E15a": {"ece_uncalibrated": raw, "tasks_above_0.05": above, "needed": 2,
                 "supported": above >= 2},
        "E15b": {"ece_isotonic": iso, "isotonic_lower": lower, "needed": 4,
                 "supported": len(lower) == 4 and all(lower.values())},
        "E15c": {"class_and_f1_unchanged": same, "needed": 4,
                 "supported": len(same) == 4 and all(same.values())},
        "E15_adopted": {"adopted": t["adopted"].to_dict()},
    }  # fmt: skip


def e15_reliability_figure(h: pd.DataFrame, runner: str, path: Path) -> None:
    """Reliability diagrams of the 4 E15 runs (2 x 2), uncalibrated and isotonic, read from each
    run's calibration/reliability_bins.csv artifact."""
    import matplotlib.pyplot as plt

    e = _e15_runs(h)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
    for ax, task in zip(axes.flat, CORE_TASKS, strict=True):
        r = e[e["params.task"] == task].iloc[0]
        art = ROOT / "mlruns" / runner / "artifacts" / r["run_id"] / "artifacts" / "calibration"
        b = pd.read_csv(art / "reliability_bins.csv")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for name, g in b[b["rows"] > 0].groupby("probabilities", sort=False):
            ece = r[f"metrics.val_ece_{name}"]
            ax.plot(g["mean_confidence"], g["accuracy"], "o-", ms=3, lw=1.2,
                    label=f"{name} (ECE {ece:.4f})")  # fmt: skip
        adopted = str(r["tags.calibration_adopted"]) == "True"
        ax.set_title(f"{task}{' - adopted' if adopted else ''}",
                     fontsize=9)  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean confidence in bin", fontsize=8)
        ax.set_ylabel("accuracy in bin", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("E15 reliability on validation (15 bins)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---- E12: test-set scores, bootstrap intervals and the 26 paired tests (E12 addendum) ------

N_BOOT = 10_000  # E12 addendum item 2: stratified paired resamples per task
BOOT_SEED = 42
HOLM_ALPHA = 0.05


def stratified_indices(y, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, len(y)) row positions: every resample draws, within each true class (in sorted
    class order), as many rows as the class has, with replacement, from numpy default_rng(seed).
    One index set per task, shared by every model of the task (paired)."""
    y = np.asarray(y).astype(str)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(y)), dtype=np.int64)
    col = 0
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        out[:, col : col + len(pos)] = pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))]
        col += len(pos)
    return out


def _codes(y, pred, classes) -> tuple[np.ndarray, np.ndarray]:
    lookup = {c: i for i, c in enumerate(classes)}
    return (np.array([lookup[str(v)] for v in y]), np.array([lookup[str(v)] for v in pred]))


def macro_f1_resampled(y, pred, idx: np.ndarray, classes) -> np.ndarray:
    """Macro F1 over `classes` on each resample (rows idx[b]); a class with no true and no
    predicted rows scores 0, as sklearn's f1_score(average="macro", zero_division=0)."""
    yi, pi = _codes(y, pred, classes)
    yt, pt = yi[idx], pi[idx]
    f1 = np.zeros((idx.shape[0], len(classes)))
    for k in range(len(classes)):
        t, p = yt == k, pt == k
        tp = (t & p).sum(axis=1)
        den = t.sum(axis=1) + p.sum(axis=1)
        f1[:, k] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1.mean(axis=1)


def accuracy_resampled(y, pred, idx: np.ndarray) -> np.ndarray:
    hit = np.asarray(y).astype(str) == np.asarray(pred).astype(str)
    return hit[idx].mean(axis=1)


def percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def paired_p(d: np.ndarray) -> float:
    """Two-sided bootstrap p for H0 d = 0 (E12 addendum item 1):
    min(1, 2 * min((1 + #{d* <= 0}) / (B + 1), (1 + #{d* >= 0}) / (B + 1)))."""
    b = len(d)
    lo = (1 + np.sum(d <= 0)) / (b + 1)
    hi = (1 + np.sum(d >= 0)) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p: dict, alpha: float = HOLM_ALPHA) -> dict:
    """Holm step-down: {name: (adjusted p, rejected)}. Adjusted p_(i) = max over j <= i of
    min(1, (m - j + 1) p_(j)) in ascending order; rejected where adjusted p <= alpha."""
    m = len(p)
    out, running = {}, 0.0
    for j, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - j) * v))
        out[k] = (running, running <= alpha)
    return out


def final_predictions(run_id: str, runner: str, name: str = "test", root: Path = ROOT):
    """A final run's saved predictions, sorted by raw_row (so every model of a task lines up)."""
    path = root / "mlruns" / runner / "artifacts" / run_id / "artifacts" / "predictions"
    return pd.read_csv(path / f"{name}_predictions.csv").sort_values("raw_row", ignore_index=True)


ROLE_ORDER = {"headline": 0, "comparison": 1, "old_rule": 2}


def e12_runs(h: pd.DataFrame, protocol: str = "final_test") -> pd.DataFrame:
    e = h[(h["tags.experiment_id"] == "E12") & (h["tags.protocol"] == protocol)].copy()
    e["type"] = e["params.model"].map(METHOD_TYPE)
    e["order"] = e["params.role"].map(ROLE_ORDER)
    return e.sort_values(["params.task", "order", "type"])


def e12_scores(
    h: pd.DataFrame, runner: str, name: str = "test", protocol: str = "final_test",
    root: Path = ROOT, n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, dict]:  # fmt: skip
    """One row per final run: test macro F1 and accuracy with 95% percentile intervals from the
    task's stratified paired resamples, against the source run's validation macro F1, plus the
    other described metrics. Returns (table, {run name: macro F1 draws}) for the paired tests.
    `name` is the scored rows' metric prefix ("rehearsal" in the scratch rehearsal)."""
    e = e12_runs(h, protocol)
    rows, draws = [], {}
    for task in CORE_TASKS:
        g = e[e["params.task"] == task]
        if g.empty:
            continue
        preds = {r["run_name"]: final_predictions(r["run_id"], runner, name, root)
                 for _, r in g.iterrows()}  # fmt: skip
        first = next(iter(preds.values()))
        for p in preds.values():  # every model of the task scored on the same rows
            if not (p["raw_row"].equals(first["raw_row"]) and p["true"].equals(first["true"])):
                raise ValueError(f"{task}: final runs were not scored on the same rows")
        y = first["true"].astype(str).to_numpy()
        classes = sorted(set(y))
        idx = stratified_indices(y, n_boot)
        for _, r in g.iterrows():
            p = preds[r["run_name"]]["predicted"].astype(str).to_numpy()
            f1 = macro_f1_resampled(y, p, idx, classes)
            acc = accuracy_resampled(y, p, idx)
            point = float(macro_f1_resampled(y, p, np.arange(len(y))[None, :], classes)[0])
            if not np.isclose(point, r[f"metrics.{name}_f1_macro"], atol=1e-12):
                raise ValueError(f"{r['run_name']}: saved predictions do not match logged F1")
            draws[r["run_name"]] = f1
            val = float(r["params.source_val_f1_macro"])
            f_lo, f_hi = percentile_interval(f1)
            a_lo, a_hi = percentile_interval(acc)
            rows.append({
                "task": task, "role": r["params.role"], "method type": r["type"],
                "run": r["run_name"], "source run": r["params.builds_on"],
                "val macro F1": round(val, 4),
                f"{name} macro F1": round(point, 4),
                "95% interval": f"{f_lo:.4f}-{f_hi:.4f}",
                f"{name} - val": round(point - val, 4),
                f"{name} accuracy": round(r[f"metrics.{name}_accuracy"], 4),
                "accuracy 95% interval": f"{a_lo:.4f}-{a_hi:.4f}",
                f"{name} weighted F1": round(r[f"metrics.{name}_f1_weighted"], 4),
                f"{name} balanced accuracy": round(r[f"metrics.{name}_balanced_accuracy"], 4),
                "CV macro F1 (fit rows)": round(r["metrics.cv_f1_macro_mean"], 4),
                f"CV - {name}": round(r["metrics.cv_f1_macro_mean"] - point, 4),
                "calibration": r["params.calibrate"],
                "_val": val, "_point": point,  # unrounded, for the verdicts; not printed
            })  # fmt: skip
    return pd.DataFrame(rows), draws


def e12_tests(scores: pd.DataFrame, draws: dict) -> pd.DataFrame:
    """The 26 confirmatory paired tests (E12 addendum item 1): per task the headline model
    against every other final run (6 comparison members + the old-rule pick where it is not a
    member); d = headline - other macro F1 on the same resamples; two-sided bootstrap p; Holm
    over all of them at 0.05."""
    rows = []
    for task, g in scores.groupby("task", sort=False):
        head = g[g["role"] == "headline"].iloc[0]
        for _, o in g[g["role"] != "headline"].iterrows():
            d = draws[head["run"]] - draws[o["run"]]
            lo, hi = percentile_interval(d)
            rows.append({"task": task, "headline": head["run"], "other": o["run"],
                         "other role": o["role"], "other type": o["method type"],
                         "d (headline - other)": round(head["_point"] - o["_point"], 4),
                         "d 95% interval": f"{lo:.4f}-{hi:.4f}", "p": paired_p(d)})  # fmt: skip
    t = pd.DataFrame(rows)
    adj = holm(dict(zip(t["other"], t["p"], strict=True)))
    t["Holm p"] = t["other"].map(lambda k: adj[k][0])
    t["different (Holm, 0.05)"] = t["other"].map(lambda k: adj[k][1])
    return t


def e12_mc_error(tests: pd.DataFrame, draws: dict, alpha: float = HOLM_ALPHA) -> pd.DataFrame:
    """Monte Carlo error of each paired test's p from its own B draws (described, not a new test):
    the smaller tail share q = (1 + k) / (B + 1) has standard error sqrt(q (1 - q) / B); the band
    is p = 2q +- 2 standard errors, times the test's Holm rank multiplier (m - rank). A test is
    settled when the whole Holm band lies on one side of alpha."""
    t = tests.sort_values("p", kind="mergesort").reset_index(drop=True)
    rows = []
    for i, r in t.iterrows():
        d = draws[r["headline"]] - draws[r["other"]]
        b, k = len(d), int(min(np.sum(d <= 0), np.sum(d >= 0)))
        q = (1 + k) / (b + 1)
        se = np.sqrt(q * (1 - q) / b)
        mult = len(t) - i
        lo, hi = min(1.0, mult * max(0.0, 2 * (q - 2 * se))), min(1.0, mult * 2 * (q + 2 * se))
        rows.append({"task": r["task"], "other": r["other"], "smaller tail count": k,
                     "p": f"{r['p']:.4f}", "Holm p": f"{r['Holm p']:.4f}",
                     "Holm p band (+-2 SE)": f"{lo:.4f}-{hi:.4f}",
                     "different (Holm, 0.05)": r["different (Holm, 0.05)"],
                     "settled by B resamples": lo > alpha or hi <= alpha})  # fmt: skip
    return pd.DataFrame(rows)


def verdict_summary(paths) -> pd.DataFrame:
    """Every pre-stated threshold check in the generated tables files (their JSON blocks), one row
    per experiment: which parts were supported and which were not."""
    rows = {}
    for path in paths:
        text = Path(path).read_text()
        for block in re.findall(r"```json\n(.*?)\n```", text, re.S):
            for key, v in json.loads(block).items():
                if "supported" not in v:
                    continue  # notes such as E09's stop rule or E15's adoption table
                exp = re.match(r"E\d+", key).group(0)
                part = key[len(exp) :].lstrip("_") or "single check"
                r = rows.setdefault(exp, {"experiment": exp, "supported": [], "not supported": []})
                r["supported" if v["supported"] else "not supported"].append(part)
    out = pd.DataFrame(sorted(rows.values(), key=lambda r: int(r["experiment"][1:])))
    for c in ("supported", "not supported"):
        out[c] = out[c].map(lambda x: ", ".join(x) if x else "none")
    return out


def e12_confusion_pairs(h: pd.DataFrame, runner: str, k: int = 5, root: Path = ROOT):
    """Top-k confusion pairs (true -> predicted, by flow count) of each headline model on the
    test rows, from its saved predictions, with the share of that true class."""
    e = e12_runs(h)
    rows = []
    for task in CORE_TASKS:
        r = e[(e["params.task"] == task) & (e["params.role"] == "headline")].iloc[0]
        p = final_predictions(r["run_id"], runner, root=root)
        t, q = p["true"].astype(str), p["predicted"].astype(str)
        wrong = pd.DataFrame({"true": t[t != q], "predicted": q[t != q]})
        c = wrong.value_counts().reset_index(name="flows")
        c = c.sort_values(["flows", "true", "predicted"], ascending=[False, True, True]).head(k)
        size = t.value_counts()
        for i, x in enumerate(c.itertuples(index=False)):
            rows.append({"task": task, "run": r["run_name"], "rank": i + 1, "true": x.true,
                         "predicted": x.predicted, "flows": int(x.flows),
                         "share of true class": round(x.flows / size[x.true], 4)})  # fmt: skip
    return pd.DataFrame(rows)


def e12_comparison_figure(scores: pd.DataFrame, path: Path) -> None:
    """Test macro F1 with its 95% interval per final model and task (headline in red), with the
    validation score as a hollow marker."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, task in zip(axes.flat, CORE_TASKS, strict=True):
        g = scores[scores["task"] == task].sort_values("_point")
        lo, hi = zip(*[map(float, v.split("-")) for v in g["95% interval"]], strict=True)
        y = np.arange(len(g))
        col = ["#d62728" if r == "headline" else "#1f77b4" for r in g["role"]]
        for yi, a, b, c in zip(y, lo, hi, col, strict=True):
            ax.plot([a, b], [yi, yi], color=c, lw=1.5)
            ax.plot([a, a, np.nan, b, b], [yi - 0.15, yi + 0.15, np.nan, yi - 0.15, yi + 0.15],
                    color=c, lw=1)  # fmt: skip
        ax.scatter(g["_point"], y, color=col, zorder=3, label="test (95% interval)")
        ax.scatter(g["_val"], y, facecolors="none", edgecolors="black", zorder=3,
                   label="validation")  # fmt: skip
        names = [f"{t} (old rule)" if r == "old_rule" else t
                 for t, r in zip(g["method type"], g["role"], strict=True)]  # fmt: skip
        ax.set_yticks(y, names, fontsize=8)
        ax.set_title(f"{task}: test macro F1 (headline in red)", fontsize=10)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("E12: every final model scored once on the test rows (10,000 stratified paired "
                 "resamples, seed 42)", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def e12_verdicts(scores: pd.DataFrame, h: pd.DataFrame) -> dict:
    """(a) |test - val| <= 0.03 for every headline model; (b) the headline above the old rule's
    pick in at least 3 of 4 tasks (point estimates); (c) Spearman >= 0.8 between the validation
    and test order of the 7 method types in at least 3 of 4 tasks."""
    from scipy.stats import spearmanr

    head = scores[scores["role"] == "headline"].set_index("task")
    gap_full = (head["_point"] - head["_val"]).to_dict()
    gap = {k: round(v, 4) for k, v in gap_full.items()}
    old = old_rule_choice(selection_runs(h)).set_index("task")["run"]
    beats, rho = {}, {}
    for task, g in scores.groupby("task", sort=False):
        pick = g[g["source run"] == old[task]].iloc[0]
        beats[task] = bool(head.loc[task, "_point"] > pick["_point"])
        types = g[g["role"] != "old_rule"]
        rho[task] = round(float(spearmanr(types["_val"], types["_point"]).statistic), 4)
    return {
        "E12a": {"headline_minus_val": gap, "needed": "all |gap| <= 0.03",
                 "supported": all(abs(v) <= 0.03 for v in gap_full.values())},
        "E12b": {"headline_above_old_rule_pick": beats, "tasks": sum(beats.values()),
                 "needed": 3, "supported": sum(beats.values()) >= 3},
        "E12c": {"spearman_val_vs_test_7_types": rho,
                 "tasks_at_least_0.8": sum(v >= 0.8 for v in rho.values()), "needed": 3,
                 "supported": sum(v >= 0.8 for v in rho.values()) >= 3},
    }  # fmt: skip


def e12_headline_details(h: pd.DataFrame) -> pd.DataFrame:
    """The described (untested) extras of the 4 headline test runs: test ECE with the
    probabilities used (calibrated where E15 adopted it) and uncalibrated, mean confidence, best
    and worst class by F1 and recall, the top-5 confusion pairs, model file size and cost."""
    e = e12_runs(h)
    rows = []
    for _, r in e[e["params.role"] == "headline"].iterrows():
        classes = r["params.classes"].split(",")
        f1 = {c: r[f"metrics.test_f1.{c}"] for c in classes}
        rec = {c: r[f"metrics.test_recall.{c}"] for c in classes}
        best_f, worst_f = max(f1, key=f1.get), min(f1, key=f1.get)
        best_r, worst_r = max(rec, key=rec.get), min(rec, key=rec.get)
        rows.append({
            "task": r["params.task"], "run": r["run_name"], "calibration": r["params.calibrate"],
            "test ECE (probabilities used)": round(r["metrics.test_ece_used"], 4),
            "test ECE uncalibrated": round(r["metrics.test_ece_uncalibrated"], 4),
            "mean confidence": round(r["metrics.test_mean_confidence_used"], 4),
            "test accuracy": round(r["metrics.test_accuracy"], 4),
            "weighted - macro F1 gap": round(r["metrics.test_f1_gap_weighted_minus_macro"], 4),
            "best class F1": f"{best_f} {f1[best_f]:.4f}",
            "worst class F1": f"{worst_f} {f1[worst_f]:.4f}",
            "best class recall": f"{best_r} {rec[best_r]:.4f}",
            "worst class recall": f"{worst_r} {rec[worst_r]:.4f}",
            "top-5 confusion pairs (true -> predicted: flows)": r["tags.top_confusion_pairs"],
            "model file MiB": round(r["metrics.model_file_bytes"] / 2**20, 2),
            "train time s": round(r["metrics.train_time_s"], 2),
            "predict ms per 1,000 flows": round(r["metrics.predict_ms_per_1000"], 1),
        })  # fmt: skip
    t = pd.DataFrame(rows)
    return t.sort_values("task", key=lambda c: c.map(CORE_TASKS.index), ignore_index=True)


def e12_per_class(h: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per-class test precision, recall and F1 of the task's headline model."""
    e = e12_runs(h)
    r = e[(e["params.task"] == task) & (e["params.role"] == "headline")].iloc[0]
    counts = json.loads(r["params.class_counts.test"])
    return pd.DataFrame([
        {"class": c, "test rows": counts[c],
         "precision": round(r[f"metrics.test_precision.{c}"], 4),
         "recall": round(r[f"metrics.test_recall.{c}"], 4),
         "F1": round(r[f"metrics.test_f1.{c}"], 4)}
        for c in r["params.classes"].split(",")
    ])  # fmt: skip
