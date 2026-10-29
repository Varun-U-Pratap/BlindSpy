"""Write notebooks/03_models.ipynb (E04-E06) and notebooks/04_tuning.ipynb (E07-E17).

The notebooks only READ results: reports/history.csv (every MLflow run, exported by
scripts/export_history.py), the generated CSVs in reports/ and, for E10, the confusion matrices
in the official MLflow store. They train nothing. The findings under each chart are written
here from the executed notebook's output; rerun this script and execute the notebooks to check
them. Figures are saved to reports/figures/.

Usage: uv run python scripts/make_notebooks.py
       uv run jupyter nbconvert --to notebook --execute --output-dir <dir> notebooks/0[34]_*.ipynb
"""

import subprocess

import nbformat

from blindspy.data import ROOT

SETUP = """import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blindspy import results as R
from blindspy.data import DATA_NAMES
from blindspy.tracking import load_yaml

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 80)
FIG = ROOT / "reports" / "figures"
RUNNER = load_yaml(ROOT / "configs" / "base.yaml")["official_runner"]
h = R.load_history()
s = R.selection_runs(h)


def verdict_table(v: dict, prefixes: tuple) -> pd.DataFrame:
    rows = [{"check": k, "supported": x.get("supported"),
             "details": json.dumps({a: b for a, b in x.items() if a != "supported"},
                                   default=lambda o: o.item() if hasattr(o, "item") else str(o))}
            for k, x in v.items() if k.startswith(prefixes)]  # fmt: skip
    return pd.DataFrame(rows)"""

NB03 = [
    (
        "markdown",
        """# 03 — Model improvements at 15 s (E04–E06)

This notebook **reads** results; it trains nothing. Every run was made by
`scripts/run_experiment.py` from a committed config (`configs/experiments/E0[4-6]_*.yaml`, written
by `scripts/make_configs.py`), logged to MLflow and exported to `reports/history.csv` by
`scripts/export_history.py`. The same tables, in text, are in `reports/phase3_tables.md`
(`scripts/results_tables.py --phase 3`). This notebook is written by `scripts/make_notebooks.py`.

Protocol (data v1): stratified 70/15/15 split with seed 42, 5-fold CV on the training set, fit
on the training set, score on the validation set; the test set is not touched. Each experiment
changes one thing against the current best and runs first at 15 s for all four tasks. A model
is rerun at 30, 60 and 120 s only if its 15 s validation macro F1 beats the task's current best
by at least 0.03 (the winner rule, decided 28 Oct).

- **E04:** KNN with a signed log, sign(x)·log1p(|x|), then StandardScaler or RobustScaler.
- **E05:** Random Forest, 300 trees, default settings.
- **E06:** XGBoost and LightGBM, default settings.""",
    ),
    (
        "code",
        SETUP.replace("DATA_NAMES", "CORE_TASKS, LABELS, ROOT")
        + """
EXPS = ["E04", "E05", "E06"]
p3 = s[s["tags.experiment_id"].isin(EXPS)]
print(len(p3), "runs of E04-E06 (15 s screen and winner reruns); data versions:",
      sorted(set(p3["tags.data_version"])), "; commits:",
      sorted(set(p3["tags.git_commit"].str[:7])))""",
    ),
    ("markdown", "## 1. Screen at 15 s: each experiment against the best run before it"),
    (
        "code",
        """for exp in EXPS:
    print(exp)
    display(R.screen(s, exp))
print("Current best at 15 s after E06")
display(R.current_best(s, "E06"))""",
    ),
    ("markdown", "F03_SCREEN"),
    ("markdown", "## 2. Model comparison at 15 s: validation macro F1 with CV mean ± std"),
    (
        "code",
        """MODELS = ["E02-tree", "E03-knn", "E03-logreg", "E03-svm_rbf", "E04-knn_log_standard",
          "E04-knn_log_robust", "E05-rf", "E06-xgb", "E06-lgbm"]
COLORS = ["#bbbbbb"] * 4 + ["#ff7f0e"] * 2 + ["#2ca02c"] + ["#1f77b4"] * 2
q = s[(s["params.timeout"] == 15) & s["label"].isin(MODELS)]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
for ax, task in zip(axes, CORE_TASKS, strict=True):
    t = q[q["params.task"] == task].set_index("label").loc[MODELS]
    x = np.arange(len(MODELS))
    ax.bar(x, t["metrics.val_f1_macro"], color=COLORS)
    ax.errorbar(x, t["metrics.cv_f1_macro_mean"], yerr=t["metrics.cv_f1_macro_std"], fmt="o",
                color="black", ms=3, capsize=2)
    ax.set_xticks(x, MODELS, rotation=60, ha="right", fontsize=8)
    ax.set_title(f"{task}, 15 s")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("macro F1\\n(bar: validation; dot: 5-fold CV mean +- std)")
fig.suptitle("E02-E03 baselines (grey) and E04-E06 at 15 s, data v1")
fig.tight_layout()
fig.savefig(FIG / "phase3_model_comparison_E04-E06.png", dpi=120)""",
    ),
    ("markdown", "Winner reruns at every timeout (the winner rule fired for these two only):"),
    (
        "code",
        """display(R.timeout_table(s, "E04-knn_log_standard", "A2-NonVPN"))
display(R.timeout_table(s, "E05-rf", "B"))""",
    ),
    ("markdown", "F03_COMPARE"),
    ("markdown", "## 3. Per-class F1 at 15 s (validation)"),
    (
        "code",
        """fig, axes = plt.subplots(2, 2, figsize=(17, 10))
for ax, task in zip(axes.flat, CORE_TASKS, strict=True):
    t = q[q["params.task"] == task].set_index("label").loc[MODELS]
    classes = LABELS[task]
    m = t[[f"metrics.val_f1.{c}" for c in classes]].to_numpy(dtype=float)
    ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)), MODELS, fontsize=8)
    ax.set_title(f"{task}, 15 s: validation F1 per class")
fig.tight_layout()
fig.savefig(FIG / "per_class_f1_E04-E06.png", dpi=120)""",
    ),
    ("markdown", "F03_CLASSES"),
    ("markdown", "## 4. Cost: training and prediction time, model size, memory"),
    (
        "code",
        """for exp in EXPS:
    print(exp)
    display(R.run_details(s, exp))""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey="row")
for c, task in enumerate(CORE_TASKS):
    t = q[q["params.task"] == task].set_index("label").loc[MODELS]
    for r, (col, xlabel) in enumerate([("metrics.train_time_s", "training time (s, log)"),
                                      ("metrics.predict_ms_per_1000",
                                       "prediction time per 1,000 flows (ms, log)")]):
        ax = axes[r][c]
        for (label, row), color in zip(t.iterrows(), COLORS, strict=True):
            ax.scatter(row[col], row["metrics.val_f1_macro"], color=color, s=30)
            ax.annotate(label.split("-", 1)[1], (row[col], row["metrics.val_f1_macro"]),
                        fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{task}, 15 s")
    axes[0][0].set_ylabel("validation macro F1")
    axes[1][0].set_ylabel("validation macro F1")
fig.tight_layout()
fig.savefig(FIG / "cost_E04-E06.png", dpi=120)""",
    ),
    ("markdown", "F03_COST"),
    ("markdown", "## 5. Threshold checks (pre-stated in E04–E06)"),
    ("code", """verdict_table(R.phase3_verdicts(s, RUNNER), ("E04", "E05", "E06"))"""),
    ("markdown", "F03_VERDICTS"),
]

NB04 = [
    (
        "markdown",
        """# 04 — Tuning, balancing, structure and analysis (E07–E17)

This notebook **reads** results; it trains nothing (see notebook 03 for how runs are made and
exported). It is written by `scripts/make_notebooks.py`; the same tables, in text, are in
`reports/phase3_tables.md`. Section 3 (E10) reads confusion matrices from the official MLflow
store `mlruns/<runner>/` (git-ignored), so it runs only on the machine that holds the store.

- **E07:** Optuna, 50 trials per search (TPESampler seed 42, 5-fold CV macro F1 on train), on
  the best 2 of Random Forest / XGBoost / LightGBM per task plus KNN.
- **E08:** class balancing on the current best per task: none → class weights → random
  oversampling → SMOTE, stopping when a step does not help.
- **E10:** task B as one 14-class model or two steps (VPN / non-VPN, then the type).
- **E17:** the same 50-trial search for every method type not yet tuned (decided 28 Oct after
  E10), then the final choice per task and the 28-model comparison set.
- **E09, E13, E14:** feature-group ablation, learning curve and split-seed stability of the
  4 headline models (analysis only; they change nothing).
- **E11** does not run: the ARFF files hold no session information (E00).
- **E16:** Isnan et al. (2026) reproduced on the Kaggle CSV (its own data version kaggle-k1),
  then corrected, with a training-size control (d).""",
    ),
    (
        "code",
        SETUP.replace("DATA_NAMES", "CORE_TASKS, ROOT, TIMEOUTS")
        + """
print(len(h), "runs in history.csv (newest per name);",
      int((h["tags.trial"].astype(str) == "True").sum()), "of them Optuna trials")""",
    ),
    ("markdown", "## 1. E07: what was tuned, the searches, and the screen"),
    (
        "code",
        """display(R.e07_candidates(s))
display(R.e07_searches(s, "E07"))
display(R.screen(s, "E07"))""",
    ),
    ("markdown", "Optuna histories: CV macro F1 of each trial (dots) and the best so far (line)."),
    (
        "code",
        """trials = h[h["tags.trial"].astype(str) == "True"].copy()
trials["search"] = trials["run_name"].str.rsplit("-trial-", n=1).str[0]
trials["number"] = trials["params.trial_number"].astype(int)


def plot_histories(exp: str, ncols: int, path: str) -> None:
    names = sorted(trials.loc[trials["tags.experiment_id"] == exp, "search"].unique())
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        t = trials[trials["search"] == name].sort_values("number")
        ax.scatter(t["number"], t["metrics.cv_f1_macro_mean"], s=8, color="#1f77b4")
        ax.plot(t["number"], t["metrics.cv_f1_macro_mean"].cummax(), color="#d62728")
        ax.set_title(name.removesuffix("-15s"), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.supxlabel("trial")
    fig.supylabel("5-fold CV macro F1 (train)")
    fig.tight_layout()
    fig.savefig(FIG / path, dpi=110)


plot_histories("E07", 4, "optuna_histories_E07.png")""",
    ),
    ("markdown", "F04_E07"),
    ("markdown", "## 2. E08: the balancing ladder on the current best per task"),
    (
        "code",
        """ladder = R.e08_ladder(s, R.E08_BASE)
display(ladder)
display(R.screen(s, "E08"))
fig, ax = plt.subplots(figsize=(9, 3.6))
steps = ["class_weight", "oversample", "smote"]
for i, step in enumerate(steps):
    t = ladder[ladder["step"] == step].set_index("task").reindex(CORE_TASKS)
    ax.bar(np.arange(4) + (i - 1) * 0.25, t["change vs none"].astype(float), 0.25, label=step)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(4), CORE_TASKS)
ax.set_ylabel("validation macro F1 change vs none")
ax.set_title("E08 balancing ladder (missing bar = step not run: an earlier step did not help)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "e08_ladder.png", dpi=120)""",
    ),
    ("markdown", "F04_E08"),
    ("markdown", "## 3. E10: task B, direct 14-class model vs two steps"),
    (
        "code",
        """display(R.screen(s, "E10"))
R.e10_table(s, RUNNER)""",
    ),
    ("markdown", "F04_E10"),
    ("markdown", "## 4. E17: the same 50-trial search for every method type"),
    (
        "code",
        """e17 = R.e07_searches(s, "E17")
display(e17)
display(R.screen(s, "E17"))
plot_histories("E17", 4, "optuna_histories_E17.png")""",
    ),
    (
        "code",
        """tuned = pd.concat([R.e07_searches(s, "E07"), e17])
tuned["type"] = tuned["model"].str.split("-", n=1).str[1]
val = tuned["val untuned -> tuned"].str.split(" -> ", expand=True).astype(float)
tuned["untuned"], tuned["tuned"] = val[0], val[1]
types = sorted(tuned["type"].unique())
fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for ax, task in zip(axes, CORE_TASKS, strict=True):
    t = tuned[tuned["task"] == task].set_index("type").reindex(types)
    x = np.arange(len(types))
    ax.bar(x - 0.2, t["untuned"], 0.4, color="#bbbbbb", label="untuned")
    ax.bar(x + 0.2, t["tuned"], 0.4, color="#1f77b4", label="tuned (50 trials)")
    ax.set_xticks(x, types, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{task}, 15 s")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("validation macro F1")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=8)
fig.suptitle("Every Optuna search of E07 and E17: untuned vs tuned (validation)")
fig.tight_layout()
fig.savefig(FIG / "tuned_vs_untuned_E07_E17.png", dpi=120)""",
    ),
    ("markdown", "F04_E17"),
    ("markdown", "## 5. Final choice per task, the comparison set and the old rule's picks"),
    (
        "code",
        """display(R.final_choice(s))
comp = R.comparison_set(s)
display(comp)
display(R.old_rule_choice(s))
by_id = s.set_index("run_id")
order = ["decision tree", "KNN", "logistic regression", "SVM-RBF", "Random Forest", "XGBoost",
         "LightGBM"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
for ax, task in zip(axes, CORE_TASKS, strict=True):
    t = comp[comp["task"] == task].set_index("method type").loc[order]
    cv = by_id.loc[t["run_id"]]
    x = np.arange(len(order))
    ax.bar(x, t["val macro F1"], color="#1f77b4")
    ax.errorbar(x, cv["metrics.cv_f1_macro_mean"], yerr=cv["metrics.cv_f1_macro_std"], fmt="o",
                color="black", ms=3, capsize=2)
    ax.set_xticks(x, order, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{task}, 15 s")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("macro F1\\n(bar: validation; dot: 5-fold CV mean +- std)")
fig.suptitle("Comparison set: best 15 s variant of each method type (scored on test in Phase 4)")
fig.tight_layout()
fig.savefig(FIG / "comparison_set_val_15s.png", dpi=120)""",
    ),
    ("markdown", "F04_FINAL"),
    ("markdown", "## 6. E09: feature-group ablation on the headline models"),
    (
        "code",
        """t9 = R.e09_table(h, s)
display(t9)
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
sets = [x for x in t9["feature set"].unique() if x != "all"]
for ax, task in zip(axes, CORE_TASKS, strict=True):
    t = t9[t9["task"] == task].set_index("feature set").reindex(sets)
    ch = t["change vs all"].astype(float)
    ax.barh(range(len(sets)), ch, color=["#d62728" if v < 0 else "#2ca02c" for v in ch])
    ax.set_yticks(range(len(sets)), sets, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"{task}: change vs all 23")
    ax.invert_yaxis()
axes[0].set_xlabel("validation macro F1 change")
fig.tight_layout()
fig.savefig(FIG / "e09_ablation.png", dpi=120)
verdict_table(R.e09_verdicts(t9), ("E09",))""",
    ),
    ("markdown", "F04_E09"),
    ("markdown", "## 7. E13: learning curve of the headline models"),
    (
        "code",
        """t13 = R.e13_table(h, s)
display(t13)
fig, ax = plt.subplots(figsize=(8, 4.2))
for task in CORE_TASKS:
    t = t13[t13["task"] == task]
    ax.errorbar(t["training rows"], t["val macro F1 mean"],
                yerr=pd.to_numeric(t["val macro F1 std"], errors="coerce").fillna(0),
                marker="o", capsize=3, label=task)
ax.set_xscale("log")
ax.set_xlabel("training rows (log; 10%-100% of the training set)")
ax.set_ylabel("validation macro F1 (mean +- std, 5 seeds)")
ax.set_title("E13 learning curve, 15 s (100% = the headline run)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "e13_learning_curve.png", dpi=120)
verdict_table(R.e13_verdicts(t13), ("E13",))""",
    ),
    ("markdown", "F04_E13"),
    ("markdown", "## 8. E14: split-seed stability of the headline models"),
    (
        "code",
        """t14 = R.e14_table(h, s)
display(t14)
e14 = R._analysis_runs(h, "E14")
fig, ax = plt.subplots(figsize=(8, 4.2))
data = [e14[e14["params.task"] == t]["metrics.val_f1_macro"] for t in CORE_TASKS]
ax.boxplot(data, tick_labels=CORE_TASKS)
off = t14.set_index("task").loc[CORE_TASKS, "official split (seed 42) val macro F1"]
ax.scatter(range(1, 5), off, color="#d62728", marker="x", s=60, zorder=3,
           label="official split (headline run)")
ax.set_ylabel("validation macro F1")
ax.set_title("E14: 10 resplits of train + validation (seeds 42-51), 15 s")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "e14_seed_stability.png", dpi=120)
verdict_table(R.e14_verdicts(t14), ("E14",))""",
    ),
    (
        "code",
        """widths = {"width_ratios": [1, 2, 2, 4]}
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), gridspec_kw=widths)
for ax, task in zip(axes, CORE_TASKS, strict=True):
    cm = pd.read_csv(ROOT / "reports" / "e14_mean_confusion" / f"{task}.csv", index_col=0)
    norm = cm.div(cm.sum(axis=1), axis=0)
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm.index)), cm.index, fontsize=7)
    ax.set_title(f"{task}: mean over 10 resplits,\\nrow-normalised", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "e14_mean_confusion.png", dpi=110)""",
    ),
    ("markdown", "F04_E14"),
    ("markdown", "## 9. E16: Isnan et al. (2026) on the Kaggle CSV, then corrected"),
    (
        "code",
        """display(R.e16_table(h))
size = R.e16_size_table(h)
display(size)
e16 = h[h["tags.experiment_id"] == "E16"].set_index("run_name")
acc = e16["metrics.kaggle_test_accuracy"]
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(4)
ax.bar(x - 0.2, [acc[f"E16-per_timeout-rf-{t}s"] for t in TIMEOUTS], 0.4, color="#2ca02c",
       label="(b) their forest, one timeout")
ax.bar(x + 0.2, [acc[f"E16-per_timeout-xgb-{t}s"] for t in TIMEOUTS], 0.4, color="#1f77b4",
       label="(c) our tuned XGBoost, one timeout")
ax.axhline(acc["E16-isnan-rf"], color="#d62728", ls="--",
           label=f"(a) their setup, timeouts mixed ({acc['E16-isnan-rf']:.4f})")
ax.axhline(R.ISNAN["accuracy"], color="black", ls=":", lw=2,
           label=f"Isnan et al. as published ({R.ISNAN['accuracy']:.4f})")
ax.set_xticks(x, [f"{t} s" for t in TIMEOUTS])
ax.set_ylim(0.7, 0.95)
ax.set_ylabel("test accuracy (Kaggle 80/20 test rows)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "e16_isnan.png", dpi=120)
verdict_table(R.e16_verdicts(h), ("E16",))""",
    ),
    ("markdown", "F04_E16"),
    ("markdown", "## 10. Threshold checks (pre-stated in E07–E17)"),
    ("code", """verdict_table(R.phase3_verdicts(s, RUNNER), ("E07", "E08", "E10", "E17"))"""),
    ("markdown", "F04_VERDICTS"),
]

FINDINGS: dict[str, str] = {
    "F03_SCREEN": """**Findings (screen at 15 s, validation, data v1).**

- **E04:** the signed log with StandardScaler lifts KNN above the E03 KNN in all four tasks
  (+0.0801 to +0.1040) and beats the RobustScaler variant in all four (by 0.0118–0.0224). It
  also passes the E02 tree everywhere (+0.0059 to +0.0376), so it becomes the current best in
  every task. Only A2-NonVPN (+0.0376) clears the 0.03 winner margin; A2-VPN misses it by 0.0004
  (+0.0296).
- **E05:** the 300-tree Random Forest is the current best in all four tasks (+0.0135 to +0.0503
  over the E04 KNN); only B (+0.0503) is a winner.
- **E06:** default XGBoost is the current best in all four tasks (+0.0067 to +0.0156 over the
  forest); LightGBM is 0.0032–0.0102 below XGBoost in every task and below the forest in A1
  (−0.0022). No winner.
- Current best after E06: XGBoost with default settings everywhere (A1 0.9271, A2-VPN 0.8758,
  A2-NonVPN 0.9015, B 0.8303).""",
    "F03_COMPARE": """**Findings.**

- From the E04 KNN to XGBoost the improved models lie close together: their spread per task is
  0.034 (A1: 0.8929–0.9271), 0.039 (A2-VPN), 0.048 (A2-NonVPN) and 0.088 (B), against
  0.31–0.46 between the E02 tree and the weakest E03 baseline.
- Cross-validation (dots) and validation (bars) differ by −0.021 to +0.024 (CV − validation).
  In A2-NonVPN CV is higher than validation for every E04–E06 model (+0.011 to +0.024); in the
  other tasks the difference is between −0.021 and +0.015. E14 later shows why: the official
  A2-NonVPN validation rows are harder than typical.
- Winner reruns: the signed-log KNN in A2-NonVPN scores 0.8523 / 0.8590 / 0.8491 at 30 / 60 /
  120 s and stays the best run at each timeout (by 0.0240 / 0.0344 / 0.0098); the forest in B
  scores 0.8114 / 0.7771 / 0.7835 and leads the tree by 0.0638 / 0.0713 / 0.0677. Both score
  less at 60 and 120 s than at 15 s. The class mix differs between timeouts and the 60 s files
  hold differently shaped VOIP and FT flows (E00 addendum), so this is not read as a timeout
  effect.""",
    "F03_CLASSES": """**Findings (values rounded to 2 decimals in the chart).**

- VoIP is the easiest traffic type for every E04–E06 model (F1 0.96–1.00 for VOIP and
  VPN-VOIP).
- The hardest classes are the mail and chat classes and non-VPN streaming: in B, MAIL
  0.56–0.72, VPN-MAIL 0.58–0.66 and VPN-CHAT 0.56–0.72; in A2-VPN, VPN-CHAT 0.63–0.72; in
  A2-NonVPN, STREAMING 0.73–0.82.
- The gains over the E02 tree come mostly from these weak classes, e.g. XGBoost in B: CHAT
  0.70 → 0.80, VPN-CHAT 0.58 → 0.72, MAIL 0.61 → 0.72; in A2-NonVPN: CHAT 0.76 → 0.88, FT
  0.76 → 0.88.
- The signed log repairs the classes the E03 KNN got worst: A2-NonVPN CHAT 0.63 → 0.83 and
  STREAMING 0.63 → 0.74, B BROWSING 0.64 → 0.83.""",
    "F03_COST": """**Findings (15 s runs; one thread per model, the official machine).**

- Training: the KNN only stores the rows (0.014–0.033 s); XGBoost takes 0.256–2.340 s,
  LightGBM 0.191–2.015 s and the forest 2.038–6.059 s.
- Prediction per 1,000 flows: XGBoost 1.13–11.76 ms, KNN 7.75–11.50 ms, the forest
  21.91–27.28 ms, LightGBM 5.12–98.28 ms. LightGBM grows one tree per class per round (100 trees
  in A1, 1,400 in B), so its prediction time grows most with the number of classes.
- Model files (compressed): the forest 5.31–22.44 MiB (171.86 MiB uncompressed for B), XGBoost
  0.12–1.06 MiB, LightGBM 0.14–2.01 MiB. The forest's fit needs 33.1–187.9 MiB in a fresh process
  (child RSS), which tracemalloc does not see (1.4–2.5 MiB).
- So default XGBoost is the best E04–E06 model and also one of the cheapest to store and to run.""",
    "F03_VERDICTS": """**Summary of the pre-stated thresholds:** all six are supported
(E04a, E04b, E05a, E05b, E06a, E06b). The current best after E06 is default XGBoost in every
task.""",
    "F04_E07": """**Findings (E07).**

- 12 searches (per task the best 2 of Random Forest / XGBoost / LightGBM, plus the signed-log
  KNN): 50 trials and 250 fits each, 600 trials and 3,000 fits in all. In every search the
  retrained model's CV equals its best trial's exactly.
- Tuning raised the 5-fold CV score in 12 of 12 searches but the validation score in only 9 of
  12 (the A1 forest −0.0022, A1 XGBoost −0.0011, A2-VPN XGBoost −0.0097): the best of 50 CV
  scores is an optimistic estimate.
- The largest validation gains are for KNN (B +0.0361, A2-NonVPN +0.0260); for the ensembles
  the gains are at most +0.0151.
- Best trials came anywhere from trial 11 to trial 49; 4 of 12 searches found their best after
  trial 40, so more trials might still help a little. Two picks sit at a range edge (A1 XGBoost
  max_depth 12, A2-NonVPN KNN k = 1); the ranges, fixed before running, were not widened.
- Current best after E07: A1 still E06 XGBoost 0.9271; A2-VPN E07 LightGBM 0.8768; A2-NonVPN
  E07 LightGBM 0.9028; B E07 XGBoost 0.8428 (+0.0125). No winner.""",
    "F04_E08": """**Findings (E08).**

- Class weights hurt in A1 (−0.0034) and B (−0.0064), so the ladder stops there.
- A2-VPN: class weights +0.0062; random oversampling gives the same score (0.00001 lower), so
  the ladder stops.
- A2-NonVPN: class weights +0.0022, random oversampling +0.0047, SMOTE +0.0019 (below
  oversampling), so it stops after SMOTE.
- The weakest class gains where balancing helps (A2-VPN VPN-CHAT 0.7343 → 0.7463; A2-NonVPN
  STREAMING 0.8095 → 0.8372) and loses in B (VPN-MAIL 0.6496 → 0.6290).
- Every change is within 0.01 (no winner). Current best after E08: A2-VPN LightGBM with class
  weights 0.8830, A2-NonVPN LightGBM with oversampling 0.9075.""",
    "F04_E10": """**Findings (E10).**

- Two steps (VPN / non-VPN first, then the type) score 0.8318 against 0.8428 for the direct
  14-class XGBoost (−0.0110), with more errors on the same 2,711 validation rows (345 vs 327).
- The VPN / non-VPN step makes more group errors than the direct model (191 vs 180) and is the
  source of 55.36% of the two-step's errors; the direct model also makes 55.05% of its errors
  across that line. Separating VPN from non-VPN is the hard part either way.
- Task B therefore stays a direct 14-class model.""",
    "F04_E17": """**Findings (E17).**

- 16 searches, 800 trials, 4,000 fits; the retrained model's CV equals its best trial's in
  16 of 16.
- SVM-RBF gains most: +0.3243 to +0.5277 on validation once tuned. Every logistic-regression and
  SVM search chose the signed log (8 of 8). The tuned SVM reaches 0.9259 (A1), 0.8707 (A2-VPN),
  0.9058 (A2-NonVPN) and 0.8242 (B), 0.0017–0.0186 below each task's best.
- Logistic regression gains +0.0670 to +0.3515 but stays far behind (0.5862–0.8002).
- Tuning barely moves the tree (0.0000 to +0.0044) and the forest (−0.0111 to +0.0025).
- LightGBM in A1 gains +0.0127 to 0.9308, the new A1 best (+0.0037 over XGBoost). No winner.""",
    "F04_FINAL": """**Findings (final choice, decided 28 Oct after E10: the highest validation macro
F1).**

- Headline models: A1 E17 LightGBM 0.9308 (next E06 XGBoost 0.9271); A2-VPN E08 LightGBM with
  class weights 0.8830 (next the same model with oversampling, 0.00001 lower); A2-NonVPN E08
  LightGBM with oversampling 0.9075 (next E17 SVM 0.9058); B E07 XGBoost 0.8428 (next E08 XGBoost
  with class weights 0.8364). All four are 15 s models; their files are 4.31–25.12 MiB, none over
  50 MB.
- In the comparison set a boosted model is top in every task and the tuned SVM close behind;
  logistic regression is last in every task.
- The superseded tie rule would have picked KNN in A1, A2-VPN and A2-NonVPN and the SVM in B,
  0.0146–0.0286 below the headline models. Those picks are scored on the test set in Phase 4
  too.""",
    "F04_E09": """**Findings (E09, analysis only).**

- No single group comes close to all 23 features: the best single group is 0.0604–0.1385 below
  (flow inter-arrival in A1 and A2-VPN, rates + duration in A2-NonVPN and B). Active or idle
  alone scores only 0.21–0.55.
- Removing rates + duration costs most (−0.0122 A1, −0.0395 A2-VPN, −0.0301 A2-NonVPN,
  −0.0549 B). Removing the active or the idle group changes the score by at most ±0.0051.
- The timing-only set (no byte rate) loses 0.0148 (A1) to 0.0441 (B): packet sizes help.
- Verdicts: E09a supported, E09b and E09c not supported. No feature set scores 0.03 or more above
  all 23, so the stop-and-ask rule did not trigger.""",
    "F04_E13": """**Findings (E13, analysis only).**

- Every task gains steadily with more training rows: 10% → 100% +0.0584 (A1), +0.1037 (A2-VPN),
  +0.0899 (A2-NonVPN), +0.1459 (B).
- The last 20% adds +0.0038, +0.0026 and +0.0070 in A1, A2-VPN and A2-NonVPN, but +0.0131 in B:
  the 14-class task would still gain from more data.
- Small samples vary more: the std over seeds is 0.0052–0.0097 at 10% against 0.0029–0.0036 at
  80%. All three pre-stated checks are supported.""",
    "F04_E14": """**Findings (E14, analysis only).**

- Over 10 resplits of train + validation the validation macro F1 varies with a std of
  0.0042–0.0114, and CV agrees with validation on average (CV − validation −0.0032 to −0.0119).
- The official split lies inside the resplit range in A1, A2-VPN and B, but in A2-NonVPN its
  0.9075 is below all 10 resplits (0.9257–0.9468): the official A2-NonVPN validation rows are
  unusually hard (mostly MAIL and P2P, E14.md). E14c is therefore not supported.
- The averaged confusion matrices show where the errors are: A1 Non-VPN → VPN 95.7 and
  VPN → Non-VPN 83.1 flows per resplit; B BROWSING → VPN-BROWSING 30.3 and back 23.3; A2-VPN
  between VPN-FT, VPN-CHAT and VPN-BROWSING (16.0–17.4); A2-NonVPN CHAT → BROWSING 10.2.""",
    "F04_E16": """**Findings (E16; Kaggle rows, not comparable with our validation scores).**

- (a) Isnan et al.'s setup is reproduced: accuracy 0.8796 and weighted F1 0.8792 against their
  0.8801 and 0.8797, on the same 32,789 / 8,198 split. Their accuracy hides weaker small classes
  (macro F1 0.8407).
- (b) With the timeouts kept apart, the same method scores 0.8206–0.8658 (mean 0.8430).
- (c) Our tuned task-B XGBoost beats their forest at every timeout (mean 0.8647; 0.8924 at
  15 s).
- (d) With the mixed-timeout training rows cut to the per-timeout sizes, accuracy is 0.8243–0.8504
  (mean 0.8395, below (b)). So the gap between (a) and (b) comes from training-set size: stacking
  the four timeout files gives 2.3–4.3 times as many training rows. Mixing the timeouts does not
  raise the score by itself. All four checks are supported.""",
    "F04_VERDICTS": """**Summary of the pre-stated thresholds:** E07 (a) supported, (b) not
supported, (c) supported; E08 (a) not supported, (b) and (c) supported; E10 (a) and (b) not
supported, (c) supported; E17 (a), (b) and (c) supported. E09, E13, E14 and E16 are
summarised in their own sections above. E11 does not run (no session information in the
ARFF files, E00).""",
}


def build(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"}}  # fmt: skip
    for kind, src in cells:
        src = FINDINGS.get(src, src)
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "markdown"
                        else nbformat.v4.new_code_cell(src))  # fmt: skip
    for i, c in enumerate(nb.cells):
        c["id"] = str(i)  # the ids nbstripout gives on commit
    return nb


def main() -> None:
    paths = []
    for name, cells in (("03_models", NB03), ("04_tuning", NB04)):
        paths.append(ROOT / "notebooks" / f"{name}.ipynb")
        nbformat.write(build(cells), paths[-1])
        print("wrote", paths[-1].relative_to(ROOT))
    # the repository's code style (ruff) applies to notebook cells too
    subprocess.run(["ruff", "format", "--quiet", *paths], check=True)
    subprocess.run(["ruff", "check", "--quiet", "--fix", *paths], check=True)


if __name__ == "__main__":
    main()
