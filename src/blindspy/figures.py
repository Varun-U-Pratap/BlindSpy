"""Report graphs drawn by scripts/export_history.py from reports/history.csv (plan: Metrics and
report graphs). Graphs drawn elsewhere (notebooks 01-05, results_tables.py, explain_final.py) are
listed in reports/figures/INDEX.md, which `write_index` generates."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from blindspy import results as R  # noqa: E402
from blindspy.data import CORE_TASKS, ROOT  # noqa: E402

FIG = ROOT / "reports" / "figures"
TYPE_COLORS = {"decision tree": "#8c564b", "KNN": "#ff7f0e", "logistic regression": "#7f7f7f",
               "SVM-RBF": "#9467bd", "Random Forest": "#2ca02c", "XGBoost": "#1f77b4",
               "LightGBM": "#17becf", "majority": "#000000"}  # fmt: skip


def progress_table(s: pd.DataFrame) -> pd.DataFrame:
    """Per task and model-selection experiment (in order): the best 15 s validation macro F1 of
    that experiment and the best so far. Experiments without a run for a task carry the best so
    far forward."""
    s15 = s[s["params.timeout"] == 15]
    rows = []
    for task in CORE_TASKS:
        best = -np.inf
        for exp in R.SELECTION_EXPS:
            g = s15[(s15["params.task"] == task) & (s15["tags.experiment_id"] == exp)]
            here = g["metrics.val_f1_macro"].max() if len(g) else np.nan
            best = max(best, here) if len(g) else best
            rows.append({"task": task, "experiment": exp, "best in experiment": here,
                         "best so far": best})  # fmt: skip
    return pd.DataFrame(rows)


def progress_chart(h: pd.DataFrame, path=FIG / "progress_chart.png") -> pd.DataFrame:
    s = R.selection_runs(h)
    t = progress_table(s)
    test = R.e12_runs(h)
    test = test[test["params.role"] == "headline"].set_index("params.task")
    versions = sorted(set(s["tags.data_version"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(R.SELECTION_EXPS))
    for ax, task in zip(axes.flat, CORE_TASKS, strict=True):
        g = t[t["task"] == task]
        ax.plot(x, g["best so far"], "-o", color="#1f77b4", label="best so far (validation)")
        ax.plot(x, g["best in experiment"], "x", color="#7f7f7f", label="best of the experiment")
        base = g.set_index("experiment").loc["E02", "best in experiment"]
        ax.axhline(base, ls=":", color="#8c564b", label="baseline (E02 tree)")
        ax.plot(len(x), test.loc[task, "metrics.test_f1_macro"], "*", ms=13, color="#d62728",
                label="headline on test (E12)")  # fmt: skip
        # the E01 majority floor is far below the rest: start the axis below the E02 baseline
        low = base - 0.1
        ax.set_ylim(low, 1.0)
        floor = g.set_index("experiment").loc["E01", "best in experiment"]
        ax.annotate(f"E01 majority {floor:.3f}\n(below the axis)", (0, low), (3.6, low + 0.012),
                    fontsize=7)  # fmt: skip
        ax.axvline(-0.5, color="black", lw=1)
        ax.text(-0.4, 0.985, f"data {', '.join(versions)}", fontsize=7, va="top")
        ax.set_xticks([*x, len(x)], [*R.SELECTION_EXPS, "E12"], rotation=45, fontsize=8)
        ax.set_title(f"{task}: 15 s validation macro F1 by experiment", fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=len(labels), loc="lower center")
    fig.suptitle("Progress: best model so far per task (one data version, v1, for every "
                 "experiment;\nE16's Kaggle rows are a separate data version and not shown)",
                 fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return t


def accuracy_vs_cost(h: pd.DataFrame, path=FIG / "accuracy_vs_cost.png") -> None:
    s = R.selection_runs(h)
    s = s.assign(type=s["params.model"].map(R.METHOD_TYPE).fillna("majority"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, task in zip(axes.flat, CORE_TASKS, strict=True):
        g = s[s["params.task"] == task]
        for typ, d in g.groupby("type"):
            ax.scatter(d["metrics.train_time_s"], d["metrics.val_f1_macro"], s=18, alpha=0.75,
                       color=TYPE_COLORS[typ], label=typ)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("training time, s (log scale)")
        ax.set_title(f"{task}: {len(g)} model-selection runs (all timeouts)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("validation macro F1")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle("Accuracy vs cost: one dot per run", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def timeout_effect(h: pd.DataFrame, path=FIG / "timeout_effect.png") -> None:
    """Phase 2 baselines at the 4 timeouts (the only models run at every timeout), with each
    file's class mix beside them; the 60 s files differ for VOIP and FT (E00 addendum)."""
    s = R.selection_runs(h)
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())["tasks"]
    models = {"E02-tree": "decision tree", "E03-knn": "KNN", "E03-svm_rbf": "SVM-RBF",
              "E03-logreg": "logistic regression"}  # fmt: skip
    timeouts = [15, 30, 60, 120]
    fig, axes = plt.subplots(4, 3, figsize=(15, 17), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for j, task in enumerate(CORE_TASKS):
        g = s[(s["params.task"] == task) & s["label"].isin(models)]
        for i, (metric, name) in enumerate([("metrics.val_f1_macro", "macro F1"),
                                            ("metrics.val_balanced_accuracy",
                                             "balanced accuracy")]):  # fmt: skip
            ax = axes[j, i]
            for lab, typ in models.items():
                d = g[g["label"] == lab].set_index("params.timeout")[metric].reindex(timeouts)
                ax.plot(range(4), d.to_numpy(), "-o", color=TYPE_COLORS[typ], label=typ)
            ax.axvspan(1.6, 2.4, color="#f9eeda")
            ax.set_xticks(range(4), [f"{t} s" for t in timeouts])
            ax.set_title(f"{task}: validation {name}", fontsize=10)
        mix = pd.DataFrame({t: manifest[f"{task}_{t}s"]["final_class_counts"] for t in timeouts})
        mix = (mix / mix.sum()).T
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[: mix.shape[1]]
        mix.plot.bar(stacked=True, ax=axes[j, 2], color=colors, width=0.8, legend=False)
        axes[j, 2].set_xticklabels([f"{t} s" for t in timeouts], rotation=0)
        axes[j, 2].set_title(f"{task}: class mix per timeout file", fontsize=10)
        axes[j, 2].legend(fontsize=6, ncol=1 if mix.shape[1] < 10 else 2, loc="center left",
                          bbox_to_anchor=(1.0, 0.5))  # fmt: skip
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Timeout effect (Phase 2 baselines). Shaded: the 60 s files, whose VOIP and FT "
                 "flows look different (E00 addendum),\nso a 60 s break in the trend is not read "
                 "as a timeout effect; class mixes differ too, so the comparison is confounded.",
                 fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def final_confusion(h: pd.DataFrame, runner: str, path=FIG / "final_confusion_matrices.png"):
    """Row-normalised confusion matrices of the 4 headline models on the test rows (E12)."""
    e = R.e12_runs(h)
    e = e[e["params.role"] == "headline"].set_index("params.task")
    fig, axes = plt.subplots(2, 2, figsize=(17, 15))
    for ax, task in zip(axes.flat, CORE_TASKS, strict=True):
        p = R.final_predictions(e.loc[task, "run_id"], runner)
        labels = sorted(set(p["true"].astype(str)))
        cm = confusion_matrix(p["true"].astype(str), p["predicted"].astype(str), labels=labels)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cmn):
            if v >= 0.005:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6 if len(labels) > 7
                        else 9, color="white" if v > 0.6 else "black")  # fmt: skip
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{task}: {e.loc[task, 'run_name']}, test rows ({len(p)}), share of each "
                     "true class", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Every graph the plan lists under "Graphs for the report": file(s) and the code that draws it.
GRAPHS = [
    ("Progress chart (best macro F1 per task over E01-En, baseline and data version marked)",
     ["progress_chart.png"], "scripts/export_history.py (blindspy.figures)"),
    ("Paper vs ours (per-class precision and recall, matched 10-fold protocol; A1 only: the paper "
     "gives A2/B values only in its text, tables in notebook 02)", ["paper_vs_ours_A1.png"],
     "notebooks/02_baselines.ipynb"),
    ("Model comparison with cross-validation error bars",
     ["baseline_model_comparison.png", "phase3_model_comparison_E04-E06.png",
      "comparison_set_val_15s.png", "e12_comparison_test.png"],
     "notebooks 02, 03, 04; scripts/results_tables.py --phase 4 (test set, 95% intervals)"),
    ("Accuracy vs cost (macro F1 against training time, one dot per run)",
     ["accuracy_vs_cost.png", "cost_E04-E06.png"], "scripts/export_history.py; notebook 03"),
    ("Timeout effect with class mix", ["timeout_effect.png", "timeout_class_mix.png"],
     "scripts/export_history.py; notebooks/01_eda.ipynb"),
    ("Feature-group ablation (E09)", ["e09_ablation.png", "shap_groups_vs_e09.png"],
     "notebooks/04_tuning.ipynb; scripts/explain_final.py"),
    ("Confusion matrices of the final models, normalised", ["final_confusion_matrices.png"],
     "scripts/export_history.py (blindspy.figures)"),
    ("SHAP summary, feature importance and worked examples",
     ["shap_summary.png", "shap_beeswarm_A1.png", "shap_examples_A1.png",
      "shap_examples_A2-VPN.png", "shap_examples_A2-NonVPN.png", "shap_examples_B.png"],
     "scripts/explain_final.py (notebook 05)"),
    ("Class and feature distributions, data funnel",
     ["class_counts_v1.png", "feature_boxes_A1_15s.png", "data_funnel_v1.png",
      "data_funnel_by_class_v1.png"], "notebooks/01_eda.ipynb"),
    ("Optuna tuning history", ["optuna_histories_E07.png", "optuna_histories_E17.png"],
     "notebooks/04_tuning.ipynb"),
    ("Learning curve (E13)", ["e13_learning_curve.png"], "notebooks/04_tuning.ipynb"),
    ("Seed-stability box plot (E14)", ["e14_seed_stability.png", "e14_mean_confusion.png"],
     "notebooks/04_tuning.ipynb"),
    ("Reliability diagram per final model (E15)", ["e15_reliability.png"],
     "scripts/results_tables.py --phase 4"),
    ("Top-5 confusion-pair table per task", ["(table) reports/phase4_tables.md, section "
     "'E12 top-5 confusion pairs'"], "scripts/results_tables.py --phase 4"),
]  # fmt: skip


def write_index(path=FIG / "INDEX.md") -> list[str]:
    """reports/figures/INDEX.md; returns the listed PNG files that do not exist."""
    lines = ["# Report graphs", "",
             "Generated by `scripts/export_history.py`. Every graph the plan lists under "
             "\"Graphs for the report\", the file(s) in this folder and the code that draws it.",
             "", "| graph | file(s) | drawn by |", "|---|---|---|"]  # fmt: skip
    missing = []
    for name, files, by in GRAPHS:
        missing += [f for f in files if f.endswith(".png") and not (FIG / f).exists()]
        lines.append(f"| {name} | {', '.join(files)} | {by} |")
    path.write_text("\n".join(lines) + "\n")
    return missing


def draw_all(h: pd.DataFrame, runner: str) -> list[str]:
    progress_chart(h)
    accuracy_vs_cost(h)
    timeout_effect(h)
    if (R.e12_runs(h)["params.role"] == "headline").sum() == len(CORE_TASKS):
        final_confusion(h, runner)
    return write_index()
