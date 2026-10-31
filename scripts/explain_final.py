"""SHAP explanations of the 4 final models (models/final) on their test rows, for notebook 05
and the report. No score is computed and nothing is chosen from these rows: the test scores are
E12's, from its saved predictions.

Writes reports/shap_importance.csv (mean |SHAP| per feature), reports/shap_groups.csv (per E09
feature group, next to E09's validation changes), reports/shap_class_top.csv (top 3 per
class), reports/shap_examples.csv (4 worked flows per
task, picked by the rule in blindspy.explain.pick_examples) and the figures
reports/figures/shap_summary.png, shap_beeswarm_A1.png, shap_groups_vs_e09.png and
shap_examples_<task>.png.

Usage: uv run python scripts/explain_final.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from blindspy import calibration, data  # noqa: E402
from blindspy import explain as E  # noqa: E402
from blindspy import results as R  # noqa: E402
from blindspy.features import signed_log  # noqa: E402
from blindspy.predict import load_final  # noqa: E402

OUT = data.ROOT / "reports"
FIG = OUT / "figures"


def main() -> None:
    h = R.load_history()
    e09 = R.e09_table(h, R.selection_runs(h)).set_index(["task", "feature set"])["change vs all"]
    imps, groups, examples, per_class, class_top = [], [], [], {}, []
    fig_ex = {}
    for task in data.CORE_TASKS:
        model, meta = load_final(task)
        test = data.load_task(task, 15, data_version="v1")[2]
        X, y = test.drop(columns="label"), test["label"].astype(str).to_numpy()
        label, conf, _ = calibration.predict_with_confidence(model, X)
        v, classes = E.shap_values(model, X)
        imp = E.importance(v, X.columns)
        # A1 only (binary): direction of each feature's effect, as the rank correlation of its
        # value with its SHAP value towards VPN
        rho = ({f: spearmanr(X[f], v[:, j, classes.index("VPN")]).statistic
                for j, f in enumerate(X.columns)} if task == "A1" else {})  # fmt: skip
        imps += [{"task": task, "rank": i + 1, "feature": f, "mean |SHAP|": round(x, 4),
                  "rank corr. value vs SHAP towards VPN (A1)":
                      round(float(rho[f]), 4) if f in rho else None}
                 for i, (f, x) in enumerate(imp.items())]  # fmt: skip
        g = E.group_table(imp)
        for _, r in g.iterrows():
            groups.append({"task": task, "group": r["group"], "features": r["features"],
                           "mean |SHAP| (sum)": round(r["mean |SHAP| (sum)"], 4),
                           "SHAP share": round(r["share"], 4),
                           "E09 drop this group (val change)": e09[(task, "no_" + r["group"])],
                           "E09 this group alone (val change)":
                               e09[(task, "only_" + r["group"])]})  # fmt: skip
        f1 = pd.Series(f1_score(y, label, labels=classes, average=None), index=classes)
        weakest = f1.idxmin()
        per_class[task] = pd.DataFrame(np.abs(v).mean(axis=0), index=X.columns, columns=classes)
        for c in classes[1:] if len(classes) == 2 else classes:  # binary: one class says it all
            t3 = per_class[task][c].sort_values(ascending=False, kind="mergesort").head(3)
            class_top.append({"task": task, "class": c, "test F1": round(float(f1[c]), 4),
                              **{f"feature {i + 1}": f"{n} ({x:.3f})"
                                 for i, (n, x) in enumerate(t3.items())}})  # fmt: skip
        picks = E.pick_examples(y, label, conf, weakest)
        fig_ex[task] = []
        for kind, pos in picks.items():
            rows = E.example_rows(v, classes, X, pos, label[pos])
            head = {"task": task, "example": kind, "raw_row": int(test.index[pos]), "true": y[pos],
                    "predicted": label[pos], "confidence": round(float(conf[pos]), 4),
                    "weakest class": weakest}  # fmt: skip
            examples += [{**head, **r} for r in rows.to_dict("records")]
            fig_ex[task].append((head, rows))
        print(f"{task}: {len(X)} test rows, weakest class {weakest} (F1 {f1.min():.4f}), "
              f"top feature {imp.index[0]}")  # fmt: skip
        if task == "A1":
            beeswarm(v[..., classes.index("VPN")], X)
    pd.DataFrame(imps).to_csv(OUT / "shap_importance.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "shap_groups.csv", index=False)
    pd.DataFrame(class_top).to_csv(OUT / "shap_class_top.csv", index=False)
    ex = pd.DataFrame(examples)
    ex["value"] = ex["value"].round(4)
    ex["SHAP"] = ex["SHAP"].round(4)
    ex.to_csv(OUT / "shap_examples.csv", index=False)
    summary(per_class)
    groups_vs_e09(pd.DataFrame(groups))
    for task, items in fig_ex.items():
        example_figure(task, items)
    print(
        "wrote reports/shap_importance.csv, shap_groups.csv, shap_class_top.csv, "
        "shap_examples.csv and figures"
    )


def summary(per_class: dict) -> None:
    """Mean |SHAP| per feature, stacked by class (top 12 features per task)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (task, t) in zip(axes.flat, per_class.items(), strict=True):
        if t.shape[1] == 2:  # binary: both columns are the same; show one
            t = t[["VPN"]]
        top = t.mean(axis=1).sort_values(ascending=False).index[:12][::-1]
        t = t.loc[top] / (1 if t.shape[1] == 1 else t.shape[1])
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[: t.shape[1]]
        t.plot.barh(stacked=True, ax=ax, color=colors, width=0.8, legend=t.shape[1] > 1)
        ax.set_title(f"{task}: mean |SHAP| per feature (test rows)")
        ax.set_xlabel("mean |SHAP|, averaged over classes (log-odds units)")
        if t.shape[1] > 1:
            ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "shap_summary.png", dpi=120)
    plt.close(fig)


def beeswarm(values: np.ndarray, X: pd.DataFrame) -> None:
    import shap

    # colour = signed log of the value (E04), since raw values span orders of magnitude; the
    # SHAP values themselves are unchanged
    np.random.seed(42)  # shap places the dots with numpy's global random generator
    shap.summary_plot(values, signed_log(X), show=False, max_display=15, plot_size=(9, 7))
    plt.title("A1: SHAP towards VPN per flow (test rows; colour = signed log of the value)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_beeswarm_A1.png", dpi=120)
    plt.close("all")


def groups_vs_e09(g: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in [(axes[0], "SHAP share", "Share of mean |SHAP| per feature group "
                            "(test rows, final models)"),
                           (axes[1], "E09 drop this group (val change)", "E09: validation macro "
                            "F1 change when the group is dropped")]:  # fmt: skip
        g.pivot(index="group", columns="task", values=col)[data.CORE_TASKS].plot.bar(ax=ax)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", lw=0.8)
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / "shap_groups_vs_e09.png", dpi=120)
    plt.close(fig)


def example_figure(task: str, items: list) -> None:
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 3.6))
    for ax, (head, rows) in zip(np.atleast_1d(axes), items, strict=True):
        r = rows[::-1]
        ax.barh(
            [f"{f} = {v:.4g}" for f, v in zip(r["feature"], r["value"], strict=True)],
            r["SHAP"],
            color=["#1f77b4" if s >= 0 else "#d62728" for s in r["SHAP"]],
        )
        ax.set_title(f"{head['example']}\ntrue {head['true']} -> predicted "
                     f"{head['predicted']} ({head['confidence']:.3f})", fontsize=9)  # fmt: skip
        ax.set_xlabel("SHAP towards the predicted class")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"{task}: worked examples (test rows; top 5 features)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"shap_examples_{task}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
