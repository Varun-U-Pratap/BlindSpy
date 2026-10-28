"""Result tables and threshold checks (on a synthetic history, no MLflow store needed)."""

import json

import pandas as pd
import pytest

from blindspy import results as R
from blindspy.data import CORE_TASKS, TIMEOUTS


def fake_majority_runs(shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for task in CORE_TASKS:
        for t in TIMEOUTS:
            counts = {"A": 60, "B": 30, "C": 10}
            p = 0.6
            rows.append({
                "params.model": "majority", "params.task": task, "params.timeout": t,
                "params.class_counts.val": json.dumps(counts),
                "metrics.val_accuracy": p + shift,
                "metrics.val_f1_macro": 2 * p / ((1 + p) * 3),
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_majority_check_formula():
    ok = R.majority_check(fake_majority_runs())
    assert ok["ok"].all() and ok["expected_macro_f1"].iloc[0] == round(1.2 / 4.8, 4)
    assert not R.majority_check(fake_majority_runs(shift=0.01))["ok"].any()


def test_missing_or_duplicate_run_is_an_error():
    runs = fake_majority_runs()
    with pytest.raises(ValueError, match="found 0"):
        R._cell(runs, "tree", "A1", 15)
    with pytest.raises(ValueError, match="found 2"):
        R._cell(pd.concat([runs, runs]), "majority", "A1", 15)


def test_paper_task_table_places_paper_values():
    rows = []
    for algo in R.BENCH_ALGO:
        for t in TIMEOUTS:
            rows.append({
                "algo": algo, "tags.rows_source": "published", "params.task": "A1",
                "params.timeout": t,
                "params.class_counts.all": json.dumps({"Non-VPN": 5, "VPN": 5}),
                **{f"metrics.bench_{m}.{c}": 0.5 for m in ("precision", "recall")
                   for c in ("VPN", "Non-VPN")},
                "metrics.bench_precision_macro": 0.5, "metrics.bench_recall_macro": 0.5,
            })  # fmt: skip
    paper = pd.DataFrame([
        {"task": "A1", "class": "VPN", "algorithm": "C4.5", "timeout_s": "15",
         "metric": "precision", "value": 0.89, "source": "stated_in_text"},
        {"task": "A1", "class": "VPN", "algorithm": "C4.5", "timeout_s": "15",
         "metric": "precision", "value": 0.89, "source": "read_from_chart"},
        {"task": "A1", "class": "VPN", "algorithm": "KNN", "timeout_s": "30",
         "metric": "recall", "value": 0.86, "source": "read_from_chart"},
    ])  # fmt: skip
    tab = R.paper_task_table(pd.DataFrame(rows), paper, "A1", "published").set_index(
        ["class", "metric"]
    )
    assert tab.loc[("VPN", "precision"), "C4.5 15 s"] == "0.500 (paper 0.890 C+T)"
    assert tab.loc[("VPN", "recall"), "KNN 30 s"] == "0.500 (paper 0.860 C)"
    assert tab.loc[("Non-VPN", "recall"), "KNN 30 s"] == "0.500"


def fake_selection(rows):
    return pd.DataFrame([{
        "run_name": f"{exp}-{model}-{task}-{t}s", "tags.protocol": "holdout",
        "tags.experiment_id": exp, "params.task": task, "params.timeout": t,
        "metrics.val_f1_macro": f1, "metrics.cv_f1_macro_mean": f1,
        "metrics.cv_f1_macro_std": 0.01,
        "tags.trial": "True" if model == "trial" else "",
    } for exp, model, task, t, f1 in rows])  # fmt: skip


def test_model_label_strips_task_and_timeout():
    r = pd.Series({"run_name": "E04-knn_log_standard-A2-NonVPN-15s",
                   "params.task": "A2-NonVPN", "params.timeout": 15})  # fmt: skip
    assert R.model_label(r) == "E04-knn_log_standard"


def test_current_best_winner_flag_and_trials_excluded():
    rows = [("E02", "tree", task, 15, 0.80) for task in CORE_TASKS]
    rows += [
        ("E04", "knn", "A1", 15, 0.83),
        ("E04", "knn", "B", 15, 0.8299),
        ("E04", "knn", "A2-VPN", 15, 0.80),
        ("E05", "trial", "A1", 15, 0.99),
    ]
    s = R.selection_runs(fake_selection(rows))
    assert "E05-trial" not in set(s["label"])
    sc = R.screen(s, "E04").set_index("task")
    assert sc.loc["A1", "winner (gain >= 0.03)"]  # exactly +0.03 counts
    assert not sc.loc["B", "winner (gain >= 0.03)"]
    # a tie keeps the earlier experiment as the current best
    assert R.best_run(s, "A2-VPN", 15)["label"] == "E02-tree"
    assert R.best_run(s, "A1", 15)["label"] == "E04-knn"
    assert R.best_run(s, "A1", 15, upto="E03")["label"] == "E02-tree"


def test_e07_verdicts_compare_with_untuned_and_best_before():
    rows = [
        (e, m, t, 15, 0.70)
        for t in CORE_TASKS
        for e, m in [("E02", "tree"), ("E03", "knn"), ("E04", "knn_log_robust")]
    ]
    rows += [(e, m, t, 15, 0.80) for t in CORE_TASKS
             for e, m in [("E04", "knn_log_standard"), ("E05", "rf"), ("E06", "xgb"),
                          ("E06", "lgbm")]]  # fmt: skip
    rows += [("E07", "xgb", t, 15, 0.81) for t in CORE_TASKS]
    rows += [("E07", "knn_log_standard", t, 15, 0.79) for t in CORE_TASKS]
    s = R.selection_runs(fake_selection(rows))
    v = R.phase3_verdicts(s)
    assert v["E07a"]["searches_above"] == 4 and v["E07a"]["of"] == 8
    assert not v["E07a"]["supported"]
    assert v["E07b"]["tasks_with_gain_ge_0.01"] == 4 and v["E07b"]["supported"]
    assert v["E07c"]["knn_minus_best_tuned_ensemble"] == [-0.02] * 4 and v["E07c"]["supported"]


def test_timeout_table_ignores_later_experiments():
    rows = [("E02", "tree", "B", t, 0.70) for t in TIMEOUTS]
    rows += [("E05", "rf", "B", t, 0.80) for t in TIMEOUTS]
    rows += [("E07", "xgb", "B", 30, 0.95)]
    s = R.selection_runs(fake_selection(rows))
    tab = R.timeout_table(s, "E05-rf", "B").set_index("timeout")
    assert tab.loc["30 s", "best other run"] == "E02-tree"


def test_run_settings_copies_logged_strings_exactly(tmp_path):
    h = pd.DataFrame([
        {"run_id": "a", "run_name": "E07-xgb-B-15s", "status": "FINISHED", "start_time": 1,
         "tags.git_dirty": "False", "tags.git_pushed": "True", "params.timeout": "15",
         "params.model": "xgb", "params.model.learning_rate": "0.1",
         "params.model.n_estimators": "100"},
        {"run_id": "b", "run_name": "E07-xgb-B-15s", "status": "FINISHED", "start_time": 2,
         "tags.git_dirty": "False", "tags.git_pushed": "True", "params.timeout": "15",
         "params.model": "xgb", "params.model.learning_rate": "0.10125778281260306",
         "params.model.n_estimators": "505"},
    ])  # fmt: skip
    h.to_csv(tmp_path / "h.csv", index=False)
    got = R.run_settings("E07-xgb-B-15s", tmp_path / "h.csv")
    assert got == {
        "model": "xgb",
        "params": {"learning_rate": 0.10125778281260306, "n_estimators": 505},
    }


def test_e08_ladder_stops_at_the_first_step_that_does_not_help():
    def run(exp, model, task, f1):
        return {
            "run_name": f"{exp}-{model}-{task}-15s",
            "tags.protocol": "holdout",
            "tags.experiment_id": exp,
            "params.task": task,
            "params.timeout": 15,
            "params.model": model.split("_")[0],
            "metrics.val_f1_macro": f1,
            "tags.worst_class_f1": "C",
            "metrics.val_f1.C": f1 - 0.2,
            "tags.trial": "",
        }

    rows = []
    for task in CORE_TASKS:
        rows.append(run("E06", "xgb", task, 0.80))
    rows += [run("E08", "xgb_class_weight", "A1", 0.79)]  # did not help -> stop
    rows += [
        run("E08", "xgb_class_weight", "A2-VPN", 0.81),  # helped -> next step
        run("E08", "xgb_class_weight", "A2-NonVPN", 0.81),
        run("E08", "xgb_oversample", "A2-NonVPN", 0.805),
    ]  # below class_weight -> stop
    rows += [run("E08", f"xgb_{st}", "B", f) for st, f in
             [("class_weight", 0.81), ("oversample", 0.82), ("smote", 0.83)]]  # fmt: skip
    s = R.selection_runs(pd.DataFrame(rows))
    lad = R.e08_ladder(s, {t: "E06-xgb" for t in CORE_TASKS})
    last = lad.groupby("task").tail(1).set_index("task")
    assert last.loc["A1", "next"] == "stop: this step did not help"
    assert last.loc["A2-VPN", "step"] == "oversample"
    assert last.loc["A2-VPN", "next"] == "run this step"
    assert last.loc["A2-NonVPN", "next"] == "stop: this step did not help"
    assert last.loc["B", "next"] == "ladder complete"
    assert lad.set_index(["task", "step"]).loc[("B", "smote"), "change vs none"] == 0.03


def test_e08_verdicts_use_the_best_step_that_ran(monkeypatch):
    # bases from an experiment with no verdict block of its own
    monkeypatch.setattr(R, "E08_BASE", {t: "E03-xgb" for t in CORE_TASKS})

    def run(label, task, f1, weak):
        exp, model = label.split("-", 1)
        return {
            "run_name": f"{label}-{task}-15s",
            "tags.protocol": "holdout",
            "tags.experiment_id": exp,
            "params.task": task,
            "params.timeout": 15,
            "params.model": model.split("_")[0],
            "metrics.val_f1_macro": f1,
            "tags.worst_class_f1": "C",
            "metrics.val_f1.C": weak,
            "tags.trial": "",
        }

    rows = [run(R.E08_BASE[t], t, 0.80, 0.60) for t in CORE_TASKS]
    m = {t: R.E08_BASE[t].split("-", 1)[1] for t in CORE_TASKS}
    rows += [run(f"E08-{m['A1']}_class_weight", "A1", 0.795, 0.6)]
    rows += [
        run(f"E08-{m['A2-VPN']}_class_weight", "A2-VPN", 0.815, 0.65),
        run(f"E08-{m['A2-VPN']}_oversample", "A2-VPN", 0.812, 0.70),
    ]
    rows += [run(f"E08-{m['A2-NonVPN']}_class_weight", "A2-NonVPN", 0.805, 0.62)]
    rows += [run(f"E08-{m['B']}_class_weight", "B", 0.79, 0.55)]  # fmt: skip
    v = R.phase3_verdicts(R.selection_runs(pd.DataFrame(rows)))
    assert v["E08a"]["best_step"]["A2-VPN"] == "class_weight"
    assert v["E08a"]["tasks_with_gain_ge_0.01"] == 1 and not v["E08a"]["supported"]
    assert v["E08b"]["supported"]
    assert v["E08c"]["weakest_class_f1_change_with_best_step"] == {
        "A2-VPN": 0.05,
        "A2-NonVPN": 0.02,
        "B": -0.05,
    }
    assert v["E08c"]["supported"]


def test_vpn_group_errors_splits_cross_group_and_type_errors():
    names = ["CHAT", "MAIL", "VPN-CHAT", "VPN-MAIL"]
    cm = pd.DataFrame([[8, 1, 1, 0],
                       [0, 9, 0, 1],
                       [2, 0, 7, 1],
                       [0, 0, 0, 10]], index=names, columns=names)  # fmt: skip
    g = R.vpn_group_errors(cm)
    assert g["rows"] == 40 and g["errors"] == 6
    assert g["group errors (VPN / non-VPN wrong)"] == 4
    assert g["type errors (right group, wrong type)"] == 2
    assert g["group accuracy"] == 0.9 and g["share of errors from the group"] == round(4 / 6, 4)


def test_e10_verdicts_from_confusion_matrices(monkeypatch):
    names = ["CHAT", "VPN-CHAT"]
    cms = {"d": pd.DataFrame([[9, 1], [1, 9]], index=names, columns=names),
           "t": pd.DataFrame([[8, 2], [1, 9]], index=names, columns=names)}  # fmt: skip
    monkeypatch.setattr(R, "run_confusion", lambda rid, runner: cms[rid])
    monkeypatch.setattr(R, "E10_DIRECT", "E03-xgb")  # no verdict block of its own
    rows = [
        {"run_name": "E03-xgb-B-15s", "run_id": "d", "metrics.val_f1_macro": 0.90},
        {"run_name": "E10-xgb_two_step-B-15s", "run_id": "t", "metrics.val_f1_macro": 0.885},
    ]
    s = pd.DataFrame([{**r, "tags.protocol": "holdout", "tags.experiment_id": r["run_name"][:3],
                       "params.task": "B", "params.timeout": 15, "metrics.val_accuracy": 0.9,
                       "metrics.cv_f1_macro_mean": 0.9, "metrics.cv_f1_macro_std": 0.01,
                       "tags.trial": ""} for r in rows])  # fmt: skip
    v = R.phase3_verdicts(R.selection_runs(s), runner="x")
    assert v["E10a"] == {"two_step_minus_direct": -0.015, "supported": False}
    assert v["E10b"]["two_step_group_errors"] == 3 and not v["E10b"]["supported"]
    assert v["E10c"]["share_of_two_step_errors_from_step_1"] == 1.0
    assert "E10a" not in R.phase3_verdicts(R.selection_runs(s))  # no runner -> skipped


def test_e17_verdicts_against_untuned_versions():
    rows = []
    for t in CORE_TASKS:
        rows += [("E02", "tree", t, 15, 0.80), ("E03", "logreg", t, 15, 0.50),
                 ("E03", "svm_rbf", t, 15, 0.50), ("E17", "tree", t, 15, 0.82),
                 ("E17", "logreg", t, 15, 0.70), ("E17", "svm_rbf", t, 15, 0.49)]  # fmt: skip
    df = fake_selection(rows)
    df["params.model.signed_log"] = [
        "True" if n.startswith("E17-logreg") else ("False" if n.startswith("E17-svm") else "")
        for n in df["run_name"]
    ]
    v = R.phase3_verdicts(R.selection_runs(df))
    assert v["E17a"]["searches_above"] == 8 and v["E17a"]["of"] == 12
    assert v["E17b"]["best_after_E17"]["A1"] == "E17-tree" and not v["E17b"]["supported"]
    assert v["E17c"]["searches_with_log"] == 4 and not v["E17c"]["supported"]
