"""Result tables and threshold checks (on a synthetic history, no MLflow store needed)."""

import json

import pandas as pd
import pytest

from blindspy import results as R
from blindspy.data import CORE_TASKS, ROOT, TIMEOUTS


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
         "params.model.n_estimators": "505", "params.model.max_depth": "None"},
    ])  # fmt: skip
    h.to_csv(tmp_path / "h.csv", index=False)
    got = R.run_settings("E07-xgb-B-15s", tmp_path / "h.csv")
    assert got == {
        "model": "xgb",
        "params": {"learning_rate": 0.10125778281260306, "n_estimators": 505, "max_depth": None},
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


def test_e17_searches_table_shows_pipeline_options():
    rows = [("E03", "svm_rbf", "A1", 15, 0.6), ("E17", "svm_rbf", "A1", 15, 0.9)]
    df = fake_selection(rows)
    df["params.search.space"] = json.dumps({"signed_log": {}, "C": {}, "max_iter": {}})
    df["params.estimator_params"] = json.dumps({"C": 5.0, "kernel": "rbf"})
    df["params.model.signed_log"] = True
    for m in ("search_n_trials", "search_total_fits", "search_best_trial", "search_time_s"):
        df[f"metrics.{m}"] = 1.0
    df["metrics.search_best_cv_f1_macro_mean"] = df["metrics.cv_f1_macro_mean"]
    tab = R.e07_searches(R.selection_runs(df), "E17")
    df["params.model.max_iter"] = float("nan")  # a None setting, absent from estimator_params
    tab = R.e07_searches(R.selection_runs(df), "E17")
    assert tab.loc[0, "chosen settings"] == "signed_log=True, C=5, max_iter=None"


def test_final_choice_takes_the_top_score_and_skips_files_over_50_mb():
    rows = [("E06", "xgb", t, 15, 0.80) for t in CORE_TASKS]
    rows += [("E07", "lgbm", "A1", 15, 0.82), ("E07", "lgbm", "B", 30, 0.85),
             ("E01", "majority", "A2-VPN", 15, 0.95)]  # fmt: skip
    df = fake_selection(rows)
    df["run_id"] = df["run_name"]
    df["params.model"] = [n.split("-")[1] for n in df["run_name"]]
    df["metrics.model_file_bytes"] = [
        60e6 if n == "E07-lgbm-A1-15s" else 1e6 for n in df["run_name"]
    ]
    f = R.final_choice(R.selection_runs(df)).set_index("task")
    assert f.loc["A1", "run"] == "E06-xgb-A1-15s"  # the 60 MB LightGBM is skipped
    assert f.loc["A1", "skipped for file size"] == "E07-lgbm-A1-15s"
    assert f.loc["B", "run"] == "E07-lgbm-B-30s" and f.loc["B", "timeout"] == 30  # any timeout
    assert f.loc["A2-VPN", "run"] == "E06-xgb-A2-VPN-15s"  # majority is not a candidate


def test_comparison_set_takes_the_best_15s_variant_of_every_method_type():
    models_ = ["tree", "knn", "knn_log_standard", "logreg", "svm_rbf", "rf", "xgb",
               "xgb_two_step", "lgbm"]  # fmt: skip
    rows = [("E05", m, t, 15, 0.70) for t in CORE_TASKS for m in models_]
    rows += [("E07", "knn_log_standard", "A1", 15, 0.75), ("E10", "xgb_two_step", "A1", 15, 0.72),
             ("E07", "xgb", "A1", 30, 0.99)]  # fmt: skip
    df = fake_selection(rows)
    df["run_id"] = df["run_name"]
    df["params.model"] = [n.split("-")[1] for n in df["run_name"]]
    c = R.comparison_set(R.selection_runs(df))
    assert len(c) == 28 and c.groupby("task").size().eq(7).all()
    a1 = c[c["task"] == "A1"].set_index("method type")
    assert a1.loc["KNN", "run"] == "E07-knn_log_standard-A1-15s"
    assert a1.loc["XGBoost", "run"] == "E10-xgb_two_step-A1-15s"  # 30 s run not eligible
    assert a1.loc["KNN", "variants compared"] == 3


def test_old_rule_picks_the_simplest_run_within_003():
    rows = [("E06", "xgb", t, 15, 0.90) for t in CORE_TASKS]
    rows += [("E04", "knn_log_standard", t, 15, 0.875) for t in CORE_TASKS]
    rows += [("E07", "knn_log_standard", t, 15, 0.88) for t in CORE_TASKS]
    rows += [("E02", "tree", "A1", 15, 0.86)]  # 0.04 below: not a tie
    df = fake_selection(rows)
    df["run_id"] = df["run_name"]
    df["params.model"] = [n.split("-")[1] for n in df["run_name"]]
    df["tags.tuned"] = ["True" if n.startswith("E07") else "" for n in df["run_name"]]
    df["params.class_balancing"] = "none"
    df["metrics.predict_ms_per_1000"] = 1.0
    o = R.old_rule_choice(R.selection_runs(df)).set_index("task")
    assert o.loc["A1", "run"] == "E04-knn_log_standard-A1-15s"  # KNN, and default < tuned
    assert o.loc["A1", "below the best by"] == 0.025 and o.loc["A1", "runs within 0.03"] == 3


def test_e09_verdicts_on_a_table():
    rows = []
    cases = [("A1", -0.10, -0.01, 0.0), ("A2-VPN", -0.2, -0.03, 0.005),
             ("A2-NonVPN", -0.05, -0.015, -0.02), ("B", 0.035, 0.0, 0.0)]  # fmt: skip
    for task, only_best, worst_drop, timing in cases:
        rows += [
            {"task": task, "feature set": "all", "change vs all": 0.0},
            {"task": task, "feature set": "only_fiat", "change vs all": only_best},
            {"task": task, "feature set": "only_idle", "change vs all": only_best - 0.1},
            {"task": task, "feature set": "no_fiat", "change vs all": worst_drop},
            {"task": task, "feature set": "timing_only", "change vs all": timing},
        ]
    v = R.e09_verdicts(pd.DataFrame(rows))
    assert v["E09a"]["tasks_all_groups_below"] == 3 and not v["E09a"]["supported"]
    assert v["E09b"]["tasks_within_0.02"] == 3 and v["E09b"]["supported"]
    assert v["E09c"]["tasks_within_0.01"] == 3 and v["E09c"]["supported"]
    assert v["E09_stop_and_ask"]["sets_at_least_0.03_above_all"]["B"] == ["only_fiat"]


def test_e13_verdicts():
    rows = []
    for task, at10, at80, at100, s10, s80 in [("A1", 0.80, 0.92, 0.925, 0.02, 0.005),
                                              ("A2-VPN", 0.70, 0.86, 0.88, 0.03, 0.01),
                                              ("A2-NonVPN", 0.75, 0.90, 0.905, 0.01, 0.02),
                                              ("B", 0.65, 0.835, 0.84, 0.02, 0.01)]:  # fmt: skip
        rows += [{"task": task, "train fraction": 0.1, "val macro F1 mean": at10,
                  "val macro F1 std": s10},
                 {"task": task, "train fraction": 0.8, "val macro F1 mean": at80,
                  "val macro F1 std": s80},
                 {"task": task, "train fraction": 1.0, "val macro F1 mean": at100,
                  "val macro F1 std": float("nan")}]  # fmt: skip
    v = R.e13_verdicts(pd.DataFrame(rows))
    assert v["E13a"]["supported"]
    assert v["E13b"]["tasks_below_0.01"] == 3 and v["E13b"]["supported"]  # A2-VPN gains 0.02
    assert v["E13c"]["tasks_10_above_80"] == 3 and v["E13c"]["supported"]


def test_e14_verdicts():
    t = pd.DataFrame([
        {"task": "A1", "val macro F1 std": 0.004, "CV - val mean": 0.002, "val macro F1 min": 0.92,
         "val macro F1 max": 0.935, "official split (seed 42) val macro F1": 0.93},
        {"task": "B", "val macro F1 std": 0.012, "CV - val mean": -0.015, "val macro F1 min": 0.82,
         "val macro F1 max": 0.84, "official split (seed 42) val macro F1": 0.843},
    ])  # fmt: skip
    v = R.e14_verdicts(t)
    assert v["E14a"]["tasks_below_0.01"] == 1 and v["E14b"]["tasks_below_0.01"] == 1
    assert v["E14c"]["official_inside_range"] == {"A1": True, "B": False}
    assert not v["E14c"]["supported"]


def test_e16_verdicts():
    rows = [
        {
            "run_name": "E16-isnan-rf",
            "tags.experiment_id": "E16",
            "metrics.kaggle_test_accuracy": 0.885,
            "metrics.kaggle_test_f1_weighted": 0.884,
        }
    ]
    rows += [
        {
            "run_name": f"E16-per_timeout-rf-{t}s",
            "tags.experiment_id": "E16",
            "metrics.kaggle_test_accuracy": a,
            "metrics.kaggle_test_f1_weighted": a,
        }
        for t, a in [(15, 0.86), (30, 0.87), (60, 0.85), (120, 0.86)]
    ]
    rows += [{"run_name": f"E16-per_timeout-xgb-{t}s", "tags.experiment_id": "E16",
              "metrics.kaggle_test_accuracy": a, "metrics.kaggle_test_f1_weighted": a}
             for t, a in [(15, 0.87), (30, 0.87), (60, 0.86), (120, 0.85)]]  # fmt: skip
    v = R.e16_verdicts(pd.DataFrame(rows))
    assert v["E16a"]["supported"]  # 0.885 vs 0.8801, 0.884 vs 0.8797
    assert v["E16b"]["gap"] == 0.025 and v["E16b"]["supported"]
    assert v["E16c"]["supported"]  # 0.8625 >= 0.86


def test_e16_size_control_verdict_and_table():
    base = [
        {
            "run_name": "E16-isnan-rf",
            "tags.experiment_id": "E16",
            "params.n_train": "1000",
            "metrics.kaggle_test_accuracy": 0.88,
            "metrics.kaggle_test_f1_weighted": 0.88,
        }
    ]
    base += [{"run_name": f"E16-per_timeout-rf-{t}s", "tags.experiment_id": "E16",
              "params.n_train": str(n), "metrics.kaggle_test_accuracy": a,
              "metrics.kaggle_test_f1_weighted": a}
             for t, n, a in [(15, 400, 0.85), (120, 200, 0.83)]]  # fmt: skip
    base += [{"run_name": f"E16-per_timeout-xgb-{t}s", "tags.experiment_id": "E16",
              "metrics.kaggle_test_accuracy": 0.86, "metrics.kaggle_test_f1_weighted": 0.86}
             for t in (15, 120)]  # fmt: skip
    assert "E16d" not in R.e16_verdicts(pd.DataFrame(base))  # before (d) runs exist
    size = [
        {
            "run_name": "E16-size-full",
            "tags.experiment_id": "E16",
            "params.n_train": "1000",
            "metrics.kaggle_test_accuracy": 0.88,
            "metrics.kaggle_test_accuracy_15s": 0.9,
        }
    ]
    size += [{"run_name": f"E16-size-{n}-s{seed}", "tags.experiment_id": "E16",
              "params.n_train": str(n), "metrics.kaggle_test_accuracy": a,
              "metrics.kaggle_test_accuracy_15s": a}
             for n, seed, a in [(400, 42, 0.85), (400, 43, 0.87),
                                (200, 42, 0.84), (200, 43, 0.84)]]  # fmt: skip
    h = pd.DataFrame(base + size)
    v = R.e16_verdicts(h)["E16d"]
    # mixed mean = mean(0.86, 0.84) = 0.85; (b) mean 0.84; rest 0.01 -> not below 0.01
    assert v["size_matched_mixed_mean"] == 0.85 and v["size_part"] == 0.03
    assert v["rest"] == 0.01 and not v["supported"] and v["full_reproduces_a"]
    t = R.e16_size_table(h)
    assert list(t["training rows (timeouts mixed)"]) == ["1,000", "400", "200"]
    assert list(t["(b) same rows, one timeout"]) == ["", "15s: 0.8500", "120s: 0.8300"]
    assert t.loc[1, "test accuracy mean"] == 0.86 and t.loc[1, "15s test rows"] == 0.86


def test_md_writes_missing_values_as_empty_cells():
    out = R.md(pd.DataFrame([{"a": 1, "b": float("nan"), "c": 0.5}]))
    assert out.splitlines()[-1] == "| 1 |  | 0.5 |"


def _e15_history(adopted: dict) -> pd.DataFrame:
    import yaml

    rows = []
    for task, (raw, iso) in {"A1": (0.06, 0.01), "A2-VPN": (0.07, 0.02),
                             "A2-NonVPN": (0.04, 0.01), "B": (0.03, 0.01)}.items():  # fmt: skip
        src = yaml.safe_load((ROOT / "configs" / "final" / f"{task}.yaml").read_text())
        rows.append({"tags.experiment_id": "E15", "tags.protocol": "calibration",
                     "params.task": task, "run_name": f"E15-{task}-15s", "run_id": task,
                     "params.builds_on": src["builds_on"], "metrics.val_accuracy": 0.9,
                     "metrics.val_mean_confidence_uncalibrated": 0.95,
                     "metrics.val_mean_confidence_isotonic": 0.9,
                     "metrics.val_ece_uncalibrated": raw, "metrics.val_ece_isotonic": iso,
                     "tags.calibration_adopted": adopted[task],
                     "metrics.val_label_agreement_base_vs_plain": 1.0,
                     "metrics.val_calibrated_argmax_differs": 0.0,
                     "metrics.val_f1_macro": src["source_val_f1_macro"]})  # fmt: skip
    return pd.DataFrame(rows)


def test_e15_adopted_read_from_bool_or_string_tags():
    # history.csv round trip turns "True"/"False" tags into booleans; both must read the same
    for kind in (bool, str):
        flags = {"A1": True, "A2-VPN": True, "A2-NonVPN": False, "B": False}
        h = _e15_history({k: kind(v) for k, v in flags.items()})
        t = R.e15_table(h).set_index("task")
        assert t["adopted"].to_dict() == flags
        v = R.e15_verdicts(h)
        assert v["E15a"]["tasks_above_0.05"] == 2 and v["E15a"]["supported"]
        assert v["E15b"]["supported"] and v["E15c"]["supported"]
        assert v["E15_adopted"]["adopted"] == flags
