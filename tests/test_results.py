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
