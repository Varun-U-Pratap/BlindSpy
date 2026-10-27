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
