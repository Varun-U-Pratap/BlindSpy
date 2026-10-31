"""app/demo.py: the Streamlit page, run headless with streamlit.testing (models/final and
app/samples as committed)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from blindspy import data
from blindspy.tracking import ROOT

DEMO = str(ROOT / "app" / "demo.py")


def _demo_module():
    spec = importlib.util.spec_from_file_location("demo", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(task=None, sample=None):
    at = AppTest.from_file(DEMO, default_timeout=120)
    at.run()
    if task is not None:
        at.selectbox(key="task").set_value(task).run()
    if sample is not None:
        at.selectbox(key="sample").set_value(sample).run()
    assert not at.exception, at.exception
    return at


def test_five_samples():
    names = _demo_module().sample_files()
    assert len(names) == 5 and "malformed_renamed_column.csv" in names
    assert all(f"{t}_test_sample.csv" in names for t in data.CORE_TASKS)


@pytest.mark.parametrize("task", data.CORE_TASKS)
def test_task_sample_gives_a_prediction_table(task):
    at = _run(task, f"{task}_test_sample.csv")
    assert not at.error
    table = at.dataframe[0].value
    assert len(table) == 20
    assert {"label", "confidence", "true_label"} <= set(table.columns)


def test_malformed_sample_is_rejected():
    at = _run("B", "malformed_renamed_column.csv")
    assert len(at.error) == 1
    assert "missing ['flowBytesPerSecond']" in at.error[0].value
    assert not at.dataframe


@pytest.mark.parametrize("task", ["A1", "B"])
def test_row_explanation_matches_predict(task):
    # the chart for a clicked row: SHAP towards the predicted label, largest first; its top 3 are
    # predict.py's top-3 features for that row
    demo = _demo_module()
    from blindspy import predict as P

    sample = pd.read_csv(ROOT / "app" / "samples" / f"{task}_test_sample.csv")
    out = P.predict(task, sample)
    for pos in (0, 7):
        ex = demo.explain_row(task, sample, pos, out["label"].iloc[pos])
        assert len(ex) == 10 and np.all(np.diff(ex["SHAP"].to_numpy()) <= 0)
        tops = [out[f"top_feature_{n}"].iloc[pos].split(" (")[0] for n in (1, 2, 3)]
        assert list(ex["feature"].head(3)) == tops
