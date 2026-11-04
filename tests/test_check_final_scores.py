"""scripts/check_final_scores.py: a reproducer's E12 and E18 test scores against
reports/history.csv and the committed test predictions."""

import importlib.util

import pandas as pd
import pytest

from blindspy.tracking import ROOT


def _cli():
    path = ROOT / "scripts" / "check_final_scores.py"
    spec = importlib.util.spec_from_file_location("check_final_scores", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_official_scores_are_the_30_e12_and_2_e18_runs():
    off = _cli().official()
    assert len(off) == 32 and off["run_name"].str[:4].value_counts().to_dict() == {
        "E12-": 30,
        "E18-": 2,
    }
    a1 = off.set_index("run_name").loc["E12-final-A1"]
    assert round(a1["test_f1_macro"], 4) == 0.9434 and round(a1["test_accuracy"], 4) == 0.9435


def test_compare():
    cli = _cli()
    off = cli.official()
    same = off.copy()
    out = cli.compare(same, off)
    assert out["exact"].all() and out["rounded"].all() and len(out) == 32
    near = off.copy()
    near.loc[0, "test_f1_macro"] += 2e-5  # last digits differ, 4 decimals agree or not
    out = cli.compare(near, off)
    assert not out["exact"].iloc[0] and out["exact"].iloc[1:].all()
    missing = cli.compare(off.iloc[1:], off)
    assert missing["test_f1_macro"].isna().iloc[0] and not missing["rounded"].iloc[0]
    assert isinstance(missing, pd.DataFrame)


@pytest.mark.skipif(not (ROOT / "mlruns" / "utsav-s-r" / "mlflow.db").exists(),
                    reason="needs the official MLflow store")  # fmt: skip
def test_same_labels_official_store_against_archive():
    cli = _cli()
    off = cli.official()
    same = cli.same_labels("utsav-s-r", off, off)
    assert len(same) == 32 and all(same.values())
    assert cli.same_labels("utsav-s-r", off.iloc[1:], off)[off["run_name"].iloc[0]] is False
