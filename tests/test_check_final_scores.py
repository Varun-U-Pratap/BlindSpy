"""scripts/check_final_scores.py: a reproducer's E12 test scores against reports/history.csv."""

import importlib.util

import pandas as pd

from blindspy.tracking import ROOT


def _cli():
    path = ROOT / "scripts" / "check_final_scores.py"
    spec = importlib.util.spec_from_file_location("check_final_scores", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_official_scores_are_the_30_e12_runs():
    off = _cli().official()
    assert len(off) == 30 and off["run_name"].str.startswith("E12-").all()
    a1 = off.set_index("run_name").loc["E12-final-A1"]
    assert round(a1["test_f1_macro"], 4) == 0.9434 and round(a1["test_accuracy"], 4) == 0.9435


def test_compare():
    cli = _cli()
    off = cli.official()
    same = off.copy()
    out = cli.compare(same, off)
    assert out["exact"].all() and out["rounded"].all() and len(out) == 30
    near = off.copy()
    near.loc[0, "test_f1_macro"] += 2e-5  # last digits differ, 4 decimals agree or not
    out = cli.compare(near, off)
    assert not out["exact"].iloc[0] and out["exact"].iloc[1:].all()
    missing = cli.compare(off.iloc[1:], off)
    assert missing["test_f1_macro"].isna().iloc[0] and not missing["rounded"].iloc[0]
    assert isinstance(missing, pd.DataFrame)
