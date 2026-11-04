"""E18 (post-hoc calibration of A2-NonVPN and B): same labels and F1 as the uncalibrated run,
its own protocol, the E18 configs, and the before/after table and verdicts."""

import shutil

import numpy as np
import pandas as pd
import yaml
from test_final import _parts

from blindspy import calibration, train
from blindspy import results as R
from blindspy.data import CORE_TASKS, ROOT


def _fit(tmp_path, calibrate, protocol=None):
    tr, va, te = _parts()
    fit, score, name = train.final_parts((tr, va, te), rehearsal=False)
    cfg = {"task": "A2-VPN", "timeout": 15, "model": "lgbm", "params": {"n_estimators": 20},
           "calibrate": calibrate, **({"protocol": protocol} if protocol else {})}  # fmt: skip
    res = train.final_run(cfg, fit, score, name, 42, 5, tmp_path)
    return res, pd.read_csv(tmp_path / "predictions" / "test_predictions.csv")


def test_calibration_keeps_labels_and_f1_and_lowers_nothing_else(tmp_path):
    plain, p_plain = _fit(tmp_path / "plain", "none")
    cal, p_cal = _fit(tmp_path / "cal", "isotonic", R.E18_PROTOCOL)
    assert plain["tags"]["protocol"] == "final_test"  # E12 path unchanged
    assert cal["tags"]["protocol"] == R.E18_PROTOCOL
    assert plain["params"]["protocol"].endswith("score test, once")
    assert "again after E12 (post-hoc" in cal["params"]["protocol"]
    assert (p_cal["predicted"] == p_plain["predicted"]).all()
    assert cal["metrics"]["test_f1_macro"] == plain["metrics"]["test_f1_macro"]
    assert cal["metrics"]["test_ece_uncalibrated"] == plain["metrics"]["test_ece_used"]
    correct = p_cal["predicted"] == p_cal["true"]
    # ECE recomputed from the saved CSV equals the logged value
    assert np.isclose(calibration.ece(p_cal["confidence"], correct),
                      cal["metrics"]["test_ece_used"], rtol=0, atol=1e-12)  # fmt: skip


def test_e18_configs_differ_from_e12_only_in_four_keys():
    exp = ROOT / "configs" / "experiments"
    for task in ("A2-NonVPN", "B"):
        e12 = yaml.safe_load((exp / f"E12_final_{task}.yaml").read_text())
        e18 = yaml.safe_load((exp / f"E18_final_{task}.yaml").read_text())
        assert e12["calibrate"] == "none"
        assert e18 == {**e12, "experiment_id": "E18", "run_name": f"E18-final-{task}",
                       "calibrate": "isotonic", "protocol": R.E18_PROTOCOL}  # fmt: skip
    assert not (exp / "E18_final_A1.yaml").exists()
    assert not (exp / "E18_final_A2-VPN.yaml").exists()


def test_e12_runs_are_the_30_e12_runs():
    h = R.load_history()
    e = R.e12_runs(h)
    assert len(e) == 30 and set(e["tags.experiment_id"]) == {"E12"}
    assert not e["run_name"].str.startswith("E18").any()


def test_e18_table_and_verdicts(tmp_path, monkeypatch):
    plain, p_plain = _fit(tmp_path / "plain", "none")
    cal, p_cal = _fit(tmp_path / "cal", "isotonic", R.E18_PROTOCOL)
    runner = "fake"
    rows = []
    for task in CORE_TASKS:
        for exp, res, src, calib in (("E12", plain, "plain", "none"),
                                     ("E18", cal, "cal", "isotonic")):  # fmt: skip
            if exp == "E18" and task not in ("A2-NonVPN", "B"):
                continue
            run_id = f"{exp}{task}"
            art = tmp_path / "mlruns" / runner / "artifacts" / run_id / "artifacts" / "predictions"
            art.mkdir(parents=True)
            shutil.copy(tmp_path / src / "predictions" / "test_predictions.csv", art)
            rows.append({
                "run_id": run_id, "run_name": f"{exp}-final-{task}",
                "tags.experiment_id": exp,
                "tags.protocol": "final_test" if exp == "E12" else R.E18_PROTOCOL,
                "params.task": task, "params.role": "headline", "params.model": "lgbm",
                "params.calibrate": calib,
                "metrics.test_f1_macro": res["metrics"]["test_f1_macro"],
                "metrics.test_ece_used": res["metrics"]["test_ece_used"],
            })  # fmt: skip
    h = pd.DataFrame(rows)
    assert len(R.e12_runs(h)) == 4  # the E18 rows never enter the E12 selection
    monkeypatch.setattr(R, "sample_rows", lambda task: pd.Index(p_plain["raw_row"][:20]))
    t = R.e18_table(h, runner, root=tmp_path).set_index("task")
    assert list(t.index) == CORE_TASKS
    assert t.loc["A1", "confidence from"] == "E12-final-A1"
    assert t.loc["B", "confidence from"] == "E18-final-B"
    assert t.loc["B", "same label as E12 (share of test rows)"] == 1.0
    assert t.loc["B", "calibration before -> after"] == "none -> isotonic"
    assert t.loc["B", "demo sample >= 0.99 before"].endswith("of 20")
    v = R.e18_verdicts(t.reset_index())
    assert v["E18a"]["supported"] and v["E18b"]["supported"]
    lower = cal["metrics"]["test_ece_used"] < plain["metrics"]["test_ece_used"]
    assert v["E18c"]["supported"] == lower
    assert "supported" not in v["E18d"]  # reported only
