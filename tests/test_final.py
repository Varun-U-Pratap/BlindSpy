"""E12 final run (rehearsal never touches the test rows) and the bootstrap / Holm statistics."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

from blindspy import results as R
from blindspy import train


def _parts(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = np.where(X["a"] + 0.5 * rng.normal(size=n) > 0, "VPN-CHAT", "VPN-FT").astype(object)
    y[X["b"] > 1] = "VPN-MAIL"
    df = X.assign(label=y)
    df.index = np.arange(1000, 1000 + n)  # raw_row ids
    train_df, val, test = df.iloc[:400], df.iloc[400:500], df.iloc[500:].copy()
    return train_df, val, test


def test_final_parts():
    tr, va, te = _parts()
    fit, score, name = train.final_parts((tr, va, te), rehearsal=True)
    assert name == "rehearsal" and fit is tr and score is va
    fit, score, name = train.final_parts((tr, va, te), rehearsal=False)
    assert name == "test" and score is te and list(fit.index) == [*tr.index, *va.index]


def test_rehearsal_never_predicts_test_rows(tmp_path):
    tr, va, te = _parts()
    te[list("abcd")] = np.nan  # any use of test features would fail or show up
    fit, score, name = train.final_parts((tr, va, te), rehearsal=True)
    cfg = {"task": "A2-VPN", "timeout": 15, "model": "lgbm", "params": {"n_estimators": 20}}
    res = train.final_run(cfg, fit, score, name, 42, 5, tmp_path)
    preds = pd.read_csv(tmp_path / "predictions" / "rehearsal_predictions.csv")
    assert list(preds["raw_row"]) == list(va.index)
    assert not set(preds["raw_row"]) & set(te.index)
    assert res["tags"]["protocol"] == "final_rehearsal"
    assert "rehearsal_f1_macro" in res["metrics"] and not any(
        k.startswith("test_") for k in res["metrics"]
    )
    assert res["params"]["n_fit"] == len(tr)


def test_final_run_calibrated_keeps_the_plain_class(tmp_path):
    tr, va, te = _parts()
    fit, score, name = train.final_parts((tr, va, te), rehearsal=False)
    cfg = {"task": "A2-VPN", "timeout": 15, "model": "lgbm", "params": {"n_estimators": 20},
           "calibrate": "isotonic"}  # fmt: skip
    res = train.final_run(cfg, fit, score, name, 42, 5, tmp_path)
    preds = pd.read_csv(tmp_path / "predictions" / "test_predictions.csv")
    model = joblib.load(tmp_path / "model" / "model.joblib")
    assert isinstance(model, CalibratedClassifierCV) and model.cv.shuffle
    plain = model.calibrated_classifiers_[0].estimator
    X = te.drop(columns="label")
    assert (preds["predicted"] == plain.predict(X)).all()
    cal = model.predict_proba(X)
    cols = np.searchsorted(model.classes_, preds["predicted"])
    assert np.allclose(preds["confidence"], cal[np.arange(len(X)), cols])
    m = res["metrics"]
    assert {"test_ece_used", "test_ece_uncalibrated", "calibration_fit_time_s"} <= set(m)
    assert res["params"]["calibrate"] == "isotonic"


def test_final_run_without_probabilities(tmp_path):
    tr, va, te = _parts()
    fit, score, name = train.final_parts((tr, va, te), rehearsal=False)
    cfg = {"task": "A2-VPN", "timeout": 15, "model": "svm_rbf", "params": {"C": 1.0}}
    res = train.final_run(cfg, fit, score, name, 42, 5, tmp_path)
    preds = pd.read_csv(tmp_path / "predictions" / "test_predictions.csv")
    assert preds["confidence"].isna().all() and "decision.VPN-FT" in preds
    assert "test_ece_used" not in res["metrics"] and res["params"]["score_kind"] == "decision"


def test_stratified_indices_keep_class_counts_and_are_seeded():
    y = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    idx = R.stratified_indices(y, 200, 42)
    assert idx.shape == (200, 10)
    for row in idx:
        assert (pd.Series(y[row]).value_counts().to_dict()) == {"a": 5, "b": 3, "c": 2}
    assert np.array_equal(idx, R.stratified_indices(y, 200, 42))
    assert not np.array_equal(idx, R.stratified_indices(y, 200, 43))


def test_macro_f1_resampled_equals_sklearn():
    rng = np.random.default_rng(1)
    classes = ["x", "y", "z"]
    y = rng.choice(classes, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.choice(classes, 300))
    idx = R.stratified_indices(y, 50, 42)
    got = R.macro_f1_resampled(y, pred, idx, classes)
    want = [f1_score(y[r], pred[r], labels=classes, average="macro", zero_division=0) for r in idx]
    assert np.allclose(got, want)
    acc = R.accuracy_resampled(y, pred, idx)
    assert np.allclose(acc, [(y[r] == pred[r]).mean() for r in idx])


def test_paired_p_formula():
    d = np.array([0.1] * 9 + [-0.1])  # one draw at or below 0
    assert R.paired_p(d) == pytest.approx(2 * (1 + 1) / 11)
    assert R.paired_p(np.full(10_000, 0.2)) == pytest.approx(2 / 10_001)  # smallest possible
    assert R.paired_p(np.zeros(10)) == 1.0


def test_holm_hand_example():
    adj = R.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted a 0.01*3 = 0.03; c 0.03*2 = 0.06; b max(0.06, 0.04*1) = 0.06
    assert adj["a"] == (pytest.approx(0.03), True)
    assert adj["c"][0] == pytest.approx(0.06) and not adj["c"][1]
    assert adj["b"][0] == pytest.approx(0.06) and not adj["b"][1]


def test_e12_tests_pairs_headline_with_every_other_run():
    scores = pd.DataFrame({
        "task": ["A1"] * 3, "role": ["headline", "comparison", "old_rule"],
        "method type": ["LightGBM", "XGBoost", "KNN"], "run": ["h", "c", "o"],
        "_point": [0.9, 0.85, 0.8], "_val": [0.9, 0.86, 0.81],
    })  # fmt: skip
    draws = {"h": np.full(100, 0.9), "c": np.full(100, 0.85), "o": np.full(100, 0.91)}
    t = R.e12_tests(scores, draws).set_index("other")
    assert list(t.index) == ["c", "o"]
    assert t.loc["c", "p"] == pytest.approx(2 / 101) and t.loc["o", "p"] == pytest.approx(2 / 101)
    assert t.loc["c", "Holm p"] == pytest.approx(4 / 101)


def test_rehearsal_refused_for_official_runner(monkeypatch):
    import importlib.util

    from blindspy import tracking

    spec = importlib.util.spec_from_file_location(
        "run_experiment", tracking.ROOT / "scripts" / "run_experiment.py"
    )
    run_experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_experiment)
    official = tracking.load_yaml(tracking.ROOT / "configs" / "base.yaml")["official_runner"]
    monkeypatch.setattr(tracking, "load_runner", lambda: official)
    with pytest.raises(SystemExit, match="Refusing --rehearsal"):
        run_experiment.main(["configs/experiments/E12_final_A1.yaml", "--rehearsal"])
