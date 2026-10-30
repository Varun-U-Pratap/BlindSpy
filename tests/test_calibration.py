import numpy as np
import pandas as pd
import pytest

from blindspy import calibration, models, train


def test_ece_hand_computed():
    # bin [0.8, 0.867): 2 rows, confidence 0.85, accuracy 0.5 -> gap 0.35
    # bin [0.933, 1]: 2 rows, confidence 1.0, accuracy 1.0 -> gap 0
    conf = [0.85, 0.85, 1.0, 1.0]
    correct = [True, False, True, True]
    assert calibration.ece(conf, correct) == pytest.approx(0.5 * 0.35)
    t = calibration.reliability_bins(conf, correct)
    assert t["rows"].sum() == 4 and len(t) == 15
    assert t.iloc[-1]["rows"] == 2  # confidence 1.0 lands in the last bin


def test_ece_perfect_is_zero():
    assert calibration.ece([0.5] * 4, [True, False, True, False]) == pytest.approx(0)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(np.where(X["a"] + 0.5 * rng.normal(size=n) > 0, "VPN-CHAT", "VPN-FT"))
    y[X["b"] > 1] = "VPN-MAIL"
    return X, y


def test_predicted_label_is_the_uncalibrated_one():
    X, y = _toy()
    pipe = models.build_model("lgbm", {"n_estimators": 20}, 42).fit(X, y)
    cal = calibration.fit_calibrated(pipe, X, y, 42)
    label, conf, proba = calibration.predict_with_confidence(cal, X)
    assert (label == pipe.predict(X)).all()  # base model fitted on all rows = the plain fit
    cols = np.searchsorted(cal.classes_, label)
    assert np.allclose(conf, proba[np.arange(len(X)), cols])
    # an uncalibrated model gives its own probability of its own prediction
    label2, conf2, _ = calibration.predict_with_confidence(pipe, X)
    assert (label2 == label).all() and np.allclose(conf2, pipe.predict_proba(X).max(axis=1))


def test_calibration_run_never_uses_test_rows(tmp_path):
    X, y = _toy(450)
    df = X.assign(label=y)
    train_df, val, test = df.iloc[:300], df.iloc[300:375], df.iloc[375:].copy()
    test[list("abcd")] = np.nan  # any use of test features would fail or show up
    cfg = {"task": "A2-VPN", "timeout": 15, "model": "lgbm", "params": {"n_estimators": 20}}
    res = train.calibration_run(cfg, (train_df, val, test), 42, tmp_path)
    preds = pd.read_csv(tmp_path / "calibration" / "val_predictions.csv")
    assert set(preds["raw_row"]) == set(val.index)
    m = res["metrics"]
    assert m["val_label_agreement_base_vs_plain"] == 1.0
    assert res["tags"]["calibration_adopted"] == str(
        m["val_ece_uncalibrated"] > 0.05 and m["val_ece_isotonic"] < m["val_ece_uncalibrated"]
    )
    assert (tmp_path / "model" / "model.joblib").exists()


def test_calibration_folds_are_shuffled():
    # cv=5 alone would be unshuffled folds: on file-ordered rows each fold is a block of the file
    X, y = _toy()
    pipe = models.build_model("lgbm", {"n_estimators": 20}, 42)
    cal = calibration.fit_calibrated(pipe, X, y, 42)
    assert cal.cv.shuffle and cal.cv.random_state == 42 and cal.cv.n_splits == 5
    assert cal.ensemble is False and cal.method == "isotonic"
    # same seed -> identical calibrated probabilities
    again = calibration.fit_calibrated(pipe, X, y, 42)
    assert np.array_equal(cal.predict_proba(X), again.predict_proba(X))
