"""predict.py: final model files, input checks, label + confidence + top-3 SHAP features."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from blindspy import calibration, data, models
from blindspy import predict as P
from blindspy.tracking import ROOT


def _small(task, model, params, balance=None, calibrate=False, n=600):
    train = data.load_task(task, 15)[0].sample(n, random_state=0)
    X, y = train.drop(columns="label"), train["label"]
    pipe = models.build_model(model, params, 42, balance).fit(X, y)
    return (calibration.fit_calibrated(pipe, X, y, 42) if calibrate else pipe), X, y


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    out = tmp_path_factory.mktemp("final")
    for task, model, params, bal, cal in [
        ("A1", "lgbm", {"n_estimators": 20}, None, True),
        ("A2-NonVPN", "lgbm", {"n_estimators": 20}, "oversample", False),
        ("B", "xgb", {"n_estimators": 10}, None, False),
    ]:
        m, _, _ = _small(task, model, params, bal, cal)
        P.write_final(m, {"task": task, "timeout": 15}, out)
    return out


def test_json_lists_features_units_and_hash(folder):
    _, meta = P.load_final("A1", folder)
    names = [f["name"] for f in meta["features"]]
    assert names == list(data.load_task("A1", 15)[0].drop(columns="label").columns)
    units = {f["name"]: f["unit"] for f in meta["features"]}
    assert units["duration"] == "microseconds" and units["std_fiat"] == "microseconds"
    assert units["flowPktsPerSecond"] == "packets per second"
    assert meta["class_labels"] == ["Non-VPN", "VPN"] and len(meta["model_file_sha256"]) == 64
    _, meta2 = P.load_final("A2-NonVPN", folder)
    assert "total_fiat" in [f["name"] for f in meta2["features"]]  # A2 layout (E00)


@pytest.mark.parametrize("task", ["A1", "A2-NonVPN", "B"])
def test_predict_label_confidence_and_top3(folder, task):
    test = data.load_task(task, 15)[2].head(40)
    out = P.predict(task, test, folder)
    model, _ = P.load_final(task, folder)
    X = test.drop(columns="label")
    label, conf, _ = calibration.predict_with_confidence(model, X)
    assert (out["label"] == label).all() and np.allclose(out["confidence"], conf)
    assert (out["label"] == calibration.base_model(model).predict(X)).all()
    assert out["true_label"].tolist() == test["label"].tolist()
    assert out["top_feature_1"].str.match(r"^\w+ \([+-]\d+\.\d{3}\)$").all()
    assert len(out) == 40


def test_top_features_are_the_largest_shap_values(folder):
    import shap

    model, _ = P.load_final("B", folder)
    X = data.load_task("B", 15)[2].drop(columns="label").head(5)
    lab = model.predict(X)
    tops = P.top_features(model, X, lab)
    v = shap.TreeExplainer(model[-1])(X).values
    for i in range(5):
        j = list(model.classes_).index(lab[i])
        assert [t[0] for t in tops[i]] == list(X.columns[np.argsort(-v[i, :, j])[:3]])


def _bad(folder, df, match):
    with pytest.raises(P.InputError, match=match):
        P.predict("A1", df, folder)


def test_malformed_input_is_rejected(folder):
    X = data.load_task("A1", 15)[2].drop(columns="label").head(5)
    _bad(folder, X.drop(columns="duration"), "missing \\['duration'\\]")
    _bad(folder, X.assign(port=1), "unexpected \\['port'\\]")
    _bad(folder, X[X.columns[::-1]], "wrong order")
    _bad(folder, X.assign(duration=["x"] * 5), "non-numeric")
    _bad(folder, X.assign(duration=[np.nan] + [1.0] * 4), "missing or infinite value in column "
         "duration, data row 1")  # fmt: skip
    _bad(folder, X.head(0), "no rows")


def test_changed_model_file_is_not_loaded(folder, tmp_path):
    import shutil

    for f in ("A1.joblib", "A1.json"):
        shutil.copy(folder / f, tmp_path / f)
    with open(tmp_path / "A1.joblib", "ab") as fh:
        fh.write(b"0")
    with pytest.raises(P.InputError, match="SHA-256"):
        P.load_final("A1", tmp_path)


def test_cli_writes_csv_and_exits_2_on_bad_input(folder, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("predict_cli", ROOT / "scripts" / "predict.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    test = data.load_task("A1", 15)[2].head(8)
    test.drop(columns="label").to_csv(tmp_path / "in.csv", index=False)
    argv = ["--task", "A1", "--input", str(tmp_path / "in.csv"),
            "--output", str(tmp_path / "out.csv"), "--models", str(folder)]  # fmt: skip
    assert cli.main(argv) == 0
    out = pd.read_csv(tmp_path / "out.csv")
    assert list(out.columns) == ["row", "label", "confidence", "top_feature_1", "top_feature_2",
                                 "top_feature_3"]  # fmt: skip
    test.drop(columns=["label", "duration"]).to_csv(tmp_path / "bad.csv", index=False)
    argv[3] = str(tmp_path / "bad.csv")
    assert cli.main(argv) == 2
    assert "Error:" in capsys.readouterr().err
