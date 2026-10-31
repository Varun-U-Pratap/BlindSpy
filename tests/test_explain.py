import numpy as np
import pandas as pd
import pytest

from blindspy import calibration, data
from blindspy import explain as E
from blindspy import predict as P


def test_importance_averages_rows_then_classes():
    v = np.zeros((2, 3, 2))
    v[:, 0, 0] = [1, -3]  # feature a: class 0 mean |SHAP| 2, class 1 0 -> 1
    v[:, 1, :] = 0.5  # feature b: 0.5
    imp = E.importance(v, ["a", "b", "c"])
    assert list(imp.index) == ["a", "b", "c"]
    assert imp.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_group_table_uses_the_e09_groups_and_shares_add_up():
    cols = data.load_task("A1", 15)[2].drop(columns="label").columns
    imp = pd.Series(np.arange(1, len(cols) + 1, dtype=float), index=cols)
    g = E.group_table(imp)
    assert g["features"].sum() == len(cols) and g["share"].sum() == pytest.approx(1.0)
    assert set(g["group"]) == {"fiat", "biat", "flowiat", "active", "idle", "rates_duration"}


def test_pick_examples_follow_the_fixed_rule():
    y = ["a", "a", "b", "b", "b", "a"]
    label = ["a", "b", "b", "b", "b", "a"]
    conf = [0.9, 0.95, 0.6, 0.7, 0.8, 0.99]
    p = E.pick_examples(y, label, conf, weakest="b")
    assert p == {E.EXAMPLES[0]: 5, E.EXAMPLES[1]: 2, E.EXAMPLES[2]: 1, E.EXAMPLES[3]: 3}
    assert E.EXAMPLES[2] not in E.pick_examples(y, y, conf, weakest="b")  # no error to show


@pytest.mark.parametrize("task", ["A1", "B"])
def test_shap_values_add_up_to_the_margin_and_match_predict(task):
    model, _ = P.load_final(task)
    X = data.load_task(task, 15)[2].drop(columns="label").head(20)
    v, classes = E.shap_values(model, X)
    assert v.shape == (20, 23, len(classes))
    base = calibration.base_model(model)
    label = calibration.predict_with_confidence(model, X)[0]
    tops = P.top_features(base, X, label)
    for i in range(20):
        rows = E.example_rows(v, classes, X, i, label[i], k=3)
        assert rows["feature"].tolist() == [t[0] for t in tops[i]]
        assert rows["SHAP"].to_numpy() == pytest.approx([t[1] for t in tops[i]])
    if task == "A1":  # binary: the two columns are mirror images
        assert np.allclose(v[..., 0], -v[..., 1])
