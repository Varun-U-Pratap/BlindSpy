"""E16 data version kaggle-k1 (src/blindspy/kaggle.py); needs datasets/kaggle (skipped without)."""

import pandas as pd
import pytest

from blindspy import kaggle

pytestmark = pytest.mark.skipif(not kaggle.CSV.exists(), reason="Kaggle CSV not in datasets/")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("k")
    return root, kaggle.build("k1", root=root)


def test_counts_match_isnan_and_e00(built):
    _, m = built
    assert m["rows"] == 59706 and len(m["labels"]) == 14
    assert m["isnan"] == {"protocol": "isnan", "rows": 59706, "after_dedup": 40987,
                          "train": 32789, "test": 8198}  # fmt: skip
    by_t = m["distinct_rows_by_number_of_timeouts"]
    assert by_t["2"] + by_t["3"] + by_t["4"] == 7093


def test_splits_disjoint_unsorted_and_within_one_timeout(built):
    root, _ = built
    tr, te = kaggle.load_split("k1", "isnan", root=root)
    assert not set(tr.index) & set(te.index)
    assert list(tr.index) != sorted(tr.index)  # kept in train_test_split's order
    for t in (15, 30, 60, 120):
        tr, te = kaggle.load_split("k1", f"per_timeout_{t}s", root=root)
        assert set(tr["timeout"]) == {t} and set(te["timeout"]) == {t}
        assert not set(tr.index) & set(te.index)
        both = pd.concat([tr, te])
        assert not both.duplicated(subset=[*kaggle.FEATURES, "label"]).any()  # no copy in both


def test_refuses_to_overwrite(built):
    root, _ = built
    with pytest.raises(FileExistsError):
        kaggle.build("k1", root=root)
