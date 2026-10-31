"""The committed archive of run artifacts matches its index and the run history."""

import hashlib
import tarfile

import pandas as pd

from blindspy.tracking import ROOT

ARCHIVE = ROOT / "reports" / "run_artifacts_text.tar.gz"
INDEX = ROOT / "reports" / "run_artifacts_index.csv"


def test_archive_matches_index_and_history():
    index = pd.read_csv(INDEX).set_index("path")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        assert sorted(m.name for m in members) == sorted(index.index)
        for m in members:
            data = tar.extractfile(m).read()
            assert len(data) == index.loc[m.name, "bytes"]
            assert hashlib.sha256(data).hexdigest() == index.loc[m.name, "sha256"]
    hist = pd.read_csv(ROOT / "reports" / "history.csv", usecols=["run_id", "run_name"])
    names = dict(zip(hist["run_id"], hist["run_name"], strict=True))
    assert all(names[r] == n for r, n in zip(index["run_id"], index["run_name"], strict=True))
    preds = index[index.index.str.endswith("predictions/test_predictions.csv")]
    assert set(preds["experiment_id"]) == {"E12"} and len(preds) == 30
