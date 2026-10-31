import pandas as pd
import pytest

from blindspy import figures as F
from blindspy import results as R
from blindspy.tracking import ROOT, load_yaml

RUNNER = load_yaml(ROOT / "configs" / "base.yaml")["official_runner"]


@pytest.fixture(scope="module")
def h():
    return R.load_history()


def test_progress_table_never_drops_and_ends_at_the_final_choice(h):
    s = R.selection_runs(h)
    t = F.progress_table(s)
    final = R.final_choice(s).set_index("task")
    for task, g in t.groupby("task"):
        best = g["best so far"].to_numpy()
        assert (best[1:] >= best[:-1]).all()
        assert best[-1] == pytest.approx(final.loc[task, "val macro F1"])


def test_every_listed_graph_exists():
    assert F.write_index() == []


STORE = ROOT / "mlruns" / RUNNER / "mlflow.db"


@pytest.mark.skipif(not STORE.exists(), reason="official MLflow store not on this machine")
def test_confusion_pairs_match_the_headline_details(h):
    pairs = R.e12_confusion_pairs(h, RUNNER)
    details = R.e12_headline_details(h).set_index("task")
    col = [c for c in details.columns if c.startswith("top-5 confusion pairs")][0]
    for task, g in pairs.groupby("task"):
        text = "; ".join(f"{a} -> {b}: {n}" for a, b, n in
                         zip(g["true"], g["predicted"], g["flows"], strict=True))  # fmt: skip
        assert text == details.loc[task, col]
    assert isinstance(pairs, pd.DataFrame) and (pairs["share of true class"] <= 1).all()
