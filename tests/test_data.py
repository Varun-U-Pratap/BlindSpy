"""Parser, column map, cleaned data v1, splits and cleaning log.

Tests that need the raw dataset or data/processed/v1 skip when those are missing
(build them with: uv run python scripts/prepare_data.py --version v1).
"""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from blindspy import checks, splits
from blindspy.data import (
    CORE_TASKS,
    FEATURES,
    HEADERS,
    LABELS,
    RAW,
    ROOT,
    TASK_FILES,
    TIMEOUTS,
    _clean_text,
    header_kind,
    load_arff,
    load_task,
    processed_dir,
)

V1 = processed_dir("v1")
needs_raw = pytest.mark.skipif(not RAW.exists(), reason="dataset not in datasets/CSVs")
needs_v1 = pytest.mark.skipif(not V1.exists(), reason="data/processed/v1 not built")
PAIRS = [(task, t) for task in CORE_TASKS for t in TIMEOUTS]
HAND_ROWS = json.loads((ROOT / "tests" / "fixtures" / "hand_rows.json").read_text())

spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


def manifest() -> dict:
    return json.loads((V1 / "manifest.json").read_text())


# --- parser ---------------------------------------------------------------------------


def test_clean_text_strips_header_commas_and_comma_only_lines():
    text = (
        "@RELATION r,,,\n@ATTRIBUTE a NUMERIC,,,\n@ATTRIBUTE class1 {x,y},,\n@DATA,,,\n,,,\n1,x\n"
    )
    fixed, dropped = _clean_text(text)
    assert dropped == 1
    assert fixed == "@RELATION r\n@ATTRIBUTE a NUMERIC\n@ATTRIBUTE class1 {x,y}\n@DATA\n1,x\n"


def test_unknown_header_is_rejected():
    with pytest.raises(ValueError, match="unknown header"):
        header_kind(list(reversed(HEADERS["T"])))


def test_fixture_covers_all_20_files():
    assert len(HAND_ROWS) == 20
    assert all(len(rows) == 3 for rows in HAND_ROWS.values())


@needs_raw
@pytest.mark.parametrize("file", sorted(HAND_ROWS))
def test_parser_matches_hand_checked_rows(file):
    df = load_arff(ROOT / file)
    for item in HAND_ROWS[file]:
        *values, label = item["line"].split(",")
        row = df.iloc[item["row"]]
        assert row["label"] == label
        np.testing.assert_array_equal(row.iloc[:-1].to_numpy(float), np.array(values, float))


@needs_raw
def test_only_the_60s_a1_and_b_files_use_the_std_header():
    s_files = {f for f in HAND_ROWS if header_kind(list(load_arff(ROOT / f).columns[:-1])) == "S"}
    assert {f.rsplit("-Dataset-", 1)[1] for f in s_files} == {
        "60s-VPN.arff",
        "60s.arff",
        "60s-AllinOne.arff",
    }
    assert all("A2-ARFF" not in f for f in s_files)


# --- cleaned data v1 ------------------------------------------------------------------


@needs_v1
def test_twenty_files_inventoried_with_expected_tasks():
    files = manifest()["files"]
    assert len(files) == 20
    per_task = pd.Series([f["task"] for f in files]).value_counts().to_dict()
    assert per_task == {task: 4 for task in TASK_FILES}
    assert sum(not f["core_task"] for f in files) == 4  # B-AllinOne, inventoried only


@needs_v1
@pytest.mark.parametrize("task,timeout", PAIRS)
def test_row_counts_labels_and_columns(task, timeout):
    info = manifest()["tasks"][f"{task}_{timeout}s"]
    df = pd.read_parquet(V1 / info["parquet"])
    assert len(df) == info["summary"]["final_rows"]
    assert df.index.name == "raw_row" and df.index.is_unique
    assert set(df["label"]) <= set(LABELS[task])
    assert list(df.columns) == [*FEATURES[TASK_FILES[task][0]], "label"]
    raw_rows = next(
        f["rows"] for f in manifest()["files"] if f["task"] == task and f["timeout"] == timeout
    )
    assert info["summary"]["initial_rows"] - info["summary"]["comma_only_lines_skipped"] == raw_rows


@needs_v1
@pytest.mark.parametrize("task,timeout", PAIRS)
def test_column_map_passes_consistency_checks(task, timeout):
    df = pd.read_parquet(V1 / f"{task}_{timeout}s.parquet")
    for name, pct in checks.consistency(df).items():
        if pct is None:
            continue
        if name.startswith("bytes per packet"):
            assert pct >= 95, (name, pct)  # header-only and merged segments are real
        else:
            assert pct == 100, (name, pct)


@needs_v1
def test_a1_120s_flow_pkts_restored_from_b():
    a1 = pd.read_parquet(V1 / "A1_120s.parquet")
    b = pd.read_parquet(V1 / "B_120s.parquet")
    common = a1.index.intersection(b.index)
    assert len(common) > 9000
    pd.testing.assert_series_equal(
        a1.loc[common, "flowPktsPerSecond"], b.loc[common, "flowPktsPerSecond"]
    )


# --- splits ---------------------------------------------------------------------------


def test_make_split_is_stratified_and_repeatable():
    df = pd.DataFrame({"label": ["a"] * 140 + ["b"] * 60}, index=range(100, 300))
    first = splits.make_split(df, 0.7, 0.15, 0.15, seed=42)
    assert first == splits.make_split(df, 0.7, 0.15, 0.15, seed=42)
    assert sorted(first["train"] + first["val"] + first["test"]) == list(df.index)
    assert df.loc[first["test"], "label"].value_counts().to_dict() == {"a": 21, "b": 9}


@needs_v1
@pytest.mark.parametrize("task,timeout", PAIRS)
def test_splits_disjoint_complete_and_stratified(task, timeout):
    df = pd.read_parquet(V1 / f"{task}_{timeout}s.parquet")
    train, val, test = load_task(task, timeout)
    parts = [set(train.index), set(val.index), set(test.index)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert parts[0] | parts[1] | parts[2] == set(df.index)
    for part, frac in [(train, 0.70), (val, 0.15), (test, 0.15)]:
        share = part["label"].value_counts() / df["label"].value_counts()
        assert (share - frac).abs().max() < 0.05, share


@needs_v1
@pytest.mark.parametrize("task,timeout", PAIRS)
def test_no_exact_duplicate_crosses_train_and_test(task, timeout):
    train, _, test = load_task(task, timeout)
    cols = list(train.columns)
    both = test.merge(train.drop_duplicates(), on=cols, how="inner")
    assert len(both) == 0
    info = manifest()["tasks"][f"{task}_{timeout}s"]
    assert info["exact_duplicates_across_train_test"] == 0


# --- cleaning log ---------------------------------------------------------------------


@needs_v1
def test_cleaning_log_adds_up_per_file_and_per_class():
    log = pd.read_csv(V1 / "cleaning_log.csv")
    for (task, timeout), steps in log.groupby(["task", "timeout"]):
        steps = steps.sort_values("step")
        assert list(steps["step"]) == list(range(1, len(steps) + 1))
        before = steps["rows_before"].to_numpy()
        after = steps["rows_after"].to_numpy()
        assert (before[1:] == after[:-1]).all()
        assert (steps["rows_removed"] == steps["rows_before"] - steps["rows_after"]).all()
        per_class = [json.loads(s) for s in steps["rows_after_per_class"]]
        for prev, step in zip(per_class[:-1], steps.iloc[1:].itertuples(), strict=True):
            removed = json.loads(step.removed_per_class)
            now = json.loads(step.rows_after_per_class)
            for label in prev:
                assert prev[label] - removed.get(label, 0) == now.get(label, 0)
        parquet = pd.read_parquet(V1 / f"{task}_{timeout}s.parquet")
        assert after[-1] == len(parquet)
        assert per_class[-1] == parquet["label"].value_counts().sort_index().to_dict()
        assert sum(steps["rows_removed"]) == before[0] - len(parquet)


@needs_v1
def test_every_cleaning_rule_is_logged_for_every_file():
    log = pd.read_csv(V1 / "cleaning_log.csv")
    rules = list(manifest()["cleaning_rules"])
    for _, steps in log.groupby(["task", "timeout"]):
        assert list(steps.sort_values("step")["step_name"]) == rules


def test_refuses_to_overwrite_an_existing_version(tmp_path):
    (tmp_path / "data" / "processed" / "v1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="never overwritten"):
        prepare_data.build("v1", root=tmp_path)


@needs_v1
def test_changelog_v1_table_matches_manifest():
    text = (ROOT / "reports" / "DATA_CHANGELOG.md").read_text()
    section = text.split("## v1 ", 1)[1]
    tasks = manifest()["tasks"]
    rows = [
        line for line in section.splitlines() if line.startswith("| A") or line.startswith("| B")
    ]
    assert len(rows) == 16
    for line in rows:
        name, raw, removed, final, parts = [c.strip() for c in line.strip("|").split("|")]
        task, timeout, _ = name.split()
        s = tasks[f"{task}_{timeout}s"]["summary"]
        num = lambda v: int(v.split()[0].replace(",", ""))  # noqa: E731
        assert num(raw) == s["initial_rows"]
        assert num(removed) == s["exact_duplicates_removed"]
        assert num(final) == s["final_rows"]
        assert [num(p) for p in parts.split("/")] == [s["train"], s["val"], s["test"]]
