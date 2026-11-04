"""Compare the test scores of your own E12 and E18 reruns with the official ones.

After rerunning the 30 E12 configs and the 2 E18 configs under your own runner name
(configs/local.yaml), this reads your MLflow store (mlruns/<runner>/mlflow.db; the newest run per
name) and compares each run's test macro F1, test accuracy and test ECE (the confidence's
calibration error; empty for the 6 models without probabilities) with the official values in
reports/history.csv (committed, full precision), and each run's predicted label on every test row
with the official test predictions in reports/run_artifacts_text.tar.gz (committed). It prints
one line per run and exits 0 only if all 32 runs are present, agree to 4 decimals (the precision
of the report's tables) and predict the same label on every test row.

Usage: uv run python scripts/check_final_scores.py --runner <your name>
"""

import argparse
import sqlite3
import sys
import tarfile

import pandas as pd

from blindspy.data import ROOT

METRICS = ["test_f1_macro", "test_accuracy", "test_ece_used"]
EXPERIMENTS = ("E12", "E18")
ARCHIVE = ROOT / "reports" / "run_artifacts_text.tar.gz"
PRED = "predictions/test_predictions.csv"


def official() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "reports" / "history.csv", low_memory=False)
    h = h[h["tags.experiment_id"].astype(str).isin(EXPERIMENTS)]
    h = h.sort_values("start_time").drop_duplicates("run_name", keep="last")
    out = h[["run_name", "run_id"] + [f"metrics.{m}" for m in METRICS]]
    return out.set_axis(["run_name", "run_id", *METRICS], axis=1).sort_values(
        "run_name", ignore_index=True
    )


def yours(runner: str) -> pd.DataFrame:
    db = ROOT / "mlruns" / runner / "mlflow.db"
    if not db.exists():
        sys.exit(f"No MLflow store at {db.relative_to(ROOT)}; run the E12 and E18 configs first.")
    with sqlite3.connect(db) as con:
        rows = pd.read_sql(
            "select r.name as run_name, r.run_uuid as run_id, r.start_time, m.key, m.value "
            "from runs r join latest_metrics m on m.run_uuid = r.run_uuid "
            "where (r.name like 'E12-%' or r.name like 'E18-%') and r.status = 'FINISHED' "
            "and m.key in (?, ?, ?)",
            con, params=METRICS,
        )  # fmt: skip
    newest = rows.groupby("run_name")["start_time"].transform("max") == rows["start_time"]
    wide = rows[newest].pivot(index=["run_name", "run_id"], columns="key", values="value")
    return wide.reindex(columns=METRICS).reset_index()


def same_labels(runner: str, mine: pd.DataFrame, off: pd.DataFrame) -> dict:
    """{run name: True if your saved test predictions give the official label on every row}."""
    out = {}
    with tarfile.open(ARCHIVE) as tar:
        for _, o in off.iterrows():
            m = mine[mine["run_name"] == o["run_name"]]
            path = ROOT / "mlruns" / runner / "artifacts"
            if m.empty or not (path / m["run_id"].iloc[0] / "artifacts" / PRED).exists():
                out[o["run_name"]] = False
                continue
            ref = pd.read_csv(tar.extractfile(f"{o['run_id']}/{PRED}"))
            got = pd.read_csv(path / m["run_id"].iloc[0] / "artifacts" / PRED)
            ref, got = (d.sort_values("raw_row", ignore_index=True) for d in (ref, got))
            out[o["run_name"]] = bool(
                ref["raw_row"].equals(got["raw_row"])
                and (ref["predicted"].astype(str) == got["predicted"].astype(str)).all()
            )
    return out


def compare(mine: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    """One row per official run: the scores, and whether they are equal exactly / to 4 decimals
    (a missing run counts as not equal; a score both sides lack, such as the ECE of a model
    without probabilities, counts as equal)."""
    m = off.drop(columns="run_id").merge(
        mine.drop(columns="run_id"), on="run_name", how="left", suffixes=("_official", "")
    )
    exact = pd.Series(True, index=m.index)
    rounded = pd.Series(True, index=m.index)
    for k in METRICS:
        both_missing = m[k].isna() & m[f"{k}_official"].isna()
        exact &= (m[k] == m[f"{k}_official"]) | both_missing
        rounded &= (m[k].round(4) == m[f"{k}_official"].round(4)) | both_missing
    present = m["run_name"].isin(mine["run_name"])
    return m.assign(exact=exact & present, rounded=rounded & present)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    runner = ap.parse_args().runner
    mine, off = yours(runner), official()
    out = compare(mine, off)
    out["same labels"] = out["run_name"].map(same_labels(runner, mine, off))
    with pd.option_context("display.width", 220, "display.max_rows", 40):
        print(out.round(6).to_string(index=False))
    n = len(out)
    print(f"\n{int(out['exact'].sum())} of {n} runs identical to the official scores; "
          f"{int(out['rounded'].sum())} of {n} equal to 4 decimals; "
          f"{int(out['same labels'].sum())} of {n} with the same label on every test "
          "row.")  # fmt: skip
    return 0 if (out["rounded"] & out["same labels"]).all() else 1


if __name__ == "__main__":
    sys.exit(main())
