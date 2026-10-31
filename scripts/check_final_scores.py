"""Compare the test scores of your own E12 reruns with the official ones.

After rerunning the 30 E12 configs under your own runner name (configs/local.yaml), this reads
your MLflow store (mlruns/<runner>/mlflow.db; the newest run per name) and compares each run's
test macro F1 and test accuracy with the official values in reports/history.csv (committed,
full precision). It prints one line per run and exits 0 only if all 30 runs are present and
agree to 4 decimals (the precision of the report's tables).

Usage: uv run python scripts/check_final_scores.py --runner <your runner>
"""

import argparse
import sqlite3
import sys

import pandas as pd

from blindspy.data import ROOT

METRICS = ["test_f1_macro", "test_accuracy"]


def official() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "reports" / "history.csv", low_memory=False)
    h = h[h["tags.experiment_id"].astype(str) == "E12"]
    h = h.sort_values("start_time").drop_duplicates("run_name", keep="last")
    out = h[["run_name"] + [f"metrics.{m}" for m in METRICS]]
    return out.set_axis(["run_name", *METRICS], axis=1).sort_values("run_name", ignore_index=True)


def yours(runner: str) -> pd.DataFrame:
    db = ROOT / "mlruns" / runner / "mlflow.db"
    if not db.exists():
        sys.exit(f"No MLflow store at {db.relative_to(ROOT)}; run the E12 configs first.")
    with sqlite3.connect(db) as con:
        rows = pd.read_sql(
            "select r.name as run_name, r.start_time, m.key, m.value from runs r "
            "join latest_metrics m on m.run_uuid = r.run_uuid "
            "where r.name like 'E12-%' and r.status = 'FINISHED' and m.key in (?, ?)",
            con, params=METRICS,
        )  # fmt: skip
    newest = rows.groupby("run_name")["start_time"].transform("max") == rows["start_time"]
    wide = rows[newest].pivot(index="run_name", columns="key", values="value")
    return wide.reindex(columns=METRICS).reset_index()


def compare(mine: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    """One row per official run: both scores, and whether they are equal exactly / to 4
    decimals (a missing run counts as not equal)."""
    m = off.merge(mine, on="run_name", how="left", suffixes=("_official", ""))
    exact = pd.Series(True, index=m.index)
    rounded = pd.Series(True, index=m.index)
    for k in METRICS:
        exact &= m[k] == m[f"{k}_official"]
        rounded &= m[k].round(4) == m[f"{k}_official"].round(4)
    return m.assign(exact=exact, rounded=rounded)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    runner = ap.parse_args().runner
    out = compare(yours(runner), official())
    with pd.option_context("display.width", 200, "display.max_rows", 40):
        print(out.round(6).to_string(index=False))
    print(f"\n{int(out['exact'].sum())} of {len(out)} runs identical to the official scores; "
          f"{int(out['rounded'].sum())} of {len(out)} equal to 4 decimals.")  # fmt: skip
    return 0 if out["rounded"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
