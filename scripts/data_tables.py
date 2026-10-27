"""Print the report's Section 2 tables (markdown) from a data version's manifest and log.

Usage: uv run python scripts/data_tables.py v1 > /tmp/tables.md
The report draft pastes these tables unchanged, so no dataset number is typed by hand.
"""

import json
import sys

import pandas as pd

from blindspy.data import CORE_TASKS, LABELS, TIMEOUTS, processed_dir

ODD_STEPS = [
    "fiat_not_measurable", "biat_not_measurable", "flowiat_not_measurable",
    "active_idle_not_measurable", "single_packet", "negative_flowiat", "rounding", "packet_size",
]  # fmt: skip


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append(
            "| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


def main(version: str) -> None:
    folder = processed_dir(version)
    m = json.loads((folder / "manifest.json").read_text())
    log = pd.read_csv(folder / "cleaning_log.csv")
    files = [(t, s) for t in CORE_TASKS for s in TIMEOUTS]

    print("### Inventory of the 20 ARFF files\n")
    inv = pd.DataFrame(m["files"])
    inv["file"] = inv["file"].str.replace("datasets/CSVs/", "", regex=False)
    inv["classes"] = inv["label_counts"].map(len)
    print(md(inv[["file", "task", "timeout", "header", "rows", "classes", "core_task"]]))

    print("\n### Column-name check (% of applicable rows passing, 15 s files)\n")
    rows = []
    for task in CORE_TASKS:
        for kind, checks in m["column_name_check"][f"{task}_15s"].items():
            for name, pct in checks.items():
                rows.append({"file": f"{task} 15 s", "names": kind, "check": name, "pct": pct})
    table = pd.DataFrame(rows).pivot_table(
        index="check", columns=["file", "names"], values="pct", aggfunc="first", sort=False
    )
    table.columns = [f"{f} ({n})" for f, n in table.columns]
    print(md(table.reset_index().astype(object).fillna("-")))

    print("\n### Dataset summary per task and timeout\n")
    summary = pd.DataFrame(
        [{"file": f"{t} {s} s", **m["tasks"][f"{t}_{s}s"]["summary"]} for t, s in files]
    )
    print(md(summary))

    print("\n### Data funnel per file (rows after each step that removes rows)\n")
    funnel = []
    for t, s in files:
        steps = log[(log.task == t) & (log.timeout == s)].set_index("step_name")
        info = m["tasks"][f"{t}_{s}s"]["summary"]
        funnel.append({
            "file": f"{t} {s} s", "raw data lines": int(steps.loc["parse", "rows_before"]),
            "parsed": int(steps.loc["parse", "rows_after"]),
            "after duplicates": int(steps.loc["exact_duplicates", "rows_after"]),
            "final": info["final_rows"], "train": info["train"], "val": info["val"],
            "test": info["test"],
        })  # fmt: skip
    print(md(pd.DataFrame(funnel)))

    print("\n### Exact duplicates removed per class\n")
    for task in CORE_TASKS:
        per = (
            pd.DataFrame(
                {
                    f"{s} s": m["tasks"][f"{task}_{s}s"]["exact_duplicates_removed_per_class"]
                    for s in TIMEOUTS
                }
            )
            .reindex(LABELS[task])
            .fillna(0)
            .astype(int)
        )
        raw = (
            pd.DataFrame(
                {
                    f"{s} s": json.loads(
                        log[(log.task == task) & (log.timeout == s) & (log.step_name == "parse")][
                            "rows_after_per_class"
                        ].iloc[0]
                    )
                    for s in TIMEOUTS
                }
            )
            .reindex(LABELS[task])
            .fillna(0)
            .astype(int)
        )
        cells = per.map(lambda v: f"{v:,}") + " of " + raw.map(lambda v: f"{v:,}")
        print(f"\n{task}\n")
        print(md(cells.reset_index(names="class")))

    print("\n### Rows each kept-as-is rule applies to (summed over the 4 timeouts)\n")
    odd = (
        log[log.step_name.isin(ODD_STEPS)]
        .pivot_table(index="step_name", columns="task", values="rows_affected", aggfunc="sum")
        .reindex(ODD_STEPS)[CORE_TASKS]
        .astype(int)
    )
    print(md(odd.reset_index(names="rule")))

    print("\n### Conflicting duplicates kept (rows, per file)\n")
    print(md(pd.DataFrame([
        {"file": f"{t} {s} s", "groups": m["tasks"][f"{t}_{s}s"]["summary"]["conflicting_groups"],
         "rows": m["tasks"][f"{t}_{s}s"]["summary"]["conflicting_rows_kept"],
         "test rows with a conflicting twin in train":
             m["tasks"][f"{t}_{s}s"]["test_rows_with_conflicting_twin_in_train"]}
        for t, s in files
    ])))  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
