"""Build one cleaned data version from the raw ARFF files.

Usage:
  uv run python scripts/prepare_data.py --version v1      # build data/processed/v1/
  uv run python scripts/prepare_data.py --export-csv v1   # rewrite data/processed/csv/ from v1

Writes, per task x timeout: <task>_<timeout>s.parquet and splits/<task>_<timeout>s.json,
plus cleaning_log.csv and a manifest.json copy, into data/processed/<version>/ (never
overwriting an existing version), data/manifest.json, and CSV copies for viewing in
data/processed/csv/. Rerunning gives byte-identical output.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from blindspy import checks, splits
from blindspy.data import (
    COLUMN_MAP,
    CORE_TASKS,
    FEATURES,
    HEADERS,
    LABELS,
    RAW,
    ROOT,
    TASK_FILES,
    TIMEOUTS,
    header_kind,
    load_arff,
    load_raw,
    processed_dir,
    raw_path,
)

SOURCE = "https://www.unb.ca/cic/datasets/vpn.html"
KAGGLE = ROOT / "datasets" / "kaggle" / "consolidated_traffic_data.csv"
LOG_COLUMNS = [
    "step", "step_name", "file", "task", "timeout", "rows_before", "rows_after",
    "rows_removed", "rows_changed", "rows_affected", "removed_per_class",
    "rows_after_per_class", "rule", "reason",
]  # fmt: skip
# Columns that are means, stds or rates, i.e. the ones the source rounded to 3 sig. figures.
_COMPUTED = [c for c in HEADERS["S"] if c.startswith(("mean_", "std_")) or c.endswith("PerSecond")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(labels: pd.Series) -> dict:
    return {k: int(v) for k, v in sorted(labels.value_counts().items())}


class Log:
    """Collects cleaning_log.csv rows for one file and checks they chain up."""

    def __init__(self, rules: dict, task: str, timeout: int, file: str):
        self.rules, self.task, self.timeout, self.file = rules, task, timeout, file
        self.rows: list[dict] = []

    def add(self, name: str, before: pd.DataFrame, after: pd.DataFrame, changed=0, affected=0):
        removed = _counts(before["label"])
        for label, n in _counts(after["label"]).items():
            removed[label] -= n
        self.rows.append({
            "step": len(self.rows) + 1, "step_name": name, "file": self.file,
            "task": self.task, "timeout": self.timeout,
            "rows_before": len(before), "rows_after": len(after),
            "rows_removed": len(before) - len(after), "rows_changed": int(changed),
            "rows_affected": int(affected),
            "removed_per_class": json.dumps({k: v for k, v in removed.items() if v}),
            "rows_after_per_class": json.dumps(_counts(after["label"])),
            "rule": self.rules[name]["rule"], "reason": self.rules[name]["reason"],
        })  # fmt: skip


def clean_file(task: str, timeout: int, rules: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """Apply every cleaning step to one task file; return data, log rows, extra facts."""
    path = raw_path(task, timeout)
    log = Log(rules, task, timeout, str(path.relative_to(ROOT)))
    body = path.read_text().split("@DATA", 1)[1].splitlines()[1:]
    raw_lines = [line for line in body if line.strip()]
    parsed = load_arff(path)
    parsed.index.name = "raw_row"
    # Rows before parsing: every non-empty data line, labelled for the per-class count.
    labels = [line.rsplit(",", 1)[-1].strip() or "(comma-only line)" for line in raw_lines]
    before = pd.DataFrame({"label": labels})
    log.add("parse", before, parsed, affected=parsed.attrs["blank_lines_dropped"])

    family = TASK_FILES[task][0]
    mapping = COLUMN_MAP[(family, header_kind(list(parsed.columns[:-1])))]
    df = parsed.rename(columns=mapping)
    renamed = any(a != b for a, b in mapping.items())
    log.add("column_map", parsed, df, affected=len(df) if renamed else 0)

    changed = 0
    if task == "A1" and timeout == 120:
        donor = load_raw("B", 120)
        others = [c for c in df.columns[:-1] if c != "flowPktsPerSecond"]
        if not np.array_equal(df[others].to_numpy(), donor[others].to_numpy()):
            raise SystemExit("A1 120 s and B 120 s no longer agree outside flowPktsPerSecond")
        changed = int(
            (df["flowPktsPerSecond"].to_numpy() != donor["flowPktsPerSecond"].to_numpy()).sum()
        )
        df["flowPktsPerSecond"] = donor["flowPktsPerSecond"].to_numpy()
    log.add("restore_flow_pkts", df, df, changed=changed, affected=changed)

    features = list(df.columns[:-1])
    x = df[features]
    for name, mask in [
        ("missing", x.isna().any(axis=1)),
        ("infinite", np.isinf(x).any(axis=1)),
        ("fiat_not_measurable", df["min_fiat"] == -1),
        ("biat_not_measurable", df["min_biat"] == -1),
        ("flowiat_not_measurable", df["max_flowiat"] == -1),
        ("active_idle_not_measurable", (df["min_active"] == -1) | (df["min_idle"] == -1)),
        ("single_packet", df["duration"] == 0),
        ("negative_flowiat", (df["min_flowiat"] < 0) & (df["max_flowiat"] != -1)),
        ("rounding", (x[[c for c in features if c in _COMPUTED]].abs() >= 1e7).any(axis=1)),
        ("packet_size", (df["flowPktsPerSecond"] > 0)
            & ~(df["flowBytesPerSecond"] / df["flowPktsPerSecond"]).between(40, 1500)),
    ]:  # fmt: skip
        if name in ("missing", "infinite") and mask.any():
            raise SystemExit(f"{task} {timeout}s: {int(mask.sum())} rows with {name} values")
        log.add(name, df, df, affected=int(mask.sum()))

    deduped = df[~df.duplicated(subset=[*features, "label"], keep="first")]
    log.add("exact_duplicates", df, deduped, affected=len(df) - len(deduped))

    groups = deduped.groupby(features)["label"].transform("nunique")
    conflicting = deduped[groups > 1]
    log.add("conflicting_duplicates", deduped, deduped, affected=len(conflicting))
    for name in ["no_normalization", "no_feature_selection", "no_outlier_removal", "no_resampling"]:
        log.add(name, deduped, deduped)

    unknown = set(deduped["label"]) - set(LABELS[task])
    if unknown:
        raise SystemExit(f"{task} {timeout}s: unexpected labels {unknown}")
    removed = _counts(df["label"])
    for label, n in _counts(deduped["label"]).items():
        removed[label] -= n
    facts = {
        "exact_duplicates_removed_per_class": {k: v for k, v in removed.items() if v},
        "conflicting_rows_per_class": _counts(conflicting["label"]),
        "conflicting_groups": int(conflicting.groupby(features).ngroups),
    }
    return deduped, log.rows, facts


def odd_values(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns[:-1]:
        v = df[c]
        out[c] = {
            "negative": int((v < 0).sum()), "exactly_-1": int((v == -1).sum()),
            "zero": int((v == 0).sum()), "missing": int(v.isna().sum()),
            "infinite": int(np.isinf(v).sum()), "min": float(v.min()), "max": float(v.max()),
        }  # fmt: skip
    return out


def describe(df: pd.DataFrame, rows: list[dict], facts: dict, split: dict) -> dict:
    """Manifest facts for one cleaned file: summary table row, splits, duplicates, checks."""
    parts = {p: df.loc[split[p]] for p in ("train", "val", "test")}
    features = list(df.columns[:-1])

    def row_set(part: pd.DataFrame, with_label: bool) -> list:
        rows_ = list(map(tuple, part[features].to_numpy()))
        return list(zip(rows_, part["label"], strict=True)) if with_label else rows_

    train_x, train_xy = set(row_set(parts["train"], False)), set(row_set(parts["train"], True))
    counts = df["label"].value_counts()
    removed = sum(r["rows_removed"] for r in rows if r["step_name"] == "exact_duplicates")
    return {
        "features": features,
        "summary": {
            "initial_rows": rows[0]["rows_before"],
            "comma_only_lines_skipped": rows[0]["rows_removed"],
            "exact_duplicates_removed": removed,
            "conflicting_rows_kept": sum(facts["conflicting_rows_per_class"].values()),
            "conflicting_groups": facts["conflicting_groups"],
            "final_rows": rows[-1]["rows_after"],
            "missing_values": 0,
            "features": len(features),
            "classes": int(df["label"].nunique()),
            "largest_to_smallest_class": f"{counts.max() / counts.min():.2f}:1",
            **{p: len(parts[p]) for p in parts},
        },
        "class_counts": {p: _counts(parts[p]["label"]) for p in parts},
        "final_class_counts": _counts(df["label"]),
        "exact_duplicates_removed_per_class": facts["exact_duplicates_removed_per_class"],
        "conflicting_rows_per_class": facts["conflicting_rows_per_class"],
        "exact_duplicates_across_train_test": sum(
            r in train_xy for r in row_set(parts["test"], True)
        ),
        "test_rows_with_conflicting_twin_in_train": sum(
            r in train_x for r in row_set(parts["test"], False)
        ),
        "odd_values": odd_values(df),
        "consistency_checks": checks.consistency(df),
    }


def inventory() -> list[dict]:
    files = []
    for task, (family, _, _) in TASK_FILES.items():
        for timeout in TIMEOUTS:
            path = raw_path(task, timeout)
            df = load_arff(path)
            files.append({
                "file": str(path.relative_to(ROOT)), "task": task, "family": family,
                "timeout": timeout, "core_task": task in CORE_TASKS,
                "header": header_kind(list(df.columns[:-1])), "rows": len(df),
                "comma_only_lines_skipped": df.attrs["blank_lines_dropped"],
                "label_counts": _counts(df["label"]), "sha256": sha256(path),
            })  # fmt: skip
    return files


def build(version: str, root: Path = ROOT, with_kaggle: bool = True) -> Path:
    out = processed_dir(version, root)
    if out.exists():
        raise SystemExit(
            f"{out} already exists. Data versions are never overwritten: "
            "bump data_version (configs/base.yaml) and build a new one."
        )
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    rules, split_cfg, seed = base["cleaning"], base["split"], base["seed"]
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    log_rows, tasks = [], {}
    for task in CORE_TASKS:
        for timeout in TIMEOUTS:
            df, rows, facts = clean_file(task, timeout, rules)
            log_rows += rows
            name = f"{task}_{timeout}s"
            parquet = tmp / f"{name}.parquet"
            df.to_parquet(parquet, engine="pyarrow", compression="zstd")
            split = splits.make_split(
                df, split_cfg["train"], split_cfg["val"], split_cfg["test"], seed
            )
            split_file = splits.save_split(split, tmp, task, timeout)
            tasks[name] = {
                "task": task,
                "timeout": timeout,
                "parquet": parquet.name,
                "parquet_sha256": sha256(parquet),
                "split_sha256": sha256(split_file),
                **describe(df, rows, facts, split),
            }

    log = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    log.to_csv(tmp / "cleaning_log.csv", index=False)

    manifest = {
        "data_version": version,
        "built_by": "scripts/prepare_data.py",
        "source": SOURCE,
        "seed": seed,
        "split": split_cfg,
        "features_per_family": FEATURES,
        "column_map": {
            f"{family}/{kind}": {a: b for a, b in m.items() if a != b}
            for (family, kind), m in COLUMN_MAP.items()
        },
        "header_names": HEADERS,
        "label_revealing_columns": "none: each file holds only 23 numeric timing features + label",
        "session_information": False,
        "session_information_reason": "no IDs, IPs, ports, timestamps or capture-file names",
        "cleaning_rules": rules,
        "files": inventory(),
        "tasks": tasks,
        "cleaning_log_sha256": sha256(tmp / "cleaning_log.csv"),
        "column_name_check": {
            f"{task}_{t}s": checks.name_hypotheses(task, t) for task in CORE_TASKS for t in TIMEOUTS
        },
        "family_match": {f"{t}s": checks.family_match(t) for t in TIMEOUTS},
        "timeouts": timeout_table(tasks),
    }
    if with_kaggle:
        manifest["kaggle"] = {
            "file": str(KAGGLE.relative_to(ROOT)),
            **checks.kaggle_compare(KAGGLE),
        }
    text = json.dumps(manifest, indent=1, allow_nan=False) + "\n"
    (tmp / "manifest.json").write_text(text)
    tmp.rename(out)
    if root == ROOT:
        (ROOT / "data" / "manifest.json").write_text(text)
        export_csv(version, root)
    return out


def timeout_table(tasks: dict) -> dict:
    """Per task: rows and class mix (%) at each timeout, before and after cleaning."""
    table = {}
    for task in CORE_TASKS:
        table[task] = {}
        for t in TIMEOUTS:
            info = tasks[f"{task}_{t}s"]
            final = info["final_class_counts"]
            total = sum(final.values())
            table[task][f"{t}s"] = {
                "initial_rows": info["summary"]["initial_rows"],
                "final_rows": total,
                "class_mix_pct": {k: round(v / total * 100, 2) for k, v in final.items()},
                "features_in_order": info["features"],
                "units": "times in microseconds, rates per second",
            }
    return table


def export_csv(version: str, root: Path = ROOT) -> None:
    """Rewrite data/processed/csv/ from one version (for viewing only; code reads Parquet)."""
    csv_dir = root / "data" / "processed" / "csv"
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True)
    for parquet in sorted(processed_dir(version, root).glob("*.parquet")):
        pd.read_parquet(parquet).to_csv(csv_dir / f"{parquet.stem}.csv")
    (csv_dir / "VERSION.txt").write_text(f"{version}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="data version to build, e.g. v1")
    group.add_argument(
        "--export-csv", metavar="VERSION", help="rewrite the CSV copies from a version"
    )
    args = parser.parse_args(argv)
    if not RAW.exists():
        sys.exit(f"{RAW} not found: put the dataset there first (see README).")
    if args.export_csv:
        export_csv(args.export_csv)
        return
    out = build(args.version)
    print(f"Built {out.relative_to(ROOT)}; manifest written to data/manifest.json.")


if __name__ == "__main__":
    main()
