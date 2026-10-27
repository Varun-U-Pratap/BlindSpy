"""Extra E00 evidence for a data version, written to reports/data_checks_<version>.json.

Adds what data/manifest.json does not hold: each feature's value range checked against
the paper's Table 2 definitions (with the units shown by the packet-count identity),
and the published-header comparison of A1 and A2 rows (exact and value-set matches).
It reads the version's Parquet files and the raw ARFF files and changes neither.

Usage: uv run python scripts/data_checks.py v1
"""

import json
import sys

import pandas as pd

from blindspy.checks import ROUNDING
from blindspy.data import (
    CORE_TASKS,
    HEADERS,
    ROOT,
    TIMEOUTS,
    load_arff,
    load_raw,
    processed_dir,
    raw_path,
)

# Paper Table 2, in its order: duration, fiat/biat/flowiat/active/idle x (mean, min, max,
# std), flow bytes per second, flow packets per second.
GROUPS = ["fiat", "biat", "flowiat", "active", "idle"]
TABLE2 = (
    ["duration"]
    + [f"{s}_{g}" for g in GROUPS for s in ["mean", "min", "max", "std"]]
    + ["flowBytesPerSecond", "flowPktsPerSecond"]
)
# Every gap, active or idle statistic is a time inside the flow, so it can't exceed duration.
WITHIN_DURATION = [f"{s}_{g}" for g in GROUPS for s in ["min", "mean", "max"]] + [
    "total_fiat",
    "total_biat",
]


def ranges(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns[:-1]:
        v = df[c]
        measured = v[v >= 0]
        q = measured.quantile([0.01, 0.5, 0.99])
        row = {
            "min": float(v.min()),
            "p01": float(q[0.01]),
            "median": float(q[0.5]),
            "p99": float(q[0.99]),
            "max": float(v.max()),
            "exactly_-1": int((v == -1).sum()),
            "other_negative": int(((v < 0) & (v != -1)).sum()),
        }
        if c in WITHIN_DURATION:
            is_measured = (v >= 0).to_numpy()
            ok = v.to_numpy() <= df["duration"].to_numpy() * (1 + ROUNDING)
            row["pct_measured_values_<=_duration"] = round(float(ok[is_measured].mean() * 100), 2)
        out[c] = row
    return out


def published_header_matches(timeout: int) -> dict:
    """A1 rows vs A2 rows as published (names ignored): exact and as unordered value sets."""
    a1 = load_arff(raw_path("A1", timeout))
    a2 = pd.concat([load_arff(raw_path(t, timeout)) for t in ["A2-NonVPN", "A2-VPN"]])
    x1, x2 = a1.iloc[:, :23].to_numpy(), a2.iloc[:, :23].to_numpy()
    exact = set(map(tuple, x2))
    sets = {tuple(sorted(r)) for r in x2}
    return {
        "A1 rows": len(a1),
        "A1 rows equal to an A2 row in all 23 values": int(sum(tuple(r) in exact for r in x1)),
        "A1 rows equal to an A2 row as an unordered set of values": int(
            sum(tuple(sorted(r)) in sets for r in x1)
        ),
        "A1 unique rows (features + label)": int(len(a1.drop_duplicates())),
        "A1 unique feature rows (label ignored)": int(len(a1.iloc[:, :23].drop_duplicates())),
    }


def cross_timeout_labels() -> dict:
    """Flows found (all 23 values equal) in more than one A2 timeout file: same label everywhere?"""
    parts = []
    for timeout in TIMEOUTS:
        for task in ["A2-NonVPN", "A2-VPN"]:
            part = load_raw(task, timeout)
            part["timeout"] = timeout
            parts.append(part)
    a2 = pd.concat(parts, ignore_index=True)
    a2["key"] = a2[HEADERS["T"]].map(lambda v: f"{v:.12g}").agg("|".join, axis=1)
    per_flow = a2.groupby("key").agg(
        timeouts=("timeout", "nunique"), labels=("label", lambda s: "/".join(sorted(set(s))))
    )
    multi = per_flow[per_flow["timeouts"] > 1]
    changed = multi[multi["labels"].str.contains("/")]
    packets = a2.drop_duplicates("key").set_index("key").loc[changed.index]
    n_pkts = packets["flowPktsPerSecond"] * packets["duration"] / 1e6
    return {
        "flows found in more than one timeout file": len(multi),
        "of these, flows whose label differs between timeouts": len(changed),
        "median packets in those flows": float(n_pkts.median().round(1)),
        "most common label sets": changed["labels"].value_counts().head(10).to_dict(),
    }


def class_profile(version: str) -> dict:
    """Per timeout: medians and shares that show how VOIP and FT flows differ at 60 s."""
    out = {}
    for task, labels in [("A2-NonVPN", ["VOIP", "FT"]), ("A2-VPN", ["VPN-VOIP", "VPN-FT"])]:
        for label in labels:
            for timeout in TIMEOUTS:
                df = pd.read_parquet(processed_dir(version) / f"{task}_{timeout}s.parquet")
                d = df[df["label"] == label]
                gap = d["mean_fiat"]
                out[f"{label} {timeout}s"] = {
                    "flows": len(d),
                    "median packets per second": round(float(d["flowPktsPerSecond"].median()), 1),
                    "median mean_fiat (us)": round(float(gap.median()), 1),
                    "% mean_fiat 15-25 ms": round(float(gap.between(15e3, 25e3).mean() * 100), 1),
                    "% no active/idle period": round(
                        float((d["min_active"] == -1).mean() * 100), 1
                    ),
                }
    return out


def main(version: str) -> None:
    folder = processed_dir(version)
    manifest = json.loads((folder / "manifest.json").read_text())
    data = {
        (t, s): pd.read_parquet(folder / f"{t}_{s}s.parquet") for t in CORE_TASKS for s in TIMEOUTS
    }
    units = {
        f"{t}_{s}s": manifest["tasks"][f"{t}_{s}s"]["consistency_checks"][
            "mean_flowiat x (packets - 1) = duration (+-1%)"
        ]
        for t, s in data
    }
    out = {
        "data_version": version,
        "table2_features": TABLE2,
        "A1/B feature list equals Table 2 (as a set)": set(HEADERS["S"]) == set(TABLE2),
        "A2 feature list equals Table 2 (as a set)": set(HEADERS["T"]) == set(TABLE2),
        "A2 features not in Table 2": sorted(set(HEADERS["T"]) - set(TABLE2)),
        "units": (
            "duration and all gap/active/idle features in microseconds, rates per second: "
            "packets = flowPktsPerSecond x duration / 1e6 makes mean_flowiat x (packets - 1) "
            "equal duration within 1% on the % of rows below, per file"
        ),
        "units_identity_pct_per_file": units,
        "feature_ranges": {
            "A1/B layout (B files, all timeouts)": ranges(
                pd.concat([data[("B", s)] for s in TIMEOUTS])
            ),
            "A2 layout (A2-VPN + A2-NonVPN, all timeouts)": ranges(
                pd.concat([data[(t, s)] for t in ["A2-VPN", "A2-NonVPN"] for s in TIMEOUTS])
            ),
        },
        "published_header_matches_A1_vs_A2": {
            f"{s}s": published_header_matches(s) for s in TIMEOUTS
        },
        "cross_timeout_label_consistency_A2": cross_timeout_labels(),
        "voip_ft_profile_per_timeout": class_profile(version),
    }
    path = ROOT / "reports" / f"data_checks_{version}.json"
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
