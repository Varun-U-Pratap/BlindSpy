"""Evidence for the column map and the file-family decision (E00).

Every number here is written into data/manifest.json by prepare_data.py, so the
report quotes the manifest, never hand-typed values.
"""

import numpy as np
import pandas as pd

from blindspy.data import HEADERS, TIMEOUTS, load_arff, load_raw, raw_path

# Means, stds and rates of 1e7 or more are stored to 3 significant figures in the
# source files, so a mean can sit up to 0.5% outside [min, max].
ROUNDING = 5e-3
GROUPS = ["fiat", "biat", "flowiat", "active", "idle"]
# Integer-valued columns present in both families; used to pair rows across files.
MATCH_KEY = [
    "duration",
    "min_fiat",
    "max_fiat",
    "min_biat",
    "max_biat",
    "min_flowiat",
    "max_flowiat",
]
SHARED = [c for c in HEADERS["S"] if c in HEADERS["T"]]


def _pct(ok: pd.Series, applicable: pd.Series) -> float | None:
    """% of applicable rows passing; None when no row is applicable."""
    n = int(applicable.sum())
    return round(float(ok[applicable].mean() * 100), 2) if n else None


def consistency(df: pd.DataFrame) -> dict[str, float]:
    """% of applicable rows passing each check, for columns named as in df."""
    out = {}
    for g in GROUPS:
        lo, mean, hi = df[f"min_{g}"], df[f"mean_{g}"], df[f"max_{g}"]
        measured = (lo >= 0) & (hi >= 0)  # -1 means "not measurable"
        ok = (lo * (1 - ROUNDING) <= mean) & (mean <= hi * (1 + ROUNDING))
        out[f"{g}: min <= mean <= max"] = _pct(ok, measured)
    n_pkts = df["flowPktsPerSecond"] * df["duration"] / 1e6
    ok = (df["mean_flowiat"] * (n_pkts - 1) / df["duration"]).between(0.99, 1.01)
    out["mean_flowiat x (packets - 1) = duration (+-1%)"] = _pct(ok, n_pkts > 2)
    size = df["flowBytesPerSecond"] / df["flowPktsPerSecond"]
    out["bytes per packet in 40-1500"] = _pct(size.between(40, 1500), df["flowPktsPerSecond"] > 0)
    measured = (df["min_fiat"] >= 0) & (df["min_biat"] >= 0)
    ok = df["min_flowiat"] <= np.minimum(df["min_fiat"], df["min_biat"])
    out["min_flowiat <= min(min_fiat, min_biat)"] = _pct(ok, measured)
    if "std_fiat" in df:
        for g in ["fiat", "biat"]:
            spread = df[f"max_{g}"] - df[f"min_{g}"]
            ok = (df[f"std_{g}"] >= 0) & (df[f"std_{g}"] <= spread * (1 + ROUNDING) + 1)
            out[f"0 <= std_{g} <= max - min"] = _pct(ok, df[f"min_{g}"] >= 0)
    if "total_fiat" in df:
        out["total_fiat <= duration"] = _pct(
            df["total_fiat"] <= df["duration"], df["duration"] >= 0
        )
        n_fwd = df["total_fiat"] / df["mean_fiat"] + 1
        n_bwd = df["total_biat"] / df["mean_biat"] + 1
        ok = ((n_fwd + n_bwd) / n_pkts).between(0.98, 1.02)
        out["total/mean + 1, fwd + bwd = packets (+-2%)"] = _pct(
            ok, (df["mean_fiat"] > 0) & (df["mean_biat"] > 0) & (n_pkts > 2)
        )
    return out


def name_hypotheses(task: str, timeout: int) -> dict[str, dict[str, float]]:
    """consistency() of one raw file read under each of the two header name lists.

    Under the "T" names it also tests the plan's packet-count reading of total_fiat
    (mean_fiat x (total_fiat - 1) = duration), which the data rules out.
    """
    df = load_arff(raw_path(task, timeout))
    values = df.drop(columns="label").to_numpy()
    out = {}
    for kind, names in HEADERS.items():
        named = pd.DataFrame(values, columns=names)
        out[kind] = consistency(named)
        if "total_fiat" in named:
            k, mean = named["total_fiat"], named["mean_fiat"]
            ok = (mean * (k - 1) / named["duration"]).between(0.9, 1.1)
            out[kind]["total_fiat as packet count (+-10%)"] = _pct(ok, (k > 1) & (mean > 0))
    return out


def _keys(df: pd.DataFrame) -> pd.Series:
    return df[MATCH_KEY].round(0).astype("int64").astype(str).agg("_".join, axis=1)


def family_match(timeout: int) -> dict:
    """Do the A1, B and A2 files at one timeout hold the same flows?"""
    a1, b = load_raw("A1", timeout), load_raw("B", timeout)
    a2 = pd.concat([load_raw("A2-NonVPN", timeout), load_raw("A2-VPN", timeout)], ignore_index=True)
    a2_vpn = np.where(a2["label"].str.startswith("VPN"), "VPN", "Non-VPN")
    k1, k2 = _keys(a1), _keys(a2)

    def label_mismatch_keys(left_keys, left_labels, right_keys, right_labels) -> int:
        left = pd.crosstab(left_keys.to_numpy(), np.asarray(left_labels))
        right = pd.crosstab(right_keys.to_numpy(), np.asarray(right_labels))
        left, right = left.align(right, fill_value=0)
        return int((left != right).any(axis=1).sum())

    unique1 = k1[~k1.duplicated(keep=False)]
    unique2 = k2[~k2.duplicated(keep=False)]
    pairs = pd.DataFrame({"k": unique1.values, "i1": unique1.index}).merge(
        pd.DataFrame({"k": unique2.values, "i2": unique2.index}), on="k"
    )
    left = a1.loc[pairs["i1"], SHARED].to_numpy()
    right = a2.loc[pairs["i2"], SHARED].to_numpy()
    agree = np.isclose(left, right, rtol=ROUNDING, atol=1e-6).mean(axis=0) * 100
    a1_vs_b = (a1.drop(columns="label").to_numpy() != b.drop(columns="label").to_numpy()).sum(0)
    return {
        "rows": {"A1": len(a1), "B": len(b), "A2 (VPN + NonVPN)": len(a2)},
        "A1 rows whose key is in A2 (%)": round(float(k1.isin(set(k2)).mean() * 100), 2),
        "keys where A1 labels differ from A2 VPN/Non-VPN labels": label_mismatch_keys(
            k1, a1["label"], k2, a2_vpn
        ),
        "keys where B labels differ from A2 labels": label_mismatch_keys(
            _keys(b), b["label"], k2, a2["label"]
        ),
        "rows paired on a unique key": len(pairs),
        "shared columns agreeing on paired rows (%)": {
            c: round(float(v), 2) for c, v in zip(SHARED, agree, strict=True)
        },
        "A1 vs B cells differing, per column": {
            c: int(n) for c, n in zip(a1.columns[:-1], a1_vs_b, strict=True) if n
        },
        "B labels differ from A1 labels mapped to VPN/Non-VPN": int(
            (np.where(b["label"].str.startswith("VPN"), "VPN", "Non-VPN") != a1["label"]).sum()
        ),
    }


def kaggle_compare(path) -> dict:
    """Is the Kaggle CSV the eight A2 files stacked? Compared as multisets of rows."""
    kaggle = pd.read_csv(path)
    features = HEADERS["T"]
    parts = []
    for timeout in TIMEOUTS:
        for task in ["A2-NonVPN", "A2-VPN"]:
            part = load_raw(task, timeout)
            part["timeout"] = timeout
            parts.append(part)
    a2 = pd.concat(parts, ignore_index=True)

    def row_keys(df: pd.DataFrame, labels: pd.Series) -> pd.Series:
        text = df[features].map(lambda v: f"{v:.12g}").agg("|".join, axis=1)
        return text + "#" + labels.astype(str).to_numpy()

    a2_keys = row_keys(a2, a2["label"])
    kk = row_keys(kaggle, kaggle.iloc[:, -1]).value_counts()
    ka = a2_keys.value_counts()
    kk, ka = kk.align(ka, fill_value=0)
    timeouts_per_row = pd.DataFrame({"k": a2_keys, "t": a2["timeout"]}).groupby("k")["t"].nunique()
    a2["file"] = a2["timeout"].astype(str) + a2["label"].str.startswith("VPN").astype(str)
    within_file_dups = int(a2.duplicated(subset=[*features, "label", "file"]).sum())
    return {
        "rows": len(kaggle),
        "header equals the A2 header": list(kaggle.columns[:-1]) == features,
        "unique rows (features + label)": int(len(kaggle.drop_duplicates())),
        "unique feature rows (label ignored)": int(len(kaggle.iloc[:, :-1].drop_duplicates())),
        "A2 rows stacked": len(a2),
        "same multiset of rows as the 8 A2 files (12 significant digits)": bool((kk == ka).all()),
        "Kaggle rows without an A2 partner": int((kk - ka).clip(lower=0).sum()),
        "distinct rows by number of timeout files they appear in": {
            str(n): int((timeouts_per_row == n).sum()) for n in range(1, 5)
        },
        "exact duplicates within single A2 files (sum of 8)": within_file_dups,
    }
