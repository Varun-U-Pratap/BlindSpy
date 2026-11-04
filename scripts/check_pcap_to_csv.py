"""Check scripts/pcap_to_csv.py against the dataset's own captures.

Converts each capture in datasets/PCAPs/ up to --max-mb with the converter and looks for every
flow in the dataset's ARFF file of the same timeout: A2 (VPN or NO-VPN by folder, total_* layout)
and B (std_* layout). A flow counts as reproduced when one row of the file has the same duration
and all 23 values agree (relative 1e-6, or 5e-3 for values >= 1e7, which the source files store
to 3 significant figures). Flows of 1-2 packets are counted apart: many identical ones exist.

    uv run python scripts/check_pcap_to_csv.py --output reports/pcap_to_csv_check.txt
"""

import argparse
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from blindspy.data import ROOT, load_raw

conv = runpy.run_path(str(ROOT / "scripts" / "pcap_to_csv.py"))
PCAPS = ROOT / "datasets" / "PCAPs"


def close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    big = np.maximum(np.abs(a), np.abs(b)) >= 1e7
    rtol = np.where(big, 5e-3, 1e-6)
    return np.abs(a - b) <= rtol * np.maximum(np.abs(a), np.abs(b)) + 1e-6


def match(ours: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each of our flows: (a row has the same duration, a row agrees on all columns)."""
    cols = list(ours.columns)
    by_dur = {d: g[cols].to_numpy(float) for d, g in rows.groupby("duration")}
    seen = np.zeros(len(ours), bool)
    hit = np.zeros(len(ours), bool)
    for i, v in enumerate(ours[cols].to_numpy(float)):
        cand = by_dur.get(v[0])
        if cand is not None:
            seen[i] = True
            hit[i] = close(cand, v[None, :]).all(axis=1).any()
    return seen, hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=30)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(p for p in PCAPS.rglob("*") if p.suffix in (".pcap", ".pcapng")
                   and p.stat().st_size <= a.max_mb * 1e6)  # fmt: skip
    if not files:
        print(f"no captures in {PCAPS}", file=sys.stderr)
        return 2
    ref = {t: load_raw(t, a.timeout) for t in ["A2-VPN", "A2-NonVPN", "B"]}
    lines = []
    tot = {"flows": 0, "big": 0, "seen": 0, "a2": 0, "b": 0, "a2_small": 0, "small": 0}
    t0 = time.time()
    for p in files:
        a2 = "A2-VPN" if p.parent.name.upper().startswith("VPN") else "A2-NonVPN"
        flows = conv["split_flows"](conv["read_packets"](p), a.timeout)
        feats = pd.DataFrame([conv["flow_features"](f) for f in flows])
        n = np.array([len(f) for f in flows])
        seen, h2 = match(conv["features_for_task"](feats, a2), ref[a2])
        _, hb = match(conv["features_for_task"](feats, "B"), ref["B"])
        big = n >= 3
        tot["flows"] += len(n)
        tot["big"] += int(big.sum())
        tot["seen"] += int(seen[big].sum())
        tot["a2"] += int(h2[big].sum())
        tot["b"] += int(hb[big].sum())
        tot["small"] += int((~big).sum())
        tot["a2_small"] += int(h2[~big].sum())
        counts = [len(n), big.sum(), seen[big].sum(), h2[big].sum(), hb[big].sum()]
        lines.append("\t".join([str(p.relative_to(ROOT)), a2, *(str(int(c)) for c in counts)]))
    head = ["capture\tA2 file\tflows\tflows >= 3 packets\tduration in A2\tfound in A2"
            "\tfound in B"]  # fmt: skip
    big = max(tot["big"], 1)
    summary = [
        f"Captures: {len(files)} (<= {a.max_mb:g} MB) of datasets/PCAPs/, timeout {a.timeout} s, "
        f"{time.time() - t0:.0f} s",
        f"Flows: {tot['flows']}; with >= 3 packets: {tot['big']}",
        f"Flows >= 3 packets whose duration occurs in the A2 file: {tot['seen']}",
        f"Of those, reproduced exactly (all 23 values): A2 layout {tot['a2']} "
        f"({tot['a2'] / max(tot['seen'], 1):.1%}), B layout {tot['b']} "
        f"({tot['b'] / max(tot['seen'], 1):.1%})",
        f"Of all flows >= 3 packets: A2 {tot['a2'] / big:.1%}, B {tot['b'] / big:.1%} (the rest"
        " have no row with that duration in the file: see the per-capture table)",
        f"Flows of 1-2 packets found in A2: {tot['a2_small']} of {tot['small']}",
    ]
    a.output.write_text("\n".join(summary + ["", *head, *lines]) + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
