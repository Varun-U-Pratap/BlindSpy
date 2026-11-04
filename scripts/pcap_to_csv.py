"""Turn a packet capture into a CSV of flows with a final model's 23 features.

For the demo only: the CSV can be uploaded in app/demo.py ("Upload a CSV"). It is never used
for training or for any reported score.

    uv run python scripts/pcap_to_csv.py capture.pcapng --task B --label STREAMING \
        --output app/samples/custom/youtube_novpn_B.csv

Packets are read with tshark (installed with Wireshark). Flows and features follow the
dataset's own tool (ISCXFlowMeter), as checked against the dataset's captures with
scripts/check_pcap_to_csv.py:
- a flow = IPv4 TCP or UDP packets with the same two (address, port) ends and protocol, in
  both directions; the first packet's sender is the forward direction;
- packets are taken in file order (not sorted by time), so a gap can be negative, as in the
  dataset;
- a flow ends when a packet arrives more than `timeout` seconds after the flow's first packet
  (that packet starts a new flow, which keeps the old flow's forward direction), or with a TCP
  packet carrying FIN (the next packet starts a new flow in its own direction);
- times in microseconds; bytes = IP total length; rates per second (0 when duration is 0);
- gaps of more than 1 s between packets are idle periods; an active period is the time from
  the first packet of one active stretch to the first packet of the next, so it includes the
  idle gap between them (the tool's own definition; none when the flow never goes idle);
- an empty list of gaps or periods gives min = max = -1 and total = mean = std = 0.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blindspy.data import CORE_TASKS, LABELS

ROOT = Path(__file__).resolve().parents[1]
TSHARK = shutil.which("tshark") or "/Applications/Wireshark.app/Contents/MacOS/tshark"
FIELDS = ["frame.time_epoch", "ip.src", "ip.dst", "ip.proto", "tcp.srcport", "tcp.dstport",
          "udp.srcport", "udp.dstport", "ip.len", "tcp.flags.fin"]  # fmt: skip
IDLE_GAP = 1_000_000  # microseconds
TASK_LABELS = {t: LABELS[t] for t in CORE_TASKS}


def read_packets(pcap: Path, tshark: str = TSHARK) -> pd.DataFrame:
    """IPv4 TCP/UDP packets in file order: ts (us), src, dst, proto, sport, dport, size, fin."""
    cmd = [tshark, "-r", str(pcap), "-T", "fields", "-E", "separator=,"]
    for f in FIELDS:
        cmd += ["-e", f]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"tshark not found at {tshark}: install Wireshark, or pass --tshark "
                           "with the path of tshark") from None  # fmt: skip
    if run.returncode != 0:
        raise RuntimeError(f"tshark failed on {pcap}: {run.stderr.strip()}")
    rows = []
    for line in run.stdout.splitlines():
        p = line.split(",")
        if len(p) != len(FIELDS) or not p[1] or p[3] not in ("6", "17"):
            continue
        sport, dport = (p[4], p[5]) if p[3] == "6" else (p[6], p[7])
        if not sport or not dport:  # a fragment without ports
            continue
        rows.append((round(float(p[0]) * 1e6), p[1], p[2], int(p[3]), int(sport), int(dport),
                     int(p[8].split(";")[0]), p[9] in ("1", "True")))  # fmt: skip
    cols = ["ts", "src", "dst", "proto", "sport", "dport", "size", "fin"]
    return pd.DataFrame(rows, columns=cols)


def split_flows(packets: pd.DataFrame, timeout: float) -> list[pd.DataFrame]:
    """Group packets into flows (see the module text); each flow keeps its packets in order
    and a `fwd` column (True = same direction as the flow's first packet)."""
    limit = timeout * 1e6
    done, open_ = [], {}
    for i, (ts, src, dst, proto, sport, dport, fin) in enumerate(zip(
            packets.ts, packets.src, packets.dst, packets.proto, packets.sport,
            packets.dport, packets.fin, strict=True)):  # fmt: skip
        key = (frozenset([(src, sport), (dst, dport)]), proto)
        flow = open_.get(key)
        fwd_end = (src, sport)
        if flow is not None and ts - flow["start"] > limit:
            done.append(flow)
            fwd_end = flow["fwd_end"]  # the next flow keeps the connection's direction
            flow = None
        if flow is None:
            flow = open_[key] = {"start": ts, "fwd_end": fwd_end, "rows": [], "fwd": []}
        flow["rows"].append(i)
        flow["fwd"].append((src, sport) == flow["fwd_end"])
        if fin and proto == 6:
            done.append(open_.pop(key))
    done += open_.values()
    done.sort(key=lambda f: f["rows"][0])
    return [packets.iloc[f["rows"]].assign(fwd=f["fwd"]) for f in done]


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"total": 0.0, "min": -1.0, "max": -1.0, "mean": 0.0, "std": 0.0}
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return {"total": float(x.sum()), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": std}  # fmt: skip


def active_idle(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Idle = every gap above IDLE_GAP; active = from the first packet of one active stretch to
    the first packet of the next, one per idle gap (none when the flow never goes idle)."""
    gaps = np.diff(ts)
    idle_at = np.flatnonzero(gaps > IDLE_GAP)
    idle = gaps[idle_at].astype(float)
    starts = np.append(ts[0], ts[idle_at + 1]).astype(float)  # first packet of each stretch
    return np.diff(starts), idle


def flow_features(flow: pd.DataFrame) -> dict:
    """All features of one flow: both layouts' fiat/biat columns (std_* and total_*)."""
    ts = flow.ts.to_numpy(dtype=float)
    fwd = flow.fwd.to_numpy()
    duration = ts[-1] - ts[0]
    out = {"duration": duration}
    for name, part in [("fiat", ts[fwd]), ("biat", ts[~fwd])]:
        s = _stats(np.diff(part))
        out |= {f"{k}_{name}": s[k] for k in ["total", "min", "max", "mean", "std"]}
    secs = duration / 1e6
    out["flowPktsPerSecond"] = len(ts) / secs if secs > 0 else 0.0
    out["flowBytesPerSecond"] = float(flow["size"].sum()) / secs if secs > 0 else 0.0
    active, idle = active_idle(ts)
    for name, x in [("flowiat", np.diff(ts)), ("active", active), ("idle", idle)]:
        s = _stats(x)
        out |= {f"{k}_{name}": s[k] for k in ["min", "max", "mean", "std"]}
    return out


_ONE_PACKET = pd.DataFrame({"ts": [0], "size": [0], "fwd": [True]})


def features_for_task(feats: pd.DataFrame, task: str) -> pd.DataFrame:
    """The task's 23 columns in the order of models/final/<task>.json."""
    meta = json.loads((ROOT / "models" / "final" / f"{task}.json").read_text())
    return feats[[f["name"] for f in meta["features"]]]


def convert(pcap: Path, task: str, timeout: float = 15, label: str | None = None,
            tshark: str = TSHARK) -> pd.DataFrame:  # fmt: skip
    flows = split_flows(read_packets(pcap, tshark), timeout)
    feats = pd.DataFrame([flow_features(f) for f in flows])
    if feats.empty:  # no flows: the right columns, no rows
        feats = pd.DataFrame(columns=list(flow_features(_ONE_PACKET)))
    out = features_for_task(feats, task)
    if label is not None:
        out = out.assign(label=label)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pcap", type=Path)
    ap.add_argument("--task", required=True, choices=sorted(TASK_LABELS))
    ap.add_argument("--label", help="true class, written as a last `label` column (optional)")
    ap.add_argument("--timeout", type=float, default=15,
                    help="flow timeout in seconds (the final models use 15)")  # fmt: skip
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tshark", default=TSHARK)
    a = ap.parse_args(argv)
    if a.label is not None and a.label not in TASK_LABELS[a.task]:
        print(f"--label {a.label!r} is not a {a.task} class: {', '.join(TASK_LABELS[a.task])}",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        df = convert(a.pcap, a.task, a.timeout, a.label, a.tshark)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    if df.empty:
        print(f"no IPv4 TCP/UDP flows in {a.pcap}", file=sys.stderr)
        return 2
    a.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"{a.output}: {len(df)} flows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
