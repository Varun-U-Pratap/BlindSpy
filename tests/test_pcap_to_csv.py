"""scripts/pcap_to_csv.py on made-up packet tables (no tshark or capture file needed)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blindspy.data import CORE_TASKS
from blindspy.predict import validate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pcap_to_csv", ROOT / "scripts" / "pcap_to_csv.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

A, B = ("10.0.0.1", 5000), ("10.0.0.2", 443)


def packets(rows):
    """rows: (ts in us, sender end, size, fin) -> the table read_packets returns (TCP)."""
    out = []
    for ts, frm, size, fin in rows:
        to = B if frm == A else A
        out.append((ts, frm[0], to[0], 6, frm[1], to[1], size, fin))
    return pd.DataFrame(out, columns=["ts", "src", "dst", "proto", "sport", "dport", "size", "fin"])


def test_flow_features_known_gaps():
    # a real dataset flow (AIMchat1.pcapng): 0, 53,064, 8,479,284, 8,499,516 us; one idle gap
    pk = packets([(0, A, 100, False), (53_064, B, 40, False),
                  (8_479_284, B, 40, False), (8_499_516, A, 100, False)])  # fmt: skip
    [flow] = conv.split_flows(pk, 15)
    f = conv.flow_features(flow)
    assert f["duration"] == 8_499_516
    assert (f["total_fiat"], f["min_fiat"], f["std_fiat"]) == (8_499_516, 8_499_516, 0)
    assert (f["total_biat"], f["mean_biat"]) == (8_426_220, 8_426_220)
    assert f["min_flowiat"] == 20_232 and f["max_flowiat"] == 8_426_220
    assert f["min_active"] == f["max_active"] == 8_479_284  # first packet to first after idle
    assert f["min_idle"] == f["max_idle"] == 8_426_220
    assert f["flowPktsPerSecond"] == pytest.approx(4 / 8.499516)
    assert f["flowBytesPerSecond"] == pytest.approx(280 / 8.499516)


def test_empty_lists_and_zero_duration():
    [flow] = conv.split_flows(packets([(5, A, 60, False)]), 15)
    f = conv.flow_features(flow)
    assert f["duration"] == 0 and f["flowPktsPerSecond"] == f["flowBytesPerSecond"] == 0
    for name in ["fiat", "biat", "flowiat", "active", "idle"]:
        assert f[f"min_{name}"] == f[f"max_{name}"] == -1
        assert f[f"mean_{name}"] == f[f"std_{name}"] == 0
    assert f["total_fiat"] == f["total_biat"] == 0


def test_no_idle_gap_means_no_active_period():
    [flow] = conv.split_flows(packets([(0, A, 60, False), (999_999, B, 60, False)]), 15)
    f = conv.flow_features(flow)
    assert f["min_active"] == -1 and f["min_idle"] == -1


def test_std_uses_sample_formula():
    pk = packets([(0, A, 60, False), (10, A, 60, False), (30, A, 60, False), (60, A, 60, False)])
    [flow] = conv.split_flows(pk, 15)
    assert conv.flow_features(flow)["std_fiat"] == pytest.approx(np.std([10, 20, 30], ddof=1))


def test_timeout_split_keeps_direction_and_fin_restarts():
    pk = packets([(0, A, 60, False), (1_000, B, 60, False),
                  (16_000_000, B, 60, False), (16_000_500, A, 60, True),  # FIN ends flow 2
                  (16_001_000, B, 60, False)])  # fmt: skip
    flows = conv.split_flows(pk, 15)
    assert [len(f) for f in flows] == [2, 2, 1]
    assert flows[1].fwd.tolist() == [False, True]  # B's packet, but A stays forward
    assert flows[2].fwd.tolist() == [True]  # after FIN: the new sender is forward


def test_packets_kept_in_file_order():
    pk = packets([(100, A, 60, False), (90, B, 60, False), (200, A, 60, False)])
    [flow] = conv.split_flows(pk, 15)
    assert conv.flow_features(flow)["min_flowiat"] == -10  # negative gap, as in the dataset


@pytest.mark.parametrize("task", CORE_TASKS)
def test_task_columns_pass_the_model_check(task):
    pk = packets([(0, A, 100, False), (2_000_000, B, 40, False), (2_000_100, A, 40, False)])
    feats = pd.DataFrame([conv.flow_features(f) for f in conv.split_flows(pk, 15)])
    out = conv.features_for_task(feats, task).assign(label=conv.TASK_LABELS[task][0])
    meta = json.loads((ROOT / "models" / "final" / f"{task}.json").read_text())
    assert list(validate(out, meta).columns) == [f["name"] for f in meta["features"]]
    assert list(out.columns)[-1] == "label"


def test_cli_rejects_a_label_of_another_task(capsys):
    assert conv.main(["missing.pcap", "--task", "A1", "--label", "STREAMING",
                      "--output", "x.csv"]) == 2  # fmt: skip
    assert "not a A1 class" in capsys.readouterr().err


def test_no_flows_gives_an_empty_table_with_the_task_columns(monkeypatch):
    monkeypatch.setattr(conv, "read_packets", lambda *a, **k: packets([]))
    out = conv.convert(Path("empty.pcap"), "A2-VPN", label="VPN-CHAT")
    assert out.empty and list(out.columns)[:2] == ["duration", "total_fiat"]


def test_cli_explains_a_missing_tshark(capsys, tmp_path):
    pcap = tmp_path / "x.pcap"
    pcap.write_bytes(b"")
    rc = conv.main([str(pcap), "--task", "B", "--output", str(tmp_path / "x.csv"),
                    "--tshark", str(tmp_path / "no-tshark")])  # fmt: skip
    assert rc == 2 and "tshark not found" in capsys.readouterr().err
