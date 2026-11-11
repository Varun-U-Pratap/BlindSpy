"""The research-paper version (reports/paper/BlindSpy_paper.md) repeats numbers from the report and
the generated tables; every number must exist there, and the paper must build."""

import importlib.util
import re

from blindspy.data import ROOT

SOURCE = ROOT / "reports" / "paper" / "BlindSpy_paper.md"
spec = importlib.util.spec_from_file_location("build_paper", ROOT / "scripts" / "build_paper.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

CORPUS = [
    *sorted((ROOT / "reports" / "report" / "drafts").glob("0*.md")),
    *sorted((ROOT / "reports" / "experiments" / "utsav-s-r").glob("E*.md")),
    *(ROOT / "reports" / f for f in ("baseline_tables.md", "phase3_tables.md", "phase4_tables.md")),
]
# decimals (0.9434, 0.07), numbers with thousands separators (18,758) and percentages (88.01%)
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+)(?![\w])")


def values(text: str) -> set[float]:
    return {round(float(n.replace(",", "")), 4) for n in NUMBER.findall(text)}


def test_every_number_in_the_paper_comes_from_a_source_file():
    corpus = values("\n".join(p.read_text() for p in CORPUS))
    body = SOURCE.read_text().split("\n", 3)[3]  # skip the title block (it holds a URL)
    missing = sorted(v for v in values(body) if v not in corpus)
    assert missing == []


def test_paper_builds_with_valid_cross_references_and_citations():
    built = paper.build_spec(SOURCE.read_text())  # asserts every Section/Table/Fig. reference
    labels = [b["label"] for b in built["blocks"] if b["t"] in ("table", "img")]
    assert labels == [f"TABLE {r}" for r in ("I", "II")] + ["Fig. 1."] + [
        "TABLE III", "TABLE IV", "Fig. 2.", "TABLE V", "Fig. 3.", "TABLE VI"]  # fmt: skip
    numbers = [r["n"] for r in built["references"]]
    assert numbers == list(range(1, len(paper.references()) + 1))  # every reference cited once


def test_paper_is_written_in_the_team_voice():
    text = SOURCE.read_text()
    assert re.findall(r"(?i)\buser\b|\buser OK\b|\bthe session\b", text) == []
