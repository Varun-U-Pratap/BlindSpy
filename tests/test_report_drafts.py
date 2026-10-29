"""The report drafts copy generated tables; each copied table must still match its source."""

from blindspy.data import ROOT

DRAFTS = ROOT / "reports" / "report" / "drafts"
SOURCES = [ROOT / "reports" / "baseline_tables.md", ROOT / "reports" / "phase3_tables.md"]


def tables(text: str) -> list[str]:
    out, cur = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            out.append("\n".join(cur))
            cur = []
    return out


def test_section_4_tables_match_generated_files():
    source = "\n".join(p.read_text() for p in SOURCES)
    copied = tables((DRAFTS / "04_experiments.md").read_text())
    assert len(copied) >= 30
    assert [t.splitlines()[0] for t in copied if t not in source] == []
