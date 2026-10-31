"""scripts/build_report.py: the appendix's whole-file includes and the git tag table."""

import importlib.util

from blindspy.tracking import ROOT


def _builder():
    spec = importlib.util.spec_from_file_location("build_report",
                                                  ROOT / "scripts" / "build_report.py")  # fmt: skip
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_include_drops_title_and_lowers_headings(tmp_path):
    f = tmp_path / "x.md"
    f.write_text("# Title\n\n## Part\n\n### Entry\n\n#### Addendum\n\ntext\n")
    assert _builder().include(f) == "\n### Part\n\n### Entry\n\n### Addendum\n\ntext"


def test_appendix_expands_includes_and_tags():
    b = _builder()
    text = b.appendix(ROOT / "reports" / "report" / "drafts" / "appendix.md")
    assert "{{" not in text
    assert "| v1.0-final | 3c31bad |" in text
    log = (ROOT / "reports" / "EXPERIMENT_LOG.md").read_text().splitlines()
    assert log[5] in text  # a line from the experiment log, in full
    assert text.count("\n| E12-final-B |") == 1  # the run table
    blocks = b.parse(text, ROOT / "reports" / "report" / "drafts")
    assert all(blk["level"] <= 3 for blk in blocks if blk["t"] == "h")
