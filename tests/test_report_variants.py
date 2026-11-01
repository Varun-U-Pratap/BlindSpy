"""The two reader editions of the report (scripts/make_report_variants.py)."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_report_variants.py"
spec = importlib.util.spec_from_file_location("variants", SCRIPT)
variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variants)


@pytest.mark.parametrize("variant", ["full", "standalone"])
def test_committed_editions_match_a_fresh_build(variant, tmp_path):
    variants.build(variant, tmp_path / variant)
    committed = ROOT / "reports" / "report" / f"drafts_{variant}"
    fresh = sorted(p.name for p in (tmp_path / variant).iterdir())
    assert fresh == sorted(p.name for p in committed.iterdir())
    for name in fresh:
        assert (tmp_path / variant / name).read_text() == (committed / name).read_text(), name


def test_key_lists_every_experiment_in_order_of_its_id():
    page = variants.key_page()
    ids = re.findall(r"^\| (E\d\d) \|", page, flags=re.M)
    assert ids == [f"E{n:02d}" for n in range(18)]
    assert "not in the order the work was done" in page


def test_full_edition_is_the_master_text_plus_the_reading_aids(tmp_path):
    variants.build("full", tmp_path)
    order = variants.SECTION4_ORDER.format(
        order="\n".join(f"{n}. {s}" for n, s in enumerate(variants.ORDER, start=1))
    )
    for master in sorted(variants.DRAFTS.glob("[0-9][0-9]_*.md")):
        text = (tmp_path / master.name).read_text().replace(order, "")
        for eid, (name, *_) in variants.EXPERIMENTS.items():
            text = text.replace(f"{eid} ({name})", eid).replace(f"{eid}, {name}", eid)
        assert text == variants.drop_working_note(master.read_text()), master.name


def test_standalone_edition_has_no_git_details(tmp_path):
    variants.build("standalone", tmp_path)
    for path in tmp_path.glob("*.md"):
        assert variants.leftovers(path.read_text()) == [], path.name


def test_a_replacement_must_match_exactly():
    with pytest.raises(SystemExit):
        variants.sub("one two", "three", "x")
    assert variants.sub("one\ntwo", "one two", "x") == "x"
