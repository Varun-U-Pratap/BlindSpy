import importlib.util

import pytest

from blindspy.tracking import ROOT, check_git_state

spec = importlib.util.spec_from_file_location("export_history", ROOT / "scripts/export_history.py")
export_history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_history)


def test_clean_and_pushed_passes():
    assert check_git_state(dirty=False, pushed=True, official=True) is None


@pytest.mark.parametrize("dirty,pushed", [(True, True), (False, False), (True, False)])
def test_official_runner_refused_unless_committed_and_pushed(dirty, pushed):
    with pytest.raises(SystemExit, match="commit|push"):
        check_git_state(dirty=dirty, pushed=pushed, official=True)


def test_personal_track_only_warns():
    msg = check_git_state(dirty=True, pushed=False, official=False, branch="dev")
    assert msg.startswith("WARNING") and "git push origin dev" in msg


def test_experiment_log_newest_first_and_skips_template(tmp_path):
    for name, body in [("E02.md", "### E02"), ("E10.md", "### E10"), ("E_TEMPLATE.md", "x")]:
        (tmp_path / name).write_text(body)
    log = export_history.build_log(tmp_path, "r")
    assert log.index("### E10") < log.index("### E02")
    assert "\nx\n" not in log
