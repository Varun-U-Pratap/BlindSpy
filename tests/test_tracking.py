import importlib.util

import pytest

from blindspy.tracking import ROOT, check_git_state, load_runner

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


def test_runner_required(tmp_path):
    (tmp_path / "configs").mkdir()
    with pytest.raises(SystemExit, match="local.yaml not found"):
        load_runner(tmp_path)
    (tmp_path / "configs" / "local.yaml").write_text("runner: your-github-username\n")
    with pytest.raises(SystemExit, match="must set runner"):
        load_runner(tmp_path)
    (tmp_path / "configs" / "local.yaml").write_text("runner: someone\n")
    assert load_runner(tmp_path) == "someone"


def test_history_csv_one_row_per_run(tmp_path):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("x", artifact_location=(tmp_path / "a").as_uri())
    mlflow.set_experiment("x")
    for i in range(2):
        with mlflow.start_run(
            run_name=f"E0{i + 1}-tree-A1-15s", tags={"experiment_id": f"E0{i + 1}"}
        ):
            mlflow.log_params({"task": "A1", "model": "tree"})
            mlflow.log_metric("val_f1_macro", 0.5 + i / 10)
    hist = export_history.history_frame(uri, "x")
    assert len(hist) == 2 and list(hist["run_name"]) == ["E01-tree-A1-15s", "E02-tree-A1-15s"]
    assert hist.columns[0] == "run_id" and "metrics.val_f1_macro" in hist.columns
    assert not any(c.startswith("tags.mlflow.") for c in hist.columns)
    assert list(hist["metrics.val_f1_macro"]) == [0.5, 0.6]
