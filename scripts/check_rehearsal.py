"""E12 rehearsal check (E12 addendum item 4), before the official test runs.

The rehearsal runs every E12 config with `run_experiment.py --rehearsal` in a scratch clone whose
runner is "rehearsal" (its own MLflow store): each model is fitted on the training rows and
scored on the validation rows; no test row is used. This script, run from the main folder with
REHEARSAL=<clone path>, checks that store: all 30 runs, every logged field, the prediction files,
the intervals, all 26 paired tests with raw and Holm p, data and split hashes, commit and lock
hash. Every check prints PASS or FAIL; the output is kept in reports/e12_rehearsal_check.txt.

Usage: REHEARSAL=<clone> uv run python scripts/check_rehearsal.py > reports/e12_rehearsal_check.txt
"""

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from blindspy import results as R
from blindspy.tracking import ROOT, file_sha256

CLONE = Path(os.environ["REHEARSAL"])
RUNNER = "rehearsal"
fails = []


def check(ok: bool, what: str) -> None:
    print("PASS" if ok else "FAIL", what)
    if not ok:
        fails.append(what)


spec = importlib.util.spec_from_file_location("eh", ROOT / "scripts" / "export_history.py")
eh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eh)
h = eh.history_frame(f"sqlite:///{CLONE / 'mlruns' / RUNNER / 'mlflow.db'}", "blindspy")
h.to_csv(CLONE / "rehearsal_history.csv", index=False)
h = R.load_history(CLONE / "rehearsal_history.csv")
e = R.e12_runs(h, "final_rehearsal")
cfgs = {yaml.safe_load(p.read_text())["run_name"]: yaml.safe_load(p.read_text())
        for p in (ROOT / "configs" / "experiments").glob("E12_*.yaml")}  # fmt: skip

check(len(e) == 30 and set(e["run_name"]) == set(cfgs), f"30 runs, names = configs ({len(e)})")
check((e["status"] == "FINISHED").all(), "all FINISHED")
check((e["params.role"] == "headline").sum() == 4, "4 headline runs")
check(e["tags.git_commit"].nunique() == 1, f"one commit {set(e['tags.git_commit'].str[:7])}")
check((e["tags.uv_lock_sha256"] == file_sha256(ROOT / "uv.lock")).all(), "lock hash = main uv.lock")
check((e["tags.git_dirty"].astype(str) == "False").all(), "clean tree")
check((e["tags.data_version"] == "v1").all(), "data v1")
man = json.loads((ROOT / "data" / "processed" / "v1" / "manifest.json").read_text())["tasks"]
ok = all(r["tags.data_sha256"] == man[f"{r['params.task']}_15s"]["parquet_sha256"]
         and r["tags.data_split_sha256"] == man[f"{r['params.task']}_15s"]["split_sha256"]
         for _, r in e.iterrows())  # fmt: skip
check(ok, "data + split hashes = manifest")
test_cols = [c for c in h.columns if c.startswith("metrics.test_")]
check(not test_cols, f"no test metric anywhere ({test_cols[:3]})")

# the rehearsal fits on train and scores validation = exactly what each source run did
diff = (e["metrics.rehearsal_f1_macro"] - e["params.source_val_f1_macro"].astype(float)).abs()
check((diff == 0).all(), f"rehearsal F1 == source val F1 exactly (max diff {diff.max():.2e})")
cal = e[e["params.calibrate"] == "isotonic"]["params.task"].tolist()
check(sorted(cal) == ["A1", "A2-VPN"], f"calibrated runs: {cal}")
check(e[e["params.calibrate"] == "isotonic"]["metrics.rehearsal_ece_used"].notna().all(),
      "calibrated runs log ECE used")  # fmt: skip
head = e[e["params.role"] == "headline"]
check(head["metrics.rehearsal_ece_uncalibrated"].notna().all(), "headline ECE logged")

tracked = ["metrics.cv_f1_macro_mean", "metrics.cv_f1_macro_std", "metrics.train_time_s",
           "metrics.predict_ms_per_1000", "metrics.model_file_bytes", "metrics.model_size_bytes",
           "metrics.peak_memory_mb", "metrics.child_peak_rss_mb", "metrics.rehearsal_accuracy",
           "metrics.rehearsal_balanced_accuracy", "metrics.rehearsal_f1_weighted",
           "params.features", "params.seed", "params.class_balancing", "params.model_compress",
           "params.n_fit", "params.n_rehearsal", "tags.runner", "tags.machine.cpu",
           "tags.config_file", "tags.top_confusion_pairs", "tags.worst_class_f1",
           "params.source_run_id", "params.builds_on"]  # fmt: skip
missing = {c: int(e[c].isna().sum()) for c in tracked if c not in e or e[c].isna().any()}
check(not missing, f"tracked fields present ({missing})")
a1 = e[e["params.task"] == "A1"]
check(a1["metrics.rehearsal_roc_auc"].notna().all(), "ROC-AUC for every A1 run")

art = CLONE / "mlruns" / RUNNER / "artifacts"
n_files = 0
for _, r in e.iterrows():
    base = art / r["run_id"] / "artifacts"
    need = ["predictions/rehearsal_predictions.csv", "model/model.joblib",
            "eval/confusion_matrix.csv", "eval/per_class_report.csv", "config"]  # fmt: skip
    if all((base / f).exists() for f in need):
        n_files += 1
check(n_files == 30, f"every run has predictions, model, eval files, config ({n_files}/30)")

from blindspy import data  # noqa: E402

ok = True
for task in R.CORE_TASKS:
    tr, va, te = data.load_task(task, 15)
    for _, r in e[e["params.task"] == task].iterrows():
        p = R.final_predictions(r["run_id"], RUNNER, "rehearsal", CLONE)
        ok &= set(p["raw_row"]) == set(va.index) and not set(p["raw_row"]) & set(te.index)
check(ok, "every prediction file = the validation rows, no test row")

scores, draws = R.e12_scores(h, RUNNER, "rehearsal", "final_rehearsal", CLONE)
tests = R.e12_tests(scores, draws)
check(len(scores) == 30 and scores["95% interval"].notna().all(), "intervals for all 30 models")
check(len(tests) == 26, f"26 paired tests ({len(tests)})")
check(tests["p"].between(2 / 10_001 - 1e-12, 1).all() and tests["Holm p"].ge(tests["p"]).all(),
      "p in [2/10001, 1], Holm p >= raw p")  # fmt: skip
v = R.e12_verdicts(scores, R.load_history())  # old-rule picks from the official history
check(set(v) == {"E12a", "E12b", "E12c"}, "verdicts (a)-(c) computable")
pd.set_option("display.width", 250)
print(scores.drop(columns=["_val", "_point"]).to_string())
print(tests.to_string())
print(json.dumps(v, indent=1))
print("FAILED:", fails if fails else "none")
