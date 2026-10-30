"""Save the 4 headline models of E12 to models/final/<task>.joblib + <task>.json (committed).

Each model is the one trained in its official E12-final-<task> run (on train + validation),
loaded from that run's MLflow artifact. Before saving, the model must reproduce the run's saved
test predictions (label and confidence) exactly. A file over 50 MB (GitHub's warning size)
stops the script: such a model needs a decision on how to ship it (release asset or Git LFS).

Usage: uv run python scripts/save_final_models.py
"""

import sys

import joblib
import numpy as np
import pandas as pd

from blindspy import calibration, data
from blindspy import results as R
from blindspy.predict import FINAL_DIR, write_final
from blindspy.tracking import ROOT, load_yaml

LIMIT = 50 * 10**6


def main() -> int:
    base = load_yaml(ROOT / "configs" / "base.yaml")
    runner = base["official_runner"]
    h = R.load_history()
    e = R.e12_runs(h)
    for task in data.CORE_TASKS:
        r = e[(e["params.task"] == task) & (e["params.role"] == "headline")].iloc[0]
        art = ROOT / "mlruns" / runner / "artifacts" / r["run_id"] / "artifacts"
        model = joblib.load(art / "model" / "model.joblib")  # the project's own run artifact
        test = data.load_task(task, 15, data_version=r["tags.data_version"])[2]
        saved = R.final_predictions(r["run_id"], runner)
        X = test.loc[saved["raw_row"]].drop(columns="label")
        label, conf, _ = calibration.predict_with_confidence(model, X)
        same = (label == saved["predicted"].astype(str).to_numpy()).all() and np.allclose(
            conf, saved["confidence"].to_numpy(), rtol=0, atol=1e-12
        )
        if not same:
            print(f"{task}: the saved model does not reproduce its run's test predictions")
            return 1
        meta = {
            "task": task,
            "timeout": int(r["params.timeout"]),
            "mlflow_run_name": r["run_name"],
            "mlflow_run_id": r["run_id"],
            "git_commit": r["tags.git_commit"],
            "training_date": pd.to_datetime(r["start_time"], utc=True).isoformat(),
            "config_file": r["tags.config_file"],
            "source_run": r["params.builds_on"],
            "model": r["params.model"],
            "class_balancing": r["params.class_balancing"],
            "pipeline_steps": r["params.pipeline_steps"],
            "trained_on": f"train + validation rows of the seed-42 split ({r['params.n_fit']} "
            f"rows), data version {r['tags.data_version']}",
            "data_sha256": r["tags.data_sha256"],
            "calibration": r["params.calibrate"],
            "confidence": (
                "calibrated (isotonic, E15) probability of the predicted class"
                if r["params.calibrate"] == "isotonic"
                else "the model's probability of the predicted class"
            ),
            "test_macro_f1": float(r["metrics.test_f1_macro"]),
        }
        meta = write_final(model, meta, FINAL_DIR, base["model_compress"])
        mib = meta["model_file_bytes"] / 2**20
        print(f"{task}: {r['run_name']} -> models/final/{task}.joblib ({mib:.2f} MiB)")
        if meta["model_file_bytes"] > LIMIT:
            print(f"STOP: {task}.joblib is over 50 MB; decide how to ship it before committing")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
