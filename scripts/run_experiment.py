"""Run one experiment from a YAML config and log it to the runner's MLflow store.

Usage: uv run python scripts/run_experiment.py configs/experiments/E02_tree_A1_15s.yaml
Several configs can be given; each becomes its own run.

Config kinds: "smoke" (setup check), "train" (5-fold CV on train, fit on train, score on
validation; the test set is never used) and "paper_benchmark" (the paper's 10-fold CV on
the whole file, rows from data v1 or as published; never used for model selection).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from blindspy import data, tracking, train  # noqa: E402


def run_smoke(cfg: dict, base: dict) -> None:
    mlflow.log_params(cfg.get("params", {}))
    mlflow.log_metric("smoke_metric", 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_title("smoke test")
        png = Path(tmp) / "smoke.png"
        fig.savefig(png, dpi=80)
        plt.close(fig)
        mlflow.log_artifact(str(png))


def run_train(cfg: dict, base: dict) -> None:
    version = base["data_version"]
    parts = data.load_task(cfg["task"], cfg["timeout"], data_version=version)
    mlflow.set_tags(tracking.data_tags(cfg["task"], cfg["timeout"], "v1", version))
    with tempfile.TemporaryDirectory() as tmp:
        result = train.train_and_score(cfg, parts, base["seed"], base["cv_folds"], Path(tmp),
                                       compress=base.get("model_compress", 0))  # fmt: skip
        _log_data_rules(version)
        tracking.log_result(result, Path(tmp))


def run_paper_benchmark(cfg: dict, base: dict) -> None:
    version = base["data_version"]
    if cfg["rows"] == "published":
        # every row as published (duplicates kept, values untouched); names from COLUMN_MAP
        df = data.load_raw(cfg["task"], cfg["timeout"])
    else:
        df = pd.concat(data.load_task(cfg["task"], cfg["timeout"], data_version=version))
    mlflow.set_tags(tracking.data_tags(cfg["task"], cfg["timeout"], cfg["rows"], version))
    with tempfile.TemporaryDirectory() as tmp:
        result = train.paper_benchmark(cfg, df, base["seed"], cfg["folds"], Path(tmp),
                                       compress=base.get("model_compress", 0))  # fmt: skip
        if cfg["rows"] != "published":
            _log_data_rules(version)
        tracking.log_result(result, Path(tmp))


def _log_data_rules(version: str) -> None:
    manifest = data.processed_dir(version) / "manifest.json"
    rules = json.loads(manifest.read_text())["cleaning_rules"]
    mlflow.log_dict(rules, "data/cleaning_rules.json")


KINDS = {"smoke": run_smoke, "train": run_train, "paper_benchmark": run_paper_benchmark}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="+")
    args = parser.parse_args(argv)

    base = tracking.load_yaml(tracking.ROOT / "configs" / "base.yaml")
    runner = tracking.load_runner()
    official = runner == base["official_runner"]

    git = tracking.git_state()
    warning = tracking.check_git_state(git["dirty"], git["pushed"], official, git["branch"])
    if warning:
        print(warning, file=sys.stderr)

    uri = tracking.setup_mlflow(runner, base["mlflow_experiment"])
    for path in args.config:
        cfg = tracking.load_yaml(path)
        tags = {
            **tracking.identity_tags(runner, cfg["experiment_id"], path, git),
            **tracking.package_versions(),
        }
        name = cfg.get("run_name", cfg["experiment_id"])
        with mlflow.start_run(run_name=name, tags=tags) as run:
            mlflow.log_artifact(str(path), artifact_path="config")
            KINDS[cfg["kind"]](cfg, base)
        print(f"Logged run {run.info.run_id} ({name}) to {uri}", flush=True)


if __name__ == "__main__":
    main()
