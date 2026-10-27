"""Run one experiment from a YAML config and log it to the runner's MLflow store.

Usage: uv run python scripts/run_experiment.py configs/experiments/E_smoke.yaml
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from blindspy import tracking  # noqa: E402


def run_smoke(cfg: dict) -> None:
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


KINDS = {"smoke": run_smoke}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)

    base = tracking.load_yaml(tracking.ROOT / "configs" / "base.yaml")
    cfg = tracking.load_yaml(args.config)
    runner = tracking.load_runner()
    official = runner == base["official_runner"]

    git = tracking.git_state()
    warning = tracking.check_git_state(git["dirty"], git["pushed"], official, git["branch"])
    if warning:
        print(warning, file=sys.stderr)

    uri = tracking.setup_mlflow(runner, base["mlflow_experiment"])
    tags = tracking.identity_tags(runner, cfg["experiment_id"], args.config, git)
    with mlflow.start_run(run_name=cfg["experiment_id"], tags=tags) as run:
        mlflow.log_artifact(str(args.config), artifact_path="config")
        KINDS[cfg["kind"]](cfg)
    print(f"Logged run {run.info.run_id} ({cfg['experiment_id']}) to {uri}")


if __name__ == "__main__":
    main()
