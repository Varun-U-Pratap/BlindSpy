"""MLflow helpers: per-runner store, run identity tags, git state checks."""

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_runner(root: Path = ROOT) -> str:
    """Runner name from the git-ignored configs/local.yaml; refuses without it."""
    local = root / "configs" / "local.yaml"
    if not local.exists():
        raise SystemExit(
            "configs/local.yaml not found. Copy configs/local.example.yaml to "
            "configs/local.yaml and set runner: <your GitHub username>."
        )
    runner = load_yaml(local).get("runner")
    if not runner or runner == "your-github-username":
        raise SystemExit("configs/local.yaml must set runner: <your GitHub username>.")
    return runner


def _git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state(root: Path = ROOT) -> dict:
    """Commit hash, branch, uncommitted changes, and whether HEAD is on a remote branch."""
    return {
        "commit": _git("rev-parse", "HEAD", root=root),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", root=root),
        "dirty": bool(_git("status", "--porcelain", root=root)),
        "pushed": bool(_git("branch", "-r", "--contains", "HEAD", root=root)),
    }


def check_git_state(dirty: bool, pushed: bool, official: bool, branch: str = "dev") -> str | None:
    """Return a warning for personal tracks; raise SystemExit for the official runner.

    Official runs must come from code that is committed and pushed, so every result
    can be traced to code anyone can check out.
    """
    if not dirty and pushed:
        return None
    steps = []
    if dirty:
        steps.append("commit your changes (git add … && git commit)")
    steps.append(f"push them (git push origin {branch})")
    steps.append("run the experiment again")
    msg = "Code is not committed and pushed. To fix: " + ", then ".join(steps) + "."
    if official:
        raise SystemExit("Refusing official run. " + msg)
    return "WARNING (personal track): " + msg


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    ram_gb = None
    if platform.system() == "Darwin":
        cpu = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            or cpu
        )
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "ram_gb": str(ram_gb),
        "python": platform.python_version(),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_mlflow(runner: str, experiment: str, root: Path = ROOT) -> str:
    """Point MLflow at mlruns/<runner>/mlflow.db with artifacts in mlruns/<runner>/artifacts."""
    store = root / "mlruns" / runner
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow.get_tracking_uri()


def identity_tags(runner: str, experiment_id: str, config_path: Path, git: dict) -> dict:
    m = machine_info()
    return {
        "experiment_id": experiment_id,
        "runner": runner,
        "machine.os": m["os"],
        "machine.cpu": m["cpu"],
        "machine.ram_gb": m["ram_gb"],
        "python_version": m["python"],
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": str(git["dirty"]),
        "git_pushed": str(git["pushed"]),
        "uv_lock_sha256": file_sha256(ROOT / "uv.lock"),
        "config_file": str(Path(config_path).resolve().relative_to(ROOT)),
    }
