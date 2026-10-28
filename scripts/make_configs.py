"""Write the experiment configs (one YAML per run) to configs/experiments/.

Baselines (E01-E03): 5 models x 4 tasks x 4 timeouts = 80 configs, protocol "holdout".
Paper benchmark (inside E02 and E03): the J48 approximation and IBk (Weka defaults) x 16 files
x 2 row sources (data v1, and every row as published) = 64 configs, protocol "paper_10fold".
Phase 3 (E04 on): each model first at 15 s for all 4 tasks; a winner (15 s validation macro F1
at least 0.03 above the task's current best) is rerun at 30, 60 and 120 s with the same
settings, listed in RERUNS.

Usage: uv run python scripts/make_configs.py
"""

import yaml

from blindspy.data import CORE_TASKS, ROOT, TIMEOUTS

OUT = ROOT / "configs" / "experiments"

# experiment ID, registry name, hyperparameters (everything not listed is sklearn's default)
BASELINES = [
    ("E01", "majority", {"strategy": "most_frequent"}),
    ("E02", "tree", {"criterion": "entropy"}),
    ("E03", "knn", {"n_neighbors": 5}),
    ("E03", "logreg", {"max_iter": 1000}),
    ("E03", "svm_rbf", {"C": 1.0, "gamma": "scale"}),
]

# Weka defaults (Weka Javadoc): J48 -C 0.25 -M 2 (pruned); IBk -K 1 with LinearNNSearch and
# EuclideanDistance, which normalises each attribute to its range. sklearn has no C4.5
# pruning, so the J48 stand-in is an unpruned entropy tree with at least 2 rows per leaf.
BENCHMARK = [
    ("E02", "j48", "tree", {"criterion": "entropy", "min_samples_leaf": 2}),
    ("E03", "ibk", "knn_minmax", {"n_neighbors": 1}),
]
ROWS = ["published", "v1"]

# Phase 3 screening at 15 s: experiment ID, registry name, hyperparameters
SCREEN = [
    ("E04", "knn_log_standard", {"n_neighbors": 5}),
    ("E04", "knn_log_robust", {"n_neighbors": 5}),
    ("E05", "rf", {"n_estimators": 300}),  # default depth, class_weight=None, n_jobs=1
]
# Winner reruns at the other timeouts: (experiment ID, registry name, task) -> timeouts
RERUNS: dict[tuple[str, str, str], list[int]] = {
    ("E04", "knn_log_standard", "A2-NonVPN"): [30, 60, 120],  # +0.0376 over E02 tree at 15 s
}


def write(name: str, cfg: dict) -> None:
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for exp, model, params in BASELINES:
        for task in CORE_TASKS:
            for t in TIMEOUTS:
                run = f"{exp}-{model}-{task}-{t}s"
                write(f"{exp}_{model}_{task}_{t}s", {
                    "experiment_id": exp, "run_name": run, "kind": "train", "task": task,
                    "timeout": t, "model": model, "params": params, "save_model": True,
                })  # fmt: skip
                n += 1
    for exp, alias, model, params in BENCHMARK:
        for rows in ROWS:
            for task in CORE_TASKS:
                for t in TIMEOUTS:
                    run = f"{exp}-paper-{alias}-{rows}-{task}-{t}s"
                    write(f"{exp}_paper_{alias}_{rows}_{task}_{t}s", {
                        "experiment_id": exp, "run_name": run, "kind": "paper_benchmark",
                        "task": task, "timeout": t, "model": model, "params": params,
                        "rows": rows, "folds": 10, "save_model": True,
                    })  # fmt: skip
                    n += 1
    for exp, model, params in SCREEN:
        for task in CORE_TASKS:
            for t in [15, *RERUNS.get((exp, model, task), [])]:
                run = f"{exp}-{model}-{task}-{t}s"
                write(f"{exp}_{model}_{task}_{t}s", {
                    "experiment_id": exp, "run_name": run, "kind": "train", "task": task,
                    "timeout": t, "model": model, "params": params, "save_model": True,
                })  # fmt: skip
                n += 1
    print(f"Wrote {n} configs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
