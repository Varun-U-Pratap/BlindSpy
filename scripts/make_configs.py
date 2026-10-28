"""Write the experiment configs (one YAML per run) to configs/experiments/.

Baselines (E01-E03): 5 models x 4 tasks x 4 timeouts = 80 configs, protocol "holdout".
Paper benchmark (inside E02 and E03): the J48 approximation and IBk (Weka defaults) x 16 files
x 2 row sources (data v1, and every row as published) = 64 configs, protocol "paper_10fold".
Phase 3 (E04 on): each model first at 15 s for all 4 tasks; a winner (15 s validation macro F1
at least 0.03 above the task's current best) is rerun at 30, 60 and 120 s with the same
settings, listed in RERUNS. E08 runs a balancing ladder on the current best per task, copying
its model and settings from reports/history.csv.

Usage: uv run python scripts/make_configs.py
"""

import yaml

from blindspy import results
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
    ("E06", "xgb", {}),  # XGBoost defaults (n_jobs=1, random_state=42 set by the builder)
    ("E06", "lgbm", {}),  # LightGBM defaults (n_jobs=1, random_state=42, verbose=-1)
]
# E07 Optuna search ranges, fixed in the plan before running (50 trials each, TPESampler(seed=42),
# maximising 5-fold CV macro F1 on train).
SEARCH = {
    "rf": {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "max_depth": {"type": "int_or_none", "low": 5, "high": 40},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "choice_or_float", "choices": ["sqrt", "log2"], "low": 0.3,
                         "high": 1.0},
    },
    "xgb": {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 50},
    },
    "lgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 15, "high": 255},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 1, "high": 50},
    },
    "knn_log_standard": {
        "n_neighbors": {"type": "int", "low": 1, "high": 50},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "metric": {"type": "categorical", "choices": ["euclidean", "manhattan"]},
    },
}  # fmt: skip
# LightGBM ignores subsample unless subsample_freq > 0, so its searches fix subsample_freq=1
FIXED = {"lgbm": {"subsample_freq": 1}}
# E07 scope from reports/phase3_tables.md ("E07 scope"): best 2 of rf / xgb / lgbm per task by
# 15 s validation macro F1, plus the KNN preprocessing with the higher 15 s score (E04 standard)
TUNE = {
    "A1": ["xgb", "rf", "knn_log_standard"],
    "A2-VPN": ["xgb", "lgbm", "knn_log_standard"],
    "A2-NonVPN": ["xgb", "lgbm", "knn_log_standard"],
    "B": ["xgb", "lgbm", "knn_log_standard"],
}
TRIALS = 50

# Winner reruns at the other timeouts: (experiment ID, registry name, task) -> timeouts
RERUNS: dict[tuple[str, str, str], list[int]] = {
    ("E04", "knn_log_standard", "A2-NonVPN"): [30, 60, 120],  # +0.0376 over E02 tree at 15 s
    ("E05", "rf", "B"): [30, 60, 120],  # +0.0503 over E04 knn_log_standard at 15 s
}

# E08 builds on the current best per task after E07 (reports/phase3_tables.md, "Current best
# at 15 s after E07"); one change: class balancing, ladder none -> class_weight -> oversample
# -> smote. A step runs only if every earlier step helped (E08.md); steps run so far per task:
E08_BASE = {"A1": "E06-xgb", "A2-VPN": "E07-lgbm", "A2-NonVPN": "E07-lgbm", "B": "E07-xgb"}
E08_STEPS = {task: ["class_weight"] for task in CORE_TASKS}


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
    for task, names in TUNE.items():
        for model in names:
            run = f"E07-{model}-{task}-15s"
            write(f"E07_{model}_{task}_15s", {
                "experiment_id": "E07", "run_name": run, "kind": "optuna", "task": task,
                "timeout": 15, "model": model, "trials": TRIALS,
                "search_space": SEARCH[model], "fixed_params": FIXED.get(model, {}),
                "save_model": True,
            })  # fmt: skip
            n += 1
    for task, steps in E08_STEPS.items():
        base = f"{E08_BASE[task]}-{task}-15s"
        settings = results.run_settings(base)
        for step in steps:
            run = f"E08-{settings['model']}_{step}-{task}-15s"
            write(f"E08_{settings['model']}_{step}_{task}_15s", {
                "experiment_id": "E08", "run_name": run, "kind": "train", "task": task,
                "timeout": 15, **settings, "balance": step, "builds_on": base,
                "save_model": True,
            })  # fmt: skip
            n += 1
    print(f"Wrote {n} configs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
