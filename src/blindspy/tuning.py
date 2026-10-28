"""Hyperparameter search with Optuna (E07): TPE sampler with a fixed seed, trials one at a time,
each trial scored by 5-fold stratified CV macro F1 on the training rows only.

The search space comes from the config. Parameter types:
  int / float (low, high, optional log), categorical (choices),
  int_or_none (an int in [low, high], or None = no limit),
  choice_or_float (one of the choices, or a float in [low, high]).
"""

import time

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from blindspy import models

CV_N_JOBS = -1  # the folds of one trial run in parallel; trials run one after another

optuna.logging.set_verbosity(optuna.logging.WARNING)


def sample(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, s in space.items():
        kind = s["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, s["low"], s["high"], log=s.get("log", False))
        elif kind == "float":
            params[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, s["choices"])
        elif kind == "int_or_none":
            limited = trial.suggest_categorical(f"{name}_limited", [True, False])
            params[name] = trial.suggest_int(name, s["low"], s["high"]) if limited else None
        elif kind == "choice_or_float":
            pick = trial.suggest_categorical(f"{name}_kind", [*s["choices"], "fraction"])
            params[name] = (trial.suggest_float(name, s["low"], s["high"]) if pick == "fraction"
                            else pick)  # fmt: skip
        else:
            raise ValueError(f"unknown search type {kind!r} for {name}")
    return params


def search(model: str, fixed: dict, space: dict, X, y, n_trials: int, seed: int, folds: int,
           on_trial=None) -> dict:  # fmt: skip
    """Run the study. Returns the trials table, the best settings (fixed + sampled) and a
    summary: trials, total fits (trials x folds), best CV macro F1 mean and std, search time.
    on_trial(number, params, result) is called after each trial (used to log it)."""
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))  # fmt: skip
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed, **sample(trial, space)}
        t0 = time.perf_counter()
        res = cross_validate(models.build_model(model, params, seed), X, y, cv=cv,
                             scoring="f1_macro", n_jobs=CV_N_JOBS)  # fmt: skip
        out = {"cv_f1_macro_mean": float(res["test_score"].mean()),
               "cv_f1_macro_std": float(res["test_score"].std()),
               "seconds": time.perf_counter() - t0}  # fmt: skip
        rows.append({"number": trial.number, "params": params, **out})
        if on_trial:
            on_trial(trial.number, params, out)
        return out["cv_f1_macro_mean"]

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    seconds = time.perf_counter() - t0
    trials = pd.DataFrame(rows)
    best = trials.loc[trials["number"] == study.best_trial.number].iloc[0]
    summary = {
        "search_n_trials": len(trials),
        "search_total_fits": len(trials) * folds,
        "search_best_trial": int(best["number"]),
        "search_best_cv_f1_macro_mean": best["cv_f1_macro_mean"],
        "search_best_cv_f1_macro_std": best["cv_f1_macro_std"],
        "search_time_s": seconds,
    }
    return {"trials": trials, "best_params": best["params"], "summary": summary}


def plot_history(trials: pd.DataFrame, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(trials["number"], trials["cv_f1_macro_mean"], s=12, label="trial")
    ax.plot(trials["number"], trials["cv_f1_macro_mean"].cummax(), color="C1", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("5-fold CV macro F1 (train)")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
