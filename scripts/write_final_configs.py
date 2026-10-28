"""Write the final configs (decided 28 Oct after E10) to configs/final/, generated from
reports/history.csv, before the test set is opened:

- configs/final/<task>.yaml: the headline model per task, the highest validation macro F1 of all
  model-selection runs (any experiment, any timeout; only fallback: a saved file over 50 MB);
- configs/final/compare/<task>_<type>.yaml: the comparison set, the best 15 s validation variant
  of each of the 7 method types per task (28 configs);
- configs/final/old_rule/<task>.yaml: what the superseded 28 Oct tie rule would pick, where that
  run is not already in the comparison set, so its test score shows what the old rule would cost.

Each is a "train" config that run_experiment.py runs unchanged and that reproduces its source
run's validation score; it names that run in `builds_on` and gives the reason. Phase 4 (E12)
retrains them on train + validation and scores each once on the test set.

Usage: uv run python scripts/write_final_configs.py
"""

import yaml

from blindspy import results as R
from blindspy.data import ROOT

OUT = ROOT / "configs" / "final"


def config(run: str, run_id: str, name: str, reason: str, h) -> dict:
    r = h.loc[run]
    settings = R.run_settings(run)
    cfg = {"experiment_id": "E12", "run_name": name, "kind": "train", "task": r["params.task"],
           "timeout": int(r["params.timeout"]), **settings}  # fmt: skip
    if r["params.class_balancing"] != "none":
        cfg["balance"] = r["params.class_balancing"]
    return {**cfg, "builds_on": run, "source_run_id": run_id,
            "source_val_f1_macro": float(r["metrics.val_f1_macro"]), "reason": reason,
            "save_model": True}  # fmt: skip


def write(path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=100))


def main() -> None:
    h = R.load_history()
    s = R.selection_runs(h)
    hs = h.set_index("run_name")
    for _, f in R.final_choice(s).iterrows():
        reason = (f"Highest validation macro F1 of all model-selection runs for {f['task']}: "
                  f"{f['val macro F1']:.4f} ({f['run']}); next best {f['next best F1']:.4f} "
                  f"({f['next best']}). Rule: 'Decided 28 Oct, after E10' (highest validation "
                  f"macro F1, any experiment and timeout; only fallback a saved file over 50 MB; "
                  f"skipped for size: {f['skipped for file size']}). Saved model file "
                  f"{f['model file MiB']} MiB.")  # fmt: skip
        write(
            OUT / f"{f['task']}.yaml",
            config(f["run"], f["run_id"], f"E12-final-{f['task']}", reason, hs),
        )
    for _, c in R.comparison_set(s).iterrows():
        slug = c["method type"].lower().replace(" ", "_").replace("-", "_")
        reason = (f"Comparison set: the best 15 s validation variant of {c['method type']} for "
                  f"{c['task']} ({c['val macro F1']:.4f}, {c['run']}; {c['variants compared']} "
                  f"variants compared). Scored once on the test set in Phase 4 as comparison "
                  f"only; never changes the headline model.")  # fmt: skip
        write(
            OUT / "compare" / f"{c['task']}_{slug}.yaml",
            config(c["run"], c["run_id"], f"E12-compare-{slug}-{c['task']}", reason, hs),
        )
    compare = set(R.comparison_set(s)["run"])
    for _, o in R.old_rule_choice(s).iterrows():
        if o["run"] in compare:
            continue
        reason = (f"What the superseded 28 Oct tie rule would pick for {o['task']}: "
                  f"{o['run']} ({o['val macro F1']:.4f}, {o['below the best by']} below the "
                  f"best; the simplest of {o['runs within 0.03']} runs within 0.03). Scored once "
                  f"on the test set in Phase 4 to show what the old rule would cost; never "
                  f"changes the headline model.")  # fmt: skip
        write(
            OUT / "old_rule" / f"{o['task']}.yaml",
            config(o["run"], o["run_id"], f"E12-old_rule-{o['task']}", reason, hs),
        )
    n = len(list((OUT / "compare").glob("*.yaml")))
    m = len(list((OUT / "old_rule").glob("*.yaml")))
    print(f"Wrote 4 headline, {n} comparison and {m} old-rule configs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
