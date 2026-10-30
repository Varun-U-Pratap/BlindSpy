"""Evidence for the E15 correction (31 Oct, before the test set): out-of-fold predictions of each
headline config on its TRAINING rows (validation and test rows are not loaded into any model),
with unshuffled 5-fold StratifiedKFold (what `cv=5` means in scikit-learn) and with the
project's shuffled folds (seed 42). The training rows are stored in file order, so unshuffled
folds hold out whole blocks of the file.

Writes reports/fold_order_check.json. Usage: uv run python scripts/fold_order_check.py
"""

import json

import numpy as np
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from blindspy import calibration, data, models
from blindspy.data import CORE_TASKS, ROOT

OUT = ROOT / "reports" / "fold_order_check.json"
SEED = 42


def check(task: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "final" / f"{task}.yaml").read_text())
    train = data.load_task(task, cfg["timeout"])[0]
    X, y = train.drop(columns="label"), train["label"]
    pipe = models.build_model(cfg["model"], cfg.get("params"), SEED, cfg.get("balance"))
    out = {"config": f"configs/final/{task}.yaml", "train_rows": len(train),
           "train_rows_in_file_order": bool(train.index.is_monotonic_increasing)}  # fmt: skip
    folds = {"unshuffled (cv=5)": StratifiedKFold(5),
             "shuffled (seed 42)": StratifiedKFold(5, shuffle=True, random_state=SEED)}  # fmt: skip
    for name, cv in folds.items():
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
        classes = np.array(sorted(y.unique()))
        pred = classes[proba.argmax(axis=1)]
        correct = pred == y.to_numpy()
        out[name] = {
            "oof_accuracy": float(correct.mean()),
            "oof_f1_macro": float(f1_score(y, pred, average="macro")),
            "oof_mean_confidence": float(proba.max(axis=1).mean()),
            "oof_ece": calibration.ece(proba.max(axis=1), correct),
        }
    return out


def main() -> None:
    result = {task: check(task) for task in CORE_TASKS}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    for task, r in result.items():
        u, s = r["unshuffled (cv=5)"], r["shuffled (seed 42)"]
        print(f"{task}: OOF accuracy unshuffled {u['oof_accuracy']:.4f}, "
              f"shuffled {s['oof_accuracy']:.4f}")  # fmt: skip
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
