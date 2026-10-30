"""Predict traffic classes for a CSV of flows with a final model (models/final/<task>.*).

Usage: uv run python scripts/predict.py --task A1 --input flows.csv --output predictions.csv

The input holds exactly the task's 23 features in the order of models/final/<task>.json (an
optional `label` column at the end is ignored and echoed as true_label). Writes, per row, the
predicted label, its confidence and the 3 features that pushed the model most towards that label
(SHAP). A malformed file stops with an error message and exit code 2.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from blindspy.data import CORE_TASKS
from blindspy.predict import FINAL_DIR, InputError, predict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=CORE_TASKS)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--models", type=Path, default=FINAL_DIR, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        df = pd.read_csv(args.input)
        out = predict(args.task, df, args.models)
    except (InputError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"Error: {args.input}: {e}", file=sys.stderr)
        return 2
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
