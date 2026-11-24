#!/usr/bin/env bash
# Run BlindSpy predictions and report accuracy.
#
# Usage:
#   ./run.sh                              all four models (A1, A2-VPN, A2-NonVPN, B) on their
#                                         sample files, per-model results + final accuracy
#   ./run.sh all [OUTPUT_DIR]             same, writing <OUTPUT_DIR>/<task>.csv (default: predictions/)
#   ./run.sh TASK [INPUT_FILE] [OUTPUT_FILE]
#                                         one model (A1 | A2-VPN | A2-NonVPN | B) on one file
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-all}"

cd "$ROOT_DIR"

if [[ "$TASK" == "all" ]]; then
  OUTPUT_DIR="${2:-${ROOT_DIR}/predictions}"
  mkdir -p "$OUTPUT_DIR"
  echo "Running BlindSpy prediction for all models"
  echo "Output: $OUTPUT_DIR/<task>.csv"
  echo
  # Arguments: output dir, then task=input pairs.
  set -- "$OUTPUT_DIR" \
    "A1=${ROOT_DIR}/app/samples/A1_test_sample.csv" \
    "A2-VPN=${ROOT_DIR}/app/samples/A2-VPN_test_sample.csv" \
    "A2-NonVPN=${ROOT_DIR}/app/samples/A2-NonVPN_test_sample.csv" \
    "B=${ROOT_DIR}/app/samples/B_test_sample.csv"
else
  INPUT_FILE="${2:-${ROOT_DIR}/app/samples/${TASK}_test_sample.csv}"
  OUTPUT_FILE="${3:-${ROOT_DIR}/predictions.csv}"
  echo "Running BlindSpy prediction"
  echo "Task:   $TASK"
  echo "Input:  $INPUT_FILE"
  echo "Output: $OUTPUT_FILE"
  echo
  set -- "FILE:$OUTPUT_FILE" "${TASK}=${INPUT_FILE}"
fi

uv run python - "$@" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

from blindspy.data import CORE_TASKS
from blindspy.predict import FINAL_DIR, InputError, predict

dest, *pairs = sys.argv[1:]
single_file = dest.startswith("FILE:")

summary = []  # (task, correct, total, test_macro_f1)
for pair in pairs:
    task, input_file = pair.split("=", 1)
    if task not in CORE_TASKS:
        sys.exit(f"Error: unknown task '{task}' (choose from {', '.join(CORE_TASKS)} or 'all')")
    out_path = Path(dest[5:]) if single_file else Path(dest) / f"{task}.csv"

    try:
        out = predict(task, pd.read_csv(input_file), FINAL_DIR)
    except (InputError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"Error: {input_file}: {e}", file=sys.stderr)
        sys.exit(2)
    out.to_csv(out_path, index=False)

    meta = json.loads((FINAL_DIR / f"{task}.json").read_text())
    test_f1 = meta.get("test_macro_f1")

    print("=" * 60)
    print(f"Model {task}   ({len(out)} rows -> {out_path})")
    print("=" * 60)
    cols = [c for c in ["row", "true_label", "label", "confidence"] if c in out.columns]
    print(out[cols].to_string(index=False))
    print()
    if test_f1 is not None:
        print(f"Model test macro-F1 (held-out test set): {test_f1:.4f}")
    if "true_label" in out.columns:
        correct = int((out["true_label"].astype(str) == out["label"].astype(str)).sum())
        total = len(out)
        print(f"Sample accuracy: {correct}/{total} = {correct / total:.4f} ({correct / total * 100:.2f}%)")
        summary.append((task, correct, total, test_f1))
    else:
        print("Sample accuracy: unavailable (input has no 'label' column)")
        summary.append((task, None, len(out), test_f1))
    print()

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Model':<12}{'Correct':>10}{'Accuracy':>12}{'Test macro-F1':>16}")
for task, correct, total, f1 in summary:
    acc = f"{correct / total * 100:.2f}%" if correct is not None else "n/a"
    cor = f"{correct}/{total}" if correct is not None else "n/a"
    f1s = f"{f1:.4f}" if f1 is not None else "n/a"
    print(f"{task:<12}{cor:>10}{acc:>12}{f1s:>16}")

scored = [(c, t) for _, c, t, _ in summary if c is not None]
if scored:
    correct = sum(c for c, _ in scored)
    total = sum(t for _, t in scored)
    print("-" * 50)
    print(f"Final prediction accuracy: {correct}/{total} = {correct / total:.4f} ({correct / total * 100:.2f}%)")
else:
    print("\nFinal prediction accuracy: unavailable (no 'label' column in any input)")
PY
