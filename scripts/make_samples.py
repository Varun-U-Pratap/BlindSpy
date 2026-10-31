"""Write app/samples/<task>_test_sample.csv: 20 test rows per task (data v1, 15 s, seed 42), the
task's 23 features in the model's order plus the true label, for predict.py and the demo; and
app/samples/malformed_renamed_column.csv: the B sample with one column renamed
(flowBytesPerSecond -> flowBytesPerSec), which every final model must reject (exit code 2).

The rows come from the test set, which every final model saw only once, in E12.

Usage: uv run python scripts/make_samples.py
"""

from blindspy import data
from blindspy.data import CORE_TASKS, ROOT

OUT = ROOT / "app" / "samples"
ROWS = 20
MALFORMED = "malformed_renamed_column.csv"


def main() -> None:
    for task in CORE_TASKS:
        test = data.load_task(task, 15)[2]
        sample = test.sample(ROWS, random_state=42).sort_index()
        sample.to_csv(OUT / f"{task}_test_sample.csv", index=False)
        print(f"Wrote app/samples/{task}_test_sample.csv ({len(sample)} rows)")
        if task == "B":
            bad = sample.rename(columns={"flowBytesPerSecond": "flowBytesPerSec"})
            bad.to_csv(OUT / MALFORMED, index=False)
            print(f"Wrote app/samples/{MALFORMED} ({len(bad)} rows, one column renamed)")


if __name__ == "__main__":
    main()
