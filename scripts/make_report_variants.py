# ruff: noqa: E501  (the replacement tables hold report sentences that must match word for word)
"""Write the two reader editions of the report from the master drafts in reports/report/drafts/.

Both editions get the same reading aids: a "How to read this report" page with a key to the
experiment IDs (E00-E17), a note that the IDs are not in the order the work was done, the plain
name of an ID at its first mention in each section, and the real order of the work at the top of
Section 4.

- full (reports/report/drafts_full/): the master text unchanged apart from those aids.
- standalone (reports/report/drafts_standalone/): also without git details (commits, tags,
  pushed-code wording, commit columns) and without repository file paths, so it reads without the
  repository. Sentences that pointed to a file are reworded, not dropped.

The master drafts are never changed. Every replacement must match exactly once (or the stated
number of times), otherwise the script stops.

Usage: uv run python scripts/make_report_variants.py
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "reports" / "report" / "drafts"

# ID -> (plain name at first mention, what it is, order run, report sections, status)
EXPERIMENTS = {
    "E00": ("data check", "Data check: column names, file families, odd values, duplicates, sessions; data version v1", "1", "2, 4.0", "done"),
    "E01": ("majority-class baseline", "Majority-class baseline, the floor every model must beat", "2", "4.1", "done"),
    "E02": ("decision-tree baseline", "Decision tree (C4.5 approximation); also the paper benchmark with J48 settings", "2", "4.2, 4.3", "done"),
    "E03": ("KNN, logistic regression and SVM baselines", "KNN, logistic regression and SVM-RBF baselines; also the paper benchmark with IBk settings", "2", "4.2, 4.3", "done"),
    "E04": ("KNN with a signed log", "KNN with a signed log transform and two scalers", "3", "4.6", "done"),
    "E05": ("Random Forest", "Random Forest, 300 trees", "4", "4.7", "done"),
    "E06": ("gradient boosting", "Gradient boosting: XGBoost and LightGBM with default settings", "5", "4.8", "done"),
    "E07": ("tuning", "Hyperparameter tuning with Optuna of the best ensembles and KNN", "6", "4.9", "done"),
    "E08": ("class balancing", "Class balancing: class weights, oversampling, SMOTE", "7", "4.10", "done"),
    "E09": ("feature-group ablation", "Feature-group ablation of the final models (analysis only)", "11", "4.14", "done"),
    "E10": ("one model or two steps for task B", "Task B as one 14-class model or two steps", "8", "4.11", "done"),
    "E11": ("session-aware split","Session-aware split", "-", "4.17", "not run: the files hold no session information"),
    "E12": ("final test", "Final test: the 30 final models scored once on the test set", "17", "4.21, 5", "done"),
    "E13": ("learning curve", "Learning curve of the final models (analysis only)", "12", "4.15", "done"),
    "E14": ("split-seed stability", "Split-seed stability of the final models (analysis only)", "13", "4.16", "done"),
    "E15": ("calibration", "Calibration of the final models' confidence, on validation", "16", "4.20", "done"),
    "E16": ("reproducing Isnan et al.", "Reproducing Isnan et al. (2026) on the Kaggle copy, then correcting it", "15", "4.18", "done"),
    "E17": ("equal tuning budget", "The same tuning budget for every method type", "9", "4.12", "done"),
}  # fmt: skip

HOW_TO_READ = """# How to read this report

Every experiment in this project has an ID, E00 to E17, and the report uses these IDs throughout:
in headings ("4.6 E04: KNN with a signed log transform"), in the text ("E04 lifted KNN by ..."),
in table rows and in run names such as E07-xgb-B-15s, which reads "experiment E07, XGBoost, task
B, 15 s files". The table below says what each ID is, in which order the experiments were run and
where the report describes them.

| ID | What it is | Order run | Section | Status |
|---|---|---|---|---|
{rows}

**The IDs are not in the order the work was done.** Each ID was given when the experiment was
planned. The plan changed as results came in: E17 was added after E10, and the analyses E09, E13
and E14 had to wait until the final models were chosen. The real order was:

{order}

Three steps have no ID of their own: the paper benchmark (part of E02 and E03, Section 4.3), the
storage checks (Sections 4.4 and 4.19) and the final choice of models with the comparison set
(Section 4.13). Section 4 is ordered by ID; the order list at its start repeats the sequence above.
"""

ORDER = [
    "E00 (data check, data version v1)",
    "E01, E02 and E03 (baselines), with the paper benchmark",
    "E04 (KNN with a signed log)",
    "E05 (Random Forest)",
    "E06 (gradient boosting)",
    "E07 (tuning)",
    "E08 (class balancing)",
    "E10 (one model or two steps for task B)",
    "E17 (equal tuning budget)",
    "final choice of models and the comparison set (Section 4.13)",
    "E09 (feature-group ablation)",
    "E13 (learning curve)",
    "E14 (split-seed stability)",
    "E11 recorded as not run (no session information)",
    "E16 (reproducing Isnan et al.)",
    "E15 (calibration, on validation, before the test set)",
    "E12 (final test: the test set opened once)",
]

SECTION4_ORDER = """**The order in which the experiments were run** (this section is ordered by ID; see the key
before Section 1):

{order}

"""

ID = re.compile(r"(?<![\w\-–])(E\d\d)(?![\w\-–])")


def key_page() -> str:
    rows = "\n".join(f"| {k} | {v[1]} | {v[2]} | {v[3]} | {v[4]} |" for k, v in EXPERIMENTS.items())
    order = "\n".join(f"{n}. {step}" for n, step in enumerate(ORDER, start=1))
    return HOW_TO_READ.format(rows=rows, order=order)


def drop_working_note(text: str) -> str:
    """The italic 'Working draft' paragraph under each draft's title (the build drops it too)."""
    return re.sub(r"\n\*Working draft\..*?\*\n", "\n", text, count=1, flags=re.S)


def name_first_mentions(text: str) -> str:
    """Add the plain name after the first mention of each ID in a section, outside headings,
    tables, images and mentions that already name it ("E04: ..." or "E04 (...)")."""
    seen, out = set(), []
    for line in text.split("\n"):
        if line.startswith(("#", "|", "![")):
            out.append(line)
            continue

        def add(m: re.Match, line: str = line) -> str:
            eid, rest = m.group(1), line[m.end() :]
            core = EXPERIMENTS.get(eid, ("",))[0].split(",")[0].lower()
            named_before = core and core in line[max(0, m.start() - 60) : m.start()].lower()
            if (
                eid in seen
                or eid not in EXPERIMENTS
                or rest.startswith((":", " ("))
                or named_before
            ):
                seen.add(eid)
                return eid
            seen.add(eid)
            if line[: m.start()].endswith("(") and rest.startswith(")"):
                return f"{eid}, {EXPERIMENTS[eid][0]}"  # "(E04)" -> "(E04, KNN with a signed log)"
            return f"{eid} ({EXPERIMENTS[eid][0]})"

        out.append(ID.sub(add, line))
    return "\n".join(out)


def sub(text: str, old: str, new: str, count: int = 1) -> str:
    """Replace `old` (whitespace may be a line break in the draft) exactly `count` times."""
    pattern = re.compile(r"\s+".join(map(re.escape, old.split())))
    found = len(pattern.findall(text))
    if found != count:
        raise SystemExit(f"expected {count} match(es), found {found}: {old[:70]!r}")
    return pattern.sub(new.replace("\\", "\\\\"), text)


# Doc 1 only: git details and repository paths, per draft. Each pair must match once.
STANDALONE = {
    "01_introduction.md": [
        ("Paper values are taken from `reports/paper_baseline.csv`, which records each number's page and whether",
         "Each paper value is recorded with its page and whether"),
        ("A command-line predictor (`scripts/predict.py`)", "A command-line predictor (`predict.py`)"),
        ("traceable to a commit.", "traceable to the exact code that produced it."),
    ],
    "02_dataset.md": [
        ("The SHA-256 hash of every file is recorded in `data/arff_sha256.txt`.",
         "The SHA-256 hash of every file is recorded so anyone can check their copy."),
        ("`COLUMN_MAP` in `src/blindspy/data.py` renames", "The column map (`COLUMN_MAP`) renames"),
        (" (`reports/data_checks_v1.json`).", "."),
        ("Every rule is in `configs/base.yaml` and is applied by `scripts/prepare_data.py` in this order.",
         "Every rule is written in the project's configuration and is applied by the preparation script in this order."),
        ("| Rows (raw lines -> final) | Commit |\n|---|---|---|---|---|---|---|",
         "| Rows (raw lines -> final) |\n|---|---|---|---|---|---|"),
        ("| 179,120 -> 167,909 | 0a36896 |", "| 179,120 -> 167,909 |"),
        ("Figures from the exploration notebook (`notebooks/01_eda.ipynb`): class counts per task and timeout (`class_counts_v1.png`), feature distributions by class at every timeout (`feature_boxes_<task>_<timeout>s.png`, 16 figures) and feature correlations (`correlation_<task>_15s.png`, all four tasks), the data funnel by class (`data_funnel_by_class_v1.png`). The notebook records",
         "The exploration notebook adds class counts per task and timeout, feature distributions by class at every timeout (16 figures), feature correlations for all four tasks, and the data funnel by class. It records"),
    ],
    "03_method.md": [
        ("were added together in one commit before", "were added together in one change before"),
        ("recorded in `data/arff_sha256.txt` and checked on every machine.", "recorded and checked on every machine."),
        ("`scripts/prepare_data.py` parses the files", "A preparation script parses the files"),
        ("to a versioned folder `data/processed/vN/`.", "to a versioned folder."),
        ("`scripts/run_experiment.py` reads one YAML configuration file", "A run script reads one YAML configuration file"),
        ("`scripts/export_history.py` exports every run to `reports/history.csv`; `scripts/results_tables.py` builds every result table from it, and the notebooks (written for Phase 3 by `scripts/make_notebooks.py`) draw the figures from it,",
         "every run is exported to one history table; every result table is built from it by code, and the notebooks draw the figures from it,"),
        ("the git commit, whether there were uncommitted changes,", "the exact code version, whether there were unsaved changes,"),
        ("only from code that is committed and pushed to the shared repository;", "only from saved code shared with the team;"),
        ("and milestones are marked with git tags (v0.1-baseline, v0.2-tuned, v1.0-final).",
         "and each milestone (baselines, tuned models, final models) is marked so its exact code can be recovered."),
        ("built by a committed script", "built by a script"),
        (" (GitHub's warning size)", ""),
        ("Fixed in `configs/final/compare/` before", "Fixed before"),
        ("saved in the repository (`models/final/`);", "kept as final model files;"),
    ],
    "04_experiments.md": [
        ("The first Phase 2 runs (commits 1817c5a and 36c56a1) timed", "The first Phase 2 runs timed"),
        ("rerun from the fixed code (commit c450c3e) with", "rerun from the fixed code with"),
        ("The same tables for KNN, Logistic Regression and SVM-RBF, plus accuracy, balanced accuracy, weighted F1 and A1 ROC-AUC for every run, are in `reports/baseline_tables.md` (Appendix).",
         "The same tables were also produced for KNN, Logistic Regression and SVM-RBF, with accuracy, balanced accuracy, weighted F1 and A1 ROC-AUC for every run."),
        (" (also cited in `notebooks/02_baselines.ipynb`)", ""),
        ("The same tables on data v1 are in `reports/baseline_tables.md`.", "The same tables were also produced on data v1."),
        ("Every table below is copied unchanged from `reports/phase3_tables.md`, generated by `scripts/results_tables.py --phase 3` from `reports/history.csv`; the figures come from notebooks 03 and 04 (`scripts/make_notebooks.py`).",
         "Every table below is copied unchanged from tables generated by code from the run history; the figures come from notebooks 03 and 04."),
        ("It was fixed in `configs/final/compare/` before", "It was fixed before"),
        ("all below GitHub's 50 MB warning size.", "all below the 50 MB limit set for final model files."),
        ("in one batch from commit 87181d2, with", "in one batch, with"),
    ],
    "05_results.md": [
        ("with the settings fixed in `configs/final/` before", "with the settings fixed before"),
        ("An independent recomputation from the saved predictions (`scripts/check_e12_independent.py`) reproduced",
         "An independent recomputation from the saved predictions reproduced"),
        ("(`scripts/explain_final.py`, notebook `05_explain`)", "(notebook 05)"),
        ("`scripts/predict.py` checks a flow file", "The command-line predictor (`predict.py`) checks a flow file"),
        ("`app/demo.py` is a one-page Streamlit [23] app built only on `models/final/` and the code above (`blindspy.predict` for the labels and confidences, `blindspy.explain` for the SHAP values)",
         "The demo is a one-page Streamlit [23] app built only on the four final models and the code above (the predictor for the labels and confidences, the explanation code for the SHAP values)"),
        ("five sample files in `app/samples/` (20 test rows per task, written by `scripts/make_samples.py`, and one",
         "five sample files (20 test rows per task, and one"),
        ("message as `scripts/predict.py`. The page is tested headless (`tests/test_demo.py`); the screenshots come from `scripts/demo_screenshots.py`. A 1-minute video of the demo, `reports/demo/demo_video.mp4`, goes through the steps of `reports/demo/video_script.md` on the",
         "message as the command-line predictor. The page is tested headless. A 1-minute video of the demo goes through the steps of its video script on the"),
        ("it was made by `scripts/record_demo.py`, which drives", "it was made by a script that drives"),
    ],
    "07_limitations.md": [
        ("(`reports/fold_order_check.json`; A1", "(A1"),
    ],
    "08_conclusion.md": [
        ("and `scripts/predict.py` and the Streamlit demo", "and the command-line predictor and the Streamlit demo"),
        ("2,013 MLflow runs exported to `reports/history.csv`, one log entry per experiment (`reports/EXPERIMENT_LOG.md`), a changelog per data version (`reports/DATA_CHANGELOG.md`) and a git tag at each milestone (Appendix).",
         "2,013 MLflow runs in one exported history table, one log entry per experiment, a changelog per data version, and a saved copy of the exact code at each milestone (Appendix)."),
    ],
    "04_experiments.md#4": [("fixed (configs/final/B.yaml)", "fixed (final task-B settings)")],
}  # fmt: skip

STANDALONE_APPENDIX = """# Appendix

## A.1 Repository

The code, configurations, experiment notes and results are at
https://github.com/Varun-U-Pratap/BlindSpy.

## A.2 MLflow screenshots

The official MLflow store holds all 2,013 runs (A.5). Below: the run list of the `blindspy`
experiment (newest first: the 30 E12 runs), and one final run, E12-final-B, with its test metrics
and the tags every run carries (runner, machine, code version, lock-file hash, package versions,
data version and data hashes). The demo is shown in Section 5.11.

![MLflow: the run list of the blindspy experiment](../../demo/mlflow_runs.png)

![MLflow: run E12-final-B, its metrics and tags](../../demo/mlflow_run_E12-final-B.png)

## A.3 Data changelog

{{include reports/report/drafts_standalone/appendix_DATA_CHANGELOG.md}}

## A.4 Experiment log

The full experiment log, one entry per experiment (newest experiment first; failures, corrections
and rechecks included as they were written):

{{include reports/report/drafts_standalone/appendix_EXPERIMENT_LOG.md}}

## A.5 Every run

{{include reports/report/drafts_standalone/appendix_runs.md}}

Every parameter, metric and tag of each of these runs, and every text file saved with a run, are
kept with the project's files.
"""

HASH = r"[0-9a-f]{7}"
# Commit and push wording in the experiment log, applied in this order (regex, replacement).
LOG_RULES = [
    (
        rf",\s+commit\s+{HASH}\s+\(merged\s+as\s+{HASH}\);\s+all\s+8\s+runs\s+then\s+came\s+from\s+that\s+commit\.",
        "; all 8 runs then came from the corrected code.",
    ),
    (
        rf",\s+from\s+dev\s+commit\s+{HASH}\s+\(committed\s+and\s+pushed;\s+tag\s+v0\.1-baseline\)\.",
        ".",
    ),
    (rf"\(dev\s+{HASH},\s+tag\s+v0\.1\.1-baseline\)", ""),
    (rf"\s+\(({HASH}),\s+({HASH});\s+tag\s+v0\.1-baseline\)", ""),
    (r"\(committed\s+and\s+pushed;\s+tag\s+v0\.1-baseline\)", ""),
    (r"\s*\(committed\s+and\s+pushed;\s+tag\s+v0\.1-baseline\)", ""),
    (r",\s+Part\s+2\s+\(Phase\s+4,\s+tag\s+v1\.0-final\)", ", Part 2 (Phase 4)"),
    (r"Part\s+2\s+\(Phase\s+4,\s+tag\s+v1\.0-final\)", "Part 2 (Phase 4)"),
    (r"\s+on\s+branch\s+exp/E12-final", ""),
    (rf"scratch\s+clone\s+of\s+{HASH}", "scratch copy of the code"),
    (
        rf"The\s+training\s+code\s+and\s+configs\s+of\s+{HASH}\s+equal\s+those\s+of\s+{HASH}\.",
        "The training code and configs of the official runs equal those of the rehearsal.",
    ),
    (rf",\s+merged\s+as\s+{HASH}\)", ")"),
    (rf"\s+\(merged\s+into\s+dev\s+as\s+{HASH}\)", ""),
    (rf"\s+\(merged\s+as\s+{HASH}\)", ""),
    (rf"\s+\((?:dev\s+)?commit\s+{HASH}\)", ""),
    (rf",\s+(?:dev\s+)?commit\s+{HASH},", ","),
    (rf",\s+(?:dev\s+)?commit\s+{HASH}\)", ")"),
    (rf"\s+from\s+dev\s+commit\s+{HASH}", ""),
    (rf"\s+from\s+commit\s+{HASH}", ""),
    (rf"\s+from\s+(?:dev\s+)?{HASH}", ""),
    (rf"\s+at\s+dev\s+{HASH}", ""),
    (rf"Setup\s+commit\s+{HASH}", "The setup change"),
    (rf"A\s+fresh\s+scratch\s+clone\s+of\s+dev\s+at\s+{HASH}", "A fresh scratch copy of the code"),
    (rf"Commit\s+{HASH}\.\s+", ""),
    (rf"hypothesis\s+committed\s+in\s+{HASH},", "hypothesis recorded"),
    (
        rf"\(hypothesis\s+{HASH}\s+and\s+the\s+fold\s+correction\s+{HASH}\s+were\s+committed\s+and\s+pushed\s+before\s+them\)",
        "(the hypothesis and the fold correction were recorded before them)",
    ),
    (rf"Fixed\s+\(commit\s+{HASH}\):", "Fixed:"),
    (rf"commit\s+{HASH}", "the corrected code"),
    (r",\s+committed\s+and\s+pushed,", ","),
    (r"\s+committed\s+and\s+pushed,", ""),
    (r"\(committed\s+and\s+pushed;?\)", ""),
    (r"\s*\(all\s+FINISHED,\s+committed\s+and\s+pushed,", " (all FINISHED,"),
    (r"all\s+FINISHED,\s+committed\s+and\s+pushed", "all FINISHED"),
    (r"clean\s+and\s+pushed", "clean"),
    (r"clean,\s+pushed,", "clean,"),
    (r"committed\s+before", "recorded before"),
    (r"committed\s+(\d+\s+s)\s+before", r"recorded \1 before"),
    (r"corrected\s+commit;", "corrected code;"),
    (r"from\s+committed\s+and\s+pushed\s+code", "from saved, shared code"),
    (r"identical\s+committed\s+config", "identical saved config"),
    (r"the\s+git\s+commit\s+and\s+lock-file\s+hash", "the code version and lock-file hash"),
    (r"\(committed\)", "(kept with the project)"),
    (r"committed\s+as\s+an\s+ordinary\s+file", "stored as an ordinary file"),
    (r"the\s+setup\s+commit", "the setup change"),
    (r"configs\s+and\s+commits\s+are\s+kept", "configs and code versions are kept"),
    (r"pushed-code\s+guard", "shared-code guard"),
    (r"generated\s+files\s+were\s+uncommitted", "generated files were not saved yet"),
    (r"the\s+clean\s+commit", "the clean code"),
    (r"GitHub's\s+50\s+MB\s+warning\s+size", "the 50 MB size limit for final model files"),
    (r"\s*\(see\s+E02\)", " (see E02)"),
]


def strip_log(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Apply LOG_RULES paragraph by paragraph; return the text and every changed paragraph."""
    paras, changed = text.split("\n\n"), []
    for i, p in enumerate(paras):
        new = p
        for pattern, repl in LOG_RULES:
            new = re.sub(pattern, repl, new)
        new = re.sub(r"[ \t]{2,}", " ", new).replace(" )", ")").replace("( ", "(")
        new = new.replace(" ,", ",").replace(" .", ".")
        if new != p:
            changed.append((p, new))
            paras[i] = new
    return "\n\n".join(paras), changed


def drop_last_column(text: str, header_last: str) -> str:
    """Drop the last cell of every row of the table whose header ends with `header_last`."""
    out, inside = [], False
    for line in text.split("\n"):
        if line.startswith("|") and line.rstrip().endswith(f"| {header_last} |"):
            inside = True
        elif not line.startswith("|"):
            inside = False
        if inside:
            line = line.rstrip()[:-1].rsplit("|", 1)[0] + "|"
        out.append(line)
    return "\n".join(out)


GIT_WORDS = re.compile(
    r"\bgit\b|github(?!\.com/Varun)|\bcommits?\b|\bcommitted\b|\bpushed\b(?! the model)|"
    r"\bv0\.[12][.0-9]*-[a-z]+|v1\.0[.0-9]*-final|\bdev\b|\bbranch\b",
    re.I,
)
HEX = re.compile(r"(?<![\w.])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}(?![\w])")


def leftovers(text: str) -> list[str]:
    """Git words and commit-like hashes still in a standalone text (empty when clean). A
    32-character hex string is an MLflow run ID, not a commit, and is allowed."""
    found = [m.group(0) for m in GIT_WORDS.finditer(text)]
    return found + [m.group(0) for m in HEX.finditer(text) if len(m.group(0)) != 32]


def build(variant: str, out: Path) -> list[tuple[str, str]]:
    """Write one edition to `out`; return the changed experiment-log paragraphs (standalone)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "00_how_to_read.md").write_text(key_page())
    for path in sorted(DRAFTS.glob("[0-9][0-9]_*.md")):
        text = drop_working_note(path.read_text())
        if variant == "standalone":
            for old, new in STANDALONE.get(path.name, []):
                text = sub(text, old, new)
            for old, new in STANDALONE.get(path.name + "#4", []):
                text = sub(text, old, new, count=4)
        text = name_first_mentions(text)
        if path.name.startswith("04_"):
            order = "\n".join(f"{n}. {step}" for n, step in enumerate(ORDER, start=1))
            text = sub(text, "## 4.0 E00:", SECTION4_ORDER.format(order=order) + "## 4.0 E00:")
        (out / path.name).write_text(text)
    changed = []
    if variant == "full":
        (out / "appendix.md").write_text(drop_working_note((DRAFTS / "appendix.md").read_text()))
        return changed
    (out / "appendix.md").write_text(STANDALONE_APPENDIX)
    log, changed = strip_log((ROOT / "reports" / "EXPERIMENT_LOG.md").read_text())
    (out / "appendix_EXPERIMENT_LOG.md").write_text(log)
    changelog = (ROOT / "reports" / "DATA_CHANGELOG.md").read_text()
    (out / "appendix_DATA_CHANGELOG.md").write_text(drop_last_column(changelog, "Commit"))
    runs = (ROOT / "reports" / "appendix_runs.md").read_text()
    runs = sub(runs, "Generated by `scripts/results_tables.py --phase 5` from `reports/history.csv`, which holds every parameter, metric and tag of each run.",
               "Generated by code from the run history, which holds every parameter, metric and tag of each run.")  # fmt: skip
    (out / "appendix_runs.md").write_text(drop_last_column(runs, "commit"))
    for f in sorted(out.glob("*.md")):
        if bad := leftovers(f.read_text()):
            raise SystemExit(f"{f.name}: git details left: {sorted(set(bad))}")
    return changed


def main() -> None:
    for variant in ("full", "standalone"):
        out = ROOT / "reports" / "report" / f"drafts_{variant}"
        changed = build(variant, out)
        print(f"wrote {out.relative_to(ROOT)} ({len(list(out.glob('*.md')))} files)")
        for old, new in changed:
            print("--- log, before:\n" + old + "\n+++ after:\n" + new + "\n")


if __name__ == "__main__":
    main()
