# Phase 5 evidence (hand-in)

Record of how each Phase 5 item and each item of the plan's Final checklist was checked before
hand-in, 31 Oct 2026. Phase 5 added no MLflow run to the official store and changed no result:
the test set was scored only by the 30 E12 runs (commit 87181d2) in Phase 4.

## Fresh clone of dev, following the README

A fresh clone of `dev` from GitHub (commit 84b9d63, the README commit), in a scratch folder with
its own runner (`readme-check`, so its runs go to its own MLflow store) and the raw dataset linked
in. Every command was copied from the README's "Run it on any laptop" section:

| README step | Result |
|---|---|
| 2. `git clone`, `uv sync` | exit 0 |
| 3. `predict.py --task B` on `app/samples/B_test_sample.csv` | exit 0, 20 predictions written |
| 3. `predict.py --task B` on `app/samples/malformed_renamed_column.csv` | exit 2, "missing ['flowBytesPerSecond'], unexpected ['flowBytesPerSec']", no output file |
| 3. `uv run streamlit run app/demo.py` | health check `ok`; the page rendered the A1 predictions, and a click on row 10 drew its SHAP chart (headless Chrome screenshot) |
| 4. `shasum -a 256 -c data/arff_sha256.txt` | 20 of 20 `OK` |
| 4. `prepare_data.py --version v1` | exit 0; 34 files, combined SHA-256 2725f36b06b2b251b8c1f0d3f26171bf6f0717b16eb077d2c22c60c0217e0f7c, identical to the official v1 |
| 5. `run_experiment.py configs/experiments/E12_*.yaml` (runner `readme-check`) | exit 0, 30 runs; warnings only the expected personal-runner notice and the known non-convergence of the A2-NonVPN logistic regression (E17) |
| 5. `check_final_scores.py --runner readme-check` | exit 0: "30 of 30 runs identical to the official scores; 30 of 30 equal to 4 decimals" |
| "Reproduce a baseline": `run_experiment.py configs/experiments/E02_tree_A1_15s.yaml` | validation macro F1 0.890145471320929, CV macro F1 0.888418404598205, validation accuracy 0.890243902439024: identical to the official run |
| `uv run pytest` | 244 passed, 1 skipped (the test that needs the official MLflow store) |

Not tested: the Windows and Linux commands (marked "not yet tested" in the README until a
teammate runs them); scores on another operating system or CPU (not measured).

## Phase 5 items

| Plan item | Files / commits | How it was checked | Result |
|---|---|---|---|
| Report sections 6 Discussion, 7 Limitations, 8 Conclusion; section 1 updated | `reports/report/drafts/06_discussion.md`, `07_limitations.md`, `08_conclusion.md`, `01_introduction.md` (0927fb2) | Every number compared with the generated tables it repeats (`reports/phase3_tables.md`, `phase4_tables.md`, `fold_order_check.json`); three wrong statements found and fixed before the commit (the CV spread, the weak-class cause, KNN vs headline clear in 3 tasks, not "every task") | Done |
| References for Holm, the bootstrap, isotonic calibration, ECE, Spearman and Streamlit | `reports/references.md` [17]-[23] (0927fb2) | Each source checked online (journal, volume, pages; Zadrozny and Elkan without pages, which could not be confirmed); each cited in the text: [17]-[21] in Section 3, [22] in Section 4.21, [23] in Sections 5.11 and 8 | Done |
| Every number cites its experiment ID; results split into reproduced, improved and limitations | Sections 6.4, 7 | Section 6.4 lists reproduced / improved / limits; Section 7 gives each limitation with the experiment or file that measured it | Done |
| Appendix: MLflow screenshots, every run, EXPERIMENT_LOG and DATA_CHANGELOG in full, tag list, repo link | `reports/report/drafts/appendix.md`, `reports/appendix_runs.md` (`results.appendix_runs`, `results_tables.py --phase 5`), `scripts/build_report.py` (42cd63a) | Run table: 2,013 rows, one per run, a test score on exactly the 30 E12 runs (`tests/test_results.py`); generated twice, same SHA-256; includes and tag table tested (`tests/test_build_report.py`); the MLflow screenshots show no side panel | Done |
| Demo `app/demo.py` with 5 sample files | `app/demo.py`, `app/samples/` (5th file `malformed_renamed_column.csv` from `scripts/make_samples.py`, which rewrote the other 4 byte-identical), `tests/test_demo.py` (c056507) | Headless page tests (every task's sample gives 20 predictions; the malformed file gives one error and no table; the chart of a clicked row equals predict.py's top 3); checked live in the browser (row click, malformed file, no server or console errors) and from the fresh clone | Done |
| Streamlit added without changing locked versions | `pyproject.toml`, `uv.lock` (c0bb692) | `uv export` before and after: only altair 6.3.0, pydeck 0.9.3, python-multipart 0.0.32, streamlit 1.66.1, watchdog 6.0.0 (not on macOS) and websockets 17.4 added; no version changed | Done |
| Screenshot for the report | `reports/demo/*.png`, `scripts/demo_screenshots.py`, Section 5.11 | Two demo screenshots and two MLflow screenshots, light theme, looked at before committing | Done |
| README for markers, every command tested from a fresh clone | `README.md`, `scripts/check_final_scores.py` (84b9d63) | See the fresh-clone table above; Windows and Linux steps marked "not yet tested" | Done (macOS) |
| Final report files, every page checked | `reports/report/BlindSpy_report.pdf` (177 pages) and `.docx` (28eca08) | Every PDF page looked at as contact sheets (pdftoppm, 12 pages per sheet); DOCX passes the docx validator and was opened in Microsoft Word (153 pages in Word's pagination; cover, tables, figures and the last page looked at; file unchanged afterwards) | Done |
| Demo video, shrunk, under `reports/demo/` | `reports/demo/video_script.md` | Script written (one minute, click by click); the recording is made on this Mac by a team member with Cmd+Shift+5, then checked, shrunk with HandBrakeCLI and committed | Pending (recording) |
| Hand-in tag; origin/main equals it | tag `v1.0.1-final` | Placed on the final dev head once the video is committed; then Varun merges dev into main and origin/main is compared with the tag | Pending |

## Final checklist

| Final-checklist item | Evidence | Result |
|---|---|---|
| A reviewer can reproduce a baseline result from the repository | README "Reproduce a baseline"; from the fresh clone, `E02_tree_A1_15s.yaml` was rerun: validation macro F1, CV macro F1 and validation accuracy identical to the official run (table above) | Done |
| The log and progress chart show how each experiment changed the scores | `reports/EXPERIMENT_LOG.md` (every experiment, failures and corrections included; in full in Appendix A.4); `reports/figures/progress_chart.png` (Section 5.1) | Done |
| Final models are scored once, on untouched test data, with confidence intervals | `reports/history.csv`: exactly 30 runs carry a test score, all E12, all from 87181d2, clean and pushed; 95% intervals from 10,000 stratified paired resamples in `reports/phase4_tables.md` (Section 5.1, 5.3) | Done |
| A2 is evaluated as two separate 7-class models | `configs/final/A2-VPN.yaml`, `A2-NonVPN.yaml`; `models/final/A2-VPN.*`, `A2-NonVPN.*`; every table reports them separately | Done |
| Macro F1, per-class metrics and confusion matrices for every task | Section 5.1-5.2 (per-class tables of all four tasks, top-5 confusion pairs), `reports/figures/final_confusion_matrices.png` | Done |
| SHAP and the ablation show what the models learned from timing features | Section 5.6 and 6.3; `reports/shap_*.csv`; E09 | Done |
| Each final model loads, and predict.py classifies a correct flow file and rejects a malformed one | `tests/test_predict.py` (committed models on committed samples; the malformed sample exits 2 for all four tasks); fresh clone: exit 0 on `B_test_sample.csv`, exit 2 and no output file on `malformed_renamed_column.csv` | Done |
| The report separates reproduced results, improvements and limitations, incl. the 2016 lab-data caveat | Sections 1.3, 6.4, 7.1, 8 | Done |
| Every final number traces to a config, an MLflow run and a git commit made with no uncommitted changes | `models/final/<task>.json`: config file (exists), MLflow run ID (its run in `history.csv` has the same test macro F1), git commit 87181d2; in `history.csv` no run is dirty or unpushed | Done |
| Every data version has a kept snapshot and a changelog entry, and every reported run names its data version | `reports/DATA_CHANGELOG.md`: v1 and kaggle-k1; both rebuild byte-identical (v1 again from the fresh clone today), and both are in the v1.0-final backup zip; every run in `history.csv` names its data version (v1 1,918, published 64 = the paper benchmark on the rows as published, kaggle-k1 30) except the setup smoke run, which uses no data | Done |
| E13, E14 and E15 are reported for every task, and every experiment shows its pre-stated threshold and verdict | Sections 5.4-5.5, 4.15-4.16, 4.20; Section 5.9 (verdict table generated from the threshold checks of all three tables files) | Done |
| The demo runs on the presenting laptop from a fresh start, and the backup video is saved offline | Fresh clone: `uv run streamlit run app/demo.py` served the page, a row click drew its chart; the backup video is pending (script in `reports/demo/video_script.md`) | Pending (recording) |
