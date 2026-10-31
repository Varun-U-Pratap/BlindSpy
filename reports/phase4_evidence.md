# Phase 4 evidence (tag v1.0-final)

Record of how each Phase 4 item was checked before the phase was closed on 31 Oct 2026. The tag
`v1.0-final` is on commit 3c31bad, the dev head after the v1.0-final report files were committed.
The tag message lists the Phase 4 run commits and every change to shared training code.

## Verification at the tag

From a fresh clone of the repository at `v1.0-final`, with its own MLflow store (runner
`verify-clone`, so the official history is not touched) and the raw datasets linked in:

| Check | Result |
|---|---|
| Rebuild data v1 (`scripts/prepare_data.py --version v1`) | 34 files, combined SHA-256 2725f36b06b2b251b8c1f0d3f26171bf6f0717b16eb077d2c22c60c0217e0f7c, identical to the official v1 |
| `pytest` | 229 passed, 1 failed: `test_confusion_pairs_match_the_headline_details` read the official MLflow store, which a clone does not have. Fixed after the tag on dev (586d20b: the test is skipped when the store is absent, like the other tests that need local data); with the fix the clone gives 2 passed, 1 skipped for that file |
| Rerun all 30 E12 configs (fit on train + validation, score the test rows) and compare with the official runs | 30 of 30 identical: every test-prediction file byte-identical, every non-timing metric equal, so every test score reproduces exactly |
| Rerun the 4 E15 configs and 4 earlier configs (E02 tree A1, E03 SVM B, E06 XGBoost B, E08 SMOTE pipeline A2-NonVPN) | 8 of 8 identical on every non-timing metric; with the 30 E12 runs, 1,813 metric values compared, 0 differences |
| `scripts/predict.py` on the four sample files (`app/samples/`) | exit 0; output identical to the main folder's for all four tasks; a file with the wrong task's columns stops with exit code 2 |

The four earlier configs exercise the shared code that Phase 4 changed (the calibration and final
run kinds and the calibration folds, all before the E12 runs; tag message): their results are
unchanged. Timing metrics (training and prediction time, memory) differ between runs and are not
compared.

## Every Phase 4 item

| Plan item | Files / commits | How it was checked | Result |
|---|---|---|---|
| Open points settled before the test set (tests, bootstrap, calibration, rehearsal, crash rules) | E12.md addendum and correction (7891274, 9fcb4a1) | The first 14,052 bytes of today's E12.md (part 1, addendum, correction) are byte-identical to commit 87181d2, before the test runs | Done |
| E15 calibration on validation, before the test set | E15.md; 4 runs from 75929cb (pilot 6281f60 superseded) | Verdicts (a)-(c) generated; ECE recomputed independently; the predicted class unchanged on 100% of rows; reruns at the tag identical | Done |
| Retrain each final configuration on train + validation, score the test set once, bootstrap intervals | E12.md part 2; 30 runs from 87181d2; `reports/phase4_tables.md` | Only the 30 E12 runs carry a test metric in the store, all from one commit; no run started after them; rehearsal on validation passed 21 of 21 checks first (`reports/e12_rehearsal_check.txt`); `scripts/check_e12_independent.py` recomputed every score, interval, p and Holm p from the saved predictions with 0 mismatches; reruns at the tag identical | Done |
| Comparison set and the 26 paired tests (Holm) | `reports/phase4_tables.md` | 14 of 26 clear; the Monte Carlo error of each p is a generated table (2 tests on the Holm line, borderline; recheck note in E12.md) | Done |
| Save each pipeline to `models/final/` with its JSON, under 50 MB | `models/final/` (2a01b3e); `scripts/save_final_models.py` | Each model reproduces its run's test predictions; files 4.60-28.14 MiB; the SHA-256 in each JSON matches; regenerating gives byte-identical files | Done |
| `scripts/predict.py` | `src/blindspy/predict.py`, `scripts/predict.py`, `tests/test_predict.py` | Full test set through predict.py = the saved predictions (label and confidence) for all 4 tasks; malformed files rejected with exit 2; works from a fresh clone | Done |
| Notebook 05 (SHAP, compared with E09, worked examples) | `src/blindspy/explain.py`, `scripts/explain_final.py`, `notebooks/05_explain.ipynb` | SHAP values add up to the model margin; explain.py and predict.py give the same top features (test); notebook executes with 0 errors; figures byte-identical on a rerun (beeswarm seeded) | Done |
| `export_history.py` draws every report graph | `src/blindspy/figures.py`, `reports/figures/INDEX.md` | Every graph in the plan's list mapped to its file and generator; a test checks each listed file exists; reruns byte-identical | Done |
| Notebook 06 (final results) | `notebooks/06_final_results.ipynb` | Executes with 0 errors | Done |
| Report section 5 and the Phase 4 entries in section 4 | `reports/report/drafts/05_results.md`, `04_experiments.md` 4.20-4.21 | Every copied table checked against its generated source (`tests/test_report_drafts.py`); every number in the new text traced to a generated file | Done |
| Log entry E12; tag | `reports/EXPERIMENT_LOG.md`; tag `v1.0-final` on 3c31bad | Log regenerated from the E files; tag annotated and pushed | Done |
| Milestone export and backup | `reports/report/BlindSpy_report_v1.0-final.pdf` and `.docx`; `backups/blindspy_utsav-s-r_v1.0-final_2026-10-31.zip` (outside the repository) | PDF 72 pages, every page looked at; DOCX passed the XSD validator; zip 2,965,968,376 bytes, SHA-256 7587e6965c7c017657726033b1f8acc2ef3bdaf8ca4a4ee283aab43ff4330bdc, `unzip -t` reports no errors, line in `backups/SHA256SUMS.txt` | Done |
| Run store as a whole | `mlruns/utsav-s-r` (outside the repository) | 2,013 runs, all finished; Phase 4 added 35 (E15 4 + 1 superseded pilot, E12 30), all from committed and pushed code with one lock file | Done |

**Done when** (plan): a fresh clone at v1.0-final reproduces the test scores within rounding, and
`predict.py` works on a sample file. Both hold: the 30 test scores reproduce exactly and
`predict.py` gives identical output on all four sample files.

Not verified: the DOCX was checked by the validator but not opened in a word processor (none is
installed on the build machine); the PDF built from the same drafts was checked page by page.
