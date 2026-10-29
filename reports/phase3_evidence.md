# Phase 3 evidence (tag v0.2-tuned)

Record of how each Phase 3 item was checked before the phase was closed on 29 Oct 2026. The tag
`v0.2-tuned` is on commit fa4a015, the dev head at the end of Phase 3 (decided 29 Oct: E09, E13,
E14, E16, the notebooks and the report all came after `configs/final/` was added in 8c07ff2, so
the tag holds every Phase 3 result and the code that made it). The tag message lists each
experiment's run commits and every later code change.

## Verification at the tag

From a fresh clone of the repository at `v0.2-tuned`, with its own MLflow store (a runner name
other than the official one, so the official history is not touched) and the raw datasets linked
in:

| Check | Result |
|---|---|
| Rebuild data v1 (`scripts/prepare_data.py --version v1`) | 34 files, combined SHA-256 2725f36b06b2b251b8c1f0d3f26171bf6f0717b16eb077d2c22c60c0217e0f7c, identical to the official v1 |
| Rebuild kaggle-k1 (`scripts/prepare_kaggle.py --version k1`) | all 6 file hashes identical to `data/kaggle_k1_manifest.json` |
| `pytest` | 190 passed |
| Rerun 13 configs and compare with the official run of the same name | 13 of 13 identical on every non-timing metric (27-79 metrics per run) |

The 13 reruns: signed-log KNN (E04, A1), Random Forest (E05, A1), XGBoost and LightGBM (E06,
A2-VPN), the tuned headline configuration `configs/final/A1.yaml` (against its source run
E17-lgbm-A1-15s), the SMOTE pipeline (E08, A2-NonVPN), the two-step model (E10, B), a full Optuna
search (E17 logistic regression, A1: 50 of 50 trials with identical CV scores), E09 timing-only
(A1), E13 10% seed 42 (A1), E14 split seed 43 (A1), E16 (c) XGBoost at 120 s and E16 (d) 7,671
rows seed 42. Timing metrics (training and prediction time, memory) differ between machines and
runs and are not compared.

## Every Phase 3 item

| Plan item or decision | Files / commits | How it was checked | Result |
|---|---|---|---|
| Setup commit: compression level 3, XGBoost, LightGBM, Optuna, imbalanced-learn, one thread per model, child-process memory and both model sizes logged | 18316d7 (merged 5e79a74) | Phase 2 baseline rerun in a scratch clone (identical); every Phase 3 run in the store has the same lock-file hash, compression 3 and its thread setting logged | Done |
| Pilot check and the check of each new model type | E04.md (pilot by hand in the MLflow UI), E05-E10 and E17 (by query) | Gaps found were fixed before the batches (E04 `preprocessing` param, E06 `estimator_params`) | Done |
| E04, E05, E06, E07, E08, E10, E17 | E##.md; `reports/phase3_tables.md` | 31 pre-stated verdicts: generated verdicts equal the E files and report section 4; every saved model reloaded and re-scored to its logged validation score | Done |
| Winner rule (0.03) and current best | E##.md | Reruns at 30, 60 and 120 s only for the two winners (E04 A2-NonVPN, E05 B) | Done |
| Final configuration per task and the 28-model comparison set (decided 28 Oct after E10) | `configs/final/` (34 configs, 8c07ff2); E12.md part 1 | All 34 rebuilt with the code at the tag and fitted on the training rows: validation macro F1 identical in 34 of 34 | Done |
| E09, E13, E14 (analysis on the headline models) | E09.md, E13.md, E14.md | Verdicts match; one config each rerun at the tag, identical | Done |
| E11 (does not run) | E11.md | No session information in the data (E00) | Done |
| E16 with its own data version, including the (d) training-size control | E16.md, `data/kaggle_k1_manifest.json`, DATA_CHANGELOG | kaggle-k1 rebuilt identically at the tag; (c) and (d) reruns identical | Done |
| Notebooks 03 and 04, generators committed under scripts/ | `scripts/make_notebooks.py` (bce0579 and later) | Both execute with 0 errors; regenerating gives byte-identical notebooks and figures | Done |
| Generated results and report (sections 3 and 4, milestone export) | `reports/phase3_tables.md`, `reports/report/drafts/`, `reports/report/BlindSpy_report_v0.2-tuned.pdf` and `.docx` | Copied tables checked by `tests/test_report_drafts.py`; DOCX checked with an XSD validator; every PDF page looked at | Done |
| Storage check | report Section 4.19; E05-E07 | Artifacts 2,645.3 MiB and mlflow.db 35.8 MiB at the end of Phase 3, far below 20 GB; the four headline model files are 4.31-25.12 MiB, below GitHub's 50 MB warning | Done |
| Backup at the tag | `backups/blindspy_utsav-s-r_v0.2-tuned_2026-10-29.zip` (outside the repository) | 2,682,085,284 bytes, SHA-256 0e3a2bb210228086219b5dd22967a233cbf7dce9fb34ad7dce0e63a3a92ec560, `unzip -t` reports no errors | Done |
| Log entries and tag | `reports/EXPERIMENT_LOG.md`; tag `v0.2-tuned` on fa4a015 | Log regenerated from the E files; tag annotated and pushed | Done |
| Run store as a whole | `mlruns/utsav-s-r` (outside the repository) | 1,687 Phase 3 runs (287 retrained or single runs and 1,400 Optuna trials): all finished, from committed and pushed code, one lock file, data and split hashes found in the manifests, a model file on every main run and none on trials | Done |

**Done when** (plan): the four final configurations are in `configs/final/<task>.yaml` with the
reason, the 28-model comparison set is in `configs/final/` before Phase 4, and the verification
at the tag passed. All three hold.

Not verified: the DOCX was checked by the validator but not opened in a word processor (none is
installed on the build machine); the PDF built from the same drafts was checked page by page.
