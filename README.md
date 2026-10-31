# BlindSpy

Classifies encrypted network traffic from the ISCXVPN2016 dataset (University of New
Brunswick) using only 23 flow timing features, never packet contents. Three tasks, following
Draper-Gil et al. (ICISSP 2016):

- **A1**: VPN vs non-VPN (binary).
- **A2**: two separate 7-class models, one on VPN flows and one on non-VPN flows.
- **B**: one 14-class model over all traffic.

## For markers: what is here

- **Report:** `reports/report/BlindSpy_report.pdf` (and `.docx`), assembled from the markdown
  drafts in `reports/report/drafts/`. Every number in it comes from a generated file and names
  its experiment ID.
- **Final results** (test set scored once, 95% bootstrap intervals): test macro F1 A1 0.9434,
  A2-VPN 0.8652, A2-NonVPN 0.9403, B 0.8480 (`reports/phase4_tables.md`, experiment E12).
- **Final models:** `models/final/<task>.joblib` with `<task>.json` (feature names, order and
  units, class labels, the MLflow run and git commit they came from).
- **History:** `reports/EXPERIMENT_LOG.md` (one entry per experiment, failures included),
  `reports/history.csv` (all 2,013 MLflow runs, every parameter, metric and tag),
  `reports/run_artifacts_text.tar.gz` (every text file saved with a run),
  `reports/DATA_CHANGELOG.md` (data versions), and the git tags below.

| Tag | What it holds |
| --- | --- |
| `v0.1-baseline` | Phase 2: the first 80 baseline runs (E01-E03, data v1) |
| `v0.1.1-baseline` | Phase 2 rerun of all 144 configs with corrected timing (scores identical); the result tables use it |
| `v0.2-tuned` | End of Phase 3: improvement experiments E04-E17, final configs, analyses, notebooks 03-04 |
| `v1.0-final` | Phase 4: final models, the test set scored once (E12), calibration (E15), SHAP, notebooks 05-06 |
| `v1.0.1-final` | Hand-in: the final report, this README, the demo and its video; no result changed |

## Run it on any laptop

Tested on macOS (Apple Silicon) from a fresh clone. The Windows and Linux commands are
**not yet tested**; they will be marked tested once a teammate has run them. Scores were shown to
be identical on the same machine type; another operating system or CPU may differ in the last
digits (not measured).

**1. Install uv and git.** uv downloads Python 3.12 by itself.

- macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `brew install libomp` (needed
  by XGBoost and LightGBM).
- Linux (not yet tested): `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Windows (not yet tested), in PowerShell:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.

**2. Clone and install** (all systems):

```bash
git clone https://github.com/Varun-U-Pratap/BlindSpy.git
```

```bash
cd BlindSpy
```

```bash
uv sync
```

**3. Classify a flow file and launch the demo.** These need only the committed models and
sample files, not the dataset:

```bash
uv run python scripts/predict.py --task B --input app/samples/B_test_sample.csv --output predictions.csv
```

It writes one row per flow: the predicted label, its confidence and the three features that
pushed the model most towards that label. The file must hold exactly the task's 23 features in
the order listed in `models/final/<task>.json` (a `label` column at the end is optional). Tasks:
`A1`, `A2-VPN`, `A2-NonVPN`, `B`. A malformed file is rejected with a clear message and exit
code 2:

```bash
uv run python scripts/predict.py --task B --input app/samples/malformed_renamed_column.csv --output predictions.csv
```

The demo (then open http://localhost:8501 in a browser; stop it with Ctrl+C). Start it from the
repository folder, so that its settings in `.streamlit/config.toml` apply (no first-run prompt,
reachable from this laptop only):

```bash
uv run streamlit run app/demo.py
```

Choose a task, pick a sample file (or upload a CSV), and click a row of the predictions table to
see the features behind that prediction.

**4. Get the dataset** (only for rebuilding the data and reproducing scores). Download the
ISCXVPN2016 dataset from https://www.unb.ca/cic/datasets/vpn.html and put its `CSVs` folder (it
holds the ARFF files, in `Scenario A1-ARFF`, `Scenario A2-ARFF` and `Scenario B-ARFF`) at
`datasets/CSVs/`, keeping the folder names as downloaded. Check the files:

- macOS / Linux: `shasum -a 256 -c data/arff_sha256.txt` (Linux also: `sha256sum -c
  data/arff_sha256.txt`). All 20 lines must say `OK`.
- Windows (not yet tested), in PowerShell:
  `Get-Content data/arff_sha256.txt | ForEach-Object { $h, $p = $_ -split '  ', 2; if ((Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() -eq $h) { "OK  $p" } else { "FAILED  $p" } }`

Then build the cleaned data (on macOS a rebuild is byte-identical to ours; other systems not
measured):

```bash
uv run python scripts/prepare_data.py --version v1
```

**5. Reproduce the final test scores.** Set a runner name of your own (anything except
`utsav-s-r`, the official runner), so your runs go to your own MLflow store:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Edit `configs/local.yaml` to say `runner: <your name>` (Windows: `copy` instead of `cp`). Then
retrain the 30 final models on training + validation rows and score each once on the test rows
(about 20 minutes on a laptop):

```bash
uv run python scripts/run_experiment.py configs/experiments/E12_*.yaml
```

(Windows PowerShell, not yet tested: `uv run python scripts/run_experiment.py
(Get-ChildItem configs/experiments/E12_*.yaml).FullName`.) `run_experiment.py` prints a warning
because your runner is not the official one; the runs still complete. Compare with the official
scores:

```bash
uv run python scripts/check_final_scores.py --runner <your name>
```

It prints each run's test macro F1 and accuracy next to the official ones from
`reports/history.csv` and ends with how many of the 30 are identical and how many agree to 4
decimals (exit code 0 only if all 30 agree).

Note: adding Streamlit for the demo changed `uv.lock` after the final runs, so the lock-file hash
logged with your runs differs from the official ones (only Streamlit and five packages it needs
were added; no version changed). For the exact environment of the final runs, use
`git checkout v1.0-final` before `uv sync` (that tag has no demo).

## Setup for development

Requires [uv](https://docs.astral.sh/uv/) (it downloads Python 3.12 itself) and git.
On macOS also run `brew install libomp` (needed by XGBoost and LightGBM).

```bash
git clone https://github.com/Varun-U-Pratap/BlindSpy.git
cd BlindSpy
uv sync
uv run nbstripout --install
cp configs/local.example.yaml configs/local.yaml   # then set runner: <your GitHub username>
```

Put the dataset in `datasets/CSVs/`, keeping the folder names exactly as downloaded
(`Scenario A1-ARFF`, `Scenario A2-ARFF`, `Scenario B-ARFF`), and check it:

```bash
shasum -a 256 -c data/arff_sha256.txt
```

All 20 lines must say `OK` (Windows: `Get-FileHash`). The Kaggle CSV goes in
`datasets/kaggle/` and is checked with `data/kaggle_sha256.txt`.

Build the cleaned data (current version: v1, see `reports/DATA_CHANGELOG.md`):

```bash
uv run python scripts/prepare_data.py --version v1
```

This writes `data/processed/v1/` (one Parquet file per task x timeout, `splits/`,
`cleaning_log.csv`, `manifest.json`), refreshes `data/manifest.json`, and writes viewing copies
to `data/processed/csv/`. It refuses to overwrite an existing version, and a rebuild gives
byte-identical files. Code loads data only through the Parquet files:

```python
from blindspy.data import load_task

train, val, test = load_task("A1", 15, data_version="v1")
```

Column names are corrected by `COLUMN_MAP` in `src/blindspy/data.py` (9 A1/B headers shift the
fiat/biat names; A2 files hold `total_fiat`/`total_biat` instead of the std columns). Evidence:
`reports/experiments/utsav-s-r/E00.md`. To rewrite the CSV copies from another version:
`uv run python scripts/prepare_data.py --export-csv v1`.

## Check the setup (MLflow smoke run)

```bash
uv run python scripts/run_experiment.py configs/experiments/E_smoke.yaml
uv run mlflow ui --backend-store-uri sqlite:///mlruns/<runner>/mlflow.db --port 5000
```

Open http://127.0.0.1:5000: experiment **blindspy** shows an `E_smoke` run with one
parameter, one metric and `smoke.png`.

## How runs are tracked

- Each laptop is a *runner* (`configs/local.yaml`, git-ignored). Each runner has its own MLflow
  store, `mlruns/<runner>/mlflow.db` plus `mlruns/<runner>/artifacts/` (git-ignored).
- Every run is tagged with the runner, machine, git commit, uncommitted-changes flag and the
  `uv.lock` hash.
- Official runs (runner `utsav-s-r`) must come from code that is committed **and** pushed.
  Otherwise `run_experiment.py` stops with:

  ```
  Refusing official run. Code is not committed and pushed. To fix: commit your changes
  (git add … && git commit), then push them (git push origin dev), then run the experiment again.
  ```

  Other runners get the same text as a warning and the run continues.
- One notes file per experiment: `reports/experiments/<runner>/E##.md` (copy `E_TEMPLATE.md`;
  creating the file claims the ID). `uv run python scripts/export_history.py` builds
  `reports/EXPERIMENT_LOG.md` from the official runner's files and writes every MLflow run of
  that store to `reports/history.csv` (one row per run, every parameter, metric and tag).
- The files saved next to each run (confusion matrices, per-class reports, feature importances,
  test and validation predictions, Optuna trial tables, configs) are kept in git as
  `reports/run_artifacts_text.tar.gz`, listed file by file in `reports/run_artifacts_index.csv`
  (run name, experiment, size, SHA-256); `uv run python scripts/export_artifacts.py` rebuilds
  both on the official laptop. Plots and model files are not in git (only the four final
  models in `models/final/`).

## Reproduce a baseline (Phase 2, tag `v0.1.1-baseline`)

Phase 2 has two tags with identical scores: `v0.1-baseline` (the first 80 baseline runs) and
`v0.1.1-baseline` (the rerun of all 144 Phase 2 configs with corrected timing, which the
result tables use). Each run is one YAML file in `configs/experiments/` (written by
`scripts/make_configs.py`). Rerunning one gives the same scores as the official run (seed 42,
data v1):

```bash
uv run python scripts/run_experiment.py configs/experiments/E02_tree_A1_15s.yaml
```

The run trains on the training split with 5-fold cross-validation, scores the validation split
(the test split is never used) and logs everything, including the model file, to your MLflow
store. `E02_paper_j48_<rows>_<task>_<timeout>s.yaml` and `E03_paper_ibk_…` rerun the paper's
10-fold benchmark. To rebuild the result tables from the official history:
`uv run python scripts/results_tables.py` (writes `reports/baseline_tables.md`).

**Which scripts need the official MLflow store.** `results_tables.py` (Phases 2-4, which also read
the confusion matrices and predictions saved with the runs), `save_final_models.py` and
`export_artifacts.py` read `mlruns/utsav-s-r/`, which is not in git: it exists only on the official
runner's laptop (and in its backup zips). In a fresh clone they stop with "No such file", and the
committed outputs are the record; on the official laptop they rewrite those outputs byte for byte.
`results_tables.py --phase 5`, `make_notebooks.py`, `explain_final.py` (needs the dataset),
`export_history.py` (rewrites the experiment log; it leaves `history.csv` alone without the store)
and `build_report.py` work in a fresh clone.

## Phase 3 results, notebooks and report

Tag `v0.2-tuned` marks the end of Phase 3; how it was verified (data rebuilt byte-identical, 13
runs rerun from a fresh clone with identical scores) is recorded in `reports/phase3_evidence.md`.

Phase 3 (E04-E17) configs are in `configs/experiments/` too, and the chosen models in
`configs/final/` (`<task>.yaml` = headline model per task, `compare/` = the 28-model comparison
set, `old_rule/` = the picks of the superseded tie rule); `scripts/write_final_configs.py`
writes them from `reports/history.csv`. Rebuild the outputs (the tables need the official store,
see above):

```bash
uv run python scripts/results_tables.py --phase 3
```

```bash
uv run python scripts/make_notebooks.py
```

The first writes `reports/phase3_tables.md`; the second writes `notebooks/03_models.ipynb` and
`notebooks/04_tuning.ipynb` (run them in Jupyter to redraw the figures in `reports/figures/`).
The report (`reports/report/BlindSpy_report_<tag>.pdf` and `.docx`) is assembled from the
markdown drafts in `reports/report/drafts/`. It needs Google Chrome and Node.js with the `docx`
package, installed once into any folder outside the repository:

```bash
npm install --prefix ~/blindspy-report-node docx@9.15.0
```

```bash
uv run python scripts/build_report.py --tag final --node-modules ~/blindspy-report-node
```

`--tag final` writes the hand-in report, `reports/report/BlindSpy_report.pdf` and `.docx`. A
milestone's report (`--tag v0.2-tuned`, `--tag v1.0-final`) is rebuilt only from a checkout of that
tag (`git checkout v0.2-tuned` first); at a later commit it would overwrite the milestone's file
with the later drafts.

## Phase 4: final models, explanation and the final report

The final runs (E12, tag `v1.0-final`) retrained each model of `configs/final/` on training +
validation rows and scored the test rows once; how a fresh clone reproduced them is recorded in
`reports/phase4_evidence.md`. From the official history (on the official laptop, see above):

```bash
uv run python scripts/results_tables.py --phase 4
```

writes `reports/phase4_tables.md` (final scores with intervals, the 26 paired tests, calibration,
SHAP tables). `scripts/explain_final.py` writes the SHAP tables and figures,
`scripts/save_final_models.py` writes `models/final/` from the official store, `scripts/make_samples.py` writes
`app/samples/`, and `scripts/make_notebooks.py` also writes `notebooks/05_explain.ipynb` and
`notebooks/06_final_results.ipynb`. The final report is built with
`scripts/build_report.py --tag final` (as in the Phase 3 section above).

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

Branches: `chore/…`, `feat/…`, `exp/E##-…`, `docs/…`, merged into `dev`.
