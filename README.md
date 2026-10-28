# BlindSpy

Classifies encrypted network traffic from the ISCXVPN2016 dataset (University of New
Brunswick) using only 23 flow timing features, never packet contents. Three tasks, following
Draper-Gil et al. (ICISSP 2016):

- **A1**: VPN vs non-VPN (binary).
- **A2**: two separate 7-class models, one on VPN flows and one on non-VPN flows.
- **B**: one 14-class model over all traffic.

## Setup

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
  that store to `reports/history.csv`.

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

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

Branches: `chore/…`, `feat/…`, `exp/E##-…`, `docs/…`, merged into `dev`.
