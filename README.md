# BlindSpy

Classifies encrypted network traffic from the ISCXVPN2016 dataset (University of New
Brunswick) using only 23 flow timing features, never packet contents. Three tasks, following
Draper-Gil et al. (ICISSP 2016):

- **A1**: VPN vs non-VPN (binary).
- **A2**: two separate 7-class models, one on VPN flows and one on non-VPN flows.
- **B**: one 14-class model over all traffic.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it downloads Python 3.12 itself) and git.
On macOS also run `brew install libomp` (needed later by XGBoost and LightGBM).

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

Build the cleaned data (available from Phase 1):

```bash
uv run python scripts/prepare_data.py --version v1
```

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
  `reports/EXPERIMENT_LOG.md` from the official runner's files.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

Branches: `chore/…`, `feat/…`, `exp/E##-…`, `docs/…`, merged into `dev`.
