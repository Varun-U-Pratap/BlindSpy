# BlindSpy: Complete Project Context and System Design

## 1. Executive summary

BlindSpy is a reproducible machine-learning research and inference project for classifying
encrypted network traffic from the ISCXVPN2016 dataset. It uses 23 flow-level timing and rate
features and deliberately does not inspect packet payloads or packet contents.

The project has two distinct planes:

1. **Research/training plane:** raw ARFF data is validated, cleaned, versioned, split, used for
   controlled experiments, tracked with MLflow, evaluated, explained, and assembled into reports.
2. **Inference/demo plane:** committed final models accept a strict CSV contract and return
   predictions, confidence, and SHAP-based feature explanations through a command-line interface
   or a Streamlit application.

This repository is primarily an offline research/reproducibility system. It is not currently a
distributed production service. There is no request queue, service cluster, reverse proxy,
container orchestration, autoscaling policy, model server, database-backed prediction API, or
implemented load balancer. The only runtime serving surface is a local Streamlit process, while
the CLI performs local batch inference.

## 2. Project purpose

The purpose of BlindSpy is to investigate whether encrypted traffic can be classified from
observable flow metadata even when payload inspection is unavailable. The project predicts:

- whether traffic is VPN or non-VPN;
- the application/traffic family carried by the flow;
- the combined VPN/application class for the complete task.

The project does **not** prove that traffic is encrypted. It classifies traffic patterns using
aggregate flow statistics. The intended research outcomes are:

- a defensible, reproducible preprocessing pipeline;
- comparisons among classical and gradient-boosted classifiers;
- final models for four classification tasks;
- measured predictive quality, calibration, explainability, and inference cost;
- a user-facing demonstration that does not retrain models;
- a complete experiment history and report.

## 3. Classification tasks

The canonical tasks are defined in [`src/blindspy/data.py`](src/blindspy/data.py):

| Task | Meaning | Number of classes | Labels |
|---|---|---:|---|
| `A1` | VPN versus non-VPN | 2 | `Non-VPN`, `VPN` |
| `A2-VPN` | Traffic type inside VPN flows | 7 | `VPN-BROWSING`, `VPN-CHAT`, `VPN-STREAMING`, `VPN-MAIL`, `VPN-VOIP`, `VPN-P2P`, `VPN-FT` |
| `A2-NonVPN` | Traffic type inside non-VPN flows | 7 | `BROWSING`, `CHAT`, `STREAMING`, `MAIL`, `VOIP`, `P2P`, `FT` |
| `B` | VPN status and traffic type jointly | 14 | Seven non-VPN labels plus seven `VPN-` labels |

Task B is a standalone 14-class model. A2 is not implemented as one hierarchical prediction
flow; it is represented by two separate seven-class task models. An optional experimental
`B` two-step model exists in [`src/blindspy/models.py`](src/blindspy/models.py), where a binary
VPN/non-VPN router selects one of two seven-class models.

The supported flow timeout variants are 15, 30, 60, and 120 seconds. These are dataset flow
construction timeouts, not guarantees that each flow lasts exactly that long.

## 4. Repository map

```text
BlindSpy/
├── README.md                         User setup, reproduction, and usage guide
├── PROJECT_CONTEXT.md                This architecture and context document
├── pyproject.toml                    Python project metadata and dependencies
├── uv.lock                           Locked dependency resolution
├── .python-version                   Python version selector
├── about/                            Background paper and project source documents
├── configs/
│   ├── base.yaml                     Shared seed, split, path, and cleaning policy
│   ├── local.example.yaml             Template for a local MLflow runner
│   ├── local.yaml                     Git-ignored local runner configuration
│   ├── experiments/                  Experiment YAML definitions
│   └── final/                        Selected final model configurations
├── data/
│   ├── *.json, *.txt                  Checksums and manifests
│   └── processed/                     Generated versioned Parquet data (normally ignored)
├── src/blindspy/                     Reusable Python package
├── scripts/                           Command-line orchestration and report utilities
├── app/
│   ├── demo.py                       Streamlit inference and explanation UI
│   └── samples/                      Demonstration CSV inputs
├── models/final/                      Four committed model files and metadata
├── notebooks/                         EDA, modelling, tuning, explanation, final-results notebooks
├── reports/                           Evidence, result tables, experiment logs, figures, report drafts
├── tests/                             Unit, integration, artifact, and reproducibility tests
└── mlruns/                            Local runner-specific MLflow stores and artifacts
```

Large generated artifacts, downloaded datasets, local configuration, and runner-specific MLflow
stores are intentionally separated from the source code. The raw dataset is not included in the
repository and must be downloaded independently.

## 5. High-level system design

```text
                 +----------------------+
                 | ISCXVPN2016 ARFF     |
                 | 20 source files      |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | data.py + checks.py  |
                 | parse, map, validate |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | prepare_data.py     |
                 | clean + deduplicate |
                 | manifest + hashes   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Versioned Parquet   |
                 | split JSON + logs   |
                 +----------+-----------+
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
 +----------------------+          +----------------------+
 | Experiment YAML      |          | Final configurations |
 | E01-E17 and E12     |          | configs/final       |
 +----------+-----------+          +----------+-----------+
            |                                 |
            v                                 v
 +----------------------+          +----------------------+
 | run_experiment.py    |          | train.py             |
 | MLflow orchestration |          | fit and score        |
 +----------+-----------+          +----------+-----------+
            |                                 |
            +-----------------+---------------+
                              v
                    +----------------------+
                    | models.py            |
                    | sklearn pipelines,   |
                    | XGBoost, LightGBM,   |
                    | balancing, routing   |
                    +----------+-----------+
                              |
                              v
                    +----------------------+
                    | evaluate.py          |
                    | metrics, plots,      |
                    | confusion artifacts  |
                    +----------+-----------+
                              |
              +---------------+----------------+
              |                                |
              v                                v
   +----------------------+          +----------------------+
   | MLflow / reports     |          | save_final_models.py |
   | history / notebooks  |          | joblib + JSON        |
   +----------------------+          +----------+-----------+
                                                 |
                                                 v
                                      +----------------------+
                                      | predict.py           |
                                      | validate + infer     |
                                      +----------+-----------+
                                                 |
                              +------------------+------------------+
                              |                                     |
                              v                                     v
                    +----------------------+              +----------------------+
                    | scripts/predict.py   |              | app/demo.py          |
                    | batch CLI            |              | Streamlit UI          |
                    +----------------------+              +----------------------+
```

## 6. Data layer

### 6.1 Source data and file families

The expected raw directory is:

```text
datasets/CSVs/
├── Scenario A1-ARFF/
├── Scenario A2-ARFF/
└── Scenario B-ARFF/
```

`data.py` maps logical tasks to these folders and the source filename suffixes. It also defines
the 23-feature header families:

- `T`: includes `total_fiat` and `total_biat`;
- `S`: includes `std_fiat` and `std_biat`.

The source files contain inconsistent header naming. `COLUMN_MAP` maps source header names to
the verified semantic feature names for each task family. This is a semantic correction, not a
numeric transformation.

### 6.2 ARFF parsing

`data.load_arff()`:

1. reads the ARFF text;
2. removes source-export trailing commas;
3. removes comma-only rows after `@DATA`;
4. parses the result with `liac-arff`;
5. verifies that the final attribute is `class1`;
6. assigns the appropriate verified 23-feature layout plus `label`;
7. converts features to `float64` and labels to strings.

`load_raw()` then applies the task-specific column map. `load_task()` reads processed Parquet
and rehydrates the saved train/validation/test index partitions.

### 6.3 Cleaning rules

The cleaning policy is recorded in [`configs/base.yaml`](configs/base.yaml) and implemented by
[`scripts/prepare_data.py`](scripts/prepare_data.py). Important decisions are:

- strip trailing commas from headers;
- drop rows made only of commas;
- correct the A1/B header interpretation;
- preserve A2's `total_fiat` and `total_biat`;
- restore the corrupted A1 120-second `flowPktsPerSecond` column from the matching B file;
- count missing and infinite values rather than silently imputing them;
- retain `-1` values that mean a statistic is not measurable;
- retain zero-duration and single-packet flows;
- retain negative flow inter-arrival values caused by capture ordering;
- retain source rounding and unusual packet-size-derived rates;
- drop exact duplicate rows within each file before splitting;
- keep conflicting duplicate feature rows when their labels differ;
- perform no global normalization, feature selection, outlier removal, or data-level balancing.

Model-specific preprocessing and balancing happen inside pipelines and are fitted only on training
data or CV training folds.

### 6.4 Versioned outputs

`prepare_data.py` writes a versioned directory containing:

- one Parquet file per task and timeout;
- deterministic split JSON files;
- `cleaning_log.csv`;
- a data manifest with row counts and hashes;
- optional CSV viewing copies.

Existing data versions are protected from accidental overwrite. The manifests and checksums make
the data lineage auditable.

### 6.5 Split policy

The default split in [`configs/base.yaml`](configs/base.yaml) is:

- 70% train;
- 15% validation;
- 15% test;
- stratified by label;
- seed 42;
- generated separately for every task and timeout.

[`src/blindspy/splits.py`](src/blindspy/splits.py) stores raw-row identifiers in sorted split
lists. The normal training protocol never uses test rows. E13 can stratified-subsample training
rows, and E14 can re-split the combined train/validation population while leaving test rows
untouched.

### 6.6 Kaggle comparison path

[`src/blindspy/kaggle.py`](src/blindspy/kaggle.py) is a separate comparison-data path for E16.
It reconstructs the provenance of an eight-file Kaggle CSV, verifies row order against the ARFF
files, supports the published global-deduplication split and a corrected per-timeout split, and
stores its own manifest. It does not modify the primary `v1` data.

## 7. Feature layer

[`src/blindspy/features.py`](src/blindspy/features.py) provides:

- `signed_log(x) = sign(x) * log1p(abs(x))`;
- a scikit-learn `FunctionTransformer` wrapper;
- feature groups for forward gaps (`fiat`), backward gaps (`biat`), flow gaps
  (`flowiat`), active periods, idle periods, and rates/duration;
- named feature sets: `all`, `only_<group>`, `no_<group>`, and `timing_only`.

The signed-log transform compresses heavy-tailed timing values while preserving ordering and
keeping the semantic `-1` code distinct from zero. It is a fixed transform and does not learn
statistics from data.

Feature selection is applied by selecting columns before a model is fitted. Scaling is a pipeline
step, so its parameters are learned only from the rows presented to that fit.

## 8. Model layer

The model registry in [`src/blindspy/models.py`](src/blindspy/models.py) maps configuration names
to full pipelines:

| Registry name | Implementation |
|---|---|
| `majority` | Most-frequent `DummyClassifier` baseline |
| `tree` | `DecisionTreeClassifier` |
| `knn` | StandardScaler plus KNN |
| `knn_minmax` | MinMaxScaler plus KNN, used for the paper-style IBk comparison |
| `knn_log_standard` | Signed-log, StandardScaler, KNN |
| `knn_log_robust` | Signed-log, RobustScaler, KNN |
| `logreg` | StandardScaler, logistic regression |
| `svm_rbf` | StandardScaler, RBF SVM |
| `rf` | Random forest |
| `xgb` | XGBoost |
| `lgbm` | LightGBM |

Tree and boosting models use one model thread (`n_jobs=1`) so that cross-validation can parallelize
folds without uncontrolled nested parallelism.

### 8.1 XGBoost label adapter

`XGBLabelled` adapts string class labels to XGBoost's required integer labels and maps predictions
back to the original names. It can calculate balanced sample weights because XGBoost does not
provide the same `class_weight` interface as several scikit-learn estimators.

### 8.2 Class balancing

`build_model()` supports:

- `none`;
- `class_weight`;
- random oversampling;
- SMOTE.

Resampling is inserted into an imbalanced-learn pipeline immediately before the estimator. This
ensures that validation/test rows are never resampled and that each CV fold learns balancing only
from its own training portion. Balancing is rejected for the two-step router.

### 8.3 Two-step task B model

`TwoStep` is an experimental hierarchical classifier:

1. a binary model predicts VPN versus non-VPN;
2. a VPN seven-class model is trained on VPN rows;
3. a non-VPN seven-class model is trained on non-VPN rows;
4. the binary result routes each input to one seven-class model;
5. probabilities are placed into the shared 14-class output layout.

The router can amplify upstream errors: if the first step selects the wrong group, the correct
class is unavailable to the second step.

## 9. Training and experiment layer

### 9.1 Central training functions

[`src/blindspy/train.py`](src/blindspy/train.py) is the central execution layer. It separates
features from labels, applies optional feature selection, builds the configured pipeline, and
controls the relevant training protocol.

Normal holdout training:

1. load train, validation, and test partitions;
2. optionally resample train/validation according to the experiment;
3. select features if requested;
4. run stratified five-fold CV on training rows;
5. fit a fresh model on training rows;
6. measure fit and prediction cost;
7. score validation rows;
8. write evaluation artifacts;
9. never use the test set.

Final training:

1. use the selected final configuration;
2. combine training and validation rows;
3. fit the model once on that combined data;
4. optionally fit the E15 confidence calibrator using the allowed training population;
5. score the held-out test rows once;
6. save predictions, scores, metrics, and model metadata.

The training module also implements paper-style CV, Kaggle comparison runs, calibration runs, and
resource measurements.

### 9.2 Experiment orchestration

[`scripts/run_experiment.py`](scripts/run_experiment.py) reads YAML configurations and dispatches
to smoke, ordinary training, calibration, final, Optuna, Kaggle, or paper-benchmark workflows.
It also logs cleaning rules and coordinates MLflow.

Experiment YAML files under [`configs/experiments/`](configs/experiments/) encode task, timeout,
model, parameters, balancing choice, feature set, seed, data version, training fraction, and
whether a model should be saved. `configs/final/` contains the selected final configurations and
comparison/legacy selections.

### 9.3 Tuning

[`src/blindspy/tuning.py`](src/blindspy/tuning.py) uses Optuna's seeded TPE sampler. Each trial:

- samples typed parameters from the configured search space;
- runs five-fold stratified CV;
- optimizes mean training macro F1;
- runs trials sequentially;
- parallelizes folds.

The test set is not part of hyperparameter selection.

## 10. Evaluation, calibration, and explainability

### 10.1 Evaluation

[`src/blindspy/evaluate.py`](src/blindspy/evaluate.py) computes:

- accuracy;
- balanced accuracy;
- macro and weighted F1;
- macro precision and recall;
- per-class precision, recall, F1, and average precision;
- binary ROC-AUC for A1;
- confusion matrices;
- largest confusion pairs;
- precision-recall plots;
- ROC plots for binary tasks;
- tree feature-importance files where supported.

The primary final comparison metric is macro F1 because the class distribution is unequal.
Bootstrap intervals and paired comparisons are produced by the reporting/checking scripts.

### 10.2 Calibration

[`src/blindspy/calibration.py`](src/blindspy/calibration.py) provides:

- 15 equal-width confidence bins;
- expected calibration error (ECE);
- reliability tables and plots;
- isotonic `CalibratedClassifierCV` with shuffled five-fold stratification.

The uncalibrated base model determines the predicted label. Calibration changes the reported
confidence only; the calibrated argmax is intentionally not used as the label.

### 10.3 SHAP explanations

[`src/blindspy/explain.py`](src/blindspy/explain.py) and the prediction layer use SHAP
`TreeExplainer` for tree-based final models. They provide:

- mean absolute per-feature importance;
- feature-group importance;
- selected worked examples;
- the strongest features pushing toward a predicted class.

SHAP values are in the model's raw output space, such as log-odds or multi-class margins. They
describe how the model uses correlated inputs; they do not prove causal traffic behaviour.

## 11. Tracking and reproducibility

[`src/blindspy/tracking.py`](src/blindspy/tracking.py) creates a separate MLflow SQLite store per
runner:

```text
mlruns/<runner>/mlflow.db
mlruns/<runner>/artifacts/
```

Run identity includes:

- experiment ID and runner;
- operating system, CPU, memory, and Python version;
- Git commit and branch;
- dirty/pushed Git state;
- `uv.lock` SHA-256;
- configuration path;
- package versions;
- data version and data hashes.

The official runner is required to use committed and pushed code. Personal runners receive a
warning and may continue. This policy makes official experiment results traceable to repository
state.

The reporting utilities export:

- an experiment log;
- a flattened MLflow history CSV;
- run text artifacts and an index;
- phase-specific result tables;
- report PDFs and DOCX files;
- notebooks and figures.

## 12. Final model and inference design

The final-model path is:

```text
models/final/<task>.joblib
models/final/<task>.json
```

The JSON records the task, selected model, training provenance, feature names and units, class
labels, confidence policy, file size, and SHA-256.

`predict.load_final()` verifies the model file hash before loading it. This is an integrity check,
not a general security sandbox; joblib files must still come from a trusted project source.

`predict.validate()` rejects:

- empty inputs;
- missing columns;
- unexpected columns;
- wrong feature order;
- non-numeric columns;
- missing or infinite values.

An optional final `label` column is accepted as ground truth and echoed for comparison. The
prediction result contains row number, predicted label, confidence, optional true label, and the
three strongest SHAP features.

## 13. User-facing applications

### 13.1 Batch CLI

[`scripts/predict.py`](scripts/predict.py) reads an input CSV, calls the shared prediction
module, and writes a CSV result. It uses the same validation and final-model code as the demo,
so command-line and UI behaviour remain aligned.

### 13.2 Streamlit demo

[`app/demo.py`](app/demo.py):

1. lets the user choose A1, A2-VPN, A2-NonVPN, or B;
2. selects a committed sample CSV or accepts an uploaded CSV;
3. loads the corresponding final model using a cached resource;
4. validates and predicts every row;
5. displays labels, confidence, and optional correctness;
6. lets the user select a row;
7. renders a ten-feature SHAP chart for that row.

The demo does not train, tune, write to MLflow, or access the raw dataset. It is a local
single-process demonstration started with:

```bash
uv run streamlit run app/demo.py
```

The Streamlit configuration restricts access to the local laptop and avoids a first-run prompt.

## 14. Load balancing and scalability

### 14.1 What is implemented

There is no network load balancer in the current repository. The implemented parallelism is
within offline experiment execution:

- cross-validation folds run in parallel using `ProcessPoolExecutor`/scikit-learn joblib;
- individual tree/boosting models are generally constrained to one thread;
- Optuna trials run sequentially while folds within a trial can run in parallel;
- each local MLflow runner has an independent store;
- Streamlit caches the loaded model resource within its process.

These are workload and resource-isolation choices, not request-level load balancing.

### 14.2 What is not implemented

The repository does not contain:

- multiple inference worker processes;
- a reverse proxy such as Nginx or HAProxy;
- a cloud load balancer;
- a service discovery mechanism;
- a message broker or request queue;
- health checks and failover;
- autoscaling;
- rate limiting;
- GPU scheduling;
- distributed model replicas;
- a horizontally scalable prediction API.

Running multiple Streamlit instances behind a proxy would therefore be a deployment extension,
not current project behaviour. Each replica would need access to the same trusted model files and
consistent dependency versions.

### 14.3 Practical future load-balancing design

If the project is promoted to a production service, a reasonable extension would be:

```text
Clients
   |
TLS termination / API gateway
   |
Health-checked load balancer
   |
+----------+----------+----------+
| inference worker 1  | worker 2 | ... worker N
| immutable model     | immutable model
+----------+----------+----------+
              |
       shared model artifact registry
```

The workers should be stateless for prediction, validate the same 23-column contract, load a
versioned model at startup, expose readiness/health endpoints, and return model version metadata.
Round-robin is adequate for similarly sized batch requests; least-connections is preferable when
CSV batch sizes vary. A queue plus bounded workers is preferable if requests are large or
asynchronous. None of these components should be described as present until implemented.

## 15. End-to-end operational flows

### 15.1 Rebuild data

```text
Download dataset
  -> verify data/arff_sha256.txt
  -> prepare_data.py --version v1
  -> parse and correct headers
  -> apply documented cleaning rules
  -> deduplicate within file
  -> create stratified splits
  -> write Parquet, split JSON, logs, manifest, hashes
```

### 15.2 Run an experiment

```text
Experiment YAML
  -> run_experiment.py
  -> load base/local configuration
  -> validate Git/data identity
  -> load versioned task data
  -> apply optional split resampling and feature set
  -> build sklearn/imbalanced-learn pipeline
  -> cross-validation on training rows
  -> fit and validate
  -> evaluate and save artifacts
  -> log parameters, metrics, tags, and files to MLflow
```

### 15.3 Produce a final model

```text
Selected final config
  -> load train + validation
  -> fit final estimator
  -> optionally fit confidence calibrator
  -> score test once
  -> record metrics and provenance
  -> save joblib plus metadata JSON
```

### 15.4 Serve a prediction

```text
CSV
  -> load model and metadata
  -> verify model SHA-256
  -> validate exact feature contract
  -> transform with fitted pipeline
  -> predict label and probability
  -> compute confidence for predicted class
  -> compute top SHAP features
  -> return CSV or Streamlit display
```

## 16. Script inventory

| Script | Responsibility |
|---|---|
| `prepare_data.py` | Parse, clean, validate, split, and version primary data |
| `prepare_kaggle.py` | Build the Kaggle comparison data version |
| `run_experiment.py` | Dispatch configured experiment types and log MLflow runs |
| `make_configs.py` | Generate experiment configurations |
| `write_final_configs.py` | Write selected final configurations from recorded results |
| `save_final_models.py` | Export official final model files and metadata |
| `predict.py` | Batch prediction CLI |
| `make_samples.py` | Create sample input CSVs |
| `data_checks.py` | Generate data quality checks and evidence |
| `data_tables.py` | Generate data summary tables |
| `fold_order_check.py` | Verify fold-order assumptions for calibration/reproducibility |
| `check_final_scores.py` | Compare reproduced final scores with official scores |
| `check_e12_independent.py` | Independently verify final bootstrap/evaluation evidence |
| `check_rehearsal.py` | Verify rehearsal protocol and artifacts |
| `explain_final.py` | Generate final SHAP tables and figures |
| `results_tables.py` | Build phase and appendix result tables |
| `export_history.py` | Export MLflow history and experiment log |
| `export_artifacts.py` | Archive text artifacts and build an index |
| `build_report.py` | Convert report drafts and figures into PDF/DOCX |
| `make_report_variants.py` | Produce full and standalone reader report variants |
| `make_notebooks.py` | Generate analysis notebooks |
| `record_demo.py` | Create the demonstration video |
| `demo_screenshots.py` | Capture browser screenshots for the demo |
| `encode_video.swift` | Encode the recorded demo video on macOS |
| `report_docx.js` | Node.js DOCX support used by report generation |
| `check_e12_independent.py` | Independent final-result consistency checking |

## 17. Test inventory

The tests cover the principal contracts rather than only isolated helper functions:

| Test area | Coverage |
|---|---|
| `test_data.py` | ARFF/data parsing, mappings, and data assumptions |
| `test_features.py` | Signed-log transform and feature grouping |
| `test_data.py`, `test_resample.py` | Stratification-related data assumptions and train-only resampling |
| `test_models.py` | Model registry, balancing, routing, and estimator behaviour |
| `test_models.py`, `test_results.py`, `test_final.py` | Training, scoring, and final-result protocols |
| `test_calibration.py` | ECE, bins, and confidence semantics |
| `test_predict.py` | Integrity checks, input validation, predictions, and SHAP top features |
| `test_explain.py` | SHAP tables and example selection |
| `test_check_final_scores.py`, `test_results.py` | Metrics and generated evaluation/result artifacts |
| `test_kaggle.py` | Kaggle provenance and split protocols |
| `test_tracking.py` | MLflow identity, hashes, and Git-state rules |
| `test_artifacts.py` | Saved artifacts and report indexes |
| `test_final.py` | Final model and final-score contracts |
| `test_build_report.py`, `test_report_drafts.py`, `test_report_variants.py` | Report generation |
| `test_figures.py`, `test_results.py` | Result and figure generation |
| `test_demo.py` | Streamlit/demo-facing contracts |

The exact test files are under [`tests/`](tests/). The standard commands are:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## 18. Dependency context

The dependency and tooling contract is defined in [`pyproject.toml`](pyproject.toml) and locked
in [`uv.lock`](uv.lock):

- Python 3.12;
- pandas, NumPy, PyArrow, and liac-arff for data;
- scikit-learn and imbalanced-learn for pipelines, metrics, splitting, and resampling;
- XGBoost and LightGBM for boosted trees;
- Optuna for tuning;
- MLflow for experiment tracking;
- SHAP for explanations;
- Matplotlib and Seaborn for figures;
- Streamlit for the demo;
- pytest and Ruff for development validation.

The runtime environment is intended to be installed with `uv sync`. XGBoost and LightGBM may
require platform-specific native runtime support such as `libomp` on macOS.

## 19. Security and reliability boundaries

The code includes several reliability safeguards:

- input feature names and order are checked exactly;
- non-numeric, missing, and infinite values are rejected;
- final model hashes are checked before loading;
- data and run hashes are recorded;
- official runs require committed and pushed code;
- test data is protected from routine model selection;
- generated data versions are not overwritten;
- errors are surfaced to the CLI or Streamlit user.

Important boundaries remain:

- joblib deserialization is only safe for trusted model files;
- the local Streamlit demo is not an authenticated public service;
- no request-level authorization, rate limiting, audit logging, or tenant isolation is present;
- raw datasets and local MLflow stores may contain sensitive or large artifacts and are not part of
  the normal deployment bundle.

## 20. Source-of-truth module reference

| Module | Context and connection |
|---|---|
| [`data.py`](src/blindspy/data.py) | Defines task labels, source paths, headers, parsing, and processed-data loading |
| [`checks.py`](src/blindspy/checks.py) | Data invariants and cross-file/Kaggle evidence used by preparation and tests |
| [`splits.py`](src/blindspy/splits.py) | Deterministic partitions and experiment-specific train-only resampling |
| [`kaggle.py`](src/blindspy/kaggle.py) | Separate Kaggle provenance, deduplication, and split path |
| [`features.py`](src/blindspy/features.py) | Fixed feature transforms and named feature groups/sets |
| [`models.py`](src/blindspy/models.py) | Estimator registry, pipelines, balancing, label adaptation, and two-step routing |
| [`train.py`](src/blindspy/train.py) | CV, holdout, final, calibration, cost measurement, and artifact production |
| [`evaluate.py`](src/blindspy/evaluate.py) | Metrics, confusion matrices, PR/ROC plots, and per-class outputs |
| [`calibration.py`](src/blindspy/calibration.py) | Isotonic confidence calibration and reliability analysis |
| [`tuning.py`](src/blindspy/tuning.py) | Seeded Optuna search over training-fold macro F1 |
| [`predict.py`](src/blindspy/predict.py) | Trusted final-model loading, input contract, confidence, and SHAP top features |
| [`explain.py`](src/blindspy/explain.py) | SHAP aggregation, groups, and selected examples |
| [`tracking.py`](src/blindspy/tracking.py) | MLflow stores, run identity, package/data hashes, and artifact logging |
| [`results.py`](src/blindspy/results.py) | Result-history and table utilities used by reporting scripts |
| [`figures.py`](src/blindspy/figures.py) | Shared report figure helpers |
| [`__init__.py`](src/blindspy/__init__.py) | Package boundary |

## 21. Final conclusions

BlindSpy is best understood as a traceable experimental ML repository with a small local
inference demonstration. Its strongest architectural properties are deterministic data
versioning, strict separation of training/validation/test data, pipeline-local preprocessing,
tracked experiment identity, reproducible final-model export, and a shared prediction path for
CLI and UI.

Its current scalability model is vertical and offline: parallel CV folds and local process
parallelism accelerate experiments, while inference runs inside one local process. It does not
currently implement horizontal request load balancing. A production deployment would require
adding an API boundary, immutable model packaging, multiple stateless inference workers,
health-checked routing, observability, and an explicit scaling policy.
