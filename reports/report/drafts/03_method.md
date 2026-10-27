# 3 Method

*Working draft. The Google Docs report is the master copy; this text is pasted there and kept in
sync. Placeholders in [brackets] are filled when the named experiment or data version exists.*

## 3.1 Pipeline overview

All code is in the project repository (Python 3.12). Dependencies are managed with uv; exact
package versions are fixed in `uv.lock`, so every team member and every rerun uses identical
library versions. The main libraries are pandas and NumPy (data), liac-arff and pyarrow (reading
ARFF, writing Parquet), scikit-learn [3] (models, pipelines, metrics), MLflow [8] (run tracking),
matplotlib and seaborn (figures) and SHAP [6] (explanations). XGBoost [4], LightGBM [5],
Optuna [7] and imbalanced-learn [9] are added at the experiments that first need them.

The flow of data and results is:

1. **Raw data.** The 20 ARFF files of ISCXVPN2016 [2] are kept unchanged; their SHA-256 hashes are
   recorded in `data/arff_sha256.txt` and checked on every machine.
2. **Preparation.** `scripts/prepare_data.py` parses the files, applies the verified column-name map
   and the cleaning rules, and writes one Parquet file per task and timeout to a versioned folder
   `data/processed/vN/`. [Column-name check and cleaning rules: E00.]
3. **Training.** `scripts/run_experiment.py` reads one YAML configuration file, builds a
   scikit-learn Pipeline (preprocessing + model), runs 5-fold cross-validation on the training set,
   fits on the training set and scores the validation set.
4. **Tracking.** Every run is logged to MLflow (Section 3.6).
5. **Reporting.** `scripts/export_history.py` exports every run to `reports/history.csv` and draws
   the report figures, so no number in this report is copied by hand.

## 3.2 Data versions

Any change that alters the cleaned data (a cleaning rule, the column map, the parser, row
filtering or the split seed) creates a new data version (v1, v2, …) in its own folder, which is
never overwritten. Each version has a manifest (file hashes, row and label counts, column map,
cleaning rules) and a step-by-step cleaning log listing, for every step, the rows before and
after, rows removed or changed (overall and per class), the rule and its reason. Steps that change
nothing and decisions not to do something are logged too. Every run records the data version and
its hash. After a version change, the current best model is rerun as its own experiment so the
effect of the change is measured. [Data versions: see Section 2 and the data changelog.]

## 3.3 Splits and evaluation protocol

For each task and timeout the cleaned data is split once into 70% training, 15% validation and 15%
test sets, stratified by class, with random seed 42. The row indices are saved inside the data
version, so every run on that version uses exactly the same rows.

- Hyperparameters are chosen by 5-fold stratified cross-validation inside the training set only.
- Candidate models are compared on the validation set.
- The chosen configuration is retrained on training + validation data and scored **once** on the
  test set, at the final milestone, with 95% confidence intervals from 1,000 bootstrap resamples.
- Scaling, transforms and resampling are steps inside the pipeline, so they are fitted on training
  folds only and never see validation or test rows.
- Every random step (split, models, tuning sampler, oversampling, bootstrap) uses seed 42. Only the
  learning-curve and seed-stability experiments (E13, E14) use seeds 42–51, on purpose.

**Paper benchmark.** To compare with Draper-Gil et al. [1], the decision tree and KNN are also run
with the paper's own protocol: 10-fold cross-validation on the full file with Weka's default
settings (KNN with k = 1; C4.5 with pruning confidence 0.25 and at least 2 samples per leaf). These
results are reported in their own table. scikit-learn's decision tree with the entropy criterion is
a close substitute for C4.5 [11], not an exact copy, and is called a C4.5 approximation.

**Session-aware check.** Flows from one recording session may fall on both sides of a random
split and inflate scores. If the data identifies sessions, the best models are also evaluated with
a group-aware split that keeps each session on one side. [E11, depending on E00.]

## 3.4 Models

| Model | Settings | Experiment |
| --- | --- | --- |
| Majority class | Always predicts the most frequent class (the floor) | E01 |
| Decision tree | Entropy criterion (C4.5 approximation) | E02 |
| KNN | k = 5, standard scaling | E03 |
| Logistic Regression | Standard scaling | E03 |
| SVM, RBF kernel | Standard scaling | E03 |
| Random Forest | 300 trees | E05 |
| XGBoost, LightGBM | Default settings, then tuned | E06, E07 |

Every baseline is run on all tasks (A1; A2-VPN and A2-NonVPN as separate models; B) at all four
timeouts. Improvement experiments change one thing at a time against the current best model:
preprocessing for KNN (E04), tuning with Optuna, 50 trials per model with the same budget for the
best baseline (E07), class imbalance handling (E08: none, class weights, random oversampling,
SMOTE [10]), feature-group ablation (E09), direct vs two-step classification for task B (E10). Each
experiment states its hypothesis and a numeric pass/fail threshold before it is run.

## 3.5 Metrics

The main metric is **macro F1**, which weights every class equally, so small classes count as much
as large ones. We also report per-class precision, recall and F1; weighted F1; accuracy; balanced
accuracy; confusion matrices and the five most frequent confusion pairs; ROC-AUC (task A1);
cross-validation mean and standard deviation; the gap between cross-validation and held-out
scores; training time, prediction time per 1,000 flows, model size and peak memory; and, for the
final models, Expected Calibration Error with a reliability diagram (E15). Comparisons with the
paper use precision against precision and recall against recall.

## 3.6 Tracking and reproducibility

Every run is logged automatically to MLflow with: the experiment ID, the person and machine that
ran it (operating system, CPU, memory), the git commit, whether there were uncommitted changes,
the hash of the dependency lock file, the configuration file, the data version and hash, all
settings, all metrics, timing, model size and memory, plots, and the trained model file (every
official run except individual tuning trials). Official runs are made on one machine, so timings
are comparable, and only from code that is committed and pushed to the shared repository; the run
script refuses otherwise. Each experiment also has a written entry (hypothesis, threshold, change,
result, verdict, decision), and milestones are marked with git tags (v0.1-baseline, v0.2-tuned,
v1.0-final).
