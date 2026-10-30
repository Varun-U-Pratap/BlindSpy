# 3 Method

*Working draft. These markdown drafts are the master copy of the report; the full report (DOCX and
PDF) is assembled from them at each milestone. Placeholders in [brackets] are filled when the named experiment or data version exists.*

## 3.1 Pipeline overview

All code is in the project repository (Python 3.12). Dependencies are managed with uv; exact
package versions are fixed in `uv.lock`, so every team member and every rerun uses identical
library versions. The main libraries are pandas and NumPy (data), liac-arff and pyarrow (reading
ARFF, writing Parquet), scikit-learn [3] (models, pipelines, metrics), MLflow [8] (run tracking),
matplotlib and seaborn (figures) and SHAP [6] (explanations). XGBoost [4], LightGBM [5],
Optuna [7] and imbalanced-learn [9] were added together in one commit before the first Phase 3
run, so every Phase 3 run uses the same locked library versions.

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
5. **Reporting.** `scripts/export_history.py` exports every run to `reports/history.csv`;
   `scripts/results_tables.py` builds every result table from it, and the notebooks (written for
   Phase 3 by `scripts/make_notebooks.py`) draw the figures from it, so no number in this report
   is copied by hand.

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
  test set, at the final milestone, with 95% percentile intervals from 10,000 stratified bootstrap
  resamples (rows drawn with replacement within each true class, so every resample keeps the test
  set's class counts; seed 42). A comparison set (the best validation variant of each method type
  per task, Section 3.8) is fixed before the test set is opened and scored the same way, once; it
  never changes the chosen configuration. 26 differences are tested, fixed before the test set is
  opened (E12): per task the chosen model against each of the other six method types, and against
  the old-rule pick (Section 3.8) in A1 and A2-VPN. Each test is a two-sided paired bootstrap on macro F1 (the
  same resampled rows for both models), Holm-corrected over all 26 at alpha 0.05; other
  differences and metrics are only described. 10,000 resamples are needed because with 1,000 the
  smallest possible p-value (0.0020) is above Holm's first threshold (0.05 / 26 = 0.0019).
- Calibration (E15) is decided on the validation set before the test set is opened; it changes
  only the reported confidence, never the predicted class.
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
split and inflate scores. A group-aware split would keep each session on one side, but the ARFF
files hold only the 23 timing features and the label: no capture-file name, address, port or
timestamp identifies a session (E00). E11 therefore cannot run, and this is listed as a
limitation (Section 7).

## 3.4 Models

| Model | Settings | Experiment |
| --- | --- | --- |
| Majority class | Always predicts the most frequent class (the floor) | E01 |
| Decision tree | Entropy criterion (C4.5 approximation); then tuned | E02, E17 |
| KNN | k = 5, standard scaling; signed log + standard or robust scaling; then tuned | E03, E04, E07 |
| Logistic Regression | Standard scaling; then tuned, signed log optional | E03, E17 |
| SVM, RBF kernel | Standard scaling; then tuned, signed log optional | E03, E17 |
| Random Forest | 300 trees, default settings; then tuned | E05, E07, E17 |
| XGBoost, LightGBM | Default settings; then tuned | E06, E07, E17 |
| Class balancing | Class weights, random oversampling, SMOTE (inside the pipeline) | E08 |
| Two-step model (task B) | VPN / non-VPN model, then one 7-class model per group | E10 |

Every baseline is run on all tasks (A1; A2-VPN and A2-NonVPN as separate models; B) at all four
timeouts. Improvement experiments change one thing at a time against the current best model:
preprocessing for KNN (E04), tuning with Optuna, 50 trials per model with the same budget for the
best baseline (E07), class imbalance handling (E08: none, class weights, random oversampling,
SMOTE [10]), feature-group ablation (E09), direct vs two-step classification for task B (E10), and
the same 50-trial tuning budget for every remaining method type (E17). The chosen models are then
described by a learning curve (E13) and a split-seed stability check (E14), and a published result
on a copy of this dataset is reproduced and corrected (E16). Each experiment states its hypothesis
and a numeric pass/fail threshold before it is run.

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

## 3.7 Preprocessing for distance-based models: the signed log

KNN, logistic regression and the RBF-kernel SVM depend on distances or linear scales, and the raw
timing features span six or more orders of magnitude (microseconds up to about 1.2e8), so a few
very large values dominate. A log compresses them, but plain log1p is not usable here: the
features hold -1 codes meaning "not measurable" (log1p(-1) is minus infinity) and 586 negative
minimum inter-arrival times (E00). We therefore apply the **signed log** sign(x)·log1p(|x|) as a
step inside the pipeline, before scaling. It is defined for every value, keeps the order of
values, keeps -1 apart from 0, and is a fixed formula: nothing is learned from data, so it cannot
leak information from validation or test rows. Data v1 is not changed. Alternatives not chosen:
shifting each column so its minimum is 0 before the log (a shift learned from the training data,
different per column, and -1 merges into small values); logging only the non-negative columns
(inconsistent, and large values stay uncompressed); replacing -1 by 0 plus indicator columns
(changes the feature set, so E04 would change two things at once). E04 compares the signed log
with StandardScaler and with RobustScaler; in E17 logistic regression and the SVM choose it inside
their search (both chose it in every task). Tree models are not affected by any order-keeping
transform, so they get none.

## 3.8 Phase 3 rules and decisions

These rules were fixed before the Phase 3 runs (28 Oct) unless stated otherwise; each experiment
entry repeats the ones it uses before it runs.

- **One change at a time.** Each experiment changes one thing against the **current best** model
  of each task: the highest 15 s validation macro F1 so far, whatever the margin.
- **Winner rule.** Every experiment runs first at 15 s for all four tasks. A model whose 15 s
  validation macro F1 beats the task's current best by at least 0.03 is rerun at 30, 60 and
  120 s with its 15 s settings (nothing is re-tuned per timeout). Why 0.03: it is above the
  largest Phase 2 cross-validation standard deviation (0.025), so a smaller gain could be noise
  from the split; a fixed number set before running keeps the choice honest. Not chosen:
  rerunning the top two per task, or running everything at every timeout (about four times the
  runs, and no screen-then-confirm design).
- **Tuning (E07, E17).** Optuna with the TPE sampler (seed 42), 50 trials per search, trials run
  one after another (parallel trials are not reproducible). Each trial is scored by 5-fold
  cross-validation macro F1 on the training rows only. E07 tunes, per task, the best two of
  Random Forest, XGBoost and LightGBM plus KNN with the same budget, so no model wins only
  because it was tuned more. E17 (added 28 Oct after E10) gives the same budget to every other
  method type, so the comparison of method types is between tuned models. Search ranges were
  written into the configurations before running. The best trial is retrained and scored like
  any other run; trials keep their settings and scores only.
- **Balancing (E08).** Inside the pipeline (imbalanced-learn), so only training folds are
  resampled. XGBoost has no class-weight option, so its "balanced" step uses per-row sample
  weights. SMOTE uses 5 neighbours; the smallest training class in any task file has 53 rows (MAIL,
  A2-NonVPN 30 s), about 42 per cross-validation fold, so it is always possible.
- **Two-step model (E10).** Both steps train on task B's training rows only; the VPN / non-VPN
  label comes from the B label. The A1 split is never used, because A1 and B hold the same flows
  split differently, so an A1-trained model would have seen B validation flows.
- **Analysis experiments (E09, E13, E14)** describe the chosen models and change nothing. If a
  feature set in E09 had scored at least 0.03 above all 23 features, the work would have stopped
  for a decision, because choosing features on validation scores risks overfitting the choice.
  E13 and E14 use other seeds on purpose and never touch the seed-42 test rows.
- **Reproducing Isnan et al. (E16)** uses its own data version (kaggle-k1, built by a committed
  script with its own manifest and hashes); data v1 is untouched, and no E16 result feeds model
  selection.
- **Cost and resources.** Every model runs on one thread (XGBoost and LightGBM included), with the
  five cross-validation folds in parallel. This keeps results deterministic and timings comparable
  with Phase 2. Peak memory is measured twice: with Python's tracemalloc (comparable with Phase 2)
  and as the peak memory of a fresh process doing the fit, which also sees the native memory of
  XGBoost and LightGBM. Every model is saved with joblib compression level 3. This is lossless and
  cuts a 300-tree Random Forest from 171.9 MiB to 22.4 MiB. Both the compressed and the
  uncompressed size are logged.
- **Final choice (decided 28 Oct after E10, a deviation from the rule set before Phase 3).** The
  rule fixed before Phase 3 treated any gap below 0.03 as a tie, broken in favour of the simpler
  model. Applied as written, it would have chosen KNN or the SVM at 0.0146-0.0286 below the best
  model. The cross-validation standard deviations of the chosen models (0.0028-0.0058) are far
  below 0.03, and the goal is the best score on unseen traffic. The final configuration per task
  is therefore the highest validation macro F1 among all model-selection runs (any experiment, any
  timeout). The only fallback is a saved model file above 50 MB (GitHub's warning size), which none
  of the chosen models reaches. The old rule's picks are scored on the test set too (E12), so the
  cost of either rule is visible.
- **Comparison set.** Fixed in `configs/final/compare/` before the test set is opened: per task,
  the best 15 s validation variant of each of 7 method types (decision tree, KNN, logistic
  regression, SVM-RBF, Random Forest, XGBoost, LightGBM), 28 models. A type's variants are its
  default, tuned and balanced runs; KNN includes the E03 and E04 preprocessings; the E10 two-step
  model counts as XGBoost; the majority baseline is a reference, not a member. Only the four
  headline models are saved in the repository (`models/final/`); the others stay in the run
  store.
