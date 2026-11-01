# How to read this report

Every experiment in this project has an ID, E00 to E17, and the report uses these IDs throughout:
in headings ("4.6 E04: KNN with a signed log transform"), in the text ("E04 lifted KNN by ..."),
in table rows and in run names such as E07-xgb-B-15s, which reads "experiment E07, XGBoost, task
B, 15 s files". The table below says what each ID is, in which order the experiments were run and
where the report describes them.

| ID | What it is | Order run | Section | Status |
|---|---|---|---|---|
| E00 | Data check: column names, file families, odd values, duplicates, sessions; data version v1 | 1 | 2, 4.0 | done |
| E01 | Majority-class baseline, the floor every model must beat | 2 | 4.1 | done |
| E02 | Decision tree (C4.5 approximation); also the paper benchmark with J48 settings | 2 | 4.2, 4.3 | done |
| E03 | KNN, logistic regression and SVM-RBF baselines; also the paper benchmark with IBk settings | 2 | 4.2, 4.3 | done |
| E04 | KNN with a signed log transform and two scalers | 3 | 4.6 | done |
| E05 | Random Forest, 300 trees | 4 | 4.7 | done |
| E06 | Gradient boosting: XGBoost and LightGBM with default settings | 5 | 4.8 | done |
| E07 | Hyperparameter tuning with Optuna of the best ensembles and KNN | 6 | 4.9 | done |
| E08 | Class balancing: class weights, oversampling, SMOTE | 7 | 4.10 | done |
| E09 | Feature-group ablation of the final models (analysis only) | 11 | 4.14 | done |
| E10 | Task B as one 14-class model or two steps | 8 | 4.11 | done |
| E11 | Session-aware split | - | 4.17 | not run: the files hold no session information |
| E12 | Final test: the 30 final models scored once on the test set | 17 | 4.21, 5 | done |
| E13 | Learning curve of the final models (analysis only) | 12 | 4.15 | done |
| E14 | Split-seed stability of the final models (analysis only) | 13 | 4.16 | done |
| E15 | Calibration of the final models' confidence, on validation | 16 | 4.20 | done |
| E16 | Reproducing Isnan et al. (2026) on the Kaggle copy, then correcting it | 15 | 4.18 | done |
| E17 | The same tuning budget for every method type | 9 | 4.12 | done |

**The IDs are not in the order the work was done.** Each ID was given when the experiment was
planned. The plan changed as results came in: E17 was added after E10, and the analyses E09, E13
and E14 had to wait until the final models were chosen. The real order was:

1. E00 (data check, data version v1)
2. E01, E02 and E03 (baselines), with the paper benchmark
3. E04 (KNN with a signed log)
4. E05 (Random Forest)
5. E06 (gradient boosting)
6. E07 (tuning)
7. E08 (class balancing)
8. E10 (one model or two steps for task B)
9. E17 (equal tuning budget)
10. final choice of models and the comparison set (Section 4.13)
11. E09 (feature-group ablation)
12. E13 (learning curve)
13. E14 (split-seed stability)
14. E11 recorded as not run (no session information)
15. E16 (reproducing Isnan et al.)
16. E15 (calibration, on validation, before the test set)
17. E12 (final test: the test set opened once)

Three steps have no ID of their own: the paper benchmark (part of E02 and E03, Section 4.3), the
storage checks (Sections 4.4 and 4.19) and the final choice of models with the comparison set
(Section 4.13). Section 4 is ordered by ID; the order list at its start repeats the sequence above.
