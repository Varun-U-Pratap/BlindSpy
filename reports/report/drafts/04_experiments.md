# 4 Experiments

*Working draft. The Google Docs report is the master copy; this text is pasted there and kept in
sync. Every table here is copied unchanged from `reports/baseline_tables.md`, which
`scripts/results_tables.py` generates from `reports/history.csv` (all MLflow runs, exported by
`scripts/export_history.py`). Every number cites its experiment (E01-E03 entries in
`reports/EXPERIMENT_LOG.md`). Data version: v1. Code: tag `v0.1-baseline` (commit 1817c5a) for the
80 baseline runs; commit 36c56a1 for the 64 paper-benchmark runs.*

Each experiment states a hypothesis and a numeric pass/fail threshold before it runs, changes one
thing, and ends with a verdict (supported / not supported). Unless stated otherwise, scores are on
the **validation** set of the fixed 70/15/15 split (seed 42); models are chosen with 5-fold
cross-validation on the training set; the test set is not used until the final experiment (E12).

## 4.0 E00: data findings and data version v1

E00 checked the dataset before any model was trained: the corrected column names, the file-family
decision, the odd-value and duplicate rules, the session question and the Kaggle finding. It
produced data version v1 (167,909 flows in 16 task files). Section 2 reports it in full.

## 4.1 E01: majority-class baseline

**What.** A model that always predicts the largest training class (`DummyClassifier`), for every
task and timeout (16 runs).
**Why.** It is the floor every model must beat, and its scores are known in advance, so it also
tests the metrics code: accuracy must equal the largest class's share p of the validation set, and
macro F1 must equal 2p / ((1 + p) k) for k classes.
**Result.** Supported: all 16 runs match both values to 4 decimals (threshold 0.001). The floor in
macro F1 is 0.34-0.36 for A1, 0.06-0.10 for A2 and 0.02-0.03 for B. Accuracy alone would look
much better (up to 0.56 for A1 and 0.51 for A2-NonVPN 120 s), which is why macro F1 is the main
score.

| task | timeout | classes | largest_class | p | val_accuracy | expected_macro_f1 | val_macro_f1 | ok |
|---|---|---|---|---|---|---|---|---|
| A1 | 15 s | 2 | VPN | 0.5185 | 0.5185 | 0.3414 | 0.3414 | yes |
| A1 | 30 s | 2 | VPN | 0.5083 | 0.5083 | 0.337 | 0.337 | yes |
| A1 | 60 s | 2 | Non-VPN | 0.5577 | 0.5577 | 0.358 | 0.358 | yes |
| A1 | 120 s | 2 | Non-VPN | 0.5185 | 0.5185 | 0.3415 | 0.3415 | yes |
| A2-VPN | 15 s | 7 | VPN-BROWSING | 0.2665 | 0.2665 | 0.0601 | 0.0601 | yes |
| A2-VPN | 30 s | 7 | VPN-BROWSING | 0.3599 | 0.3599 | 0.0756 | 0.0756 | yes |
| A2-VPN | 60 s | 7 | VPN-BROWSING | 0.3769 | 0.3769 | 0.0782 | 0.0782 | yes |
| A2-VPN | 120 s | 7 | VPN-BROWSING | 0.4828 | 0.4828 | 0.093 | 0.093 | yes |
| A2-NonVPN | 15 s | 7 | VOIP | 0.3244 | 0.3244 | 0.07 | 0.07 | yes |
| A2-NonVPN | 30 s | 7 | BROWSING | 0.3735 | 0.3735 | 0.0777 | 0.0777 | yes |
| A2-NonVPN | 60 s | 7 | BROWSING | 0.3046 | 0.3046 | 0.0667 | 0.0667 | yes |
| A2-NonVPN | 120 s | 7 | BROWSING | 0.5061 | 0.5061 | 0.096 | 0.096 | yes |
| B | 15 s | 14 | VOIP | 0.156 | 0.156 | 0.0193 | 0.0193 | yes |
| B | 30 s | 14 | BROWSING | 0.1833 | 0.1833 | 0.0221 | 0.0221 | yes |
| B | 60 s | 14 | BROWSING | 0.1697 | 0.1697 | 0.0207 | 0.0207 | yes |
| B | 120 s | 14 | BROWSING | 0.2606 | 0.2606 | 0.0295 | 0.0295 | yes |

## 4.2 E02 and E03: decision tree, KNN, Logistic Regression and SVM-RBF

**What.** Four baselines, each a scikit-learn pipeline, run on all 16 task/timeout files (64 runs):
an unpruned entropy decision tree (E02; the closest scikit-learn model to C4.5, so a "C4.5
approximation"), KNN with k = 5 and standard scaling, Logistic Regression and an RBF-kernel SVM,
both with standard scaling (E03). Scaling is a pipeline step, fitted on training rows only.
**Why.** The tree and KNN are the paper's two algorithms. Logistic Regression and SVM-RBF match
the baselines of Isnan et al. [13]. Together they set the bar for the improvements in Phase 3.
**Hypotheses and thresholds (written before running).** E02: the tree beats the floor by at least
0.30 in every config and reaches at least 0.80 on A1 at every timeout. E03: all three beat the floor
by at least 0.10; KNN is within 0.10 of the tree in at least 12 of 16 configs; Logistic Regression
and SVM-RBF score below the tree in at least 12 of 16 configs, because the features are heavy-tailed
(microsecond times up to about 10^8 next to -1 and 0 codes).
**Result.** All supported. The tree is the best baseline in all 16 configs (macro F1 0.706-0.890),
0.499-0.765 above the floor. KNN is second everywhere, 0.039-0.101 below the tree (within 0.10 in
15 of 16). Logistic Regression (0.270-0.598) and SVM-RBF (0.274-0.621) are below the tree in all 16.

Validation macro F1 (E01-E03):

| task | timeout | majority | tree | knn | logreg | svm_rbf |
|---|---|---|---|---|---|---|
| A1 | 15 s | 0.341 | 0.890 | 0.813 | 0.585 | 0.602 |
| A1 | 30 s | 0.337 | 0.864 | 0.825 | 0.598 | 0.610 |
| A1 | 60 s | 0.358 | 0.857 | 0.813 | 0.541 | 0.545 |
| A1 | 120 s | 0.341 | 0.857 | 0.807 | 0.576 | 0.621 |
| A2-VPN | 15 s | 0.060 | 0.825 | 0.774 | 0.432 | 0.468 |
| A2-VPN | 30 s | 0.076 | 0.828 | 0.788 | 0.536 | 0.542 |
| A2-VPN | 60 s | 0.078 | 0.775 | 0.730 | 0.339 | 0.330 |
| A2-VPN | 120 s | 0.093 | 0.780 | 0.727 | 0.332 | 0.332 |
| A2-NonVPN | 15 s | 0.070 | 0.829 | 0.773 | 0.449 | 0.487 |
| A2-NonVPN | 30 s | 0.078 | 0.828 | 0.783 | 0.486 | 0.468 |
| A2-NonVPN | 60 s | 0.067 | 0.825 | 0.750 | 0.447 | 0.441 |
| A2-NonVPN | 120 s | 0.096 | 0.839 | 0.769 | 0.482 | 0.471 |
| B | 15 s | 0.019 | 0.759 | 0.660 | 0.307 | 0.296 |
| B | 30 s | 0.022 | 0.748 | 0.647 | 0.320 | 0.311 |
| B | 60 s | 0.021 | 0.706 | 0.617 | 0.271 | 0.278 |
| B | 120 s | 0.030 | 0.716 | 0.637 | 0.270 | 0.274 |

5-fold cross-validation on the training set, mean +- standard deviation:

| task | timeout | majority | tree | knn | logreg | svm_rbf |
|---|---|---|---|---|---|---|
| A1 | 15 s | 0.341 +- 0.000 | 0.888 +- 0.005 | 0.821 +- 0.002 | 0.600 +- 0.011 | 0.608 +- 0.005 |
| A1 | 30 s | 0.337 +- 0.000 | 0.873 +- 0.006 | 0.814 +- 0.010 | 0.600 +- 0.012 | 0.619 +- 0.023 |
| A1 | 60 s | 0.358 +- 0.000 | 0.852 +- 0.010 | 0.808 +- 0.009 | 0.551 +- 0.008 | 0.552 +- 0.006 |
| A1 | 120 s | 0.341 +- 0.000 | 0.857 +- 0.009 | 0.803 +- 0.002 | 0.569 +- 0.008 | 0.608 +- 0.011 |
| A2-VPN | 15 s | 0.060 +- 0.000 | 0.810 +- 0.007 | 0.767 +- 0.008 | 0.410 +- 0.014 | 0.448 +- 0.024 |
| A2-VPN | 30 s | 0.076 +- 0.000 | 0.796 +- 0.009 | 0.769 +- 0.025 | 0.495 +- 0.016 | 0.507 +- 0.021 |
| A2-VPN | 60 s | 0.078 +- 0.000 | 0.760 +- 0.016 | 0.708 +- 0.019 | 0.334 +- 0.022 | 0.331 +- 0.021 |
| A2-VPN | 120 s | 0.093 +- 0.000 | 0.725 +- 0.023 | 0.692 +- 0.016 | 0.343 +- 0.019 | 0.317 +- 0.021 |
| A2-NonVPN | 15 s | 0.070 +- 0.000 | 0.862 +- 0.009 | 0.792 +- 0.010 | 0.459 +- 0.013 | 0.479 +- 0.018 |
| A2-NonVPN | 30 s | 0.078 +- 0.000 | 0.817 +- 0.015 | 0.783 +- 0.017 | 0.459 +- 0.007 | 0.426 +- 0.008 |
| A2-NonVPN | 60 s | 0.067 +- 0.000 | 0.829 +- 0.016 | 0.740 +- 0.013 | 0.454 +- 0.015 | 0.432 +- 0.022 |
| A2-NonVPN | 120 s | 0.096 +- 0.000 | 0.807 +- 0.015 | 0.731 +- 0.013 | 0.472 +- 0.016 | 0.441 +- 0.023 |
| B | 15 s | 0.019 +- 0.000 | 0.757 +- 0.013 | 0.663 +- 0.008 | 0.299 +- 0.010 | 0.278 +- 0.008 |
| B | 30 s | 0.022 +- 0.000 | 0.733 +- 0.008 | 0.654 +- 0.016 | 0.323 +- 0.009 | 0.292 +- 0.008 |
| B | 60 s | 0.021 +- 0.000 | 0.703 +- 0.013 | 0.607 +- 0.010 | 0.275 +- 0.002 | 0.261 +- 0.009 |
| B | 120 s | 0.030 +- 0.000 | 0.684 +- 0.020 | 0.607 +- 0.015 | 0.250 +- 0.005 | 0.251 +- 0.017 |

What the tables show:

- **Cross-validation predicts validation.** For every run, |CV mean - validation| is at most 0.054,
  and CV standard deviations are at most 0.025. A gap of several points between two models is
  therefore larger than fold-to-fold noise.
- **Small classes.** Logistic Regression and SVM-RBF score F1 = 0.00 on MAIL / VPN-MAIL in most A2
  and B configs. So their macro F1 is far below their weighted F1 (gap up to 0.225); the tree's gap
  is 0.000-0.069. The SVM keeps 55-88% of the training rows as support vectors, a sign that the RBF
  kernel separates the standard-scaled features poorly. A log transform or a robust scaler is
  tested in E04.
- **Easiest and hardest classes.** VoIP is the easiest type (tree F1 0.92-0.99 for VOIP and
  VPN-VOIP). The weakest are CHAT, VPN-CHAT, STREAMING, MAIL and VPN-MAIL (tree F1 0.42-0.71).
- **Most frequent confusion.** In task B the largest confusion for the tree and KNN, at every
  timeout, is BROWSING vs VPN-BROWSING in both directions (tree: 29-45 validation flows each way).
  The model recognises browsing, but not whether it went through the VPN.
- **Timeouts.** B and A2-VPN score lower at 60 s and 120 s. The class mix differs between timeouts
  (Section 2.8), and the 60 s files hold differently shaped VOIP and FT flows (E00 addendum), so this
  is not read as an effect of the timeout.
- **Cost.** All baselines train in under 3 s. To predict 1,000 flows, the tree needs 0.30 ms
  (median), KNN 7.9 ms and the SVM 211 ms. Model files range from 1.4 KB (majority) and
  4-5 KB (Logistic Regression) to 4.2 MB (SVM). Training times were measured with Python's memory
  tracer switched on, which slows fitting (x1.05 tree, x1.00 SVM, x1.19 Logistic Regression,
  x2.3 KNN in a recheck), so they are slight overestimates; the ranking doesn't change.

Per-class extremes for the tree (E02): macro vs weighted F1, their gap, the generalisation gap, and
the best and worst class by F1 and by recall:

| task | timeout | macro F1 | weighted F1 | gap | CV - val | best F1 | worst F1 | best recall | worst recall |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 15 s | 0.89 | 0.89 | 0.0 | -0.002 | VPN 0.89 | Non-VPN 0.89 | Non-VPN 0.89 | VPN 0.89 |
| A1 | 30 s | 0.864 | 0.864 | 0.0 | 0.009 | VPN 0.87 | Non-VPN 0.86 | VPN 0.87 | Non-VPN 0.86 |
| A1 | 60 s | 0.857 | 0.859 | 0.002 | -0.005 | Non-VPN 0.87 | VPN 0.84 | Non-VPN 0.87 | VPN 0.85 |
| A1 | 120 s | 0.857 | 0.857 | 0.0 | 0.0 | Non-VPN 0.86 | VPN 0.85 | Non-VPN 0.86 | VPN 0.85 |
| A2-VPN | 15 s | 0.825 | 0.849 | 0.024 | -0.014 | VPN-VOIP 0.99 | VPN-CHAT 0.65 | VPN-VOIP 0.98 | VPN-CHAT 0.66 |
| A2-VPN | 30 s | 0.828 | 0.847 | 0.02 | -0.031 | VPN-VOIP 0.98 | VPN-CHAT 0.67 | VPN-STREAMING 0.98 | VPN-CHAT 0.70 |
| A2-VPN | 60 s | 0.775 | 0.831 | 0.056 | -0.014 | VPN-VOIP 0.92 | VPN-CHAT 0.63 | VPN-VOIP 0.92 | VPN-FT 0.65 |
| A2-VPN | 120 s | 0.78 | 0.825 | 0.045 | -0.054 | VPN-VOIP 0.98 | VPN-CHAT 0.61 | VPN-VOIP 0.98 | VPN-CHAT 0.57 |
| A2-NonVPN | 15 s | 0.829 | 0.891 | 0.061 | 0.033 | VOIP 0.99 | STREAMING 0.71 | VOIP 0.99 | STREAMING 0.70 |
| A2-NonVPN | 30 s | 0.828 | 0.889 | 0.061 | -0.011 | VOIP 0.98 | STREAMING 0.68 | VOIP 0.98 | STREAMING 0.57 |
| A2-NonVPN | 60 s | 0.825 | 0.878 | 0.053 | 0.004 | VOIP 0.97 | STREAMING 0.58 | VOIP 0.97 | STREAMING 0.55 |
| A2-NonVPN | 120 s | 0.839 | 0.893 | 0.054 | -0.033 | VOIP 0.95 | CHAT 0.56 | BROWSING 0.95 | CHAT 0.56 |
| B | 15 s | 0.759 | 0.81 | 0.051 | -0.001 | VOIP 0.98 | VPN-MAIL 0.55 | VOIP 0.99 | VPN-MAIL 0.52 |
| B | 30 s | 0.748 | 0.8 | 0.053 | -0.014 | VPN-VOIP 0.98 | MAIL 0.55 | VPN-VOIP 0.99 | MAIL 0.55 |
| B | 60 s | 0.706 | 0.775 | 0.069 | -0.002 | VOIP 0.92 | VPN-CHAT 0.42 | VOIP 0.92 | VPN-CHAT 0.42 |
| B | 120 s | 0.716 | 0.767 | 0.051 | -0.031 | VOIP 0.97 | STREAMING 0.47 | VOIP 1.00 | CHAT 0.42 |

Top 5 confusion pairs of the tree on the validation set (E02):

| task | timeout | top 5 confusion pairs (true -> predicted: flows) |
|---|---|---|
| A1 | 15 s | VPN -> Non-VPN: 158; Non-VPN -> VPN: 139 |
| A1 | 30 s | Non-VPN -> VPN: 142; VPN -> Non-VPN: 135 |
| A1 | 60 s | Non-VPN -> VPN: 161; VPN -> Non-VPN: 150 |
| A1 | 120 s | Non-VPN -> VPN: 104; VPN -> Non-VPN: 100 |
| A2-VPN | 15 s | VPN-FT -> VPN-CHAT: 28; VPN-CHAT -> VPN-BROWSING: 26; VPN-BROWSING -> VPN-CHAT: 25; VPN-CHAT -> VPN-FT: 20; VPN-FT -> VPN-BROWSING: 19 |
| A2-VPN | 30 s | VPN-BROWSING -> VPN-CHAT: 26; VPN-FT -> VPN-BROWSING: 22; VPN-BROWSING -> VPN-FT: 15; VPN-CHAT -> VPN-BROWSING: 12; VPN-CHAT -> VPN-FT: 12 |
| A2-VPN | 60 s | VPN-FT -> VPN-BROWSING: 17; VPN-BROWSING -> VPN-FT: 16; VPN-FT -> VPN-CHAT: 14; VPN-BROWSING -> VPN-VOIP: 13; VPN-CHAT -> VPN-FT: 12 |
| A2-VPN | 120 s | VPN-FT -> VPN-BROWSING: 18; VPN-BROWSING -> VPN-FT: 16; VPN-CHAT -> VPN-BROWSING: 11; VPN-FT -> VPN-P2P: 9; VPN-P2P -> VPN-BROWSING: 9 |
| A2-NonVPN | 15 s | BROWSING -> CHAT: 18; CHAT -> BROWSING: 15; BROWSING -> STREAMING: 12; FT -> BROWSING: 12; FT -> CHAT: 11 |
| A2-NonVPN | 30 s | CHAT -> BROWSING: 12; STREAMING -> BROWSING: 11; BROWSING -> CHAT: 10; BROWSING -> FT: 10; FT -> CHAT: 9 |
| A2-NonVPN | 60 s | CHAT -> BROWSING: 20; BROWSING -> CHAT: 19; MAIL -> FT: 8; FT -> BROWSING: 7; BROWSING -> MAIL: 6 |
| A2-NonVPN | 120 s | CHAT -> BROWSING: 12; P2P -> BROWSING: 8; FT -> BROWSING: 7; BROWSING -> P2P: 6; BROWSING -> CHAT: 5 |
| B | 15 s | VPN-BROWSING -> BROWSING: 41; BROWSING -> VPN-BROWSING: 39; VPN-CHAT -> VPN-FT: 25; VPN-FT -> VPN-CHAT: 24; VPN-BROWSING -> VPN-CHAT: 18 |
| B | 30 s | VPN-BROWSING -> BROWSING: 44; BROWSING -> VPN-BROWSING: 36; VPN-BROWSING -> VPN-CHAT: 13; VPN-FT -> VPN-CHAT: 12; VPN-CHAT -> VPN-BROWSING: 11 |
| B | 60 s | BROWSING -> VPN-BROWSING: 37; VPN-BROWSING -> BROWSING: 29; VPN-MAIL -> MAIL: 15; VPN-P2P -> P2P: 14; VPN-VOIP -> VOIP: 14 |
| B | 120 s | BROWSING -> VPN-BROWSING: 45; VPN-BROWSING -> BROWSING: 32; VPN-BROWSING -> VPN-FT: 17; VPN-FT -> VPN-BROWSING: 12; VPN-BROWSING -> VPN-P2P: 10 |

Model complexity (E02 tree, E03 SVM):

| task | timeout | n_train | tree depth | tree nodes | tree leaves | SVM support vectors |
|---|---|---|---|---|---|---|
| A1 | 15 s | 12,626 | 27 | 1,789 | 895 | 10,292 |
| A1 | 30 s | 9,526 | 35 | 1,601 | 801 | 7,976 |
| A1 | 60 s | 10,273 | 27 | 1,917 | 959 | 7,882 |
| A1 | 120 s | 6,666 | 25 | 1,243 | 622 | 5,501 |
| A2-VPN | 15 s | 6,566 | 23 | 1,407 | 704 | 4,497 |
| A2-VPN | 30 s | 4,860 | 20 | 1,067 | 534 | 3,108 |
| A2-VPN | 60 s | 4,565 | 24 | 1,249 | 625 | 3,265 |
| A2-VPN | 120 s | 3,252 | 21 | 985 | 493 | 2,368 |
| A2-NonVPN | 15 s | 6,085 | 16 | 743 | 372 | 3,329 |
| A2-NonVPN | 30 s | 4,685 | 19 | 751 | 376 | 2,751 |
| A2-NonVPN | 60 s | 5,744 | 24 | 1,001 | 501 | 3,803 |
| A2-NonVPN | 120 s | 3,458 | 18 | 591 | 296 | 1,907 |
| B | 15 s | 12,651 | 30 | 3,409 | 1,705 | 11,076 |
| B | 30 s | 9,546 | 23 | 2,901 | 1,451 | 8,368 |
| B | 60 s | 10,309 | 26 | 3,437 | 1,719 | 8,796 |
| B | 120 s | 6,711 | 22 | 2,481 | 1,241 | 5,813 |

Cost per model over the 16 configs:

| model | train time (s) median | train time (s) max | 5-fold CV time (s) median | 5-fold CV time (s) max | predict ms / 1,000 flows median | predict ms / 1,000 flows max | model file (KB) median | model file (KB) max | peak fit memory (MB) median | peak fit memory (MB) max |
|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.013 | 0.027 | 0.147 | 2.325 | 0.097 | 0.176 | 1.414 | 1.523 | 0.624 | 1.181 |
| tree | 0.121 | 0.312 | 0.268 | 0.493 | 0.301 | 0.534 | 134.49 | 592.916 | 1.258 | 2.393 |
| knn | 0.021 | 0.042 | 0.187 | 0.258 | 7.856 | 29.5 | 1242.967 | 2374.631 | 3.003 | 5.733 |
| logreg | 0.116 | 0.461 | 0.201 | 0.604 | 0.364 | 0.684 | 3.829 | 5.196 | 3.432 | 8.147 |
| svm_rbf | 0.636 | 2.939 | 2.05 | 3.849 | 211.011 | 599.119 | 1175.091 | 4288.896 | 3.554 | 7.783 |

Peak fit memory is measured with Python's tracemalloc, which counts NumPy arrays
but not memory allocated inside compiled libraries (for example libsvm), so SVM
memory is a lower bound. In these Phase 2 runs the timed fit also ran with
tracemalloc on, which slows fitting: in a recheck (B 15 s, median of 5 fits) by
x1.05 for the tree, x1.00 for the SVM, x1.19 for Logistic Regression and x2.3 for
KNN (0.012 s -> 0.027 s). So the training times above are slight overestimates;
from Phase 3 on, time and memory come from separate fits. The metric
process_max_rss_mb in these runs is the high-water mark of the whole batch
process (it only grows from run to run), not a per-run figure; it is not used
and is no longer logged.

The same tables for KNN, Logistic Regression and SVM-RBF, plus accuracy, balanced accuracy,
weighted F1 and A1 ROC-AUC for every run, are in `reports/baseline_tables.md` (Appendix).

![Baselines E01-E03](../../figures/baseline_model_comparison.png)

## 4.3 Paper benchmark (part of E02 and E03)

**What.** The paper's own protocol [1]: 10-fold stratified cross-validation on the whole file, with
predictions pooled over the folds as Weka reports them, for its two algorithms with Weka's default
settings. J48 (C4.5) defaults are a pruning confidence of 0.25 and at least 2 instances per leaf;
scikit-learn has no C4.5 pruning, so our stand-in is an unpruned entropy tree with at least 2 rows
per leaf. IBk (KNN) defaults are k = 1 with a Euclidean distance that normalises each attribute
to its range; our stand-in is min-max scaling followed by 1-NN. The defaults were checked in the
Weka documentation [16] (also cited in `notebooks/02_baselines.ipynb`). Each algorithm ran on two row
sources per file (64 runs): every row as published (duplicates kept, values untouched), which is
closest to the paper's input, and data v1 (deduplicated).
**Why.** It tests whether we reproduce the paper's results before we try to improve on them. We
compare precision with precision and recall with recall, never the paper's precision with our F1.
These scores are never used to choose a model.
**Hypotheses and thresholds.** On the published rows, at least 12 of the 16 A1 values of the
paper's Figure 2 are matched within +-0.05, for each algorithm. Because duplicates can put a copy
of a flow in the training folds, removing them lowers 10-fold macro F1 in at least 12 of 16 files.
**Result.** Reproduction supported: our C4.5 approximation matches all 16 A1 values (largest
difference 0.021, close to the +-0.01 reading error of the chart), and 1-NN matches 15 of 16 (the
miss is VPN recall at 120 s: 0.803 vs 0.860). The duplicate hypothesis is not supported (6 of 16
files lower for the tree, 2 of 16 for 1-NN).

A1, paper (Figure 2, read to +-0.01) vs ours (`diff` = ours - paper):

| algorithm | class | timeout | metric | paper | paper_source | ours_published | ours_v1 | diff_published | diff_v1 |
|---|---|---|---|---|---|---|---|---|---|
| C4.5 | Non-VPN | 15 | precision | 0.906 | chart +-0.01 and text | 0.886 | 0.882 | -0.02 | -0.024 |
| C4.5 | Non-VPN | 30 | precision | 0.885 | chart +-0.01 | 0.866 | 0.866 | -0.019 | -0.019 |
| C4.5 | Non-VPN | 60 | precision | 0.864 | chart +-0.01 | 0.86 | 0.876 | -0.004 | 0.012 |
| C4.5 | Non-VPN | 120 | precision | 0.887 | chart +-0.01 and text | 0.87 | 0.863 | -0.017 | -0.024 |
| C4.5 | Non-VPN | 15 | recall | 0.877 | chart +-0.01 | 0.882 | 0.903 | 0.005 | 0.026 |
| C4.5 | Non-VPN | 30 | recall | 0.846 | chart +-0.01 | 0.857 | 0.886 | 0.011 | 0.04 |
| C4.5 | Non-VPN | 60 | recall | 0.888 | chart +-0.01 | 0.889 | 0.898 | 0.001 | 0.01 |
| C4.5 | Non-VPN | 120 | recall | 0.843 | chart +-0.01 | 0.853 | 0.89 | 0.01 | 0.047 |
| C4.5 | VPN | 15 | precision | 0.89 | chart +-0.01 and text | 0.892 | 0.908 | 0.002 | 0.018 |
| C4.5 | VPN | 30 | precision | 0.868 | chart +-0.01 | 0.874 | 0.888 | 0.006 | 0.02 |
| C4.5 | VPN | 60 | precision | 0.856 | chart +-0.01 | 0.857 | 0.867 | 0.001 | 0.011 |
| C4.5 | VPN | 120 | precision | 0.863 | chart +-0.01 and text | 0.868 | 0.877 | 0.005 | 0.014 |
| C4.5 | VPN | 15 | recall | 0.917 | chart +-0.01 | 0.896 | 0.888 | -0.021 | -0.029 |
| C4.5 | VPN | 30 | recall | 0.901 | chart +-0.01 | 0.881 | 0.867 | -0.02 | -0.034 |
| C4.5 | VPN | 60 | recall | 0.827 | chart +-0.01 | 0.821 | 0.84 | -0.006 | 0.013 |
| C4.5 | VPN | 120 | recall | 0.902 | chart +-0.01 | 0.884 | 0.849 | -0.018 | -0.053 |
| KNN | Non-VPN | 15 | precision | 0.846 | chart +-0.01 and text | 0.821 | 0.846 | -0.025 | 0.0 |
| KNN | Non-VPN | 30 | precision | 0.84 | chart +-0.01 | 0.805 | 0.83 | -0.035 | -0.01 |
| KNN | Non-VPN | 60 | precision | 0.838 | chart +-0.01 | 0.831 | 0.854 | -0.007 | 0.016 |
| KNN | Non-VPN | 120 | precision | 0.837 | chart +-0.01 and text | 0.794 | 0.83 | -0.043 | -0.007 |
| KNN | Non-VPN | 15 | recall | 0.832 | chart +-0.01 | 0.857 | 0.848 | 0.025 | 0.016 |
| KNN | Non-VPN | 30 | recall | 0.808 | chart +-0.01 | 0.839 | 0.831 | 0.031 | 0.023 |
| KNN | Non-VPN | 60 | recall | 0.851 | chart +-0.01 | 0.857 | 0.849 | 0.006 | -0.002 |
| KNN | Non-VPN | 120 | recall | 0.787 | chart +-0.01 | 0.827 | 0.825 | 0.04 | 0.038 |
| KNN | VPN | 15 | precision | 0.848 | chart +-0.01 and text | 0.863 | 0.859 | 0.015 | 0.011 |
| KNN | VPN | 30 | precision | 0.835 | chart +-0.01 | 0.85 | 0.836 | 0.015 | 0.001 |
| KNN | VPN | 60 | precision | 0.812 | chart +-0.01 | 0.816 | 0.811 | 0.004 | -0.001 |
| KNN | VPN | 120 | precision | 0.815 | chart +-0.01 and text | 0.836 | 0.813 | 0.021 | -0.002 |
| KNN | VPN | 15 | recall | 0.861 | chart +-0.01 | 0.83 | 0.857 | -0.031 | -0.004 |
| KNN | VPN | 30 | recall | 0.862 | chart +-0.01 | 0.818 | 0.835 | -0.044 | -0.027 |
| KNN | VPN | 60 | recall | 0.795 | chart +-0.01 | 0.784 | 0.817 | -0.011 | 0.022 |
| KNN | VPN | 120 | recall | 0.86 | chart +-0.01 | 0.803 | 0.818 | -0.057 | -0.042 |

![Paper vs ours, A1](../../figures/paper_vs_ours_A1.png)

The plan asks for the comparison as one table per task. The tables below give our 10-fold
precision and recall for every class, both algorithms and all timeouts, on the published rows
(closest to the paper's input), with the paper's value in brackets wherever the paper gives one
(C = read from its chart, +-0.01; T = stated in its text). The same tables on data v1 are in
`reports/baseline_tables.md`.

A2-VPN:

| class | metric | C4.5 15 s | C4.5 30 s | C4.5 60 s | C4.5 120 s | KNN 15 s | KNN 30 s | KNN 60 s | KNN 120 s |
|---|---|---|---|---|---|---|---|---|---|
| VPN-BROWSING | precision | 0.872 | 0.878 | 0.873 | 0.878 | 0.830 | 0.874 | 0.881 | 0.795 |
| VPN-BROWSING | recall | 0.884 | 0.916 | 0.918 | 0.906 | 0.846 | 0.893 | 0.902 | 0.888 |
| VPN-CHAT | precision | 0.661 | 0.647 | 0.568 | 0.532 | 0.511 | 0.490 | 0.427 | 0.576 |
| VPN-CHAT | recall | 0.687 | 0.628 | 0.562 | 0.473 | 0.651 | 0.667 | 0.658 | 0.562 |
| VPN-FT | precision | 0.792 | 0.744 | 0.676 | 0.649 | 0.773 | 0.764 | 0.714 | 0.674 |
| VPN-FT | recall | 0.770 | 0.690 | 0.610 | 0.605 | 0.735 | 0.689 | 0.648 | 0.595 |
| VPN-MAIL | precision | 0.808 | 0.891 | 0.778 | 0.864 | 0.766 | 0.912 | 0.753 | 0.859 |
| VPN-MAIL | recall | 0.762 | 0.887 | 0.715 | 0.852 | 0.766 | 0.947 | 0.734 | 0.904 |
| VPN-P2P | precision | 0.827 | 0.831 | 0.793 | 0.810 | 0.736 | 0.806 | 0.727 | 0.771 |
| VPN-P2P | recall | 0.863 | 0.870 | 0.848 | 0.862 | 0.517 | 0.585 | 0.527 | 0.567 |
| VPN-STREAMING | precision | 0.922 | 0.926 | 0.875 | 0.838 | 0.906 | 0.876 | 0.838 | 0.815 |
| VPN-STREAMING | recall | 0.891 | 0.876 | 0.782 | 0.752 | 0.914 | 0.856 | 0.787 | 0.759 |
| VPN-VOIP | precision | 0.990 | 0.992 | 0.939 | 0.973 | 0.990 | 0.993 | 0.911 | 0.984 |
| VPN-VOIP | recall | 0.979 | 0.975 | 0.914 | 0.925 | 0.987 | 0.981 | 0.907 | 0.951 |
| AVERAGE (macro) | precision | 0.839 (paper 0.840 T) | 0.844 | 0.786 | 0.792 | 0.788 | 0.816 | 0.750 | 0.782 |
| AVERAGE (macro) | recall | 0.834 | 0.835 | 0.764 | 0.768 | 0.774 | 0.803 | 0.738 | 0.747 |

A2-NonVPN:

| class | metric | C4.5 15 s | C4.5 30 s | C4.5 60 s | C4.5 120 s | KNN 15 s | KNN 30 s | KNN 60 s | KNN 120 s |
|---|---|---|---|---|---|---|---|---|---|
| BROWSING | precision | 0.904 | 0.915 | 0.897 | 0.928 | 0.866 | 0.883 | 0.865 | 0.922 |
| BROWSING | recall | 0.930 | 0.934 | 0.922 | 0.948 | 0.880 | 0.912 | 0.872 | 0.932 |
| CHAT | precision | 0.813 | 0.769 | 0.738 | 0.656 | 0.558 | 0.508 | 0.483 | 0.331 |
| CHAT | recall | 0.798 | 0.793 | 0.753 | 0.694 | 0.692 | 0.662 | 0.661 | 0.599 |
| FT | precision | 0.810 | 0.794 | 0.793 | 0.796 | 0.812 | 0.769 | 0.847 | 0.791 |
| FT | recall | 0.853 | 0.824 | 0.873 | 0.851 | 0.670 | 0.629 | 0.734 | 0.571 |
| MAIL | precision | 0.849 | 0.754 | 0.866 | 0.859 | 0.709 | 0.734 | 0.762 | 0.709 |
| MAIL | recall | 0.835 | 0.605 | 0.786 | 0.789 | 0.735 | 0.716 | 0.723 | 0.697 |
| P2P | precision | 0.971 | 0.943 | 0.948 | 0.946 | 0.907 | 0.897 | 0.893 | 0.914 |
| P2P | recall | 0.955 | 0.916 | 0.919 | 0.929 | 0.890 | 0.883 | 0.897 | 0.911 |
| STREAMING | precision | 0.837 | 0.769 | 0.757 | 0.768 | 0.782 | 0.759 | 0.655 | 0.657 |
| STREAMING | recall | 0.701 | 0.664 | 0.567 | 0.558 | 0.685 | 0.617 | 0.504 | 0.524 |
| VOIP | precision | 0.992 | 0.990 | 0.973 | 0.989 | 0.990 | 0.980 | 0.968 | 0.931 |
| VOIP | recall | 0.988 | 0.979 | 0.946 | 0.941 | 0.987 | 0.977 | 0.954 | 0.929 |
| AVERAGE (macro) | precision | 0.882 (paper 0.890 T) | 0.848 | 0.853 | 0.849 | 0.803 | 0.790 | 0.782 | 0.751 |
| AVERAGE (macro) | recall | 0.866 | 0.817 | 0.824 | 0.816 | 0.791 | 0.771 | 0.764 | 0.738 |

B:

| class | metric | C4.5 15 s | C4.5 30 s | C4.5 60 s | C4.5 120 s | KNN 15 s | KNN 30 s | KNN 60 s | KNN 120 s |
|---|---|---|---|---|---|---|---|---|---|
| BROWSING | precision | 0.771 | 0.790 | 0.779 | 0.800 | 0.690 | 0.748 | 0.746 | 0.807 |
| BROWSING | recall | 0.820 | 0.831 | 0.827 | 0.847 | 0.700 | 0.758 | 0.741 | 0.794 |
| CHAT | precision | 0.677 | 0.620 | 0.631 | 0.510 | 0.349 | 0.292 | 0.336 | 0.164 |
| CHAT | recall | 0.727 | 0.667 | 0.667 | 0.554 | 0.567 | 0.536 | 0.586 | 0.488 |
| FT | precision | 0.717 | 0.666 | 0.807 | 0.687 | 0.744 | 0.685 | 0.809 | 0.683 |
| FT | recall | 0.653 | 0.614 | 0.743 | 0.535 | 0.622 | 0.552 | 0.717 | 0.487 |
| MAIL | precision | 0.633 (paper 0.671 T) | 0.617 | 0.741 | 0.680 (paper 0.790 T) | 0.618 | 0.712 | 0.674 | 0.548 |
| MAIL | recall | 0.651 | 0.617 | 0.727 | 0.665 | 0.643 | 0.642 | 0.645 | 0.616 |
| P2P | precision | 0.880 | 0.851 | 0.842 | 0.874 | 0.749 | 0.743 | 0.760 | 0.818 |
| P2P | recall | 0.893 | 0.845 | 0.863 | 0.891 | 0.732 | 0.724 | 0.752 | 0.804 |
| STREAMING | precision | 0.713 | 0.657 | 0.496 | 0.510 | 0.679 | 0.630 | 0.536 | 0.477 |
| STREAMING | recall | 0.670 | 0.623 | 0.472 | 0.495 | 0.629 | 0.532 | 0.440 | 0.394 |
| VOIP | precision | 0.983 | 0.976 | 0.912 | 0.934 | 0.980 | 0.974 | 0.923 | 0.895 |
| VOIP | recall | 0.981 | 0.971 | 0.905 | 0.903 | 0.983 | 0.972 | 0.903 | 0.895 |
| VPN-BROWSING | precision | 0.773 (paper 0.771 T) | 0.800 | 0.797 | 0.807 (paper 0.809 T) | 0.689 (paper 0.691 T) | 0.738 | 0.760 | 0.742 (paper 0.743 T) |
| VPN-BROWSING | recall | 0.768 | 0.778 | 0.802 | 0.799 | 0.711 | 0.759 | 0.780 | 0.703 |
| VPN-CHAT | precision | 0.616 | 0.607 | 0.557 | 0.534 | 0.510 (paper 0.501 T) | 0.503 | 0.460 | 0.466 (paper 0.501 T) |
| VPN-CHAT | recall | 0.571 | 0.560 | 0.477 | 0.444 | 0.507 | 0.471 | 0.442 | 0.476 |
| VPN-FT | precision | 0.759 | 0.717 | 0.639 | 0.661 | 0.693 | 0.664 | 0.584 | 0.549 |
| VPN-FT | recall | 0.712 | 0.661 | 0.532 | 0.539 | 0.637 | 0.593 | 0.528 | 0.580 |
| VPN-MAIL | precision | 0.770 (paper 0.739 T) | 0.828 | 0.702 | 0.804 (paper 0.786 T) | 0.687 (paper 0.688 T) | 0.843 | 0.599 | 0.820 (paper 0.688 T) |
| VPN-MAIL | recall | 0.670 | 0.808 | 0.608 | 0.802 | 0.644 | 0.902 | 0.554 | 0.878 |
| VPN-P2P | precision | 0.615 | 0.620 | 0.547 | 0.629 | 0.591 | 0.666 | 0.593 | 0.669 |
| VPN-P2P | recall | 0.767 | 0.771 | 0.745 | 0.806 | 0.428 | 0.498 | 0.441 | 0.514 |
| VPN-STREAMING | precision | 0.885 | 0.871 | 0.863 | 0.875 | 0.836 | 0.774 | 0.774 | 0.713 |
| VPN-STREAMING | recall | 0.823 | 0.812 | 0.766 | 0.772 | 0.848 | 0.772 | 0.782 | 0.703 |
| VPN-VOIP | precision | 0.986 | 0.980 | 0.893 | 0.942 | 0.978 | 0.963 | 0.807 | 0.921 |
| VPN-VOIP | recall | 0.973 | 0.968 | 0.845 | 0.889 | 0.972 | 0.965 | 0.809 | 0.912 |
| AVERAGE (macro) | precision | 0.770 | 0.757 | 0.729 | 0.732 | 0.700 | 0.710 | 0.669 | 0.662 |
| AVERAGE (macro) | recall | 0.763 | 0.752 | 0.713 | 0.710 | 0.687 | 0.691 | 0.651 | 0.660 |

For A2 and B the paper's Figure 3 draws two unlabelled lines per timeout, which can't be assigned
to C4.5 or KNN. So only the values the paper states in its text are compared. The paper doesn't
say whether its "average" is a plain or a class-weighted mean, so both are shown. On the published
rows the plain (macro) mean matches it within 0.013 in every case, which suggests the paper used
the plain mean.

For averages the paper does not say whether it is a plain or a class-weighted mean
(Weka prints the weighted one), so both are shown.

| task | algorithm | class | timeout | metric | paper | rows | ours | ours_kind | diff |
|---|---|---|---|---|---|---|---|---|---|
| A2-VPN | C4.5 | AVERAGE | 15 | precision | 0.84 | published | 0.839 | macro | -0.001 |
| A2-VPN | C4.5 | AVERAGE | 15 | precision | 0.84 | published | 0.853 | weighted | 0.013 |
| A2-VPN | C4.5 | AVERAGE | 15 | precision | 0.84 | v1 | 0.827 | macro | -0.013 |
| A2-VPN | C4.5 | AVERAGE | 15 | precision | 0.84 | v1 | 0.846 | weighted | 0.006 |
| A2-NonVPN | C4.5 | AVERAGE | 15 | precision | 0.89 | published | 0.882 | macro | -0.008 |
| A2-NonVPN | C4.5 | AVERAGE | 15 | precision | 0.89 | published | 0.914 | weighted | 0.024 |
| A2-NonVPN | C4.5 | AVERAGE | 15 | precision | 0.89 | v1 | 0.886 | macro | -0.004 |
| A2-NonVPN | C4.5 | AVERAGE | 15 | precision | 0.89 | v1 | 0.92 | weighted | 0.03 |
| B | C4.5 | VPN-BROWSING | 15 | precision | 0.771 | published | 0.773 | class | 0.002 |
| B | C4.5 | VPN-BROWSING | 15 | precision | 0.771 | v1 | 0.76 | class | -0.011 |
| B | C4.5 | VPN-BROWSING | 120 | precision | 0.809 | published | 0.807 | class | -0.002 |
| B | C4.5 | VPN-BROWSING | 120 | precision | 0.809 | v1 | 0.76 | class | -0.049 |
| B | C4.5 | VPN-MAIL | 15 | precision | 0.739 | published | 0.77 | class | 0.031 |
| B | C4.5 | VPN-MAIL | 15 | precision | 0.739 | v1 | 0.718 | class | -0.021 |
| B | C4.5 | VPN-MAIL | 120 | precision | 0.786 | published | 0.804 | class | 0.018 |
| B | C4.5 | VPN-MAIL | 120 | precision | 0.786 | v1 | 0.682 | class | -0.104 |
| B | C4.5 | MAIL | 15 | precision | 0.671 | published | 0.633 | class | -0.038 |
| B | C4.5 | MAIL | 15 | precision | 0.671 | v1 | 0.625 | class | -0.046 |
| B | C4.5 | MAIL | 120 | precision | 0.79 | published | 0.68 | class | -0.11 |
| B | C4.5 | MAIL | 120 | precision | 0.79 | v1 | 0.613 | class | -0.177 |
| B | KNN | VPN-BROWSING | 15 | precision | 0.691 | published | 0.689 | class | -0.002 |
| B | KNN | VPN-BROWSING | 15 | precision | 0.691 | v1 | 0.695 | class | 0.004 |
| B | KNN | VPN-BROWSING | 120 | precision | 0.743 | published | 0.742 | class | -0.001 |
| B | KNN | VPN-BROWSING | 120 | precision | 0.743 | v1 | 0.7 | class | -0.043 |
| B | KNN | VPN-CHAT | 15 | precision | 0.501 | published | 0.51 | class | 0.009 |
| B | KNN | VPN-CHAT | 15 | precision | 0.501 | v1 | 0.509 | class | 0.008 |
| B | KNN | VPN-CHAT | 120 | precision | 0.501 | published | 0.466 | class | -0.035 |
| B | KNN | VPN-CHAT | 120 | precision | 0.501 | v1 | 0.494 | class | -0.007 |
| B | KNN | VPN-MAIL | 15 | precision | 0.688 | published | 0.687 | class | -0.001 |
| B | KNN | VPN-MAIL | 15 | precision | 0.688 | v1 | 0.633 | class | -0.055 |
| B | KNN | VPN-MAIL | 120 | precision | 0.688 | published | 0.82 | class | 0.132 |
| B | KNN | VPN-MAIL | 120 | precision | 0.688 | v1 | 0.521 | class | -0.167 |
| B | C4.5 | AVERAGE_BEST_TIMEOUT | best | precision | 0.783 | published | 0.77 | macro | -0.013 |
| B | C4.5 | AVERAGE_BEST_TIMEOUT | best | precision | 0.783 | published | 0.808 | weighted | 0.025 |
| B | C4.5 | AVERAGE_BEST_TIMEOUT | best | precision | 0.783 | v1 | 0.774 | macro | -0.009 |
| B | C4.5 | AVERAGE_BEST_TIMEOUT | best | precision | 0.783 | v1 | 0.815 | weighted | 0.032 |
| B | KNN | AVERAGE_BEST_TIMEOUT | best | precision | 0.711 | published | 0.71 | macro | -0.001 |
| B | KNN | AVERAGE_BEST_TIMEOUT | best | precision | 0.711 | published | 0.745 | weighted | 0.034 |
| B | KNN | AVERAGE_BEST_TIMEOUT | best | precision | 0.711 | v1 | 0.707 | macro | -0.004 |
| B | KNN | AVERAGE_BEST_TIMEOUT | best | precision | 0.711 | v1 | 0.755 | weighted | 0.044 |

**Duplicates.** Removing the exact duplicates changed the 10-fold macro F1 by -0.018 to +0.014 for
the tree (mean +0.001) and by -0.017 to +0.040 for 1-NN (mean +0.012):

| algo | task | timeout | rows_published | rows_v1 | f1_published | f1_v1 | v1_minus_published |
|---|---|---|---|---|---|---|---|
| j48 | A1 | 15 | 18,758 | 18,038 | 0.8888 | 0.8954 | 0.0067 |
| j48 | A1 | 30 | 14,651 | 13,610 | 0.8696 | 0.8766 | 0.0069 |
| j48 | A1 | 60 | 15,515 | 14,677 | 0.8562 | 0.8702 | 0.014 |
| j48 | A1 | 120 | 10,782 | 9,524 | 0.8688 | 0.8694 | 0.0007 |
| j48 | A2-VPN | 15 | 9,793 | 9,380 | 0.836 | 0.8197 | -0.0162 |
| j48 | A2-VPN | 30 | 7,734 | 6,944 | 0.8389 | 0.821 | -0.0179 |
| j48 | A2-VPN | 60 | 6,935 | 6,523 | 0.7741 | 0.7702 | -0.0039 |
| j48 | A2-VPN | 120 | 5,631 | 4,648 | 0.779 | 0.7719 | -0.0071 |
| j48 | A2-NonVPN | 15 | 8,965 | 8,694 | 0.8731 | 0.8768 | 0.0037 |
| j48 | A2-NonVPN | 30 | 6,917 | 6,694 | 0.8303 | 0.8359 | 0.0055 |
| j48 | A2-NonVPN | 60 | 8,580 | 8,206 | 0.8359 | 0.8483 | 0.0123 |
| j48 | A2-NonVPN | 120 | 5,151 | 4,941 | 0.8294 | 0.8346 | 0.0052 |
| j48 | B | 15 | 18,758 | 18,074 | 0.765 | 0.7707 | 0.0057 |
| j48 | B | 30 | 14,651 | 13,638 | 0.7535 | 0.7488 | -0.0047 |
| j48 | B | 60 | 15,515 | 14,729 | 0.7185 | 0.7293 | 0.0107 |
| j48 | B | 120 | 10,782 | 9,589 | 0.7183 | 0.718 | -0.0003 |
| ibk | A1 | 15 | 18,758 | 18,038 | 0.8424 | 0.8524 | 0.01 |
| ibk | A1 | 30 | 14,651 | 13,610 | 0.8276 | 0.8331 | 0.0055 |
| ibk | A1 | 60 | 15,515 | 14,677 | 0.8217 | 0.8326 | 0.0109 |
| ibk | A1 | 120 | 10,782 | 9,524 | 0.8146 | 0.8218 | 0.0072 |
| ibk | A2-VPN | 15 | 9,793 | 9,380 | 0.7766 | 0.793 | 0.0164 |
| ibk | A2-VPN | 30 | 7,734 | 6,944 | 0.8047 | 0.81 | 0.0054 |
| ibk | A2-VPN | 60 | 6,935 | 6,523 | 0.7377 | 0.7552 | 0.0175 |
| ibk | A2-VPN | 120 | 5,631 | 4,648 | 0.7611 | 0.7441 | -0.017 |
| ibk | A2-NonVPN | 15 | 8,965 | 8,694 | 0.7949 | 0.8118 | 0.0169 |
| ibk | A2-NonVPN | 30 | 6,917 | 6,694 | 0.7769 | 0.7989 | 0.022 |
| ibk | A2-NonVPN | 60 | 8,580 | 8,206 | 0.7687 | 0.7899 | 0.0213 |
| ibk | A2-NonVPN | 120 | 5,151 | 4,941 | 0.7349 | 0.7751 | 0.0402 |
| ibk | B | 15 | 18,758 | 18,074 | 0.69 | 0.7071 | 0.0171 |
| ibk | B | 30 | 14,651 | 13,638 | 0.6958 | 0.6931 | -0.0027 |
| ibk | B | 60 | 15,515 | 14,729 | 0.6561 | 0.6697 | 0.0137 |
| ibk | B | 120 | 10,782 | 9,589 | 0.6531 | 0.661 | 0.0079 |

![Duplicate effect](../../figures/benchmark_dedup_effect.png)

The reason the hypothesis failed is in the rows themselves. The removed duplicates are not copies
of ordinary flows. Between 39% and 92% of them have an identical twin with a different label, against
2-10% of all rows, and their median flow has 0-2 packets (all flows: 8-43). They are near-empty
flows that timing features can't tell apart. Keeping every copy multiplies errors that no model can
avoid, which outweighs the advantage of having a test flow's copy in the training data:

Share of removed rows whose features also occur with another label, and packets per
flow. A1 120 s packet counts use its published (corrupt) flowPktsPerSecond.

| task | timeout | duplicates removed | removed with other-label twin | all rows with other-label twin | median packets, removed | median packets, all |
|---|---|---|---|---|---|---|
| A1 | 15 | 720 | 0.761 | 0.032 | 0.0 | 32.0 |
| A1 | 30 | 1,041 | 0.51 | 0.039 | 2.0 | 22.0 |
| A1 | 60 | 838 | 0.766 | 0.047 | 0.0 | 16.0 |
| A1 | 120 | 1,257 | 0.434 | 0.056 | 0.0 | 0.0 |
| A2-VPN | 15 | 413 | 0.823 | 0.04 | 0.0 | 27.0 |
| A2-VPN | 30 | 790 | 0.395 | 0.045 | 2.0 | 17.0 |
| A2-VPN | 60 | 412 | 0.748 | 0.053 | 0.0 | 8.0 |
| A2-VPN | 120 | 983 | 0.45 | 0.098 | 2.0 | 8.0 |
| A2-NonVPN | 15 | 271 | 0.742 | 0.024 | 0.0 | 43.0 |
| A2-NonVPN | 30 | 223 | 0.897 | 0.03 | 0.0 | 26.0 |
| A2-NonVPN | 60 | 374 | 0.805 | 0.039 | 0.0 | 23.0 |
| A2-NonVPN | 120 | 210 | 0.919 | 0.039 | 0.0 | 17.0 |
| B | 15 | 684 | 0.804 | 0.035 | 0.0 | 32.0 |
| B | 30 | 1,013 | 0.529 | 0.042 | 2.0 | 22.0 |
| B | 60 | 786 | 0.818 | 0.053 | 0.0 | 16.0 |
| B | 120 | 1,193 | 0.562 | 0.078 | 2.0 | 12.0 |

The duplicate rule of data v1 stays: in our own split it guarantees that no flow is scored after
being trained on, and this benchmark shows it changes 10-fold scores by only about +-0.02.

## 4.4 Storage after the baselines

Every official run saves its trained model (the plan's rule). After the 147 runs so far (80 baselines, 64 benchmark, 2 pilot, 1 setup check), the MLflow
store holds 137.8 MB of artifacts and a 5.8 MB database; the largest single run is 4.3 MB (SVM).
Per model type:

| kind | runs | total_MB | per_run_max_MB | model_file_max_MB |
|---|---|---|---|---|
| knn (holdout) | 16 | 23.075 | 2.475 | 2.319 |
| knn_minmax (paper_10fold) | 32 | 66.742 | 3.596 | 3.438 |
| logreg (holdout) | 16 | 1.64 | 0.167 | 0.005 |
| majority (holdout) | 16 | 1.395 | 0.118 | 0.001 |
| svm_rbf (holdout) | 16 | 27.418 | 4.345 | 4.188 |
| tree (holdout) | 16 | 5.457 | 0.764 | 0.579 |
| tree (paper_10fold) | 32 | 11.674 | 0.817 | 0.629 |

Whole store now: artifacts 137.8 MB, mlflow.db 5.8 MB.

Tree ensembles in Phase 3 will be much larger: a 300-tree Random Forest, measured on the largest
training set (B, 15 s), is a 171.9 MB file (22.4 MB with joblib compression level 3) [E03].
