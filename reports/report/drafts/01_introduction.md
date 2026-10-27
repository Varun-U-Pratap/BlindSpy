# 1 Introduction

*Working draft. The Google Docs report is the master copy; this text is pasted there and kept in
sync. Placeholders in [brackets] are filled when the named experiment is logged.*

## 1.1 Problem

Most network traffic is now encrypted. Deep packet inspection, which classifies traffic by reading
packet contents, no longer works when the contents are hidden. Network operators still need to know
what kind of traffic a connection carries (for quality of service, access control and security
monitoring) and whether it is routed through a VPN.

Applications leave a timing pattern even when their contents are encrypted. A voice call sends
small packets at a steady rhythm, video streaming arrives in bursts, and a file transfer runs at
full speed with almost no idle time. BlindSpy trains classifiers on these patterns only: 23 flow
timing statistics per connection, never the payload.

We use the ISCXVPN2016 dataset [2] published with Draper-Gil et al. [1]. Each row is one flow
described by 23 time-related features and a label. The dataset is provided in four versions,
built with flow timeouts of 15, 30, 60 and 120 seconds. The timeout is a setting of the flow
builder (it decides when an inactive flow is closed); it is not the duration of each flow.

## 1.2 Tasks

We follow the three tasks of the original paper [1], each scored on its own:

| Task | Description | Scored by |
| --- | --- | --- |
| A1 | One binary classifier: VPN vs non-VPN | Precision and recall per class, macro F1, ROC-AUC |
| A2 | Two separate 7-class classifiers, one trained on VPN flows and one on non-VPN flows | Each model scored separately; never merged into one score |
| B | One 14-class classifier over VPN and non-VPN traffic together | Macro F1 and per-class scores over all 14 labels |

The seven traffic types are browsing, email, chat, streaming, file transfer, VoIP and P2P; task B
uses each type twice (VPN and non-VPN).

## 1.3 What BlindSpy can and cannot claim

- It predicts the traffic type from aggregate timing statistics and never reads packet contents.
- It does not prove that traffic is encrypted.
- The data was captured in 2016, in a lab, with specific applications and one VPN setup. Results
  may not hold for current applications, VPN providers or networks. The conclusion repeats this.

## 1.4 Related work

| Study | Data and setup | Models | Reported result |
| --- | --- | --- | --- |
| Draper-Gil et al. 2016 [1] | ISCXVPN2016 ARFF files, tasks A1, A2, B, each timeout separately; Weka, 10-fold cross-validation | C4.5, KNN | A1 at 15 s, C4.5: precision 0.89 (VPN) and 0.906 (non-VPN); KNN: 0.848 and 0.846. A2 at 15 s, C4.5: average precision 0.84 (VPN model), 0.89 (non-VPN model). B, best timeout: average precision 0.783 (C4.5), 0.711 (KNN) |
| Bagui et al. 2017 [12] | Same time-related features | Random Forest, gradient boosting and others | About 90% (cited from the public abstract only; the full paper was not available to us, so no further detail is claimed) |
| Jorgensen et al. 2023 [14] | ISCXVPN2016 captures | Uncertainty-aware traffic labelling | Documents quality problems in the dataset's captures (see Section 7) |
| Isnan et al. 2026 [13] | Kaggle copy of the dataset, 59,706 rows; 18,719 duplicates removed, leaving 40,987; 14 classes; 80/20 split | Random Forest, 300 trees | 88.01% accuracy |

Paper values are taken from `reports/paper_baseline.csv`, which records each number's page and
whether it is stated in the text or read from a figure.

**Why published scores (80–99%) are not directly comparable.** Studies on this dataset differ in
the split (k-fold vs a single hold-out), the class setup (binary, 7-class or 14-class), whether
duplicates are removed, and whether flows from different timeout versions are mixed. Each of these
can move accuracy by several points, so we compare only under matched protocols.

**The Isnan et al. data.** The Kaggle file used by Isnan et al. is the eight A2 files (VPN and
non-VPN at all four timeouts) stacked together: compared as a multiset, its 59,706 rows equal the
rows of those files (5,151 + 5,631 + 8,965 + 9,793 + 6,917 + 7,734 + 8,580 + 6,935), with no row
left over and the same labels. It has 40,987 unique rows, exactly their count. 7,093 distinct
flows appear in two to four timeout files, while only 3,676 rows are duplicates within one file
(E00). So most of their 18,719 "duplicates" are the same flows repeated across timeout versions,
and a random 80/20 split of the stacked file lets the same flow (built with a different timeout)
appear in both training and test data. The file also uses the A2 feature set (`total_fiat` and
`total_biat` in place of the paper's `std_fiat` and `std_biat`; Section 2). [Reproduction and
corrected protocol: E16.]

## 1.5 Contributions

1. A reproduction of the paper's C4.5 and KNN results under its own 10-fold protocol, compared
   precision with precision and recall with recall. [E02, E03, paper benchmark]
2. A leakage-safe protocol (stratified 70/15/15 split, cross-validation inside the training set
   only, test set scored once) applied to all three tasks and all four timeouts.
3. Stronger models (Random Forest, XGBoost, LightGBM), tuned with the same search budget as the
   best baseline. [E05–E08]
4. A check of the dataset itself: whether the ARFF column names match the data, duplicates, odd
   values and the class mix across timeouts, recorded as versioned data with a step-by-step
   cleaning log. [E00]
5. Analyses of what the models learned and how far the results can be trusted: feature-group
   ablation, SHAP, learning curve, split-seed stability and calibration. [E09, E13–E15]
6. A full history: every run, data version and experiment, traceable to a commit.
