# 6 Discussion

*Working draft. These markdown drafts are the master copy of the report; the full report (DOCX and
PDF) is assembled from them at each milestone. Every number here is taken from a generated file
(`reports/phase3_tables.md`, `reports/phase4_tables.md`, `reports/baseline_tables.md`) and repeats a
value already given in Sections 4 and 5, with its experiment ID. Data version: v1.*

## 6.1 What worked

**Reproducing the paper first.** Before trying to improve anything, the paper's own protocol was
rerun (10-fold cross-validation on every published row, Weka defaults). Our C4.5 approximation
matched all 16 A1 values of the paper's Figure 2 within 0.05 and 1-NN matched 15 of 16 (E02, E03,
Section 4.3). Every later number therefore starts from a baseline that agrees with the published
one, and the changes we made (deduplication, a held-out test set, other models) can be judged
against it.

**Trees combined into ensembles, then boosted.** On validation at 15 s, the Phase 2 decision tree
(E02) scored 0.8901 (A1), 0.8247 (A2-VPN), 0.8294 (A2-NonVPN) and 0.7585 (B). The final models
score 0.9308, 0.8830, 0.9075 and 0.8428 on the same rows, so most of the gain over the paper's
algorithms came from three steps:

- a 300-tree Random Forest (+0.0302 to +0.0647 over the tree, E05);
- default gradient boosting (XGBoost +0.0067 to +0.0156 over the forest, E06);
- the same 50-trial tuning budget for every method type (E07, E17): tuning added at most +0.0151
  to an ensemble, but it lifted the SVM by +0.3243 to +0.5277 and turned it from one of the weakest
  models into one 0.0017-0.0186 below each task's best.

The progress chart (Section 5.1) shows each step per task. The order of the seven method types was
stable: by test score it agreed with the order by validation score (Spearman 0.8214-0.9643, E12
(c)), so model selection on validation carried over to new rows.

**The signed log for distance-based models.** One preprocessing step, sign(x)·log1p(|x|), lifted
KNN by 0.080-0.104 in every task (E04) and was chosen by every logistic-regression and SVM search
when it was offered (8 of 8, E17). Raw timing values span several orders of magnitude, so without
it a few very large gaps decide every distance.

**Choosing by the best validation score.** The rule fixed before Phase 3 would have picked a simpler
model whenever the gap was below 0.03. The measured spread of cross-validation was much smaller
(standard deviation 0.0028-0.0058 for the chosen models, Section 3.8), so we chose by the highest validation score
instead and kept the old rule's picks for comparison (Section 3.8). On the test rows the chosen
model scored higher in all four tasks, clearly so in A1 (+0.0322) and A2-NonVPN (+0.0242) (E12
(b), Section 5.3).

**Reproducing and correcting a recent result.** Isnan et al.'s 88.01% was reproduced to within
0.0005 (0.8796 on the same split). Keeping the timeout files apart lowered the same method to a
mean of 0.8430, and a size control showed that this drop comes from training-set size, not from
mixing the timeouts: at equal size the mixed setup scored 0.8395 (E16 (a), (b), (d)). Without the
control we would have blamed the mixing, which the first version of our E16 text did before the
recheck (Section 4.18). Our tuned task-B XGBoost beat their forest at every timeout (mean 0.8647,
E16 (c)).

**Calibration where it was needed.** All four headline models were overconfident on validation.
Isotonic calibration, adopted by a rule fixed in advance for A1 and A2-VPN, lowered the test ECE
from 0.0442 to 0.0090 and from 0.0709 to 0.0169 without changing a single prediction (E15, E12,
Section 5.4). The confidence that `predict.py` and the demo show is therefore close to the share
of correct predictions at that confidence, for those two tasks.

## 6.2 What did not work

- **Class balancing** (E08) moved macro F1 by at most +-0.0064: a little up in A2-VPN (class
  weights +0.0062) and A2-NonVPN (oversampling +0.0047), down in A1 and B. The pre-stated gain of
  0.01 was not reached (E08 (a) not supported). The weak classes (VPN-CHAT, VPN-MAIL, STREAMING)
  seem to be weak not because they are small but because their flows look like those of other
  classes; E08 does not test this directly.
- **Two steps for task B** (E10): routing each flow through a VPN / non-VPN model first scored
  0.0110 below the direct 14-class model and made more VPN / non-VPN errors (191 against 180). The
  direct model makes 55.05% of its errors across the VPN line as well, so separating VPN from
  non-VPN is the hard part of B however the model is built.
- **Tuning the ensembles** gave little: the validation score rose in 9 of 12 E07 searches although
  cross-validation rose in all 12, because the best of 50 cross-validation scores is optimistic.
  Only B improved by more than 0.01 (E07 (b) not supported).
- **Removing duplicates did not lower the paper-protocol scores**, contrary to the hypothesis
  (E02 (d), E03 (e)). The removed rows are mostly near-empty flows of 0-2 packets whose features
  also occur with another label, so keeping them adds errors no model can avoid. The duplicate
  rule stays, because it guarantees that no flow is scored after being trained on.
- **Validation was not always within 0.03 of test** (E12 (a) not supported): A2-NonVPN scored
  0.0328 higher on test. E14 had shown before the test set was opened that its official validation
  rows are harder than any of 10 resplits (0.9075 against 0.9257-0.9468), mostly for MAIL and P2P.

## 6.3 What the models learned: SHAP and the ablation

SHAP (on the test rows, no score computed) and the feature-group ablation (E09, on validation)
answer two different questions: which features a model uses, and which it cannot do without.
They agree on most points (Section 5.6):

- **The byte rate matters.** flowBytesPerSecond is the most used feature in A2-VPN, A2-NonVPN and
  B. Rates and duration together hold 21.4-26.9% of mean |SHAP|, and dropping them costs the most
  validation macro F1 in A2-VPN (0.0395), A2-NonVPN (0.0301) and B (0.0549). A timing-only model
  without the byte rate loses 0.015-0.044 (E09 (c) not supported), so the "timing only" claim needs
  the byte rate, which depends on packet sizes, not only on time.
- **In A1 the backward gaps matter more.** Dropping the backward inter-arrival times costs 0.0155,
  more than rates and duration (0.0122): whether a flow is tunnelled shows in how the replies are
  spaced.
- **Active and idle times matter least.** They hold 3.1-5.7% of mean |SHAP|, and dropping either
  group moves validation macro F1 by -0.0036 to +0.0051. A model on either group alone loses
  0.38-0.63.
- **They differ on the flow gaps.** The flow inter-arrival times hold the largest SHAP share in A1,
  A2-VPN and A2-NonVPN (25.5-29.6%), yet dropping them costs at most 0.0166, because the forward
  and backward gaps carry much the same timing. With correlated features a model can swap one
  group for another; SHAP shows the one it happened to use.

The worked examples show the limit of a confidence score: the most confident errors are flows
whose byte rate is typical of another class (a VPN-P2P flow at 365.1 bytes per second predicted as
VPN-FT with confidence 0.9986, Section 5.6). Neither SHAP nor the ablation explains what causes a
traffic pattern; both describe these models on these files.

## 6.4 Reproduced, improved, and what limits both

- **Reproduced:** the paper's A1 results under its own protocol (C4.5 16 of 16, KNN 15 of 16 within
  0.05; E02, E03) and Isnan et al.'s 88.01% on their split (0.8796; E16 (a)).
- **Improved:** on the leakage-safe protocol, test macro F1 of A1 0.9434, A2-VPN 0.8652, A2-NonVPN
  0.9403 and B 0.8480, each with a 95% interval (E12). The paper's two algorithms, even tuned
  (decision tree, KNN), score below the headline model in every task, clearly so after Holm in A1,
  A2-NonVPN and B (Section 5.3).
- **Limits:** the data are from a 2016 lab capture with known quality problems, flows from one
  capture may sit on both sides of every random split, and several of the comparisons above are
  close to the noise. Section 7 lists each limitation with its evidence.
