# 7 Limitations

*Working draft. These markdown drafts are the master copy of the report; the full report (DOCX and
PDF) is assembled from them at each milestone. Each limitation names the experiment or file that
measured it; where nothing was measured, it says so.*

## 7.1 The data

- **2016 lab data.** ISCXVPN2016 was captured in 2016, in a lab, with a fixed set of applications
  and one VPN setup [1, 2]. Applications, VPN protocols and networks have changed since, so the
  scores in Section 5 describe traffic like these captures, not current traffic in general.
- **Documented capture problems.** Jorgensen et al. [14] found readable HTML in a capture that is
  supposed to be encrypted and VPN captures that hold several connections; they also cite a
  finding that 65% of the biflows come from the BlueStacks Android emulator, and they point out
  ambiguous labels such as skype_video, which could be VoIP or browsing. We use the published
  feature files and cannot correct any of this; a label error in the source is an error in our
  training and test rows alike.
- **The 60 s files hold different VOIP and FT flows.** At 60 s, non-VPN and VPN VoIP flows almost
  swap their packet rates and gaps, and non-VPN FT drops from 1,237-3,739 to 56.5 packets per
  second (E00 addendum, Section 2.8). It is not a label swap, and the cause cannot be found from the
  ARFF files. Results at 60 s are read with care.
- **Column names had to be inferred.** 9 of the 12 A1 and B files carry a header whose fiat / biat
  names are shifted by one pair, and the A2 files hold different features (`total_fiat`,
  `total_biat`) from A1 and B (`std_fiat`, `std_biat`). The mapping was decided from value checks
  that hold on 100% of rows (E00, Sections 2.3-2.4), not from documentation of the export.
- **Partial packet captures.** Only the VPN packet captures were downloaded in full, so the flows
  could not be rebuilt from packets and no new features were computed; everything uses the
  published 23 features.

## 7.2 The protocol

- **No session-aware split.** The files have no flow IDs, addresses, timestamps or capture names,
  so flows of one recording session cannot be kept on one side of a split; E11 could not run
  (E00). The direction of the effect was measured once, as a side result of the E15 correction:
  holding out contiguous blocks of the file-ordered training rows, instead of shuffled rows, lowers
  the out-of-fold accuracy of the four headline configurations by 0.07-0.26
  (`reports/fold_order_check.json`; A1 0.9321 to 0.6701, A2-VPN 0.9048 to 0.7897, A2-NonVPN 0.9566
  to 0.8841, B 0.8738 to 0.7486). Neighbouring flows in a file, probably from the same capture,
  resemble each other more than random flows do, so scores on a random split may be optimistic
  for traffic from captures the model has never seen. This is an inference from the file order,
  not a session-aware test, and the size of the optimism is not known.
- **Timeouts are confounded.** The class mix changes with the timeout (one class's share by up to
  24.6 percentage points in A2-NonVPN), and the 60 s files differ in content (7.1). Score
  differences between timeouts are therefore not read as a timeout effect (Sections 2.8, 5.8).
- **C4.5 approximation.** scikit-learn has no C4.5 pruning, so the paper's J48 is approximated by
  an unpruned entropy tree with at least 2 rows per leaf [11, 16]. The A1 reproduction is close
  (16 of 16 values within 0.05, Section 4.3), but the two algorithms are not identical.
- **The paper's A2 and B charts cannot be read per algorithm.** Its Figure 3 draws unlabelled
  lines, so for A2 and B only values stated in its text were compared (Section 4.3).
- **One deviation from a pre-set rule.** The final model per task was chosen by the highest
  validation score, not by the tie rule fixed before Phase 3 (Section 3.8). The change was made
  after E10 and before the test set was opened, and the old rule's picks were scored too, so its
  cost is shown (Section 5.3), but it is still a change of plan after some validation results were
  known.

## 7.3 The results

- **Validation did not predict every test score within 0.03.** A2-NonVPN scored 0.0328 above its
  validation score (E12 (a) not supported). E14 shows its official validation rows are harder than
  any of 10 resplits (0.9075 against 0.9257-0.9468), so its validation score understates it.
- **Test scores outside the resplit range.** The A1 test score (0.9434) is above all 10 E14
  resplits (at most 0.9397), and the A2-VPN test score (0.8652) is below them (at least 0.8689).
  The resplit models are trained on less data than the final models and a single 15% test set can
  land outside a 10-resplit range by chance, but one test set is one sample.
- **Intervals are conditional.** Every interval comes from stratified resampling of the test rows,
  so it keeps the test set's class counts fixed; it does not include the variation from drawing a
  different test set, or from training on a different split (E14 measures that separately).
- **Two of the 26 tests are borderline.** A1 against the SVM and A2-VPN against the decision tree
  have Holm p-values within the Monte Carlo error of 0.05 (bands of about 0.03-0.07 from the 10,000
  resamples, Section 5.3). Under the rule fixed in advance both count as not clear; another draw
  of resamples could change them. No verdict depends on them.
- **Close comparisons.** The headline model is not clearly different from the SVM in any task, and
  the A2-VPN SVM (0.8670) and the B LightGBM (0.8567) scored slightly above the headline models on
  test. Nothing was swapped after the test set was seen.
- **Logistic regression did not converge in one search.** The tuned A2-NonVPN logistic regression
  (C = 985) stops at the fixed limit of 1,000 iterations (E17). Its score may be slightly below what
  a converged model would reach; it was not rerun after the result was seen.
- **Calibration is checked on one test set.** The test ECE (Section 5.4) is a single measurement
  with 15 bins and no interval.
- **"Timing only" needs the byte rate.** Without flowBytesPerSecond, which depends on packet
  sizes, validation macro F1 drops by 0.015-0.044 (E09 (c)). The models read no payload, but they
  do use the amount of data per second.

## 7.4 Related work

- **Bagui et al. 2017 [12]** is cited from its public abstract only (about 90%); the full paper was
  not available to us, so no detail beyond the abstract is claimed or compared.
- **Isnan et al. 2026 [13].** Their split stacks all four timeout files, so it mixes timeouts and is
  not comparable with a single-timeout score. E16 shows the measured gap to a per-timeout score
  comes from training-set size (Section 4.18); we do not claim that their score is inflated by the
  mixing.
