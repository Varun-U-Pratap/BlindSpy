# 8 Conclusion

*Working draft. These markdown drafts are the master copy of the report; the full report (DOCX and
PDF) is assembled from them at each milestone.*

BlindSpy classifies ISCXVPN2016 flows from 23 timing statistics, without reading any payload, in
the three tasks of the original paper: VPN or not (A1), the traffic type within VPN and within
non-VPN traffic (A2, two separate models), and all 14 classes at once (B).

We first checked the data and the paper. The ARFF headers do not always describe their columns:
most A1 and B files hold the paper's standard deviations under shifted names, and the A2 files hold
different features, which we mapped from value checks before building any model (E00). Under the
paper's own protocol our C4.5 approximation and 1-NN reproduce its A1 results (16 of 16 and 15 of
16 values within 0.05; E02, E03). A recent result on a stacked copy of the dataset (Isnan et al.,
88.01%) was reproduced as well (0.8796), and its gap to a per-timeout score was traced to
training-set size, not to mixing timeouts (E16).

On a leakage-safe protocol (deduplicated data, a stratified 70/15/15 split, every choice made on
training and validation rows, the test set scored once), gradient-boosted trees were the best
models on validation and scored test macro F1 0.9434 for A1, 0.8652 for the VPN model and 0.9403 for the non-VPN model of
A2, and 0.8480 for B, each with a 95% bootstrap interval (E12). Every method type had the same
tuning budget (E07, E17), and the order of the seven types by validation score largely held on
the test rows (Spearman 0.82-0.96). Class balancing and a two-step model for task B did not help
(E08, E10). The byte rate and the shortest gaps between packets are among the features the models
use most (SHAP), dropping the rates and duration costs the most in three of the four tasks (E09),
and both methods find that active and idle times add little (Section 5.6). Confidence is
calibrated where the validation data showed it was needed (E15), and `scripts/predict.py` and the
Streamlit demo [23] use the four final models to classify new flow files and explain each
prediction.

These results hold for this dataset only. The data were captured in a lab in 2016, with specific
applications and one VPN setup, and the captures have documented quality problems [14]. Flows from
one capture can sit on both sides of a random split, and holding out contiguous blocks of the
files lowers out-of-fold accuracy by 0.07-0.26 (Section 7.2), so performance on traffic from new
captures, new applications or today's VPN services may well be lower than the scores reported here;
it has not been measured. A session-aware split on data with capture identifiers, and a test on traffic captured
independently of ISCXVPN2016, are the next steps before any of these models is used on a real
network.

All runs, data versions and decisions are kept: 2,013 MLflow runs exported to
`reports/history.csv`, one log entry per experiment (`reports/EXPERIMENT_LOG.md`), a changelog per
data version (`reports/DATA_CHANGELOG.md`) and a git tag at each milestone (Appendix).
