# 2 Dataset

*Working draft. The Google Docs report is the master copy; this text is pasted there and kept in
sync. Every number and table here comes from files written by code for data version v1
(`data/manifest.json`, `data/processed/v1/cleaning_log.csv`, `reports/DATA_CHANGELOG.md`); the
tables are printed by `scripts/data_tables.py v1`. Evidence: experiment E00.*

## 2.1 Source and format

We use ISCXVPN2016 [2], published with Draper-Gil et al. [1] by the Canadian Institute for
Cybersecurity, University of New Brunswick. The traffic was captured in 2016 in a lab, with and
without a VPN, for seven traffic types: browsing, chat, streaming, mail, VoIP, P2P and file
transfer. The authors turned the captures into flows with ISCXFlowMeter and published the flows as
20 ARFF files (Weka's text format). Each row is one flow: 23 numeric time-related features (times
in microseconds, rates per second) and a label. We use only these files. They hold no payload,
addresses, ports or timestamps. The SHA-256 hash of every file is recorded in
`data/arff_sha256.txt`.

Each scenario is published at four flow timeouts (15, 30, 60 and 120 s). The timeout is a setting
of the flow builder, not a flow duration. Flows longer than the timeout exist (up to 646 s) because
the timeout closes a flow only after it has been inactive for that long.

## 2.2 Tasks and files

| Task | Files | Labels |
|---|---|---|
| A1 | `Scenario A1-ARFF`, 4 files | VPN, Non-VPN |
| A2-VPN | `Scenario A2-ARFF`, 4 `-VPN` files | 7 VPN traffic types |
| A2-NonVPN | `Scenario A2-ARFF`, 4 `-NO-VPN` files | 7 traffic types |
| B | `Scenario B-ARFF`, 4 files | 14 classes (7 types x VPN / non-VPN) |
| (B-AllinOne) | `Scenario B-ARFF`, 4 `-AllinOne` files | 7 types, VPN and non-VPN merged; inventoried, not a task |

| file | task | timeout | header | rows | classes | core_task |
|---|---|---|---|---|---|---|
| Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff | A1 | 15 | T | 18,758 | 2 | 1 |
| Scenario A1-ARFF/TimeBasedFeatures-Dataset-30s-VPN.arff | A1 | 30 | T | 14,651 | 2 | 1 |
| Scenario A1-ARFF/TimeBasedFeatures-Dataset-60s-VPN.arff | A1 | 60 | S | 15,515 | 2 | 1 |
| Scenario A1-ARFF/TimeBasedFeatures-Dataset-120s-VPN.arff | A1 | 120 | T | 10,782 | 2 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff | A2-VPN | 15 | T | 9,793 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-30s-VPN.arff | A2-VPN | 30 | T | 7,734 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-60s-VPN.arff | A2-VPN | 60 | T | 6,935 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-120s-VPN.arff | A2-VPN | 120 | T | 5,631 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff | A2-NonVPN | 15 | T | 8,965 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-30s-NO-VPN.arff | A2-NonVPN | 30 | T | 6,917 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-60s-NO-VPN.arff | A2-NonVPN | 60 | T | 8,580 | 7 | 1 |
| Scenario A2-ARFF/TimeBasedFeatures-Dataset-120s-NO-VPN.arff | A2-NonVPN | 120 | T | 5,151 | 7 | 1 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-15s.arff | B | 15 | T | 18,758 | 14 | 1 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-30s.arff | B | 30 | T | 14,651 | 14 | 1 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-60s.arff | B | 60 | S | 15,515 | 14 | 1 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-120s.arff | B | 120 | T | 10,782 | 14 | 1 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-15s-AllinOne.arff | B-AllinOne | 15 | T | 18,758 | 7 | 0 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-30s-AllinOne.arff | B-AllinOne | 30 | T | 14,651 | 7 | 0 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-60s-AllinOne.arff | B-AllinOne | 60 | S | 15,515 | 7 | 0 |
| Scenario B-ARFF/TimeBasedFeatures-Dataset-120s-AllinOne.arff | B-AllinOne | 120 | T | 10,782 | 7 | 0 |

Two features of the raw files matter for everything below. A1, A2 and B hold **the same flows**
at each timeout: every A1 row matches an A2 row on seven integer features at all four timeouts
(100%), and the labels agree key by key (0 disagreements). B and A1 are identical cell for cell
except for one corrupt column (Section 2.4). BROWSING and VPN-BROWSING hold exactly 2,500 flows in
every raw A2 file, which suggests the dataset authors capped those classes.

## 2.3 Column-name check

**Problem.** The first row of A1 15 s has `duration` = 9,368,711 us (9.4 s) but `total_fiat` = 16
and `total_biat` = 4. A total forward gap of 16 us in a 9-second flow is impossible. The paper's
Table 2 [1] and the authors' 2016 poster [15] describe the forward and backward inter-arrival
features as min, mean, max and std, with no totals. Yet 17 of the 20 headers list `total_fiat` and
`total_biat` and no std. The three 60 s files of A1 and B have a second header that matches the
paper.

**Method.** We read every file under both name lists. On every row we tested identities that hold
whatever the traffic: min <= mean <= max for each gap statistic, 0 <= std <= max - min, mean flow
gap x (packets - 1) = duration, and, for totals, total / mean + 1 per direction = packet count.
Means of 10^7 or more are stored to 3 significant figures in the files, so a tolerance of 0.5% is
allowed. We also tested the reading that `total_fiat` is a forward packet count.

| check | A1 15 s (T) | A1 15 s (S) | A2-VPN 15 s (T) | A2-VPN 15 s (S) | A2-NonVPN 15 s (T) | A2-NonVPN 15 s (S) | B 15 s (T) | B 15 s (S) |
|---|---|---|---|---|---|---|---|---|
| fiat: min <= mean <= max | 0.0 | 100.0 | 100.0 | 18.8 | 100.0 | 5.24 | 0.0 | 100.0 |
| biat: min <= mean <= max | 0.01 | 100.0 | 100.0 | 14.07 | 100.0 | 4.06 | 0.01 | 100.0 |
| flowiat: min <= mean <= max | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| active: min <= mean <= max | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| idle: min <= mean <= max | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| mean_flowiat x (packets - 1) = duration (+-1%) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| bytes per packet in 40-1500 | 97.72 | 97.72 | 99.26 | 99.26 | 96.06 | 96.06 | 97.72 | 97.72 |
| min_flowiat <= min(min_fiat, min_biat) | 100.0 | 100.0 | 100.0 | 80.98 | 100.0 | 83.1 | 100.0 | 100.0 |
| total_fiat <= duration | 100.0 | - | 100.0 | - | 100.0 | - | 100.0 | - |
| total/mean + 1, fwd + bwd = packets (+-2%) | 0.0 | - | 100.0 | - | 100.0 | - | 0.0 | - |
| total_fiat as packet count (+-10%) | 1.98 | - | 0.0 | - | 0.04 | - | 1.98 | - |
| 0 <= std_fiat <= max - min | - | 100.0 | - | 0.0 | - | 0.0 | - | 100.0 |
| 0 <= std_biat <= max - min | - | 100.0 | - | 0.0 | - | 0.0 | - | 100.0 |

The same pattern holds at every timeout (`column_name_check` in the manifest has all 16 files).

**Result.** The headers are wrong for 9 of the 12 A1/B files, and the fault is a shift, not a
shuffle. These files hold the paper's features in the order min, max, mean, std for fiat and
biat, but their header names them total, min, max, mean. Under the corrected names every identity
passes on 100% of rows; under the header names, min <= mean <= max passes on at most 0.02%. A2
files are named correctly and hold a different pair of features: `total_fiat` and `total_biat`
(the sum of the forward and backward gaps) instead of `std_fiat` and `std_biat`. The packet-count
reading passes on at most 2% of rows in any file, so it is ruled out. `COLUMN_MAP` in
`src/blindspy/data.py` renames the 9 shifted files. It maps `total_fiat`->`min_fiat`,
`total_biat`->`min_biat`, `min_fiat`->`max_fiat`, `min_biat`->`max_biat`, `max_fiat`->`mean_fiat`,
`max_biat`->`mean_biat`, `mean_fiat`->`std_fiat`, `mean_biat`->`std_biat`; the other 15 names are
unchanged. All feature names in this report (importances, SHAP, ablation) are the corrected ones.

## 2.4 File-family decision

Because A1/B and A2 hold the same flows with two different feature pairs, we had three options:
use each task's own files, rebuild A1 and B from the A2 files, or merge both into 25 features. We
use **each task's own files**: A1 and B with the paper's 23 features, A2 with its 23. These are
the files the paper trained on for each scenario, so our comparison with it stays matched. A
rebuild would have replaced two of the paper's features, and a merge is ambiguous wherever a key
repeats. For the two-step classifier in E10, both steps use the A1/B layout: the 7-class step is
trained on B rows split by VPN (the same flows as A2).

One column needed repair. In A1 120 s, `flowPktsPerSecond` disagrees with A2 on 99.97% of
matched rows, while B 120 s, identical to A1 120 s in the other 22 columns of every row, agrees
with A2 on 100%. We restore the column from B 120 s, which changes 10,288 of 10,782 rows.

## 2.5 Cleaning rules

Every rule is in `configs/base.yaml` and is applied by `scripts/prepare_data.py` in this order.
Each file gets one line per rule in the cleaning log, including rules that change nothing.

| # | Step | Rule | Reason |
|---|---|---|---|
| 1 | parse | Strip trailing commas from header lines; skip comma-only data lines | Export glitch; a comma-only line (one in each 60 s file of A1 and B) is not a flow |
| 2 | column map | Rename with `COLUMN_MAP` | Section 2.3 |
| 3 | restore | A1 120 s `flowPktsPerSecond` from B 120 s | Section 2.4 |
| 4-5 | missing, infinite | Count; none found | No imputation needed |
| 6-9 | -1 codes | Keep | -1 means "not measurable": fewer than 2 packets in a direction, a single-packet flow, or no active/idle period. The matching mean or total is 0 on the same rows |
| 10 | single-packet flows | Keep `duration` = 0 flows | Real flows; dropping them would bias the data toward long flows |
| 11 | negative `min_flowiat` | Keep (down to -1,247 us) | Out-of-order capture timestamps |
| 12 | rounding | Keep means, stds and rates >= 10^7 as stored | Already rounded to 3 significant figures in the source; can't be undone |
| 13 | packet size | Keep flows outside 40-1,500 B per packet | Header-only packets (down to 30 B) and segments merged at capture (up to about 7 kB) are real |
| 14 | exact duplicates | Drop within each file before splitting, keep the first copy | A copy in both train and test would score the model on rows it trained on |
| 15 | conflicting duplicates | Keep (one row per label) | Real ambiguity; removing it would make the test set easier than reality |
| 16-19 | not done | No normalization, no feature selection, no outlier removal, no class balancing | Scaling and resampling run inside the model pipeline on training folds only; all 23 features are kept |

Rows each kept-as-is rule applies to:

| rule | A1 | A2-VPN | A2-NonVPN | B |
|---|---|---|---|---|
| fiat_not_measurable | 8,846 | 5,120 | 3,726 | 8,846 |
| biat_not_measurable | 13,857 | 8,056 | 5,801 | 13,857 |
| flowiat_not_measurable | 2,108 | 1,205 | 903 | 2,108 |
| active_idle_not_measurable | 32,228 | 15,753 | 16,475 | 32,228 |
| single_packet | 2,110 | 1,207 | 903 | 2,110 |
| negative_flowiat | 586 | 300 | 286 | 586 |
| rounding | 15,217 | 7,979 | 7,234 | 15,217 |
| packet_size | 1,380 | 397 | 983 | 1,380 |

## 2.6 Dataset summary

| file | initial_rows | comma_only_lines_skipped | exact_duplicates_removed | conflicting_rows_kept | conflicting_groups | final_rows | missing_values | features | classes | largest_to_smallest_class | train | val | test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 15 s | 18,758 | 0 | 720 | 46 | 23 | 18,038 | 0 | 23 | 2 | 1.08:1 | 12,626 | 2,706 | 2,706 |
| A1 30 s | 14,651 | 0 | 1,041 | 44 | 22 | 13,610 | 0 | 23 | 2 | 1.03:1 | 9,526 | 2,042 | 2,042 |
| A1 60 s | 15,516 | 1 | 838 | 94 | 47 | 14,677 | 0 | 23 | 2 | 1.26:1 | 10,273 | 2,202 | 2,202 |
| A1 120 s | 10,782 | 0 | 1,258 | 62 | 31 | 9,524 | 0 | 23 | 2 | 1.08:1 | 6,666 | 1,429 | 1,429 |
| A2-VPN 15 s | 9,793 | 0 | 413 | 53 | 24 | 9,380 | 0 | 23 | 7 | 5.79:1 | 6,566 | 1,407 | 1,407 |
| A2-VPN 30 s | 7,734 | 0 | 790 | 39 | 17 | 6,944 | 0 | 23 | 7 | 8.39:1 | 4,860 | 1,042 | 1,042 |
| A2-VPN 60 s | 6,935 | 0 | 412 | 61 | 28 | 6,523 | 0 | 23 | 7 | 12.50:1 | 4,565 | 979 | 979 |
| A2-VPN 120 s | 5,631 | 0 | 983 | 111 | 52 | 4,648 | 0 | 23 | 7 | 15.48:1 | 3,252 | 698 | 698 |
| A2-NonVPN 15 s | 8,965 | 0 | 271 | 10 | 3 | 8,694 | 0 | 23 | 7 | 11.85:1 | 6,085 | 1,304 | 1,305 |
| A2-NonVPN 30 s | 6,917 | 0 | 223 | 9 | 3 | 6,694 | 0 | 23 | 7 | 32.47:1 | 4,685 | 1,004 | 1,005 |
| A2-NonVPN 60 s | 8,580 | 0 | 374 | 34 | 15 | 8,206 | 0 | 23 | 7 | 11.31:1 | 5,744 | 1,231 | 1,231 |
| A2-NonVPN 120 s | 5,151 | 0 | 210 | 9 | 3 | 4,941 | 0 | 23 | 7 | 14.37:1 | 3,458 | 741 | 742 |
| B 15 s | 18,758 | 0 | 684 | 100 | 41 | 18,074 | 0 | 23 | 14 | 11.85:1 | 12,651 | 2,711 | 2,712 |
| B 30 s | 14,651 | 0 | 1,013 | 83 | 33 | 13,638 | 0 | 23 | 14 | 32.47:1 | 9,546 | 2,046 | 2,046 |
| B 60 s | 15,516 | 1 | 786 | 174 | 75 | 14,729 | 0 | 23 | 14 | 12.69:1 | 10,309 | 2,210 | 2,210 |
| B 120 s | 10,782 | 0 | 1,193 | 173 | 77 | 9,589 | 0 | 23 | 14 | 17.24:1 | 6,711 | 1,439 | 1,439 |

## 2.7 Data funnel (data v1)

Only two steps remove rows: the parser (two comma-only lines in total) and exact
de-duplication (11,209 rows). Over the 16 core files, 179,120 raw data lines become 167,909
cleaned flows; no value is changed except the restored A1 120 s column.

| file | raw data lines | parsed | after duplicates | final | train | val | test |
|---|---|---|---|---|---|---|---|
| A1 15 s | 18,758 | 18,758 | 18,038 | 18,038 | 12,626 | 2,706 | 2,706 |
| A1 30 s | 14,651 | 14,651 | 13,610 | 13,610 | 9,526 | 2,042 | 2,042 |
| A1 60 s | 15,516 | 15,515 | 14,677 | 14,677 | 10,273 | 2,202 | 2,202 |
| A1 120 s | 10,782 | 10,782 | 9,524 | 9,524 | 6,666 | 1,429 | 1,429 |
| A2-VPN 15 s | 9,793 | 9,793 | 9,380 | 9,380 | 6,566 | 1,407 | 1,407 |
| A2-VPN 30 s | 7,734 | 7,734 | 6,944 | 6,944 | 4,860 | 1,042 | 1,042 |
| A2-VPN 60 s | 6,935 | 6,935 | 6,523 | 6,523 | 4,565 | 979 | 979 |
| A2-VPN 120 s | 5,631 | 5,631 | 4,648 | 4,648 | 3,252 | 698 | 698 |
| A2-NonVPN 15 s | 8,965 | 8,965 | 8,694 | 8,694 | 6,085 | 1,304 | 1,305 |
| A2-NonVPN 30 s | 6,917 | 6,917 | 6,694 | 6,694 | 4,685 | 1,004 | 1,005 |
| A2-NonVPN 60 s | 8,580 | 8,580 | 8,206 | 8,206 | 5,744 | 1,231 | 1,231 |
| A2-NonVPN 120 s | 5,151 | 5,151 | 4,941 | 4,941 | 3,458 | 741 | 742 |
| B 15 s | 18,758 | 18,758 | 18,074 | 18,074 | 12,651 | 2,711 | 2,712 |
| B 30 s | 14,651 | 14,651 | 13,638 | 13,638 | 9,546 | 2,046 | 2,046 |
| B 60 s | 15,516 | 15,515 | 14,729 | 14,729 | 10,309 | 2,210 | 2,210 |
| B 120 s | 10,782 | 10,782 | 9,589 | 9,589 | 6,711 | 1,439 | 1,439 |

![Data funnel](../../figures/data_funnel_v1.png)

Removals are uneven by class, so they change the class balance.

A1

| class | 15 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| Non-VPN | 278 of 8,965 | 229 of 6,917 | 393 of 8,580 | 216 of 5,151 |
| VPN | 442 of 9,793 | 812 of 7,734 | 445 of 6,935 | 1,042 of 5,631 |

A2-VPN

| class | 15 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| VPN-BROWSING | 0 of 2,500 | 0 of 2,500 | 37 of 2,500 | 256 of 2,500 |
| VPN-CHAT | 38 of 1,196 | 38 of 780 | 36 of 514 | 39 of 349 |
| VPN-STREAMING | 0 of 475 | 0 of 298 | 0 of 197 | 0 of 145 |
| VPN-MAIL | 59 of 491 | 483 of 848 | 59 of 383 | 420 of 722 |
| VPN-VOIP | 5 of 2,271 | 5 of 1,299 | 9 of 1,620 | 5 of 386 |
| VPN-P2P | 262 of 928 | 230 of 851 | 218 of 823 | 215 of 813 |
| VPN-FT | 49 of 1,932 | 34 of 1,158 | 53 of 898 | 48 of 716 |

A2-NonVPN

| class | 15 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| BROWSING | 0 of 2,500 | 0 of 2,500 | 0 of 2,500 | 0 of 2,500 |
| CHAT | 3 of 890 | 7 of 595 | 24 of 778 | 3 of 242 |
| STREAMING | 32 of 482 | 31 of 342 | 31 of 252 | 31 of 208 |
| MAIL | 11 of 249 | 4 of 81 | 117 of 849 | 11 of 185 |
| VOIP | 5 of 2,826 | 3 of 1,438 | 36 of 1,829 | 3 of 392 |
| P2P | 59 of 1,000 | 7 of 1,000 | 4 of 1,000 | 2 of 1,000 |
| FT | 161 of 1,018 | 171 of 961 | 162 of 1,372 | 160 of 624 |

B

| class | 15 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| BROWSING | 0 of 2,500 | 0 of 2,500 | 0 of 2,500 | 0 of 2,500 |
| CHAT | 3 of 890 | 7 of 595 | 24 of 778 | 3 of 242 |
| STREAMING | 32 of 482 | 31 of 342 | 31 of 252 | 31 of 208 |
| MAIL | 11 of 249 | 4 of 81 | 117 of 849 | 11 of 185 |
| VOIP | 5 of 2,826 | 3 of 1,438 | 36 of 1,829 | 3 of 392 |
| P2P | 59 of 1,000 | 7 of 1,000 | 4 of 1,000 | 2 of 1,000 |
| FT | 161 of 1,018 | 171 of 961 | 162 of 1,372 | 160 of 624 |
| VPN-BROWSING | 0 of 2,500 | 0 of 2,500 | 37 of 2,500 | 256 of 2,500 |
| VPN-CHAT | 38 of 1,196 | 38 of 780 | 36 of 514 | 39 of 349 |
| VPN-STREAMING | 0 of 475 | 0 of 298 | 0 of 197 | 0 of 145 |
| VPN-MAIL | 59 of 491 | 483 of 848 | 59 of 383 | 420 of 722 |
| VPN-VOIP | 5 of 2,271 | 5 of 1,299 | 9 of 1,620 | 5 of 386 |
| VPN-P2P | 262 of 928 | 230 of 851 | 218 of 823 | 215 of 813 |
| VPN-FT | 49 of 1,932 | 34 of 1,158 | 53 of 898 | 48 of 716 |

| file | groups | rows | test rows with a conflicting twin in train |
|---|---|---|---|
| A1 15 s | 23 | 46 | 4 |
| A1 30 s | 22 | 44 | 8 |
| A1 60 s | 47 | 94 | 9 |
| A1 120 s | 31 | 62 | 7 |
| A2-VPN 15 s | 24 | 53 | 9 |
| A2-VPN 30 s | 17 | 39 | 4 |
| A2-VPN 60 s | 28 | 61 | 6 |
| A2-VPN 120 s | 52 | 111 | 11 |
| A2-NonVPN 15 s | 3 | 10 | 3 |
| A2-NonVPN 30 s | 3 | 9 | 2 |
| A2-NonVPN 60 s | 15 | 34 | 6 |
| A2-NonVPN 120 s | 3 | 9 | 1 |
| B 15 s | 41 | 100 | 15 |
| B 30 s | 33 | 83 | 15 |
| B 60 s | 75 | 174 | 25 |
| B 120 s | 77 | 173 | 21 |

**In sentences.** For task A1 at 15 s, 18,758 flows were parsed. After removing 720 exact
duplicates (442 VPN, 278 non-VPN), 18,038 unique flows remained. All 23 features were kept without
feature selection or normalization, and the flows were split 12,626 / 2,706 / 2,706 into training,
validation and test sets. Duplicates are most common at the 120 s timeout: 1,258 of 10,782 A1
flows (11.67%) and 983 of 5,631 A2-VPN flows (17.46%). They hit some classes much harder than
others: VPN-MAIL lost 483 of 848 flows at 30 s and 420 of 722 at 120 s, FT lost 160-171 flows at
every timeout, and BROWSING lost none. A1 loses slightly more rows than B at the same timeout
(720 vs 684 at 15 s) because flows that share features but differ in traffic type become exact
duplicates once both are labelled with the same VPN / non-VPN label. After cleaning, no exact
duplicate crosses train and test in any file; 1-25 test flows per file have a twin in training
with the same features but a different label.

**Deliberately not done, and why.** No values were imputed (none are missing). No -1 code was
replaced and nothing was set to zero, because -1 has a meaning. No flag columns were added, since
the matching zero mean already marks the same rows and flags would change the paper's 23-feature
set. No rows were dropped as outliers, because long, bursty or single-packet flows are real
traffic. No features were scaled or selected and no classes were rebalanced at data level; all of
that happens inside model pipelines, fitted on training data only (Section 3).

## 2.8 Timeouts

Within each task the feature names, their order and their units are the same at all four
timeouts. The number of flows falls as the timeout grows (A1: 18,038 at 15 s to 9,524 at 120 s),
and the class mix moves a lot. One class's share changes by up to 24.6 percentage points in
A2-NonVPN (VOIP 32.5% at 15 s vs 7.9% at 120 s; BROWSING 28.8% vs 50.6%), 21.6 in A2-VPN, 12.2 in B
and 7.6 in A1. A difference in scores between timeouts is therefore confounded by class mix, and
we do not attribute it to the timeout alone.

![Class mix per timeout](../../figures/timeout_class_mix.png)

## 2.9 The Kaggle copy used by Isnan et al.

The Kaggle file used by Isnan et al. [13] has 59,706 rows and the A2 header. Compared as a
multiset of rows, it equals the eight A2 files (VPN and non-VPN at all four timeouts) stacked:
no row is left over at 12 significant digits, and the labels are the same. It has 40,987 unique
rows, their count after removing 18,719 duplicates. Only 3,676 of those duplicates lie within one
file; 7,093 distinct flows appear in two (2,475), three (1,286) or four (3,332) timeout files. Most
of their "duplicates" are therefore the same flow built with different timeouts, and a random
80/20 split of the stacked file puts versions of one flow in both training and test sets. The file
also carries the A2 feature pair (`total_*`), not the paper's (`std_*`). E16 reproduces their
setup and measures the effect of keeping timeouts apart.

## 2.10 Data versions

| Version | Date | Author | What changed | Files | Rows (raw lines -> final) | Commit |
|---|---|---|---|---|---|---|
| v1 | 2026-10-27 | utsav-s-r | First cleaned version | 16 core files (A1, A2-VPN, A2-NonVPN, B x 15/30/60/120 s) | 179,120 -> 167,909 | 0a36896 |

Figures from the exploration notebook (`notebooks/01_eda.ipynb`): class counts per task and timeout
(`class_counts_v1.png`), feature distributions by class (`feature_boxes_<task>_15s.png`) and
feature correlations (`correlation_B_15s.png`, `correlation_A2-VPN_15s.png`). The notebook records
two findings that matter later. Classes differ by orders of magnitude on rate and gap features
(VPN-STREAMING: median 202 packets/s; VPN-VOIP: 2.2). Many features are near-duplicates: 19 of
253 pairs in B have Spearman |rho| >= 0.9, mostly within the active/idle group. Feature
importances will therefore be shared out among correlated features.
