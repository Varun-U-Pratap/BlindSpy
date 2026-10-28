# Data changelog

One entry per cleaned-data version, newest first. A version is built with
`uv run python scripts/prepare_data.py --version vN` into `data/processed/vN/` (never
overwritten). Every number below comes from that version's `manifest.json` and
`cleaning_log.csv`.

| Version | Date | Author | What changed | Files | Rows (raw lines -> final) | Commit |
|---|---|---|---|---|---|---|
| kaggle-k1 | 2026-10-29 | utsav-s-r | Kaggle CSV of Isnan et al. (2026) as its own version, for E16 only; data v1 unchanged | 1 CSV (= the 8 A2 files stacked) | 59,706 -> 40,987 (their dedup) / 56,030 kept per timeout | 3a72f2e |
| v1 | 2026-10-27 | utsav-s-r | First cleaned version | 16 core files (A1, A2-VPN, A2-NonVPN, B x 15/30/60/120 s) | 179,120 -> 167,909 | 0a36896 |

## kaggle-k1 (2026-10-29, utsav-s-r)

**What.** The Kaggle CSV used by Isnan et al. (2026), `datasets/kaggle/consolidated_traffic_data.csv`
(SHA-256 in `data/kaggle_sha256.txt`), as its own data version for E16. Built with
`uv run python scripts/prepare_kaggle.py --version k1` into `data/processed/kaggle/k1/` (never
overwritten); manifest copied to `data/kaggle_k1_manifest.json`. Every number below comes from
that manifest.

- The CSV is the eight A2 files (VPN and NonVPN x 4 timeouts) stacked in their original row order:
  A2-NonVPN 120 s (5,151 rows), A2-VPN 120 s (5,631), A2-NonVPN 15 s (8,965), A2-VPN 15 s (9,793),
  A2-NonVPN 30 s (6,917), A2-VPN 30 s (7,734), A2-NonVPN 60 s (8,580), A2-VPN 60 s (6,935). Each
  block was checked against its ARFF file (23 features at 12 significant digits and the label),
  so every row's source file, timeout and row number are known (`rows.parquet`).
- Labels: the 14 task-B classes. Features: the 23 A2 features (total_fiat / total_biat layout).
- Split "isnan" (Isnan et al.'s setup): exact duplicates (23 features + label) dropped over the
  whole CSV, first copy kept: 59,706 -> 40,987 rows; stratified 80/20, seed 42: 32,789 / 8,198.
- Splits "per_timeout_<t>s" (corrected setup): each timeout on its own, duplicates dropped within
  it, then stratified 80/20, seed 42: 15 s 18,758 -> 18,074 (14,459 / 3,615); 30 s 14,651 ->
  13,638 (10,910 / 2,728); 60 s 15,515 -> 14,729 (11,783 / 2,946); 120 s 10,782 -> 9,589 (7,671 /
  1,918). 56,030 rows kept over the four timeouts.
- Distinct rows (23 features + label) by the number of timeout files they appear in: 1: 33,894;
  2: 2,475; 3: 1,286; 4: 3,332 (7,093 in two or more, as E00 found).
- Split files keep the row order `train_test_split` returns (a seeded forest and an unshuffled
  5-fold CV depend on it).
- Not done, on purpose: no cleaning beyond the duplicate rule (no odd-value rules, no column
  changes); the data is used as Isnan et al. used it.

**Why.** E16 reproduces Isnan et al. and then repeats their protocol with the timeouts kept apart;
their data mixes timeouts, so it needs its own version, separate from v1.

**Checks.** Two builds into separate folders and the build in the main folder gave identical
SHA-256 hashes for all 6 files; a second build into an existing folder is refused
(`tests/test_kaggle.py`).

## v1 (2026-10-27, utsav-s-r)

**What.** First cleaned version of the 16 core task files. Rules in `configs/base.yaml`
(`cleaning:`), evidence in `reports/experiments/utsav-s-r/E00.md`.

- Parser: trailing commas stripped from every header line; 2 comma-only data lines skipped
  (one each in A1 60 s and B 60 s; B-AllinOne 60 s has one too but is not a core task).
- Column names: `COLUMN_MAP` renames the 9 A1/B files whose header shifts the fiat/biat names by
  one pair. A1 and B hold the paper's Table 2 features (fiat/biat as min, max, mean, std); A2
  holds `total_fiat`/`total_biat` instead of `std_fiat`/`std_biat`.
- A1 120 s: `flowPktsPerSecond` restored from B 120 s in 10,288 of 10,782 rows (the column was
  corrupt; B 120 s is identical in the other 22 columns).
- Exact duplicates (23 features + label) dropped within each file, first copy kept: 11,209 rows
  (A1 3,857; A2-VPN 2,598; A2-NonVPN 1,078; B 3,676). Conflicting duplicates (same features,
  different label) kept: 1,102 rows over the 16 files.
- Odd values kept as-is and logged with counts: -1 "not measurable" codes, duration-0
  single-packet flows, negative `min_flowiat`, 3-significant-figure rounding of large
  means/stds/rates, bytes per packet outside 40-1,500.
- Not done, on purpose: no imputation (0 missing values), no normalization, no feature
  selection (all 23 kept), no outlier removal, no class balancing at data level.
- Splits: stratified 70/15/15, seed 42, per file, saved in `splits/`. 0 exact duplicates
  cross train and test in any file.

**Why.** The raw ARFF files can't be used as published: wrong column names in 9 files, one
corrupt column, parse glitches, and duplicate rows that would leak between train and test.

**Size.** Parquet 18.7 MB for the 16 files (20 MB folder with splits and log). The build is
deterministic: three builds gave byte-identical files.

**Per file (from the manifest):**

| File | Raw lines | Duplicates removed | Final | Train / val / test |
|---|---|---|---|---|
| A1 15 s | 18,758 | 720 | 18,038 | 12,626 / 2,706 / 2,706 |
| A1 30 s | 14,651 | 1,041 | 13,610 | 9,526 / 2,042 / 2,042 |
| A1 60 s | 15,516 (1 comma-only) | 838 | 14,677 | 10,273 / 2,202 / 2,202 |
| A1 120 s | 10,782 | 1,258 | 9,524 | 6,666 / 1,429 / 1,429 |
| A2-VPN 15 s | 9,793 | 413 | 9,380 | 6,566 / 1,407 / 1,407 |
| A2-VPN 30 s | 7,734 | 790 | 6,944 | 4,860 / 1,042 / 1,042 |
| A2-VPN 60 s | 6,935 | 412 | 6,523 | 4,565 / 979 / 979 |
| A2-VPN 120 s | 5,631 | 983 | 4,648 | 3,252 / 698 / 698 |
| A2-NonVPN 15 s | 8,965 | 271 | 8,694 | 6,085 / 1,304 / 1,305 |
| A2-NonVPN 30 s | 6,917 | 223 | 6,694 | 4,685 / 1,004 / 1,005 |
| A2-NonVPN 60 s | 8,580 | 374 | 8,206 | 5,744 / 1,231 / 1,231 |
| A2-NonVPN 120 s | 5,151 | 210 | 4,941 | 3,458 / 741 / 742 |
| B 15 s | 18,758 | 684 | 18,074 | 12,651 / 2,711 / 2,712 |
| B 30 s | 14,651 | 1,013 | 13,638 | 9,546 / 2,046 / 2,046 |
| B 60 s | 15,516 (1 comma-only) | 786 | 14,729 | 10,309 / 2,210 / 2,210 |
| B 120 s | 10,782 | 1,193 | 9,589 | 6,711 / 1,439 / 1,439 |
