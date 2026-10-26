# Paper baseline values — how they were collected

Source: Draper-Gil, G., Lashkari, A. H., Mamun, M. S. I., Ghorbani, A. A. (2016).
*Characterization of Encrypted and VPN Traffic using Time-related Features.*
ICISSP 2016, pp. 407–414. Local copy: `about/VPNNonVPN_ICISSP_2016.pdf`.

Data file: `paper_baseline.csv` (one row per value, long format).

| Column | Meaning |
| --- | --- |
| `task` | A1, A2-VPN, A2-NonVPN or B (same names as our tasks) |
| `class` | Class label, or `AVERAGE` / `AVERAGE_BEST_TIMEOUT` for averages the paper states |
| `algorithm` | `C4.5` or `KNN` (Weka defaults, 10-fold cross-validation) |
| `timeout_s` | Flow timeout in seconds (15, 30, 60, 120), or `best` |
| `metric` | `precision` or `recall` |
| `source` | `stated_in_text` = exact number written in the paper; `read_from_chart` = read by eye from a figure |
| `uncertainty` | ± error of the reading (0 for text values) |
| `page`, `location` | Where in the paper the value comes from |

## Coverage

- **A1 (VPN vs non-VPN):** complete. Figure 2 gives precision and recall for both classes,
  both algorithms and all four timeouts. Bar heights were read against the y-axis gridlines
  (±0.01). The values the text states exactly (15 s and 120 s precision) agree with the chart
  readings, which supports the reading accuracy.
- **A2 (per-class, separate VPN / non-VPN models):** only the class-averaged precision at 15 s
  (C4.5) is stated in the text. Figure 3 (a–d) shows per-class precision and recall, but it
  draws two unlabelled lines per timeout (one C4.5, one KNN), so a line can't be assigned to
  an algorithm reliably. These values are **not** in the CSV.
- **B (14 classes, one model):** the text states 12 per-class precision values and the best
  average precision for each algorithm. Figure 3 (e–h) has the same unlabelled-line problem.

## How to use in the comparison

- Compare like with like: our precision vs their precision, our recall vs their recall,
  under the paper's own protocol (10-fold CV on the full file). Never against our macro F1.
- For A2 and most B classes, compare against the stated averages only, and say in the report
  that per-class values could not be recovered from the figures.
- Chart readings are approximate: a difference smaller than the stated uncertainty is not a
  real difference.
