// Builds BlindSpy_research_paper.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, LevelFormat, TabStopType,
  Footer, PageNumber, VerticalAlign,
} = require("docx");

const FONT = "Times New Roman";
const BODY = 24;       // 12 pt
const H1 = 28;         // 14 pt
const H2 = 24;         // 12 pt
const TITLE = 32;      // 16 pt
const SMALL = 20;      // 10 pt (tables, captions)
const FIG = "/home/varun/IT/BlindSpy/reports/figures/";
const HERE = __dirname;
const TEXT_W = 9026;   // A4 with 1-inch margins, in DXA

// ---------------- references (numbered by first citation) ----------------
const REFS = {
  draper: 'G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun, and A. A. Ghorbani, "Characterization of encrypted and VPN traffic using time-related features," in *Proc. 2nd Int. Conf. on Information Systems Security and Privacy (ICISSP)*, 2016, pp. 407–414.',
  iscx: 'Canadian Institute for Cybersecurity, University of New Brunswick, "VPN-nonVPN dataset (ISCXVPN2016)." [Online]. Available: https://www.unb.ca/cic/datasets/vpn.html',
  rezaei: 'S. Rezaei and X. Liu, "Deep learning for encrypted traffic classification: An overview," *IEEE Communications Magazine*, 2019, arXiv:1810.07906.',
  quinlan: 'J. R. Quinlan, *C4.5: Programs for Machine Learning*. San Mateo, CA, USA: Morgan Kaufmann, 1993.',
  bagui: 'S. Bagui, X. Fang, E. Kalaimannan, S. C. Bagui, and J. Sheehan, "Comparison of machine-learning algorithms for classification of VPN network traffic flow using time-related features," *Journal of Cyber Security Technology*, vol. 1, no. 2, pp. 108–126, 2017, doi: 10.1080/23742917.2017.1321891.',
  wang: 'W. Wang, M. Zhu, J. Wang, X. Zeng, and Z. Yang, "End-to-end encrypted traffic classification with one-dimensional convolution neural networks," in *Proc. IEEE Int. Conf. on Intelligence and Security Informatics (ISI)*, 2017, doi: 10.1109/ISI.2017.8004872.',
  lotfollahi: 'M. Lotfollahi, M. Jafari Siavoshani, R. Shirali Hossein Zade, and M. Saberian, "Deep Packet: A novel approach for encrypted traffic classification using deep learning," *Soft Computing*, vol. 24, no. 3, pp. 1999–2012, 2020, doi: 10.1007/s00500-019-04030-2.',
  jorgensen: 'S. Jorgensen *et al.*, "Extensible machine learning for encrypted network traffic application labeling via uncertainty quantification," *IEEE Transactions on Artificial Intelligence*, 2023, doi: 10.1109/TAI.2023.3244168.',
  isnan: 'W. Isnan, Y. Fatmi, R. Darni, and V. Aulia, "Flow-based encrypted network traffic classification using Random Forest for network access control," *Intechno Journal: Information Technology Journal*, vol. 8, no. 1, 2026, doi: 10.24076/intechnojournal.2026v8i1.2922.',
  poster: 'G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun, and A. A. Ghorbani, "Characterization of encrypted and VPN traffic using time-related features" (poster), UNB Research Expo, 2016. [Online]. Available: https://www.cs.unb.ca/research-expo/expos/2016/',
  sklearn: 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
  xgb: 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.',
  lgbm: 'G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30*, 2017.',
  optuna: 'T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2019, pp. 2623–2631.',
  imblearn: 'G. Lemaître, F. Nogueira, and C. K. Aridas, "Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning," *Journal of Machine Learning Research*, vol. 18, no. 17, pp. 1–5, 2017.',
  smote: 'N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.',
  efron: 'B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.',
  holm: 'S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.',
  naeini: 'M. P. Naeini, G. F. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in *Proc. 29th AAAI Conf. on Artificial Intelligence*, 2015, pp. 2901–2907.',
  guo: 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. on Machine Learning (ICML)*, PMLR 70, 2017, pp. 1321–1330.',
  zadrozny: 'B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2002, pp. 694–699.',
  shap: 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30*, 2017.',
  mlflow: 'M. Zaharia *et al.*, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39–45, 2018.',
  streamlit: 'Snowflake Inc., "Streamlit: A faster way to build and share data apps" (open-source Python library, Apache 2.0 licence). [Online]. Available: https://streamlit.io',
  weka: 'University of Waikato, "Weka 3 API documentation (stable)," classes weka.classifiers.trees.J48, weka.classifiers.lazy.IBk, weka.core.EuclideanDistance and weka.core.NormalizableDistance. [Online]. Available: https://weka.sourceforge.io/doc.stable/ (accessed 28 Oct. 2026).',
  spearman: 'C. Spearman, "The proof and measurement of association between two things," *American Journal of Psychology*, vol. 15, no. 1, pp. 72–101, 1904.',
};
const refOrder = [];
function citeNums(keys) {
  return keys.split(",").map((k) => {
    k = k.trim();
    if (!REFS[k]) throw new Error("unknown ref " + k);
    if (!refOrder.includes(k)) refOrder.push(k);
    return refOrder.indexOf(k) + 1;
  });
}
function resolveCites(s) {
  return s.replace(/\{([a-z0-9, ]+)\}/g, (_, keys) => {
    const n = citeNums(keys).sort((a, b) => a - b);
    return "[" + n.join("], [") + "]";
  });
}

// ---------------- inline markup: **bold**, *italic*, ~sub~, ^sup^ ----------------
function runs(text, base = {}) {
  text = resolveCites(text);
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|~[^~]+~|\^[^^]+\^)/g;
  let last = 0, m;
  const push = (t, extra) => { if (t) out.push(new TextRun({ text: t, font: FONT, size: BODY, ...base, ...extra })); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("*")) push(tok.slice(1, -1), { italics: true });
    else if (tok.startsWith("~")) push(tok.slice(1, -1), { subScript: true });
    else push(tok.slice(1, -1), { superScript: true });
    last = m.index + tok.length;
  }
  push(text.slice(last), {});
  return out;
}

const body = [];
const P = (text, opts = {}) => body.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, ...opts,
  children: runs(text),
}));
const H = (text) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT, size: H1, bold: true })] }));
const HU = (text) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.LEFT, children: [new TextRun({ text, font: FONT, size: H1, bold: true })] }));
const H2_ = (text) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT, size: H2, bold: true })] }));
const BUL = (text) => body.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 300 }, children: runs(text) }));
let numInst = 0;
const NUMLIST = (items, ref = "num") => { numInst++; items.forEach((t) => body.push(new Paragraph({ numbering: { reference: ref, level: 0, instance: numInst }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 300 }, children: runs(t) }))); };
let eqNo = 0;
const EQ = (text) => { eqNo++; body.push(new Paragraph({
  tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
  spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: "\t", font: FONT, size: BODY }), ...runs(text), new TextRun({ text: `\t(${eqNo})`, font: FONT, size: BODY })],
})); };

function caption(label, text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 }, keepLines: true,
    children: [new TextRun({ text: label + " ", font: FONT, size: SMALL, bold: true }), ...runs(text, { size: SMALL })],
  });
}
let figNo = 0;
function FIGURE(file, wpx, text) {
  figNo++;
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: Math.round(wpx * h / w) },
      altText: { title: `Figure ${figNo}`, description: text.replace(/\*/g, ""), name: `fig${figNo}` } })] }));
  body.push(caption(`Fig. ${figNo}.`, text));
}

let tabNo = 0;
const THICK = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
const THIN = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(title, header, rows, widths, opts = {}) {
  tabNo++;
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
    children: [new TextRun({ text: `Table ${tabNo}. `, font: FONT, size: SMALL, bold: true }), ...runs(title, { size: SMALL })] }));
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, isHead, isLast) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: isHead ? THICK : NONE, bottom: isHead ? THIN : (isLast ? THICK : NONE), left: NONE, right: NONE },
    children: [new Paragraph({ alignment: i === 0 || (opts.leftCols || []).includes(i) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: SMALL, ...(isHead ? { bold: true } : {}) }) })],
  });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true, false)) })];
  rows.forEach((r, ri) => trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, ri === rows.length - 1)) })));
  body.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE }, rows: trs }));
  if (opts.note) body.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 40, after: 200 }, children: runs(opts.note, { size: SMALL, italics: false }) }));
  else body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
}

// =====================================================================
// FRONT MATTER
// =====================================================================
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: 300 },
  children: [new TextRun({ text: "Payload-Free Classification of VPN and Encrypted Traffic from Flow Timing Features: A Leakage-Safe Re-evaluation of ISCXVPN2016", font: FONT, size: TITLE, bold: true })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "BlindSpy Team", font: FONT, size: BODY, bold: true })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "Third-year AIML Project", font: FONT, size: BODY, italics: true })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: "Code, data versions and run history: https://github.com/Varun-U-Pratap/BlindSpy", font: FONT, size: BODY, italics: true })] }));

HU("Abstract");
P("Encryption hides the contents of network packets but not the timing of a connection. This paper asks how accurately the type of encrypted traffic, and whether it passes through a virtual private network (VPN), can be recognised from flow timing statistics alone, and how much of the performance reported on the ISCXVPN2016 benchmark survives a leakage-safe evaluation. Using only the 23 published time-related flow features and never the payload, we study the dataset's three tasks: VPN versus non-VPN traffic (A1), the seven traffic types within VPN and within non-VPN traffic (A2, two separate models) and all 14 classes jointly (B). An audit of the published files shows that 9 of the 12 A1 and B files carry column names shifted by one feature pair and that the A2 files hold a different feature pair; we recover the correct mapping from value identities that hold on 100% of rows. We then remove exact duplicates before a stratified 70/15/15 split, make every modelling decision on training and validation rows, give seven method types the same tuning budget, and score the test set once, with 95% stratified paired bootstrap intervals (10,000 resamples) and 26 Holm-corrected tests fixed in advance. Under the original study's protocol, our C4.5 approximation and 1-nearest-neighbour classifier reproduce its published A1 results. Under the leakage-safe protocol, gradient-boosted trees reach a test macro F1 of 0.9434 (A1), 0.8652 and 0.9403 (A2, VPN and non-VPN) and 0.8480 (B). Isotonic calibration lowers the test expected calibration error to 0.0090–0.0169 without changing any prediction (for two tasks, a change made after the test set was opened). Finally, we reproduce a recent 88.01% result on a stacked copy of the dataset (0.8796) and show, with a training-size control, that its advantage over single-timeout scores comes from a larger training set rather than from mixing flow timeouts.");
body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 240, line: 300 }, children: [
  new TextRun({ text: "Keywords: ", font: FONT, size: BODY, bold: true, italics: true }),
  new TextRun({ text: "encrypted traffic classification, VPN detection, flow timing features, ISCXVPN2016, gradient boosting, data leakage, reproducibility, probability calibration, SHAP.", font: FONT, size: BODY, italics: true })] }));

// =====================================================================
// 1 INTRODUCTION
// =====================================================================
H("1. Introduction");
P("Most Internet traffic is now encrypted. Deep packet inspection, which identifies an application by reading packet contents, therefore no longer works for most connections. Network operators still need to know what kind of traffic a connection carries, for quality of service, access control and security monitoring, and whether it is tunnelled through a VPN. Applications leave a pattern in time even when their contents are hidden: a voice call sends small packets at a steady rhythm, video streaming arrives in bursts, and a file transfer runs at full speed with almost no idle time. Classifiers built on such patterns need no payload and no decryption, which makes them attractive both for privacy and for deployment.");
P("Draper-Gil *et al.* {draper} published the ISCXVPN2016 dataset {iscx} to study this idea. Each flow is described by 23 time-related statistics and a label, and the flows are published at four flow timeouts (15, 30, 60 and 120 s). The timeout is a setting of the flow builder that closes an inactive flow; it is not the duration of a flow. The original study defines three tasks, which we follow and score separately: **A1**, a binary VPN / non-VPN classifier; **A2**, two separate 7-class classifiers, one for VPN and one for non-VPN traffic; and **B**, one 14-class classifier over both. The seven traffic types are browsing, chat, email, file transfer (FT), peer-to-peer (P2P), streaming and voice over IP (VoIP).");
P("Since then, scores between about 80% and 99% have been reported on this dataset, but they are rarely comparable {rezaei}. Studies differ in the split, the class setup, whether duplicate rows are removed and whether flows from different timeouts are mixed, and each of these choices can move accuracy by several points. In addition, we found that the column headers of most published files do not describe their columns, so features may be misread before any model is trained. A credible answer to the question \"how well does timing alone work?\" therefore needs a careful check of the data and an evaluation protocol in which no information from the test rows reaches any modelling decision.");
P("This paper addresses four research questions:");
NUMLIST([
  "**RQ1 (data validity).** Do the published feature files contain the features their headers name, and can the correct mapping be established from the data alone?",
  "**RQ2 (reproducibility).** Can the original C4.5 and KNN results be reproduced under the original protocol?",
  "**RQ3 (performance).** How well do modern classifiers perform on all three tasks under a leakage-safe protocol, with equal tuning budgets, statistical tests fixed in advance and calibrated confidence?",
  "**RQ4 (comparability).** Why does a recently published 14-class result on a stacked copy of the dataset differ from single-timeout scores?",
], "rq");
P("The main contributions of this work are:");
NUMLIST([
  "An audit of the published files which finds shifted column names in 9 of the 12 A1 and B files and a different feature pair in the A2 files, resolved from value identities that hold on 100% of rows (Section 3.2).",
  "A reproduction of the original C4.5 and KNN results under the original 10-fold protocol (Section 4.1).",
  "A leakage-safe protocol with a test set scored once, a fair comparison of seven method types with equal tuning budgets, and 26 paired tests fixed before the test set was opened (Sections 3.4–3.7 and 4.2–4.4).",
  "Analyses of calibration, stability, training-set size and feature use (SHAP and a feature-group ablation), together with a command-line predictor and a demo that report a calibrated confidence and the features behind each prediction (Sections 4.5–4.6).",
  "A reproduction of a recent 88.01% result and a control experiment that separates the effect of training-set size from the effect of mixing flow timeouts (Section 4.7).",
]);
P("The rest of the paper is organised as follows. Section 2 reviews related work. Section 3 describes the dataset, the column audit, the cleaning rules, the evaluation protocol, the models and the statistical methods. Section 4 presents and discusses the results, including the limitations of the study. Section 5 concludes and outlines future work.");

// =====================================================================
// 2 RELATED WORK
// =====================================================================
H("2. Related Work");
H2_("2.1 Flow-feature classification on ISCXVPN2016");
P("Draper-Gil *et al.* {draper} introduced the time-related feature set and evaluated C4.5 {quinlan} and k-nearest neighbours (KNN) with 10-fold cross-validation in Weka. At the 15 s timeout they reported an A1 precision of 0.89 (VPN) and 0.906 (non-VPN) with C4.5, and for task B an average precision of 0.783 (C4.5) and 0.711 (KNN) at the best timeout. Bagui *et al.* {bagui} compared several learners on the same features, including Random Forest and gradient boosting, and reported an accuracy of about 90%; we cite this figure from the public abstract only, since the full paper was not available to us.");
H2_("2.2 Packet- and payload-based deep learning");
P("A second line of work learns directly from raw packet bytes rather than from flow statistics. Wang *et al.* {wang} applied one-dimensional convolutional networks to the first bytes of each session, and Lotfollahi *et al.* {lotfollahi} proposed Deep Packet, which combines stacked autoencoders and convolutional networks on packet bytes; Rezaei and Liu {rezaei} survey this area. These models use the packet captures, not the published 23-feature files, and they read header and payload bytes, so their scores answer a different question from ours. Our study deliberately stays with flow timing statistics, which need no access to packet contents.");
H2_("2.3 Data quality and evaluation pitfalls");
P("Jorgensen *et al.* {jorgensen} documented quality problems in the ISCXVPN2016 captures, including readable HTML in a capture that should be encrypted, VPN captures that hold several connections and ambiguous labels. Isnan *et al.* {isnan} trained a 300-tree Random Forest on a public copy of the dataset with 59,706 rows, removed 18,719 duplicates, kept 40,987 rows and reported 88.01% accuracy for 14 classes on a single 80/20 split. We found that this copy is the eight A2 files of all four timeouts stacked together: as a multiset its 59,706 rows equal the rows of those files, and 7,093 distinct flows appear in two to four timeout files, while only 3,676 rows are duplicates within one file. Most of the removed \"duplicates\" are therefore the same flows repeated across timeouts, and a random split of the stacked file mixes the four flow-building settings in the training and test data. Section 4.7 measures what this does to the score. Table 1 summarises the studies most relevant to this paper.");
TABLE("Summary of related work on ISCXVPN2016 and encrypted traffic classification.",
  ["Study", "Input and setup", "Models", "Reported result"],
  [
    ["Draper-Gil *et al.* {draper}", "23 timing features; A1, A2, B per timeout; 10-fold CV", "C4.5, KNN", "A1 precision 0.89 / 0.906 (C4.5, 15 s); B avg. precision 0.783"],
    ["Bagui *et al.* {bagui}", "Same timing features", "RF, gradient boosting, others", "About 90% (abstract)"],
    ["Wang *et al.* {wang}", "Raw session bytes from captures", "1-D CNN", "Not comparable (payload input)"],
    ["Lotfollahi *et al.* {lotfollahi}", "Raw packet bytes from captures", "SAE, CNN", "Not comparable (payload input)"],
    ["Jorgensen *et al.* {jorgensen}", "ISCXVPN2016 captures", "Uncertainty-aware labelling", "Documents capture and label problems"],
    ["Isnan *et al.* {isnan}", "Stacked copy, 4 timeouts mixed; 80/20 split", "RF (300 trees)", "88.01% accuracy, 14 classes"],
    ["This work", "23 timing features; dedup; 70/15/15; test scored once", "7 method types, equal tuning", "Macro F1 0.9434 / 0.8652 / 0.9403 / 0.8480"],
  ], [1900, 2650, 1850, 2626], { leftCols: [1, 2, 3] });
P("**Research gap.** No prior study on the published feature files checks that the columns hold the features their headers name, removes duplicates before splitting, keeps the timeouts apart, gives every method type the same tuning budget and fixes its statistical tests before opening the test set. This paper does all five and also reports calibrated confidence and feature use.");

// =====================================================================
// 3 METHODOLOGY
// =====================================================================
H("3. Methodology");
P("Fig. 1 gives an overview of the method. The data stage checks and cleans the published files; the modelling stage makes every choice on training and validation rows only; and the test rows of each task are scored once at the end.");
FIGURE(path.join(HERE, "fig_pipeline.png"), 600, "Overview of the leakage-safe pipeline. Shaded boxes in the top row run in order for every task; the dashed boxes are analyses of the final models.");

H2_("3.1 Dataset and tasks");
P("ISCXVPN2016 was captured in 2016 in a laboratory, with and without a VPN, and converted to flows with ISCXFlowMeter {draper, iscx}. It is published as 20 ARFF files: task A1 (4 files), A2 (4 VPN and 4 non-VPN files), B (4 files) and four 7-class \"AllinOne\" files that do not belong to a task. Each row is one flow with 23 numeric features (times in microseconds, rates per second) and a label. The features describe the flow duration, the byte and packet rates, the inter-arrival times (IAT) of the whole flow (flowiat), of the forward direction (fiat) and of the backward direction (biat), and the active and idle periods. The files hold no payload, addresses, ports, timestamps or capture identifiers. A1, A2 and B hold the same flows at each timeout: every A1 row matches an A2 row on seven integer features (100%), and B is identical to A1 cell for cell except for one corrupt column.");
H2_("3.2 Column-name audit");
P("The first row of the A1 15 s file has a duration of 9,368,711 µs but a \"total forward gap\" of 16 µs, which is impossible. The original feature table and the authors' poster {poster} list the forward and backward inter-arrival times as minimum, mean, maximum and standard deviation, yet 17 of the 20 headers list *total_fiat* and *total_biat* and no standard deviation. We therefore read every file under both candidate name lists and tested identities that hold for any traffic:");
EQ("min ≤ mean ≤ max,");
EQ("0 ≤ std ≤ max − min,");
EQ("mean flowiat × (n − 1) ≈ duration  (within 1%),");
P("where *n* is the number of packets in the flow, allowing 0.5% for values that the files store to three significant figures. Under the header names, Eq. (1) holds on at most 0.02% of A1 and B rows; under names shifted by one pair (the original order min, max, mean, std) every identity holds on 100% of rows. The A2 files are named correctly but hold *total_fiat* and *total_biat* (the sum of the gaps) in place of *std_fiat* and *std_biat*: their totals reproduce the packet count on 100% of rows, while the standard-deviation check of Eq. (2) holds on 0%. Reading *total_fiat* as a packet count holds on at most 2% of rows in any file and is ruled out. Each task therefore uses its own files, with a column map that renames the 9 shifted files. One corrupt A1 120 s column (*flowPktsPerSecond*), which disagrees with A2 on 99.97% of matched rows, is restored from the B file that is identical in every other column; this changes 10,288 of 10,782 rows.");
H2_("3.3 Data cleaning");
P("Every cleaning rule is applied by one script and logged per file and per class, including rules that change nothing. Exact duplicates (23 features and label) are removed within each file before splitting, so no flow can be both trained on and scored. Of 179,120 raw lines in the 16 task files, 2 comma-only lines are skipped and 11,209 duplicates are removed, leaving 167,909 rows. Conflicting duplicates (same features, different label; 1,102 rows) are kept, because removing them would make the test set easier than reality. The code −1 (\"not measurable\": fewer than two packets in a direction, or no active or idle period), single-packet flows and negative minimum gaps (out-of-order timestamps) are kept as they are. No normalisation, feature selection, outlier removal or class balancing is applied to the data itself. Table 2 summarises the 15 s files used by the final models.");
TABLE("Data at the 15 s timeout after cleaning (data version v1).",
  ["Task", "Raw rows", "Duplicates removed", "Final rows", "Classes", "Train / validation / test"],
  [
    ["A1", "18,758", "720", "18,038", "2", "12,626 / 2,706 / 2,706"],
    ["A2-VPN", "9,793", "413", "9,380", "7", "6,566 / 1,407 / 1,407"],
    ["A2-NonVPN", "8,965", "271", "8,694", "7", "6,085 / 1,304 / 1,305"],
    ["B", "18,758", "684", "18,074", "14", "12,651 / 2,711 / 2,712"],
  ], [1400, 1200, 1500, 1250, 1050, 2626]);
H2_("3.4 Evaluation protocol");
P("For each task and timeout the cleaned data are split once into 70% training, 15% validation and 15% test rows, stratified by class, with seed 42. Hyperparameters are chosen by 5-fold stratified cross-validation inside the training rows, and models are compared on the validation rows. Scaling, transforms and resampling are pipeline steps fitted on training folds only. The final configuration of each task is retrained on the training and validation rows and scored once on its test rows. Every experiment states its hypotheses and numeric pass/fail thresholds before it runs, and changes one thing against the current best model. The main metric is the macro-averaged F1 score, which weights every class equally regardless of its size:");
EQ("F1~macro~ = (1/K) Σ~k~ 2P~k~R~k~ / (P~k~ + R~k~),");
P("where *K* is the number of classes and *P*~k~ and *R*~k~ are the precision and recall of class *k*. Accuracy is reported alongside it.");
H2_("3.5 Models and experimental design");
P("We start from a majority-class floor, a decision tree with the entropy criterion (a C4.5 approximation, since scikit-learn {sklearn} has no C4.5 pruning), KNN, logistic regression and a support vector machine (SVM) with an RBF kernel, all at the four timeouts. Improvement experiments at 15 s then add: (i) a signed logarithm before scaling for the distance-based models,");
EQ("φ(x) = sign(x) · ln(1 + |x|),");
P("which compresses timing values that span several orders of magnitude, keeps the code −1 apart from 0 and is defined for negative gaps; (ii) a 300-tree Random Forest; (iii) XGBoost {xgb} and LightGBM {lgbm} with default settings; (iv) Optuna {optuna} tuning with the TPE sampler, 50 trials per search and the same budget for every method type; (v) class balancing with class weights, random oversampling or SMOTE {imblearn, smote}; and (vi) a two-step model for task B (VPN / non-VPN first, then one 7-class model per group). A model that beat the current best at 15 s by at least 0.03, above the largest cross-validation standard deviation of the baselines (0.025), was rerun at the other timeouts.");
H2_("3.6 Model selection and pre-registered comparison set");
P("The final model of each task is the configuration with the highest validation macro F1. A rule set earlier would have broken gaps below 0.03 in favour of a simpler model, but the measured cross-validation standard deviation of the chosen models (0.0028–0.0058) is far below 0.03. The change was made before the test set was opened, and the old rule's picks were scored as well so that its cost is visible. Before the test set was opened we also fixed a comparison set: for each task, the best validation variant of each of seven method types (decision tree, KNN, logistic regression, SVM, Random Forest, XGBoost and LightGBM).");
H2_("3.7 Statistical testing");
P("Intervals are 95% percentile intervals from 10,000 stratified paired bootstrap resamples of the test rows {efron} (rows drawn within each true class, seed 42), so they are conditional on the test set's class counts. Exactly 26 differences in macro F1 were fixed in advance: for each task the final model against the six other method types, plus the old rule's pick in A1 and A2-VPN. Each is a two-sided paired bootstrap test, Holm-corrected {holm} over all 26 at α = 0.05. With 1,000 resamples the smallest possible p-value (0.0020) would exceed Holm's first threshold (0.05 / 26 = 0.0019), which is why 10,000 resamples are used.");
H2_("3.8 Probability calibration");
P("Calibration is measured by the expected calibration error (ECE) {naeini, guo} with *M* = 15 equal-width confidence bins:");
EQ("ECE = Σ~m~ (|B~m~| / N) · |acc(B~m~) − conf(B~m~)|,");
P("where *B*~m~ is the set of predictions whose confidence falls in bin *m*, *N* is the number of rows, and acc and conf are the accuracy and mean confidence in the bin. Isotonic calibration {zadrozny} is learned from shuffled 5-fold out-of-fold predictions on the rows the model is fitted on. The predicted class always stays that of the uncalibrated model, so calibration changes only the reported confidence, never a label or a macro F1 score.");
H2_("3.9 Explainability and reproducibility");
P("Feature use is explained with SHAP {shap} on the test rows and, separately, with a feature-group ablation on the validation rows, which drops one of six groups (flow, forward and backward IAT, active, idle, and rates with duration) at a time. Every run is logged to MLflow {mlflow} with its commit, dependency lock hash, data version and hash, settings, metrics, timing and trained model, and official runs are refused unless the code is committed and pushed. A fresh clone of the repository reproduced the final scores, predictions and calibration errors exactly. A command-line predictor and a Streamlit {streamlit} demo apply the four final models to a new flow file, report the label, its calibrated confidence and the features behind it, and reject a malformed file. In total the study comprises 2,015 tracked runs.");

// =====================================================================
// 4 RESULTS AND DISCUSSION
// =====================================================================
H("4. Results and Discussion");
H2_("4.1 Reproduction of the original study (RQ2)");
P("Under the original protocol (10-fold cross-validation on the full published files, Weka defaults {weka}) our C4.5 approximation matches all 16 A1 precision and recall values of the original study within 0.05, and 1-NN matches 15 of 16. Table 3 shows the 15 s values. Contrary to our hypothesis, removing duplicates did not lower these scores: the removed rows are near-empty flows (median 0–2 packets), and 39–92% of them have a twin with the same features and another label, so keeping them adds errors that no model can avoid. The duplicate rule is kept because it guarantees that no flow is scored after being trained on.");
TABLE("Task A1 at 15 s under the original 10-fold protocol: original study {draper} against our reproduction.",
  ["Algorithm", "Class", "Precision (orig.)", "Precision (ours)", "Recall (orig.)", "Recall (ours)"],
  [
    ["C4.5", "VPN", "0.890", "0.892", "0.917", "0.896"],
    ["C4.5", "Non-VPN", "0.906", "0.886", "0.877", "0.882"],
    ["KNN", "VPN", "0.848", "0.863", "0.861", "0.830"],
    ["KNN", "Non-VPN", "0.846", "0.821", "0.832", "0.857"],
  ], [1500, 1300, 1600, 1550, 1500, 1576],
  { note: "Original values other than the C4.5 and KNN precisions stated in the text are read from the original chart (±0.01)." });

H2_("4.2 Model development on validation");
P("Table 4 follows the best 15 s validation macro F1 through the experiments, and Fig. 2 shows the progress per task. A 300-tree Random Forest added +0.0302 to +0.0647 over the decision tree, and default gradient boosting added a little more. The signed logarithm of Eq. (5) lifted KNN by 0.080–0.104 in every task and was chosen by every logistic-regression and SVM search that could use it, because without it a few very large gaps decide every distance. Equal tuning added at most +0.0151 to an ensemble but lifted the SVM by +0.3243 to +0.5277, to within 0.0017–0.0186 of each task's best. Class balancing moved macro F1 by at most 0.0064, and the two-step model for task B scored 0.8318 against 0.8428 for the direct model, with more errors across the VPN line (191 against 180).");
TABLE("Best 15 s validation macro F1 after each step (majority-class floor: 0.341, 0.060, 0.070 and 0.019).",
  ["Step", "A1", "A2-VPN", "A2-NonVPN", "B"],
  [
    ["Decision tree (C4.5 approx.)", "0.8901", "0.8247", "0.8294", "0.7585"],
    ["KNN with signed log", "0.9048", "0.8544", "0.8670", "0.7644"],
    ["Random Forest", "0.9203", "0.8679", "0.8941", "0.8147"],
    ["XGBoost, default", "0.9271", "0.8758", "0.9015", "0.8303"],
    ["Final choice", "**0.9308**", "**0.8830**", "**0.9075**", "**0.8428**"],
  ], [3026, 1500, 1500, 1500, 1500]);
FIGURE(FIG + "progress_chart.png", 560, "Best validation macro F1 so far per task and experiment, with the final model's test score (star).");

H2_("4.3 Final test performance (RQ3)");
P("Table 5 gives the final models on the test rows; their validation scores are the last row of Table 4. Gradient-boosted trees were selected in all four tasks. Three tasks score above validation and one below. One pre-stated check failed: the A2-NonVPN test score is 0.0328 above validation, more than the 0.03 we allowed. A split-seed experiment run before the test set was opened had already shown that its official validation rows are harder than any of 10 resplits (0.9075 against 0.9257–0.9468). The two other checks held: the final model scored above the old rule's pick in all four tasks, and the order of the seven method types by test score followed their order by validation score (Spearman {spearman} ρ = 0.9643, 0.8214, 0.9286 and 0.9643).");
TABLE("Final models on the test rows (macro F1 with 95% bootstrap interval; ECE after calibration).",
  ["Task", "Model", "Test macro F1", "95% interval", "Accuracy", "ECE"],
  [
    ["A1", "LightGBM, tuned", "0.9434", "0.9348–0.9519", "0.9435", "0.0090"],
    ["A2-VPN", "LightGBM, class weights", "0.8652", "0.8431–0.8866", "0.8955", "0.0169"],
    ["A2-NonVPN", "LightGBM, oversampling", "0.9403", "0.9216–0.9573", "0.9609", "0.0132"],
    ["B", "XGBoost, tuned", "0.8480", "0.8305–0.8651", "0.8798", "0.0158"],
  ], [1300, 2300, 1400, 1700, 1250, 1076], { leftCols: [1] });
P("Fig. 3 shows the normalised confusion matrices. The voice classes are the easiest (VPN-VoIP F1 0.9941, VoIP 0.9964 and 0.9917), and the weakest are VPN-MAIL (0.7132 in A2-VPN, 0.7259 in B) and STREAMING in A2-NonVPN (0.8955). In task B the largest confusion is BROWSING predicted as VPN-BROWSING (28 flows) and the reverse (17), and 55.05% of the direct model's validation errors cross the VPN line: separating VPN from non-VPN traffic is the hard part of task B.");
FIGURE(FIG + "final_confusion_matrices.png", 580, "Normalised confusion matrices of the four final models on the test rows (share of each true class).");

H2_("4.4 Comparison of method types");
P("Table 6 and Fig. 4 give every member of the comparison set on the same test rows. Of the 26 pre-stated tests, 14 are significant after Holm correction. The final model is clearly better than logistic regression in every task and than the decision tree in A1, A2-NonVPN and B, but it is not clearly different from the tuned SVM in any task, nor from XGBoost or LightGBM except A1 against XGBoost. Two comparison models scored slightly higher on test, neither significantly: the A2-VPN SVM (0.8670) and the B LightGBM (0.8567); nothing was swapped after the test set was seen. Two tests sit on the Holm line (Holm p = 0.0504, Monte Carlo band about 0.03–0.07) and count as not significant. The old rule's picks scored 0.0322 (A1, significant), 0.0211, 0.0242 (significant) and 0.0069 below the final models.");
TABLE("Test macro F1 of the seven method types (bold: final model of the task).",
  ["Task", "Tree", "KNN", "LogReg", "SVM", "RF", "XGB", "LGBM"],
  [
    ["A1", "0.8982", "0.9208", "0.6804", "0.9311", "0.9289", "0.9304", "**0.9434**"],
    ["A2-VPN", "0.8308", "0.8431", "0.7179", "0.8670", "0.8594", "0.8577", "**0.8652**"],
    ["A2-NonVPN", "0.8825", "0.9160", "0.8053", "0.9244", "0.9067", "0.9346", "**0.9403**"],
    ["B", "0.7886", "0.8253", "0.6089", "0.8411", "0.8403", "**0.8480**", "0.8567"],
  ], [1460, 1081, 1081, 1081, 1081, 1081, 1081, 1080]);
FIGURE(FIG + "e12_comparison_test.png", 560, "Test macro F1 with 95% intervals (filled) and validation macro F1 (open) of every comparison model; the final model of each task is shown in red.");

H2_("4.5 Calibration, stability and training size");
P("All four final models were overconfident on validation (Fig. 5). A rule fixed before the test set adopted isotonic calibration where the uncalibrated validation ECE exceeded 0.05 (A1 and A2-VPN). After the test set had been opened, A2-NonVPN and B were calibrated in the same way; this is a post-hoc change, it changed no label or score, and the raw values remain reported in Table 7. Across 10 resplits of the training and validation rows, the validation macro F1 of the final configurations varies with a standard deviation of 0.0042–0.0114. The learning curve still rises between 80% and 100% of the training rows in every task, most in B (0.8297 to 0.8428), so more data would probably help.");
TABLE("Expected calibration error, uncalibrated (raw) and after isotonic calibration.",
  ["Task", "Val. ECE raw", "Val. ECE isotonic", "Test ECE raw", "Test ECE used", "Decided"],
  [
    ["A1", "0.0552", "0.0140", "0.0442", "0.0090", "before test"],
    ["A2-VPN", "0.0714", "0.0168", "0.0709", "0.0169", "before test"],
    ["A2-NonVPN", "0.0444", "0.0152", "0.0339", "0.0132", "after test"],
    ["B", "0.0350", "0.0099", "0.0420", "0.0158", "after test"],
  ], [1400, 1450, 1650, 1450, 1450, 1626]);
FIGURE(FIG + "e15_reliability.png", 430, "Reliability diagrams of the final models on the validation rows (15 bins): uncalibrated against isotonic.");

H2_("4.6 What the models use");
P("SHAP values of the final models on the test rows and the feature-group ablation on validation answer two different questions: which features a model uses, and which it cannot do without (Fig. 6). The byte rate is the most used feature in A2-VPN, A2-NonVPN and B, and the smallest gaps between packets are in the top five of every task. Rates and duration hold 21.4–26.9% of mean |SHAP|, and dropping them costs the most in A2-VPN, A2-NonVPN and B (0.0395, 0.0301 and 0.0549); in A1, dropping the backward gaps costs more (0.0155 against 0.0122). Active and idle times matter least (3.1–5.7% of mean |SHAP|; dropping either group changes macro F1 by −0.0036 to +0.0051). The flow inter-arrival times hold the largest SHAP share in three tasks, yet dropping them costs at most 0.0166, probably because the strongly correlated forward and backward gaps carry much the same information. A timing-only model without the byte rate loses 0.015–0.044, so the models read no payload but do use the amount of data per second.");
FIGURE(FIG + "shap_groups_vs_e09.png", 600, "Share of mean |SHAP| per feature group on the test rows (left) and the change in validation macro F1 when the group is dropped (right).");

H2_("4.7 Re-examining a published result (RQ4)");
P("On the stacked copy used by Isnan *et al.* {isnan}, with their deduplication, 80/20 split and grid search, a Random Forest reaches 0.8796 accuracy against their 0.8801 (Table 8). Keeping the timeouts apart lowers the same method to 0.8206–0.8658 (mean 0.8430). Stacking the four files, however, gives 2.3–4.3 times as many training rows, so we also trained the mixed-timeout forest on training sets cut to each per-timeout size: it then scores 0.8395 on average, no higher than the per-timeout models. The gap therefore comes from training-set size, not from mixing timeouts itself, and the stacked result is still not comparable with a single-timeout score. Our tuned task-B XGBoost beats the forest at every timeout (mean 0.8647; Fig. 7).");
TABLE("Reproduction of Isnan *et al.* on the stacked copy (test accuracy, 14 classes).",
  ["Setup", "Training rows", "Accuracy"],
  [
    ["Isnan *et al.*, published", "32,789", "0.8801"],
    ["Their setup, our run", "32,789", "0.8796"],
    ["Same forest, per timeout (mean)", "7,671–14,459", "0.8430"],
    ["Mixed timeouts, cut to per-timeout size (mean)", "7,671–14,459", "0.8395"],
    ["Our tuned XGBoost, per timeout (mean)", "7,671–14,459", "0.8647"],
  ], [5026, 2200, 1800]);
FIGURE(FIG + "e16_isnan.png", 540, "Test accuracy per timeout of the Isnan *et al.* forest and our tuned XGBoost, against their mixed-timeout setup and the published value.");

H2_("4.8 Discussion");
P("**Data validity (RQ1).** The headers of 9 of the 12 A1 and B files are shifted by one feature pair and the A2 files hold different features; the mapping is unambiguous because the identities of Eqs. (1)–(3) hold on 100% of rows under one reading and almost never under the other. Any study that takes the headers at face value reads the standard deviations of the gaps as their totals, which does not necessarily change accuracy but does change any interpretation of the features.");
P("**Reproducibility (RQ2).** The original A1 results reproduce within 0.05 under the original protocol, so every later improvement starts from a baseline that agrees with the published one.");
P("**Performance (RQ3).** Most of the gain over the original algorithms comes from ensembles and boosting, while tuning mattered mainly for the SVM, which moved from one of the weakest models to one of the strongest. Choosing by the highest validation score instead of a fixed tie margin cost nothing on the test rows and gained a significant margin in two tasks. Class balancing and the two-step task-B model did not help: the weak classes appear weak not because they are small but because their flows look like those of other classes, and the two-step model inherits the hardest decision, VPN or not, as its first step. The most confident errors come from a byte rate typical of another class (for example a VPN-P2P flow predicted as VPN-FT with confidence 0.9916), so a calibrated confidence is reliable on average but does not prove an individual prediction right.");
P("**Comparability (RQ4).** The 88.01% result reproduces, and its advantage over single-timeout scores is explained by a larger training set. Scores obtained on stacked timeouts should therefore be compared only with scores from training sets of the same size.");
H2_("4.9 Limitations and threats to validity");
BUL("**Data.** The data were captured in a laboratory in 2016 with specific applications and one VPN setup, and the captures have documented quality problems {jorgensen}; label errors in the source are errors in our training and test rows alike.");
BUL("**No session-aware split.** The files carry no session or capture identifiers. Holding out contiguous blocks of the file-ordered training rows instead of shuffled rows lowers out-of-fold accuracy by 0.07–0.26, which suggests that neighbouring flows, probably from the same capture, resemble each other; random-split scores may therefore be optimistic for traffic from new captures.");
BUL("**Timeouts are confounded.** The 60 s files hold different VoIP and file-transfer flows from the other timeouts, and the class mix changes with the timeout, so score differences between timeouts are not read as a timeout effect.");
BUL("**Approximation and deviations.** The decision tree only approximates C4.5. Two deviations from rules set in advance are reported as such: the final choice by the highest validation score, made before the test set was opened, and the post-hoc calibration of A2-NonVPN and B, which used their test rows a second time for confidence only.");
BUL("**Statistical scope.** Every interval is conditional on the test set's class counts, two of the 26 tests are borderline, and one test set is one sample.");

// =====================================================================
// 5 CONCLUSION
// =====================================================================
H("5. Conclusion");
P("This paper re-evaluated payload-free classification of encrypted and VPN traffic on ISCXVPN2016 using only 23 flow timing statistics. A check of the published files showed that most of them label their columns incorrectly, and we recovered the correct mapping from value identities that hold on every row. Under the original protocol, the original C4.5 and KNN results reproduce. Under a leakage-safe protocol that removes duplicates before splitting, makes every choice on training and validation rows and scores the test set once, gradient-boosted trees reach a test macro F1 of 0.9434 for VPN detection, 0.8652 and 0.9403 for the traffic type within VPN and non-VPN traffic, and 0.8480 for all 14 classes, each with a 95% interval and well-calibrated confidence. The byte rate and the shortest inter-packet gaps carry most of the signal, while active and idle times add little. A recent 88.01% result on stacked timeouts was reproduced, and its advantage was traced to training-set size rather than to mixing timeouts.");
P("These results hold for this dataset only. Future work should (i) evaluate on a session-aware split using data that carry capture identifiers, (ii) test the models on traffic captured independently of ISCXVPN2016, including current applications and VPN protocols, and (iii) rebuild flows from the packet captures to remove the dependence on the published feature files. These steps are needed before any of these models is used on a real network. The code, data versions, all 2,015 runs and every decision are kept in the project repository.");

// =====================================================================
// REFERENCES
// =====================================================================
HU("References");
refOrder.forEach((k, i) => {
  body.push(new Paragraph({
    alignment: AlignmentType.LEFT, spacing: { after: 80, line: 276 },
    indent: { left: 567, hanging: 567 },
    children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: BODY }), ...runs(REFS[k])],
    tabStops: [{ type: TabStopType.LEFT, position: 567 }],
  }));
});
const unused = Object.keys(REFS).filter((k) => !refOrder.includes(k));
if (unused.length) console.warn("Uncited references:", unused);

// ---------------- document ----------------
const doc = new Document({
  creator: "BlindSpy Team",
  title: "Payload-Free Classification of VPN and Encrypted Traffic from Flow Timing Features",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: H1, bold: true, color: "000000" },
        paragraph: { spacing: { before: 300, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: H2, bold: true, color: "000000" },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
    { reference: "rq", levels: [{ level: 0, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SMALL })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(path.join(HERE, "BlindSpy_research_paper.docx"), b); console.log("written, refs:", refOrder.length); });
