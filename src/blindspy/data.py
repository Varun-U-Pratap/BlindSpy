"""Load the ISCXVPN2016 ARFF files into DataFrames."""

import io
from pathlib import Path

import arff
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "CSVs"

_SHARED_TAIL = [
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]  # fmt: skip

# The two feature-name lists the 20 ARFF headers use (23 names each).
# "T" (17 files) has total_fiat/total_biat; "S" (the three 60 s files of A1 and B)
# has std_fiat/std_biat instead, as in the paper's Table 2.
HEADERS = {
    "T": [
        "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
        "max_fiat", "max_biat", "mean_fiat", "mean_biat", *_SHARED_TAIL,
    ],
    "S": [
        "duration", "min_fiat", "min_biat", "max_fiat", "max_biat",
        "mean_fiat", "mean_biat", "std_fiat", "std_biat", *_SHARED_TAIL,
    ],
}  # fmt: skip


def _clean_text(text: str) -> tuple[str, int]:
    """Fix the export glitches liac-arff can't read.

    Strips the stray trailing commas from every line up to and including @DATA, and
    drops data lines made only of commas (the 60 s files of A1 and B have one right
    after @DATA). Returns the fixed text and the number of comma-only data lines dropped.
    """
    head, sep, body = text.partition("@DATA")
    if not sep:
        raise ValueError("no @DATA line")
    lines = [line.rstrip().rstrip(",") for line in head.splitlines()]
    data_line_rest, _, body = body.partition("\n")
    if data_line_rest.strip(", \r"):
        raise ValueError(f"unexpected text after @DATA: {data_line_rest!r}")
    rows = body.splitlines()
    kept = [row for row in rows if row.strip(", \r")]
    header = "\n".join(line for line in lines if line)
    return header + "\n@DATA\n" + "\n".join(kept) + "\n", len(rows) - len(kept)


# What the 23 columns really hold, per file family (verified in E00 on every row of
# all 20 files). A1 and B files hold the paper's Table 2 features (fiat/biat as
# min, max, mean, std); A2 files hold total_fiat/total_biat (sum of the gaps) instead
# of the two std columns. 9 of the 12 A1/B files (all but the 60 s ones) carry the
# "T" header, whose fiat/biat names are shifted by one pair, so they are renamed here.
FEATURES = {"A1": HEADERS["S"], "B": HEADERS["S"], "A2": HEADERS["T"]}
COLUMN_MAP = {
    (family, kind): dict(zip(HEADERS[kind], FEATURES[family], strict=True))
    for family, kind in [("A1", "T"), ("A1", "S"), ("B", "T"), ("B", "S"), ("A2", "T")]
}


def header_kind(names: list[str]) -> str:
    """Which of HEADERS a file's 23 feature names are ("T" or "S")."""
    for kind, header in HEADERS.items():
        if names == header:
            return kind
    raise ValueError(f"unknown header {names}")


def load_arff(path: Path) -> pd.DataFrame:
    """Parse one ARFF file: 23 numeric columns under their header names + `label` (str).

    The columns keep the names the file's header gives them; see COLUMN_MAP for
    what each column really holds. `df.attrs["blank_lines_dropped"]` counts the
    comma-only data lines skipped.
    """
    text, blank = _clean_text(Path(path).read_text())
    parsed = arff.load(io.StringIO(text))
    names = [name for name, _ in parsed["attributes"]]
    if names[-1] != "class1":
        raise ValueError(f"{path}: last attribute is {names[-1]!r}, expected class1")
    features = HEADERS[header_kind(names[:-1])]
    df = pd.DataFrame(parsed["data"], columns=[*features, "label"])
    df[features] = df[features].astype("float64")
    df["label"] = df["label"].astype(str)
    df.attrs["blank_lines_dropped"] = blank
    return df


TIMEOUTS = [15, 30, 60, 120]
_A1, _A2, _B = "Scenario A1-ARFF", "Scenario A2-ARFF", "Scenario B-ARFF"
_NAME = "TimeBasedFeatures-Dataset-{t}s{suffix}.arff"

# task -> (file family, folder, file-name suffix). B-AllinOne is inventoried but is
# not one of the three core tasks.
TASK_FILES = {
    "A1": ("A1", _A1, "-VPN"),
    "A2-VPN": ("A2", _A2, "-VPN"),
    "A2-NonVPN": ("A2", _A2, "-NO-VPN"),
    "B": ("B", _B, ""),
    "B-AllinOne": ("B", _B, "-AllinOne"),
}
CORE_TASKS = ["A1", "A2-VPN", "A2-NonVPN", "B"]

_TYPES = ["BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP", "P2P", "FT"]
LABELS = {
    "A1": ["Non-VPN", "VPN"],
    "A2-VPN": [f"VPN-{t}" for t in _TYPES],
    "A2-NonVPN": _TYPES,
    "B": _TYPES + [f"VPN-{t}" for t in _TYPES],
    "B-AllinOne": _TYPES,
}


def raw_path(task: str, timeout: int) -> Path:
    _, folder, suffix = TASK_FILES[task]
    return RAW / folder / _NAME.format(t=timeout, suffix=suffix)


def load_raw(task: str, timeout: int) -> pd.DataFrame:
    """One task file parsed and renamed with COLUMN_MAP (values untouched)."""
    family = TASK_FILES[task][0]
    df = load_arff(raw_path(task, timeout))
    kind = header_kind(list(df.columns[:-1]))
    return df.rename(columns=COLUMN_MAP[(family, kind)])


def processed_dir(data_version: str, root: Path = ROOT) -> Path:
    return root / "data" / "processed" / data_version


def load_task(
    task: str, timeout: int, data_version: str = "v1", root: Path = ROOT
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cleaned train / validation / test DataFrames for one task and timeout.

    Columns are the verified feature names + `label`; the index is `raw_row`, the
    row's position in the source ARFF file.
    """
    from blindspy.splits import load_split

    folder = processed_dir(data_version, root)
    df = pd.read_parquet(folder / f"{task}_{timeout}s.parquet")
    split = load_split(folder, task, timeout)
    return tuple(df.loc[split[part]] for part in ("train", "val", "test"))
