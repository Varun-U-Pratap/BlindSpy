"""Archive every text artifact of the official MLflow store, so it is kept in git.

The run store (mlruns/<runner>/) is git-ignored: its database is exported to
reports/history.csv by export_history.py, but the files saved next to each run (confusion
matrices, per-class reports, confusion pairs, feature importances, test and validation
predictions, Optuna trial tables, configs) would otherwise exist only on the official
laptop. This script packs every .csv, .json, .yaml, .txt and .md artifact into
reports/run_artifacts_text.tar.gz (path inside: <run_id>/<artifact path>) and lists each
file in reports/run_artifacts_index.csv with its run name, experiment, size and SHA-256.
Plots (.png) and model files (.joblib) are left out: they are large, and both are kept in
the milestone backup zips (models/final/ holds the four final models).

The archive is byte-reproducible: sorted members, fixed timestamps and owners.

Usage: uv run python scripts/export_artifacts.py
"""

import gzip
import hashlib
import io
import sqlite3
import tarfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from blindspy.tracking import ROOT, load_yaml

TEXT_SUFFIXES = {".csv", ".json", ".yaml", ".txt", ".md"}
ARCHIVE = ROOT / "reports" / "run_artifacts_text.tar.gz"
INDEX = ROOT / "reports" / "run_artifacts_index.csv"


def runs(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(db)
    q = ("select r.run_uuid as run_id, r.name as run_name, r.artifact_uri, "
         "t.value as experiment_id from runs r left join tags t "
         "on t.run_uuid = r.run_uuid and t.key = 'experiment_id'")  # fmt: skip
    return pd.read_sql_query(q, con).sort_values("run_id").reset_index(drop=True)


def text_files(table: pd.DataFrame) -> list[dict]:
    rows = []
    for r in table.itertuples():
        folder = Path(unquote(urlparse(r.artifact_uri).path))
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix in TEXT_SUFFIXES:
                data = p.read_bytes()
                rows.append({"run_id": r.run_id, "run_name": r.run_name,
                             "experiment_id": r.experiment_id,
                             "path": f"{r.run_id}/{p.relative_to(folder).as_posix()}",
                             "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                             "data": data})  # fmt: skip
    return sorted(rows, key=lambda x: x["path"])


def write_archive(files: list[dict], out: Path) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for f in files:
                info = tarfile.TarInfo(f["path"])
                info.size, info.mtime, info.mode = f["bytes"], 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(f["data"]))
    out.write_bytes(buf.getvalue())


def main() -> None:
    runner = load_yaml(ROOT / "configs" / "base.yaml")["official_runner"]
    db = ROOT / "mlruns" / runner / "mlflow.db"
    if not db.exists():
        raise SystemExit(
            f"No MLflow store at {db.relative_to(ROOT)}: run it on the official laptop"
        )
    table = runs(db)
    files = text_files(table)
    write_archive(files, ARCHIVE)
    index = pd.DataFrame(files).drop(columns="data")
    index.to_csv(INDEX, index=False)
    print(f"Wrote {ARCHIVE.relative_to(ROOT)} ({ARCHIVE.stat().st_size:,} bytes) and "
          f"{INDEX.relative_to(ROOT)}: {len(index):,} files from {index['run_id'].nunique():,} "
          f"of {len(table):,} runs, {index['bytes'].sum():,} bytes before compression")  # fmt: skip


if __name__ == "__main__":
    main()
