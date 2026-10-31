"""Build the E16 data version from the Kaggle CSV: data/processed/kaggle/<version>/.

Usage: uv run python scripts/prepare_kaggle.py --version k1
Refuses to overwrite an existing version. Copies the manifest to data/kaggle_<version>_manifest.json
(committed). See src/blindspy/kaggle.py.
"""

import argparse
import json

from blindspy import kaggle
from blindspy.data import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    m = kaggle.build(ap.parse_args().version)
    path = ROOT / "data" / f"kaggle_{m['data_version'].removeprefix('kaggle-')}_manifest.json"
    path.write_text(json.dumps(m, indent=1) + "\n")
    print(
        f"Built {m['data_version']}: {m['rows']} rows; isnan split {m['isnan']['train']} / "
        f"{m['isnan']['test']} after dedup {m['isnan']['after_dedup']}; manifest {path.name}"
    )


if __name__ == "__main__":
    main()
