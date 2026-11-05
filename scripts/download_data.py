"""Download and extract the NASA C-MAPSS dataset into data/raw/CMAPSSData/.

Source: NASA Prognostics Center of Excellence data repository, mirrored at
the URL below. The outer zip contains a nested CMAPSSData.zip.
Uses only the standard library.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "CMAPSSData"


def main() -> int:
    if (DEST / "train_FD001.txt").exists():
        print(f"Data already present in {DEST}")
        return 0
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        print(f"SHA-256 mismatch: got {digest}, expected {SHA256}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        inner_bytes = outer.read(INNER)
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for member in inner.namelist():
            name = Path(member).name
            if not name or member.endswith("/"):
                continue
            (DEST / name).write_bytes(inner.read(member))
    print(f"Extracted to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
