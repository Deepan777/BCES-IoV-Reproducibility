#!/usr/bin/env python3
"""Inspect the official CooperScene mini ZIP by bounded HTTP byte ranges."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.remote_zip import inspect_remote_zip
from bces.utils.reproducibility import write_json_atomic

URL = "https://data.ucr.edu/datasets/cooperscene/mini.zip"
OUTPUT = ROOT / "data/manifests/cooperscene_mini_remote_index_v1.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("CooperScene remote index already exists")
    index = inspect_remote_zip(URL, maximum_metadata_bytes=20_000_000)
    index.update({"official_url": URL,
        "official_license_url": "https://github.com/UCR-CISL/CooperScene/blob/main/DATA_LICENSE",
        "acquisition_status": "metadata_only_no_member_download"})
    write_json_atomic(OUTPUT, index)
    top = Counter(member["name"].split("/")[0] for member in index["members"])
    suffix = Counter(Path(member["name"]).suffix.lower() for member in index["members"])
    print(json.dumps({"archive_bytes": index["archive_bytes"],
        "metadata_range_bytes_consumed": index["metadata_range_bytes_consumed"],
        "member_count": index["member_count"], "top_level": top.most_common(20),
        "suffixes": suffix.most_common(20)}, indent=2))


if __name__ == "__main__":
    main()
