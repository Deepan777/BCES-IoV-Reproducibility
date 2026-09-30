#!/usr/bin/env python3
"""Index the official V2X-Traj ZIP without downloading the full archive."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.remote_zip import inspect_remote_zip
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic

URL = "https://drive.usercontent.google.com/download?id=1-5cPcZfGx1L58aiNiHUGoJOkoTSojFtF&export=download&confirm=t"


def main() -> int:
    index = inspect_remote_zip(URL)
    prefixes = Counter(
        PurePosixPath(item["name"]).parts[0]
        for item in index["members"]
        if PurePosixPath(item["name"]).parts
    )
    index.update(
        {
            "schema_version": 1,
            "dataset_id": "v2xtraj_primary",
            "official_url": URL,
            "top_level_member_counts": dict(sorted(prefixes.items())),
        }
    )
    index["index_sha256"] = canonical_json_hash(index)
    output = ROOT / "data" / "manifests" / "v2xtraj_archive_index_v1.json"
    write_json_atomic(output, index)
    summary = {key: value for key, value in index.items() if key != "members"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
