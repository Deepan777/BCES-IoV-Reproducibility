#!/usr/bin/env python3
"""Bounded sampling of V2X-Traj ego files to map scenarios to intersections."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.remote_zip import RemoteZipMember, download_member
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic

INDEX = ROOT / "data" / "manifests" / "v2xtraj_archive_index_v1.json"
OUTPUT = ROOT / "data" / "manifests" / "v2xtraj_intersection_scan_v1.json"
SAMPLE_COUNT = 84
MAX_COMPRESSED_BYTES = 50_000_000


def main() -> int:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    expected = index.pop("index_sha256")
    if canonical_json_hash(index) != expected:
        raise ValueError("archive index hash does not verify")
    prefixes = (
        "v2x-traj/ego-trajectories/train/data/",
        "v2x-traj/ego-trajectories/val/data/",
    )
    candidates = sorted(
        (RemoteZipMember(**item) for item in index["members"] if item["name"].startswith(prefixes) and item["name"].endswith(".csv")),
        key=lambda item: ("/val/" in item.name, int(Path(item.name).stem)),
    )
    positions = sorted({round(i * (len(candidates) - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    selected = [candidates[position] for position in positions]
    compressed = sum(item.compressed_bytes for item in selected)
    if compressed > MAX_COMPRESSED_BYTES:
        raise ValueError("intersection scan exceeds its byte cap")
    session = requests.Session()
    samples = []
    for member in selected:
        payload = download_member(session, index["official_url"], member)
        row = next(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        samples.append(
            {
                "scenario_id": Path(member.name).stem,
                "intersection_id": row["intersect_id"],
                "compressed_bytes": member.compressed_bytes,
            }
        )
    report = {
        "schema_version": 1,
        "dataset_id": "v2xtraj_primary",
        "sampling": "84 evenly spaced train and validation scenario IDs",
        "compressed_bytes_consumed": compressed,
        "samples": samples,
    }
    report["scan_sha256"] = canonical_json_hash(report)
    write_json_atomic(OUTPUT, report)
    counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample["intersection_id"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"sample_count": len(samples), "compressed_bytes": compressed, "intersection_counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
