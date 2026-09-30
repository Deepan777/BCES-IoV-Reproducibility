#!/usr/bin/env python3
"""Build the exact V2X-Traj scenario/intersection inventory from a bounded range."""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.remote_zip import RemoteZipMember, download_contiguous_members
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic

INDEX = ROOT / "data" / "manifests" / "v2xtraj_archive_index_v1.json"
OUTPUT = ROOT / "data" / "manifests" / "v2xtraj_intersection_inventory_v1.json"


def main() -> int:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    expected = index.pop("index_sha256")
    if canonical_json_hash(index) != expected:
        raise ValueError("archive index hash does not verify")
    prefixes = (
        "v2x-traj/traffic-light/train/data/",
        "v2x-traj/traffic-light/val/data/",
    )
    members = [
        RemoteZipMember(**item)
        for item in index["members"]
        if item["name"].startswith(prefixes) and item["name"].endswith(".csv")
    ]
    payloads, consumed = download_contiguous_members(
        requests.Session(),
        index["official_url"],
        members,
        archive_bytes=index["archive_bytes"],
        maximum_range_bytes=25_000_000,
    )
    scenarios = []
    empty_scenarios = []
    for member in members:
        reader = csv.DictReader(io.StringIO(payloads[member.name].decode("utf-8-sig")))
        row = next(reader, None)
        if row is None:
            empty_scenarios.append(Path(member.name).stem)
            continue
        scenarios.append(
            {
                "scenario_id": Path(member.name).stem,
                "source_partition": "validation" if "/val/" in member.name else "train",
                "intersection_id": row["intersect_id"],
            }
        )
    counts = Counter(item["intersection_id"] for item in scenarios)
    report = {
        "schema_version": 1,
        "dataset_id": "v2xtraj_primary",
        "inventory_source": "all official train/validation traffic-light members",
        "range_bytes_consumed": consumed,
        "scenario_count": len(scenarios),
        "empty_traffic_light_scenario_count": len(empty_scenarios),
        "empty_traffic_light_scenarios": sorted(empty_scenarios, key=int),
        "intersection_count": len(counts),
        "intersection_scenario_counts": dict(sorted(counts.items())),
        "scenarios": sorted(scenarios, key=lambda item: (item["intersection_id"], int(item["scenario_id"]))),
    }
    report["inventory_sha256"] = canonical_json_hash(report)
    write_json_atomic(OUTPUT, report)
    print(json.dumps({key: value for key, value in report.items() if key != "scenarios"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
