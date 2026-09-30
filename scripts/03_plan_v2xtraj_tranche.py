#!/usr/bin/env python3
"""Freeze a deterministic, complete 100-scenario V2X-Traj review tranche."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic

INDEX = ROOT / "data" / "manifests" / "v2xtraj_archive_index_v1.json"
INVENTORY = ROOT / "data" / "manifests" / "v2xtraj_intersection_inventory_v1.json"
OUTPUT = ROOT / "data" / "manifests" / "v2xtraj_tranche_v1.json"
SALT = "bces-iov-v2xtraj-primary-v1"
QUOTAS = {
    "yizhuang#11-1_po": 13,
    "yizhuang#12-1_po": 13,
    "yizhuang#13-1_po": 13,
    "yizhuang#14-1_po": 12,
    "yizhuang#20-1_po": 13,
    "yizhuang#25-1_po": 12,
    "yizhuang#4-1_po": 12,
    "yizhuang#7-1_po": 12,
}
ROLE_DIRS = {
    "ego": "ego-trajectories",
    "vehicle": "vehicle-trajectories",
    "infrastructure": "infrastructure-trajectories",
    "traffic_light": "traffic-light",
}


def _verified(path: Path, hash_field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(hash_field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path.name}")
    return payload


def main() -> int:
    index = _verified(INDEX, "index_sha256")
    inventory = _verified(INVENTORY, "inventory_sha256")
    by_name = {item["name"]: item for item in index["members"]}
    selected = []
    for intersection, quota in QUOTAS.items():
        candidates = [item for item in inventory["scenarios"] if item["intersection_id"] == intersection]
        candidates.sort(key=lambda item: hashlib.sha256(f"{SALT}:{item['scenario_id']}".encode()).hexdigest())
        if len(candidates) < quota:
            raise ValueError(f"insufficient scenarios at {intersection}")
        for item in candidates[:quota]:
            source = "val" if item["source_partition"] == "validation" else "train"
            members = {
                role: f"v2x-traj/{directory}/{source}/data/{item['scenario_id']}.csv"
                for role, directory in ROLE_DIRS.items()
            }
            missing = sorted(set(members.values()) - set(by_name))
            if missing:
                raise ValueError(f"incomplete scenario bundle: {missing}")
            selected.append({**item, "members": members})
    maps = {}
    for intersection in QUOTAS:
        number = re.search(r"#(\d+)-", intersection)
        assert number is not None
        name = f"v2x-traj/maps/yizhuang_hdmap{number.group(1)}.json"
        if name not in by_name:
            raise ValueError(f"map member is absent: {name}")
        maps[intersection] = name
    member_names = [name for item in selected for name in item["members"].values()] + list(maps.values())
    compressed_bytes = sum(int(by_name[name]["compressed_bytes"]) for name in member_names)
    uncompressed_bytes = sum(int(by_name[name]["uncompressed_bytes"]) for name in member_names)
    # The extractor performs a 30-byte local-header range plus the exact compressed
    # member range, so transfer accounting is exact and does not include ZIP names.
    request_overhead_allowance = 30 * len(member_names)
    maximum = 250_000_000
    if compressed_bytes + request_overhead_allowance > maximum:
        raise ValueError(
            "review tranche exceeds the registered transfer cap: "
            f"{compressed_bytes + request_overhead_allowance} > {maximum}"
        )
    report = {
        "schema_version": 1,
        "dataset_id": "v2xtraj_primary",
        "selection_salt": SALT,
        "selection_rule": "SHA-256 rank within fixed intersection quotas",
        "scenario_count": len(selected),
        "intersection_count": len(QUOTAS),
        "intersection_quotas": QUOTAS,
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "request_overhead_allowance_bytes": request_overhead_allowance,
        "maximum_transfer_bytes": maximum,
        "maximum_initial_review_bytes": 500_000_000,
        "maps": maps,
        "scenarios": sorted(selected, key=lambda item: int(item["scenario_id"])),
    }
    if uncompressed_bytes > report["maximum_initial_review_bytes"]:
        raise ValueError("review tranche exceeds the registered initial-review cap")
    report["tranche_sha256"] = canonical_json_hash(report)
    write_json_atomic(OUTPUT, report)
    print(json.dumps({key: value for key, value in report.items() if key != "scenarios"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
