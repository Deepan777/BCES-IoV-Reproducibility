#!/usr/bin/env python3
"""Materialize the verified V2X-Traj receipt as a file-level allowlist."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash

MANIFEST_ROOT = ROOT / "data" / "manifests"
OUTPUT = ROOT / "configs" / "data" / "v2xtraj_allowlist.yaml"
WINDOW_REVIEW = ROOT / "data" / "manifests" / "v2xtraj_window_review_v1.json"
LICENSE = "https://github.com/AIR-THU/V2X-Graph/blob/main/LICENSE"


def _verified(path: Path, field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path.name}")
    return payload


def main() -> int:
    tranche_paths = sorted(MANIFEST_ROOT.glob("v2xtraj_tranche_v*.json"))
    receipt_paths = sorted(MANIFEST_ROOT.glob("v2xtraj_download_receipt_v*.json"))
    if not tranche_paths or len(tranche_paths) != len(receipt_paths):
        raise ValueError("every numbered tranche requires one numbered receipt")
    tranches = [_verified(path, "tranche_sha256") for path in tranche_paths]
    receipts = [_verified(path, "receipt_sha256") for path in receipt_paths]
    if any(receipt.get("status") != "complete" for receipt in receipts):
        raise ValueError("all tranche receipts must be complete")
    scenario_lookup = {}
    map_lookup = {}
    for tranche in tranches:
        for item in tranche["scenarios"]:
            for member in item["members"].values():
                if member in scenario_lookup:
                    raise ValueError(f"duplicate member across tranches: {member}")
                scenario_lookup[member] = (item["scenario_id"], item["intersection_id"])
        map_lookup.update(
            {member: intersection for intersection, member in tranche.get("maps", {}).items()}
        )
    reviewed_by_scene = {}
    if WINDOW_REVIEW.exists():
        review = _verified(WINDOW_REVIEW, "review_sha256")
        reviewed_by_scene = {
            str(item["scenario_id"]): item for item in review.get("records", ())
        }
    files = []
    for record in (item for receipt in receipts for item in receipt["records"]):
        member = record["member"]
        destination = record["destination"].removeprefix("data/raw/v2xtraj_primary/")
        if member in map_lookup:
            role = "map"
            intersection = map_lookup[member]
            scene = f"map-{intersection}"
        else:
            scene, intersection = scenario_lookup[member]
            role = destination.split("/", 1)[0]
        reviewed = reviewed_by_scene.get(str(scene), {})
        files.append(
            {
                "entry_id": "v2xtraj-" + hashlib.sha256(member.encode()).hexdigest()[:16],
                "url": receipts[0]["official_archive_url"],
                "archive_member": member,
                "destination": destination,
                "role": role,
                "scene_id": str(scene),
                "intersection_id": intersection,
                "selection_reason": "deterministic primary review tranche covering frozen whole-intersection partitions",
                "release": "official-archive-2025-03-31",
                "license_url": LICENSE,
                "expected_bytes": record["source_bytes"],
                "expected_sha256": record["sha256"],
                "expected_valid_windows": (
                    None if role == "map" else int(reviewed.get("valid_windows", 1))
                ),
                "behavior_families": (
                    [] if role == "map" else list(reviewed.get("behavior_families", ()))
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "dataset_id": "v2xtraj_primary",
        "evidence_role": "primary",
        "status": "ready_review_tranche",
        "reviewed_utc_date": "2026-09-04",
        "official_sources": [
            {
                "provenance_url": "https://github.com/AIR-THU/V2X-Graph",
                "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/1812042b83f20707a898ff6f8af7db84-Paper-Conference.pdf",
                "download_url": receipts[0]["official_archive_url"],
                "advertised_name": "v2x-traj.zip",
                "archive_bytes": receipts[0]["official_archive_bytes"],
                "access_granularity": "verified_http_byte_range_members",
                "license_status": "provisional_apache_2_0_from_official_project",
                "license_url": LICENSE,
            }
        ],
        "blocking_reasons": [],
        "files": sorted(files, key=lambda item: item["entry_id"]),
    }
    OUTPUT.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps({"files": len(files), "status": payload["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
