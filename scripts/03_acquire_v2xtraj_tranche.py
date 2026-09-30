#!/usr/bin/env python3
"""Acquire the frozen V2X-Traj review tranche with resumable atomic writes."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.remote_zip import RemoteZipMember, download_member
from bces.utils.budget import Budget, build_budget_report, require_headroom
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import write_json_atomic

INDEX = ROOT / "data" / "manifests" / "v2xtraj_archive_index_v1.json"
PLAN = ROOT / "data" / "manifests" / "v2xtraj_tranche_v1.json"
RECEIPT = ROOT / "data" / "manifests" / "v2xtraj_download_receipt_v1.json"
RAW_ROOT = ROOT / "data" / "raw" / "v2xtraj_primary"
MAX_WORKERS = 6
_thread_state = threading.local()


def _verified(path: Path, hash_field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(hash_field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path.name}")
    return payload


def _destination(member_name: str) -> Path:
    parts = PurePosixPath(member_name).parts
    if parts[:2] == ("v2x-traj", "maps"):
        relative = Path("map", parts[-1])
    else:
        role = {
            "ego-trajectories": "ego",
            "vehicle-trajectories": "vehicle",
            "infrastructure-trajectories": "infrastructure",
            "traffic-light": "traffic_light",
        }[parts[1]]
        relative = Path(role, parts[2], parts[-1])
    destination = (RAW_ROOT / relative).resolve()
    destination.relative_to(RAW_ROOT.resolve())
    return destination


def _session() -> requests.Session:
    if not hasattr(_thread_state, "session"):
        _thread_state.session = requests.Session()
    return _thread_state.session


def _download(url: str, member: RemoteZipMember) -> dict:
    destination = _destination(member.name)
    existing_is_valid = False
    if destination.exists():
        if destination.stat().st_size == member.uncompressed_bytes:
            payload_crc = 0
            with destination.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    payload_crc = binascii.crc32(chunk, payload_crc)
            existing_is_valid = payload_crc & 0xFFFFFFFF == member.crc32
        if not existing_is_valid:
            destination.unlink()
    if not existing_is_valid:
        payload = download_member(_session(), url, member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        with part.open("xb") as handle:
            handle.write(payload)
        os.replace(part, destination)
    return {
        "member": member.name,
        "destination": destination.relative_to(ROOT).as_posix(),
        "source_bytes": member.uncompressed_bytes,
        "compressed_bytes": member.compressed_bytes,
        "crc32": f"{member.crc32:08x}",
        "sha256": sha256_file(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--continuation-review", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = _verified(INDEX, "index_sha256")
    plan = _verified(args.plan, "tranche_sha256")
    by_name = {item["name"]: RemoteZipMember(**item) for item in index["members"]}
    names = [name for item in plan["scenarios"] for name in item["members"].values()] + list(plan.get("maps", {}).values())
    if len(names) != len(set(names)):
        raise ValueError("tranche contains duplicate members")
    members = [by_name[name] for name in names]
    transfer_bytes = sum(item.compressed_bytes + 30 for item in members)
    if transfer_bytes > plan["maximum_transfer_bytes"]:
        raise ValueError("planned transfer exceeds the registered cap")
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    current_data_bytes = build_budget_report(budget)["measurements"]["data_bytes"]
    planned_uncompressed_bytes = sum(item.uncompressed_bytes for item in members)
    existing_bytes = sum(_destination(item.name).stat().st_size for item in members if _destination(item.name).exists())
    incoming_bytes = planned_uncompressed_bytes - existing_bytes
    projected_data_bytes = current_data_bytes + incoming_bytes
    if projected_data_bytes > budget.initial_data_review_bytes:
        if args.continuation_review is None:
            raise ValueError("crossing the initial-review checkpoint requires a review record")
        continuation = _verified(
            args.continuation_review, "continuation_review_sha256"
        )
        if continuation.get("source_plan_sha256") != canonical_json_hash(plan):
            raise ValueError("continuation review does not bind this tranche")
        if continuation.get("decision") != "approved_for_one_additional_review_tranche":
            raise ValueError("continuation review does not approve one tranche")
        if continuation.get("planned_uncompressed_member_bytes") != planned_uncompressed_bytes:
            raise ValueError("continuation review planned-data count is stale")
        if current_data_bytes < continuation.get("budget_before", {}).get("data_bytes", 0):
            raise ValueError("current data count predates the continuation review")
    require_headroom(budget, incoming_bytes, preserve_reserve=True, enforce_tranche=False)
    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download, index["official_url"], item): item.name for item in members}
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed % 10 == 0:
                write_json_atomic(
                    args.receipt,
                    {
                        "schema_version": 1,
                        "dataset_id": "v2xtraj_primary",
                        "status": "in_progress",
                        "tranche_sha256": canonical_json_hash(plan),
                        "completed_files": completed,
                        "records": sorted(records, key=lambda item: item["member"]),
                    },
                )
                print(f"completed {completed}/{len(members)}", flush=True)
    receipt = {
        "schema_version": 1,
        "dataset_id": "v2xtraj_primary",
        "tranche_number": int(plan.get("tranche_number", 1)),
        "status": "complete",
        "official_archive_url": index["official_url"],
        "official_archive_bytes": index["archive_bytes"],
        "tranche_sha256": canonical_json_hash(plan),
        "scenario_count": plan["scenario_count"],
        "intersection_count": plan["intersection_count"],
        "transfer_bytes": transfer_bytes,
        "extracted_bytes": sum(item["source_bytes"] for item in records),
        "records": sorted(records, key=lambda item: item["member"]),
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    write_json_atomic(args.receipt, receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
