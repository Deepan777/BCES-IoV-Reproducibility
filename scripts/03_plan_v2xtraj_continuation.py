#!/usr/bin/env python3
"""Freeze exactly one reviewed, non-redundant V2X-Traj continuation tranche."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.budget import Budget, build_budget_report, require_headroom
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic

MANIFESTS = ROOT / "data" / "manifests"
INDEX = MANIFESTS / "v2xtraj_archive_index_v1.json"
INVENTORY = MANIFESTS / "v2xtraj_intersection_inventory_v1.json"
WINDOW_REVIEW = MANIFESTS / "v2xtraj_window_review_v1.json"
SPLIT = MANIFESTS / "v2xtraj_primary_split_v1.json"
TARGET_VALID_WINDOWS = 500
TRANSFER_CAP_BYTES = 250_000_000
REQUEST_OVERHEAD_BYTES = 30
WILSON_Z = 1.96
PARTITION_FRACTIONS = {
    "train": 0.60,
    "calibration": 0.15,
    "validation": 0.10,
    "test": 0.15,
}
ROLE_DIRS = {
    "ego": "ego-trajectories",
    "vehicle": "vehicle-trajectories",
    "infrastructure": "infrastructure-trajectories",
    "traffic_light": "traffic-light",
}


def _verified(path: Path, hash_field: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(hash_field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path.name}")
    return payload, str(expected)


def _numbered_paths(prefix: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.json$")
    result: dict[int, Path] = {}
    for path in MANIFESTS.glob(f"{prefix}_v*.json"):
        match = pattern.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def _completed_history() -> tuple[list[dict[str, Any]], list[str], set[str]]:
    plans = _numbered_paths("v2xtraj_tranche")
    receipts = _numbered_paths("v2xtraj_download_receipt")
    if not receipts:
        raise ValueError("an initial completed tranche is required")
    expected_numbers = list(range(1, max(receipts) + 1))
    if sorted(receipts) != expected_numbers:
        raise ValueError("completed receipt numbering must be contiguous")
    history: list[dict[str, Any]] = []
    hashes: list[str] = []
    downloaded: set[str] = set()
    for number in expected_numbers:
        if number not in plans:
            raise ValueError(f"missing plan for completed tranche {number}")
        plan, plan_hash = _verified(plans[number], "tranche_sha256")
        receipt, receipt_hash = _verified(receipts[number], "receipt_sha256")
        if int(plan.get("tranche_number", 1 if number == 1 else -1)) != number:
            raise ValueError(f"plan number mismatch for tranche {number}")
        if (
            int(receipt.get("tranche_number", 1 if number == 1 else -1)) != number
            or receipt.get("status") != "complete"
            or receipt.get("tranche_sha256") != plan_hash
            or int(receipt.get("scenario_count", -1)) != len(plan.get("scenarios", ()))
        ):
            raise ValueError(f"receipt does not bind a complete tranche {number}")
        scenario_ids = {str(item["scenario_id"]) for item in plan["scenarios"]}
        overlap = downloaded & scenario_ids
        if overlap:
            raise ValueError(f"scenario overlap in completed history: {sorted(overlap)[:3]}")
        downloaded |= scenario_ids
        history.append(
            {
                "tranche_number": number,
                "tranche_sha256": plan_hash,
                "receipt_sha256": receipt_hash,
                "scenario_count": len(scenario_ids),
            }
        )
        hashes.append(plan_hash)
    return history, hashes, downloaded


def _wilson_lower(successes: int, total: int, z: float = WILSON_Z) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid observations for Wilson interval")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - radius)


def _largest_remainder(
    quota: int, capacities: dict[str, int], *, salt: str
) -> dict[str, int]:
    if quota < 0 or quota > sum(capacities.values()):
        raise ValueError("quota exceeds remaining official capacity")
    if quota == 0:
        return {key: 0 for key in capacities}
    total = sum(capacities.values())
    raw = {key: quota * value / total for key, value in capacities.items()}
    result = {key: min(capacities[key], math.floor(value)) for key, value in raw.items()}
    remaining = quota - sum(result.values())
    ranked = sorted(
        capacities,
        key=lambda key: (
            -(raw[key] - math.floor(raw[key])),
            hashlib.sha256(f"{salt}:quota:{key}".encode()).hexdigest(),
        ),
    )
    while remaining:
        progressed = False
        for key in ranked:
            if result[key] < capacities[key]:
                result[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("quota exceeds remaining official capacity")
    return result


def _select(
    count: int,
    *,
    candidates_by_intersection: dict[str, list[dict[str, Any]]],
    assignments: dict[str, str],
    by_name: dict[str, dict[str, Any]],
    salt: str,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    partition_capacities = {
        partition: sum(
            len(candidates_by_intersection.get(intersection, ()))
            for intersection, assigned in assignments.items()
            if assigned == partition
        )
        for partition in PARTITION_FRACTIONS
    }
    partition_quotas = _largest_remainder(
        count, partition_capacities, salt=f"{salt}:partition"
    )
    intersection_quotas: dict[str, int] = {}
    for partition, quota in partition_quotas.items():
        capacities = {
            intersection: len(candidates_by_intersection.get(intersection, ()))
            for intersection, assigned in assignments.items()
            if assigned == partition
        }
        intersection_quotas.update(
            _largest_remainder(quota, capacities, salt=f"{salt}:{partition}")
        )

    selected: list[dict[str, Any]] = []
    for intersection, quota in sorted(intersection_quotas.items()):
        for item in candidates_by_intersection[intersection][:quota]:
            source = "val" if item["source_partition"] == "validation" else "train"
            members = {
                role: f"v2x-traj/{directory}/{source}/data/{item['scenario_id']}.csv"
                for role, directory in ROLE_DIRS.items()
            }
            missing = sorted(set(members.values()) - set(by_name))
            if missing:
                raise ValueError(f"incomplete scenario bundle: {missing}")
            selected.append({**item, "members": members})
    return selected, partition_quotas, intersection_quotas


def _transfer_size(
    selected: list[dict[str, Any]], by_name: dict[str, dict[str, Any]]
) -> tuple[int, int, int]:
    names = [name for item in selected for name in item["members"].values()]
    compressed = sum(int(by_name[name]["compressed_bytes"]) for name in names)
    uncompressed = sum(int(by_name[name]["uncompressed_bytes"]) for name in names)
    return compressed, uncompressed, compressed + REQUEST_OVERHEAD_BYTES * len(names)


def _known_behavior_gaps(review: dict[str, Any]) -> dict[str, list[str]]:
    counts = review.get("summary", {}).get("valid_windows_by_split_behavior", {})
    behaviors = sorted(review.get("summary", {}).get("behavior_counts", {}))
    return {
        partition: [behavior for behavior in behaviors if int(values.get(behavior, 0)) == 0]
        for partition, values in sorted(counts.items())
        if any(int(values.get(behavior, 0)) == 0 for behavior in behaviors)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tranche-number",
        type=int,
        help="must equal the next number after all completed receipts",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        help="override conservative Wilson-based sizing for this one review tranche",
    )
    args = parser.parse_args()

    index, index_hash = _verified(INDEX, "index_sha256")
    inventory, inventory_hash = _verified(INVENTORY, "inventory_sha256")
    review, review_hash = _verified(WINDOW_REVIEW, "review_sha256")
    split, split_hash = _verified(SPLIT, "split_sha256")
    if split.get("method") != "capacity_balanced_whole_intersection_v1":
        raise ValueError("capacity-balanced split must be frozen before continuation")

    history, prior_hashes, downloaded = _completed_history()
    next_number = len(history) + 1
    tranche_number = args.tranche_number or next_number
    if tranche_number != next_number:
        raise ValueError(f"next safe tranche number is {next_number}, not {tranche_number}")
    if int(review["summary"]["reviewed_scenarios"]) != len(downloaded):
        raise ValueError("window review is not synchronized with completed history")

    observed_valid = int(review["summary"]["valid_windows"])
    observed_total = int(review["summary"]["reviewed_scenarios"])
    shortfall = max(0, TARGET_VALID_WINDOWS - observed_valid)
    if shortfall == 0 and args.scenario_count is None:
        raise ValueError("valid-window target is already met; no automatic continuation needed")
    lower_bound = _wilson_lower(observed_valid, observed_total)
    requested_count = args.scenario_count or math.ceil(shortfall / lower_bound)
    if requested_count <= 0:
        raise ValueError("scenario count must be positive")

    salt = f"bces-iov-v2xtraj-primary-v{tranche_number}"
    candidates_by_intersection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in inventory["scenarios"]:
        if str(item["scenario_id"]) not in downloaded:
            candidates_by_intersection[str(item["intersection_id"])].append(item)
    for candidates in candidates_by_intersection.values():
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{salt}:{item['scenario_id']}".encode()
            ).hexdigest()
        )

    by_name = {item["name"]: item for item in index["members"]}
    count = min(requested_count, sum(map(len, candidates_by_intersection.values())))
    while count > 0:
        selected, partition_quotas, intersection_quotas = _select(
            count,
            candidates_by_intersection=candidates_by_intersection,
            assignments=split["assignments"],
            by_name=by_name,
            salt=salt,
        )
        compressed_bytes, uncompressed_bytes, transfer_bytes = _transfer_size(
            selected, by_name
        )
        if transfer_bytes <= TRANSFER_CAP_BYTES:
            break
        count -= 1
    else:
        raise ValueError("no scenario bundle fits the transfer cap")

    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    require_headroom(budget, uncompressed_bytes, preserve_reserve=True, enforce_tranche=False)
    budget_before = build_budget_report(budget)
    plan_output = MANIFESTS / f"v2xtraj_tranche_v{tranche_number}.json"
    review_output = MANIFESTS / f"v2xtraj_continuation_review_v{tranche_number - 1}.json"
    if plan_output.exists() or review_output.exists():
        raise FileExistsError("refusing to overwrite a frozen tranche or continuation review")

    plan = {
        "schema_version": 2,
        "dataset_id": "v2xtraj_primary",
        "tranche_number": tranche_number,
        "selection_salt": salt,
        "selection_rule": (
            "exclude every scenario in verified completed receipts; size from the 95% "
            "Wilson lower confidence bound of cumulative valid-window yield; allocate "
            "60/15/10/15 by frozen partition and by remaining official intersection "
            "capacity using largest remainder; SHA-256 rank within intersection; shrink "
            "only if required by the exact transfer cap"
        ),
        "sizing": {
            "method": "valid_window_shortfall_divided_by_95pct_wilson_lower_bound",
            "requested_scenarios": requested_count,
            "selected_scenarios": len(selected),
            "reduced_by_transfer_cap": len(selected) < requested_count,
            "observed_valid_windows": observed_valid,
            "observed_scenarios": observed_total,
            "observed_acceptance_rate": observed_valid / observed_total,
            "wilson_z": WILSON_Z,
            "wilson_lower_acceptance_rate": lower_bound,
            "valid_window_shortfall": shortfall,
        },
        "partition_quotas": partition_quotas,
        "intersection_quotas": intersection_quotas,
        "scenario_count": len(selected),
        "intersection_count": sum(value > 0 for value in intersection_quotas.values()),
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "request_overhead_allowance_bytes": REQUEST_OVERHEAD_BYTES
        * len(selected)
        * len(ROLE_DIRS),
        "transfer_bytes": transfer_bytes,
        "maximum_transfer_bytes": TRANSFER_CAP_BYTES,
        "crosses_initial_review_checkpoint": (
            budget_before["measurements"]["data_bytes"] + uncompressed_bytes > 500_000_000
        ),
        "source_index_sha256": index_hash,
        "source_inventory_sha256": inventory_hash,
        "source_prior_tranche_sha256s": prior_hashes,
        "source_window_review_sha256": review_hash,
        "source_split_sha256": split_hash,
        "scenarios": sorted(selected, key=lambda item: int(item["scenario_id"])),
    }
    plan["tranche_sha256"] = canonical_json_hash(plan)

    continuation = {
        "schema_version": 2,
        "dataset_id": "v2xtraj_primary",
        "continuation_number": tranche_number - 1,
        "decision": "approved_for_one_additional_review_tranche",
        "trigger": "manual_review_after_each_completed_continuation_tranche",
        "completed_history": history,
        "current_valid_windows": observed_valid,
        "required_valid_windows": TARGET_VALID_WINDOWS,
        "valid_window_shortfall": shortfall,
        "observed_acceptance_rate": observed_valid / observed_total,
        "wilson_95pct_lower_acceptance_rate": lower_bound,
        "planned_new_scenarios": len(selected),
        "planned_uncompressed_member_bytes": uncompressed_bytes,
        "point_expected_new_valid_windows": round(
            len(selected) * observed_valid / observed_total, 2
        ),
        "conservative_expected_new_valid_windows": round(len(selected) * lower_bound, 2),
        "known_split_behavior_gaps": _known_behavior_gaps(review),
        "reason": (
            f"The primary gate lacks {shortfall} valid windows and still lacks true "
            "cached-versus-fresh validity decisions. This single tranche uses a "
            "conservative observed-yield bound, covers all frozen partitions, excludes "
            "every completed scenario, and does not authorize unattended looping."
        ),
        "source_plan_sha256": plan["tranche_sha256"],
        "source_window_review_sha256": review_hash,
        "source_split_sha256": split_hash,
        "budget_before": budget_before["measurements"],
        "minimum_projected_data_bytes_after_extraction": (
            budget_before["measurements"]["data_bytes"] + uncompressed_bytes
        ),
        "output_reserve_preserved": True,
        "phase4_remains_prohibited": True,
    }
    continuation["continuation_review_sha256"] = canonical_json_hash(continuation)
    write_json_atomic(plan_output, plan)
    write_json_atomic(review_output, continuation)
    print(
        json.dumps(
            {
                "plan": plan_output.relative_to(ROOT).as_posix(),
                "continuation_review": review_output.relative_to(ROOT).as_posix(),
                "tranche_sha256": plan["tranche_sha256"],
                "scenario_count": len(selected),
                "partition_quotas": partition_quotas,
                "intersection_quotas": intersection_quotas,
                "transfer_bytes": transfer_bytes,
                "uncompressed_member_bytes": uncompressed_bytes,
                "wilson_lower_acceptance_rate": lower_bound,
                "conservative_expected_new_valid_windows": continuation[
                    "conservative_expected_new_valid_windows"
                ],
                "minimum_projected_data_bytes_after_extraction": continuation[
                    "minimum_projected_data_bytes_after_extraction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
