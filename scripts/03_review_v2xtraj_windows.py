#!/usr/bin/env python3
"""Review downloaded V2X-Traj scenarios for windows and behavior proxies."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.windows import WindowReviewConfig, review_scene
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "data" / "raw" / "v2xtraj_primary"
    )
    parser.add_argument(
        "--tranche",
        type=Path,
        action="append",
        help="repeat to review specific tranches; defaults to every numbered tranche",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "manifests" / "v2xtraj_primary_split_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "v2xtraj_window_review_v1.json",
    )
    return parser.parse_args()


def _verified(path: Path, hash_field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(hash_field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path.name}")
    return payload


def main() -> int:
    args = parse_args()
    tranche_paths = args.tranche or sorted(
        (ROOT / "data" / "manifests").glob("v2xtraj_tranche_v*.json")
    )
    tranches = []
    source_tranche_sha256 = []
    for path in tranche_paths:
        source_tranche_sha256.append(
            json.loads(path.read_text(encoding="utf-8"))["tranche_sha256"]
        )
        tranches.append(_verified(path, "tranche_sha256"))
    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    split_hash = split_payload.pop("split_sha256")
    if canonical_json_hash(split_payload) != split_hash:
        raise ValueError("split hash does not verify")
    assignments = split_payload["assignments"]
    config = WindowReviewConfig()

    scenarios_by_id = {}
    for tranche in tranches:
        for item in tranche["scenarios"]:
            scenario_id = str(item["scenario_id"])
            if scenario_id in scenarios_by_id:
                raise ValueError(f"scenario appears in multiple tranches: {scenario_id}")
            scenarios_by_id[scenario_id] = item
    records = []
    for item in sorted(scenarios_by_id.values(), key=lambda value: int(value["scenario_id"])):
        member = item["members"]["ego"]
        member_parts = Path(member).parts
        source_partition = member_parts[-3]
        source = args.raw_root / "ego" / source_partition / member_parts[-1]
        record = review_scene(
                source,
                scenario_id=str(item["scenario_id"]),
                intersection_id=str(item["intersection_id"]),
                split=assignments[str(item["intersection_id"])],
                config=config,
            )
        record["source_file"] = source.relative_to(args.raw_root).as_posix()
        records.append(record)

    behavior_counts = Counter(
        behavior for record in records for behavior in record["behavior_families"]
    )
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for behavior in record["behavior_families"]:
            by_split[record["split"]][behavior] += record["valid_windows"]
    invalid_reasons = Counter(
        reason for record in records for reason in record["review_reasons"]
    )
    payload = {
        "schema_version": 1,
        "dataset_id": "v2xtraj_primary",
        "review_type": "window_and_behavior_proxy_availability_not_oracle_labels",
        "config": config.to_dict(),
        "source_tranche_sha256": source_tranche_sha256,
        "source_split_sha256": split_hash,
        "summary": {
            "reviewed_scenarios": len(records),
            "valid_scenarios": sum(record["valid_windows"] > 0 for record in records),
            "valid_windows": sum(record["valid_windows"] for record in records),
            "behavior_counts": dict(sorted(behavior_counts.items())),
            "valid_windows_by_split_behavior": {
                split: dict(sorted(counts.items())) for split, counts in sorted(by_split.items())
            },
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
            "oracle_validity_decisions": 0,
        },
        "records": records,
        "scientific_guard": (
            "Behavior values are deterministic trajectory-derived proxies. They are not "
            "ground-truth driver intentions, cached-versus-fresh validity decisions, or "
            "empirical BCES performance evidence."
        ),
    }
    payload["review_sha256"] = canonical_json_hash(payload)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
