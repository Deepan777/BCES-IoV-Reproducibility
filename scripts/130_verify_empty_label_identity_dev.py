#!/usr/bin/env python3
"""Independent raw-source audit of corrected development-table identities."""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash, sha256_file

SOURCE = ROOT / "outputs/study_b/stateless_empty_labels_development_v1"
DERIVED = ROOT / "outputs/study_b/stateless_empty_labels_identity_v1"


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main():
    protocol = json.loads((DERIVED / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((DERIVED / "report.json").read_text(encoding="utf-8"))
    if sha256_file(SOURCE / "report.json") != protocol["source_report_sha256"]:
        raise AssertionError("source report changed")
    if sha256_file(SOURCE / "protocol.json") != protocol["source_protocol_sha256"]:
        raise AssertionError("source protocol changed")
    if sha256_file(ROOT / "scripts/129_materialize_empty_label_identity_dev.py") != protocol["script_sha256"]:
        raise AssertionError("materializer source changed")
    if sha256_file(ROOT / "docs/STUDY_B_EMPTY_LABEL_IDENTITY_CORRECTION_DEV_V1_PLAN.md") != protocol["plan_sha256"]:
        raise AssertionError("identity plan changed")
    if sha256_file(ROOT / "docs/STUDY_B_EMPTY_LABEL_IDENTITY_CROSS_SEED_CORRECTION.md") != protocol["cross_seed_correction_sha256"]:
        raise AssertionError("documented correction changed")
    if sha256_file(DERIVED / "protocol.json") != report["protocol_sha256"]:
        raise AssertionError("derived protocol changed")
    for name, digest in report["artifact_sha256"].items():
        if sha256_file(DERIVED / name) != digest:
            raise AssertionError("derived artifact changed: " + name)
    all_slots = rows(DERIVED / "all_slots.jsonl.gz")
    unique = rows(DERIVED / "unique_observables.jsonl.gz")
    by_slot = {row["slot_id"]: row for row in all_slots}
    if len(all_slots) != len(by_slot) or len(all_slots) != 576:
        raise AssertionError("slot IDs not unique and complete")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if source_report["artifact_sha256"] != protocol["raw_artifact_sha256"]:
        raise AssertionError("raw artifact index changed")
    groups = defaultdict(list)
    source_seen = set()
    for name, digest in sorted(protocol["raw_artifact_sha256"].items()):
        if sha256_file(SOURCE / name) != digest:
            raise AssertionError("raw artifact changed: " + name)
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        reference = artifact["references"][0]
        feature_key = reference["frozen_input"]["feature_sha256"]
        for point in artifact["points"]:
            slot_id = f"{name}:{point['current_timestamp_ms']}"
            if slot_id in source_seen or slot_id not in by_slot:
                raise AssertionError("source-to-derived slot collision or omission")
            source_seen.add(slot_id)
            saved = by_slot[slot_id]
            key = canonical_json_hash({
                "reference_feature_sha256": feature_key,
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
            })
            if (saved["observable_key"] != key
                    or saved["reference_feature_key"] != feature_key
                    or saved["original_point_id"] != point["point_id"]
                    or saved["reference_sha256"] != canonical_json_hash(reference)
                    or saved["point_sha256"] != canonical_json_hash(point)
                    or saved["seed_cluster"] != artifact["seed"]
                    or saved["valid"] != point["valid"]
                    or saved["normalized_drift"] != point["normalized_drift"]):
                raise AssertionError("derived slot differs from raw source")
            groups[key].append(saved)
    if source_seen != set(by_slot) or len(groups) != len(unique):
        raise AssertionError("source grid or observable grouping incomplete")
    unique_by_key = {row["observable_key"]: row for row in unique}
    if len(unique_by_key) != len(unique):
        raise AssertionError("observable keys repeated")
    for key, members in groups.items():
        row = unique_by_key[key]
        if row["member_slot_ids"] != sorted(item["slot_id"] for item in members):
            raise AssertionError("observable membership mismatch")
        if row["multiplicity"] != len(members):
            raise AssertionError("observable multiplicity mismatch")
        if row["member_seed_clusters"] != sorted({item["seed_cluster"] for item in members}):
            raise AssertionError("seed membership mismatch")
        if len({item["validation_group"] for item in members}) != 1:
            raise AssertionError("observable leaked across validation groups")
        if row["validation_group"] != members[0]["validation_group"]:
            raise AssertionError("unique row validation group mismatch")
        if len({item["valid"] for item in members}) != 1 or row["valid"] != members[0]["valid"]:
            raise AssertionError("conflicting observable validity")
    components = sorted({row["validation_group"] for row in all_slots})
    if components != sorted(report["validation_groups"]) or len(components) != 4:
        raise AssertionError("linked seed components wrong")
    if (report["raw_slot_count"] != 576
            or report["unique_observable_count"] != len(unique)
            or report["cross_seed_observable_groups"] != sum(
                len(row["member_seed_clusters"]) > 1 for row in unique)
            or report["raw_invalid"] != sum(not row["valid"] for row in all_slots)
            or report["unique_invalid"] != sum(not row["valid"] for row in unique)):
        raise AssertionError("derived report count mismatch")
    print(json.dumps({"status": "PASS", "raw_slots": len(all_slots),
                      "unique_observables": len(unique),
                      "validation_groups": components,
                      "report_sha256": sha256_file(DERIVED / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
