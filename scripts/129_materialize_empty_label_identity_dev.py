#!/usr/bin/env python3
"""Lossless, immutable identity correction for opened development labels."""

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
OUTPUT = ROOT / "outputs/study_b/stateless_empty_labels_identity_v1"
SOURCE_REPORT_SHA256 = "50d0c1b0ff8aec7f8c769dccf0f8cf5b8e2bf5a2a88aa9f9b17eb31628ef2fba"
PLAN = ROOT / "docs/STUDY_B_EMPTY_LABEL_IDENTITY_CORRECTION_DEV_V1_PLAN.md"
CORRECTION = ROOT / "docs/STUDY_B_EMPTY_LABEL_IDENTITY_CROSS_SEED_CORRECTION.md"


def _write_jsonl_gzip(path, rows):
    with gzip.open(path, "xt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_REPORT_SHA256:
        raise RuntimeError("source report changed")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    artifacts = source_report["artifact_sha256"]
    if len(artifacts) != 96 or {path.name for path in SOURCE.glob("*.json.gz")} != set(artifacts):
        raise RuntimeError("source artifact grid changed")
    slots = []
    groups = defaultdict(list)
    for name in sorted(artifacts):
        path = SOURCE / name
        if sha256_file(path) != artifacts[name]:
            raise RuntimeError("raw artifact changed: " + name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if len(artifact["references"]) != 1:
            raise RuntimeError("unexpected reference count: " + name)
        reference = artifact["references"][0]
        feature_key = reference["frozen_input"]["feature_sha256"]
        for point in artifact["points"]:
            slot_id = f"{name}:{point['current_timestamp_ms']}"
            observable_key = canonical_json_hash({
                "reference_feature_sha256": feature_key,
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
            })
            row = {
                "slot_id": slot_id,
                "raw_artifact": name,
                "original_point_id": point["point_id"],
                "seed_cluster": artifact["seed"],
                "timing": artifact["timing"],
                "presence": artifact["presence"],
                "continuation_mps2": artifact["continuation_mps2"],
                "reference_feature_key": feature_key,
                "observable_key": observable_key,
                "reference_sha256": canonical_json_hash(reference),
                "point_sha256": canonical_json_hash(point),
                "frozen_reference_batch": reference["frozen_input"]["batch"],
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
                "valid": point["valid"],
                "cost_regret": point["cost_regret"],
                "cached_risk": point["cached_risk"],
                "trajectory_deviation_m": point["trajectory_deviation_m"],
                "cached_status": point["cached_status"],
                "fresh_status": point["fresh_status"],
            }
            slots.append(row)
            groups[observable_key].append(row)
    if len(slots) != 576 or len({row["slot_id"] for row in slots}) != 576:
        raise RuntimeError("derived slot IDs are not unique and complete")
    if len({row["seed_cluster"] for row in slots}) != 8:
        raise RuntimeError("seed-cluster grid changed")
    if sum(not row["valid"] for row in slots) != 107:
        raise RuntimeError("source invalid count changed")
    seeds = sorted({row["seed_cluster"] for row in slots})
    neighbors = {seed: {seed} for seed in seeds}
    for members in groups.values():
        member_seeds = {row["seed_cluster"] for row in members}
        for seed in member_seeds:
            neighbors[seed].update(member_seeds)
    components = []
    unseen = set(seeds)
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            seed = pending.pop()
            if seed in component:
                continue
            component.add(seed)
            pending.extend(neighbors[seed] - component)
        unseen -= component
        components.append(sorted(component))
    if len(components) != 4 or any(len(component) != 2 for component in components):
        raise RuntimeError("unexpected observable-linked seed components")
    component_by_seed = {
        seed: "-".join(map(str, component))
        for component in components for seed in component
    }
    for row in slots:
        row["validation_group"] = component_by_seed[row["seed_cluster"]]
    unique = []
    for key, members in sorted(groups.items()):
        if len({row["valid"] for row in members}) != 1:
            raise RuntimeError("identical observables have conflicting validity: " + key)
        if len({row["validation_group"] for row in members}) != 1:
            raise RuntimeError("identical observables cross validation groups: " + key)
        representative = min(members, key=lambda row: row["slot_id"])
        unique.append({
            "observable_key": key,
            "representative_slot_id": representative["slot_id"],
            "member_slot_ids": sorted(row["slot_id"] for row in members),
            "multiplicity": len(members),
            "member_seed_clusters": sorted({row["seed_cluster"] for row in members}),
            "validation_group": representative["validation_group"],
            "reference_feature_key": representative["reference_feature_key"],
            "frozen_reference_batch": representative["frozen_reference_batch"],
            "normalized_drift": representative["normalized_drift"],
            "cache_age_s": representative["cache_age_s"],
            "valid": representative["valid"],
        })
    OUTPUT.mkdir(parents=True)
    slots.sort(key=lambda row: row["slot_id"])
    _write_jsonl_gzip(OUTPUT / "all_slots.jsonl.gz", slots)
    _write_jsonl_gzip(OUTPUT / "unique_observables.jsonl.gz", unique)
    protocol = {
        "status": "DEVELOPMENT_IDENTITY_CORRECTION_ONLY_V1",
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "source_protocol_sha256": sha256_file(SOURCE / "protocol.json"),
        "plan_sha256": sha256_file(PLAN),
        "cross_seed_correction_sha256": sha256_file(CORRECTION),
        "script_sha256": sha256_file(Path(__file__)),
        "raw_artifact_sha256": artifacts,
        "no_confirmation_accessed": True,
        "no_model_fitted": True,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": protocol["status"],
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": {
            name: sha256_file(OUTPUT / name) for name in (
                "all_slots.jsonl.gz", "unique_observables.jsonl.gz")
        },
        "raw_slot_count": len(slots),
        "unique_slot_id_count": len({row["slot_id"] for row in slots}),
        "original_point_id_count": len({row["original_point_id"] for row in slots}),
        "unique_observable_count": len(unique),
        "duplicate_observable_groups": sum(row["multiplicity"] > 1 for row in unique),
        "duplicate_observable_extra_rows": len(slots) - len(unique),
        "cross_seed_observable_groups": sum(
            len(row["member_seed_clusters"]) > 1 for row in unique),
        "validation_groups": ["-".join(map(str, group)) for group in components],
        "conflicting_validity_groups": 0,
        "raw_invalid": sum(not row["valid"] for row in slots),
        "unique_invalid": sum(not row["valid"] for row in unique),
        "seed_clusters": sorted({row["seed_cluster"] for row in slots}),
        "no_model_fitted": True,
        "independent_confirmation": False,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": report,
                      "report_sha256": sha256_file(OUTPUT / "report.json")},
                     indent=2))


if __name__ == "__main__":
    main()
