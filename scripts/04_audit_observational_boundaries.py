#!/usr/bin/env python3
"""Independently reload raw scenes and replay 100 training boundary traces."""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.oracle.boundary import audit_boundary, directional_boundary_from_dict
from bces.oracle.observational_v2 import (
    build_reference,
    load_scene,
    make_directional_oracle,
)
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

RUN = ROOT / "outputs" / "phase4" / "observational_v2"
RAW = ROOT / "data" / "raw" / "v2xtraj_primary"


def _gzip_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> int:
    output = RUN / "boundary_audit.json"
    if output.exists():
        raise FileExistsError("boundary audit is immutable")
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256")
    if canonical_json_hash(manifest) != expected or manifest["run_scope"] != "full":
        raise ValueError("full run manifest does not verify")
    scales_payload = json.loads((ROOT / "data/manifests/drift_scales_v2.json").read_text(encoding="utf-8"))
    scales = DriftScales(*map(float, scales_payload["scales"]))
    validity = yaml.safe_load((ROOT / "configs/oracle/validity_v1.yaml").read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(
        float(validity["max_cost_regret"]), float(validity["max_cached_risk"]),
        float(validity["max_trajectory_deviation_m"]),
    )
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml"))
    review = json.loads((ROOT / "data/manifests/v2xtraj_window_review_v1.json").read_text(encoding="utf-8"))
    by_scenario = {str(row["scenario_id"]): row for row in review["records"]}
    candidates = [row for row in _gzip_rows(RUN / "boundaries.jsonl.gz") if row["split"] == "train"]
    candidates.sort(key=lambda row: canonical_json_hash({"independent-boundary-audit-v2": row["boundary_id"]}))
    selected = candidates[:100]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["scenario_id"]].append(row)
    mismatches = []
    probe_count = 0
    for scenario_id, rows in grouped.items():
        record = by_scenario[scenario_id]
        ego_s, vehicle_s, infra_s, focal_id, states = load_scene(record, RAW)
        reference_time = min(states)
        for behavior in sorted({row["behavior"] for row in rows}):
            reference, _, query, cached, template = build_reference(
                scenario_id=scenario_id, intersection_id=str(record["intersection_id"]),
                split=str(record["split"]), behavior=behavior, focal_id=focal_id,
                reference_timestamp=reference_time, states=states, ego_snapshots=ego_s,
                vehicle_snapshots=vehicle_s, infrastructure_snapshots=infra_s,
                scales=scales, planner=planner, maximum_objects=int(validity["maximum_objects_per_source"]),
            )
            oracle = make_directional_oracle(
                reference=reference, query=query, cached_reference=cached, template=template,
                states=states, focal_id=focal_id, ego_snapshots=ego_s, vehicle_snapshots=vehicle_s,
                infrastructure_snapshots=infra_s, planner=planner, thresholds=thresholds,
                maximum_objects=int(validity["maximum_objects_per_source"]),
            )
            for row in (item for item in rows if item["behavior"] == behavior):
                result = audit_boundary(directional_boundary_from_dict(row), oracle)
                probe_count += result["probe_count"]
                if not result["passed"]:
                    mismatches.append({"boundary_id": row["boundary_id"], "probe_indices": result["mismatches"]})
    report = {
        "schema_version": 2, "selection_partition": "train",
        "selection": "lowest_sha256_boundary_id_before_recomputation",
        "boundary_count": len(selected), "probe_count": probe_count,
        "mismatches": mismatches, "passed": len(selected) == 100 and not mismatches,
        "test_partition_accessed": False, "source_manifest_sha256": expected,
        "git": git_state(ROOT), "completed_utc": utc_now(),
    }
    report["audit_sha256"] = canonical_json_hash(report)
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
