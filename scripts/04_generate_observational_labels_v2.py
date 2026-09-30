#!/usr/bin/env python3
"""Generate leakage-controlled reference inputs, point labels, and boundaries."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.codebooks import NORMAL_CODEBOOK_V1, codebook_sha256
from bces.geometry.drift import DriftScales
from bces.oracle.boundary import BoundaryConfig, search_boundary
from bces.oracle.observational_v2 import (
    BEHAVIORS,
    build_reference,
    load_scene,
    make_directional_oracle,
    observed_validity_point,
)
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

REVIEW = ROOT / "data" / "manifests" / "v2xtraj_window_review_v1.json"
SCALES = ROOT / "data" / "manifests" / "drift_scales_v2.json"
PLANNER = ROOT / "configs" / "planner" / "kinematic.yaml"
VALIDITY = ROOT / "configs" / "oracle" / "validity_v1.yaml"
BOUNDARY = ROOT / "configs" / "oracle" / "boundary_v2.yaml"
RAW = ROOT / "data" / "raw" / "v2xtraj_primary"
DEFAULT_OUTPUT = ROOT / "outputs" / "phase4" / "observational_v2"


def verified_json(path: Path, hash_field: str) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(hash_field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path}")
    return payload, expected


def _write_line(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_scenes is None and os.environ.get("FULL_RUN") != "1":
        raise RuntimeError("full observational generation requires FULL_RUN=1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / f"{name}.jsonl.gz" for name in ("references", "validity_points", "boundaries")}
    if any(path.exists() for path in (*paths.values(), output / "run_manifest.json")):
        raise FileExistsError("observational_v2 output is immutable")

    review, review_sha = verified_json(REVIEW, "review_sha256")
    scale_payload, scale_sha = verified_json(SCALES, "scales_sha256")
    scales = DriftScales(*map(float, scale_payload["scales"]))
    validity_payload = yaml.safe_load(VALIDITY.read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(
        float(validity_payload["max_cost_regret"]),
        float(validity_payload["max_cached_risk"]),
        float(validity_payload["max_trajectory_deviation_m"]),
    )
    boundary_payload = yaml.safe_load(BOUNDARY.read_text(encoding="utf-8"))
    boundary_config = BoundaryConfig(
        float(boundary_payload["expansion_step"]), float(boundary_payload["tolerance"]),
        float(boundary_payload["max_radius"]), int(boundary_payload["max_queries_per_direction"]),
    )
    planner = KinematicPlanner(PlannerConfig.from_yaml(PLANNER))
    records = [row for row in review["records"] if row.get("valid_windows", 0) > 0]
    records.sort(key=lambda row: str(row["scenario_id"]))
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise ValueError("max scenes must be positive")
        records = records[:args.max_scenes]

    candidate_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        for behavior in BEHAVIORS:
            reference_id = f"{record['scenario_id']}:{behavior}:candidate"
            candidate_ids[(record["split"], behavior)].append(reference_id)
    selected: set[str] = set()
    for (split, behavior), values in candidate_ids.items():
        limit = int(boundary_payload["selected_references_per_split_behavior"][split])
        ranked = sorted(values, key=lambda value: canonical_json_hash({"boundary-selection-v2": value}))
        selected.update(ranked[:limit])

    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    budget_before = build_budget_report(budget)
    counts = Counter()
    cells: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts = Counter()
    started = utc_now()
    temp_paths = {key: value.with_suffix(value.suffix + ".part") for key, value in paths.items()}
    for path in temp_paths.values():
        path.unlink(missing_ok=True)
    try:
        with (
            gzip.open(temp_paths["references"], "xt", encoding="utf-8", newline="\n") as references,
            gzip.open(temp_paths["validity_points"], "xt", encoding="utf-8", newline="\n") as points,
            gzip.open(temp_paths["boundaries"], "xt", encoding="utf-8", newline="\n") as boundaries,
        ):
            for scene_index, record in enumerate(records, 1):
                ego_s, vehicle_s, infra_s, focal_id, states = load_scene(record, RAW)
                times = sorted(states)
                reference_time = times[0]
                for behavior in BEHAVIORS:
                    try:
                        reference, _, query, cached, template = build_reference(
                            scenario_id=str(record["scenario_id"]), intersection_id=str(record["intersection_id"]),
                            split=str(record["split"]), behavior=behavior, focal_id=focal_id,
                            reference_timestamp=reference_time, states=states, ego_snapshots=ego_s,
                            vehicle_snapshots=vehicle_s, infrastructure_snapshots=infra_s,
                            scales=scales, planner=planner,
                            maximum_objects=int(validity_payload["maximum_objects_per_source"]),
                        )
                    except RuntimeError:
                        status_counts[f"reference_infeasible:{record['split']}:{behavior}"] += 1
                        continue
                    _write_line(references, reference)
                    counts["references"] += 1
                    reference_candidate = f"{record['scenario_id']}:{behavior}:candidate"
                    for age in map(float, boundary_payload["validity_ages_s"]):
                        point = observed_validity_point(
                            reference=reference, query=query, cached_reference=cached, template=template,
                            current_timestamp=reference_time + round(age * 1000), states=states,
                            focal_id=focal_id, ego_snapshots=ego_s, vehicle_snapshots=vehicle_s,
                            infrastructure_snapshots=infra_s, planner=planner, thresholds=thresholds,
                            maximum_objects=int(validity_payload["maximum_objects_per_source"]),
                        )
                        if point is not None:
                            _write_line(points, point)
                            counts["validity_points"] += 1
                            counts["valid_points" if point["valid"] else "invalid_points"] += 1
                            cells[reference["split"]][behavior] += 1
                    if reference_candidate in selected:
                        oracle = make_directional_oracle(
                            reference=reference, query=query, cached_reference=cached, template=template,
                            states=states, focal_id=focal_id, ego_snapshots=ego_s,
                            vehicle_snapshots=vehicle_s, infrastructure_snapshots=infra_s,
                            planner=planner, thresholds=thresholds,
                            maximum_objects=int(validity_payload["maximum_objects_per_source"]),
                        )
                        for direction_index, direction in enumerate(NORMAL_CODEBOOK_V1):
                            result = search_boundary(oracle, direction, boundary_config)
                            row = {
                                "boundary_id": f"{reference['reference_id']}:{direction_index}",
                                "reference_id": reference["reference_id"], "scenario_id": reference["scenario_id"],
                                "intersection_id": reference["intersection_id"], "split": reference["split"],
                                "behavior": behavior, "direction_index": direction_index,
                                **result.to_dict(),
                            }
                            _write_line(boundaries, row)
                            counts["boundaries"] += 1
                            status_counts[result.status] += 1
                if scene_index % 10 == 0:
                    print(f"processed reference scenes {scene_index}/{len(records)}", flush=True)
        for key, path in paths.items():
            os.replace(temp_paths[key], path)
    except BaseException:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise

    minimum_cell = min((value for group in cells.values() for value in group.values()), default=0)
    manifest = {
        "schema_version": 2, "phase": 4,
        "status": "PASS" if counts["references"] and counts["boundaries"] and minimum_cell >= 50 else "FAIL",
        "run_scope": "full" if args.max_scenes is None else "development_subset",
        "input_contract": "exact_reference_payload_and_frozen_policy_query_v2",
        "boundary_targets": "oracle_bracketed_and_bisected",
        "counts": dict(counts), "boundary_status_counts": dict(status_counts),
        "validity_decisions_by_split_behavior": {k: dict(v) for k, v in sorted(cells.items())},
        "minimum_split_behavior_decisions": minimum_cell,
        "policy_hash": planner.policy_hash, "normal_codebook_sha256": codebook_sha256(),
        "source_review_sha256": review_sha, "drift_scales_sha256": scale_sha,
        "validity_config_sha256": sha256_file(VALIDITY), "boundary_config_sha256": sha256_file(BOUNDARY),
        "planner_config_sha256": sha256_file(PLANNER),
        "artifact_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "test_partition_used_for_development": False,
        "git": git_state(ROOT), "budget_before": budget_before["measurements"],
        "budget_after": build_budget_report(budget)["measurements"],
        "started_utc": started, "completed_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_json_atomic(output / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
