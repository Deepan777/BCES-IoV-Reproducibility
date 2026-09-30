#!/usr/bin/env python3
"""Conditional clearance of full braking-to-stop backup paths after view loss."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.empty_view_reachability import conditional_empty_disk_clearance
from bces.oracle.planner import PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, route_for
from bces.simulation.development_stop_hold_candidate import stop_hold_points

prior = importlib.import_module("scripts.122_audit_empty_disk_reachability_dev")
PLAN = ROOT / "docs/STUDY_B_EMPTY_VIEW_BACKUP_STOP_DEV_V1_PLAN.md"
ORIGINAL = ROOT / "outputs/study_b/empty_disk_reachability_development_v1/report.json"
ORIGINAL_SHA256 = "5ec3dbea32a51255a544b7ce34de58cda7cbfefe33a7513fe061c8ee88104f33"
OUTPUT = ROOT / "outputs/study_b/empty_view_backup_stop_dev_v1"
RADII_M = (40, 80, 120)
AGE_S = 0.4
BRAKING_MPS2 = -4.0
STEP_S = 0.2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(seed, timing, condition, timestamp):
    return f"{seed}/{timing}/{condition}/{timestamp}"


def main():
    if OUTPUT.exists():
        raise RuntimeError("output exists; preserve first run")
    for path, expected in ((ORIGINAL, ORIGINAL_SHA256),
                           (prior.BASE / "report.json", prior.BASE_REPORT_SHA256),
                           (prior.REGISTRATION, prior.REGISTRATION_SHA256),
                           (prior.GEOMETRY, prior.GEOMETRY_SHA256)):
        if digest(path) != expected:
            raise RuntimeError("frozen source changed: " + str(path))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    baseline = json.loads((prior.BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(prior.REGISTRATION.read_text(encoding="utf-8"))
    if original["summary"]["0.4"]["approach_states"] != 568 or original["summary"]["0.4"]["conditional_clear_states"] != 328:
        raise RuntimeError("old 0.4-s summary changed")
    old_rows = {}
    for row in original["rows"]:
        if row["age_s"] != AGE_S:
            continue
        k = row_key(row["seed"], row["timing"], row["condition"], row["timestamp_ms"])
        if k in old_rows or not row["candidate_available"]:
            raise RuntimeError("duplicate or unavailable original candidate")
        old_rows[k] = row
    if len(old_rows) != 568:
        raise RuntimeError("old row count changed")
    for name, expected in registration["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("registered source changed: " + name)
    planner = ControlledDecisionPlanner(
        PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    if BRAKING_MPS2 not in planner.config.longitudinal_accelerations_mps2 or planner.config.step_s != STEP_S:
        raise RuntimeError("braking authority or planner step changed")
    margin = planner.config.collision_margin_m
    actor_radius = 0.5 * math.hypot(prior.ACTOR_LENGTH_M + 2 * margin,
                                    prior.ACTOR_WIDTH_M + 2 * margin)
    branch_names = sorted(name for name in baseline["artifact_sha256"]
                          if name.endswith("_absent_negative_evidence_periodic_payload.json.gz"))
    if len(branch_names) != 32:
        raise RuntimeError("branch count changed")
    rows = []
    for name in branch_names:
        path = prior.BASE / name
        if digest(path) != baseline["artifact_sha256"][name]:
            raise RuntimeError("development branch changed")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        seed = int(name.split("_")[0])
        timing = "near" if "_near_" in name else "timing_control"
        condition = "nominal" if "_nominal_" in name else "impaired"
        visibility = {item["timestamp_ms"]: item for item in branch["visibility_audit"]}
        for sample in branch["rows"]:
            timestamp = sample["timestamp_ms"]
            k = row_key(seed, timing, condition, timestamp)
            if k not in old_rows:
                continue
            ego = ego_from_sample(sample["receiver"])
            if abs(ego.speed_mps - old_rows[k]["ego_speed_mps"]) > 1e-9:
                raise RuntimeError("old speed not reproduced")
            source = visibility.get(timestamp - 400)
            if source is None or source["target_xy_m"] is not None:
                raise RuntimeError("old empty source reference not reproduced")
            route = route_for(ego, "keep", planner)
            stop_time = ego.speed_mps / -BRAKING_MPS2
            horizon = math.ceil(stop_time / STEP_S - 1e-9) * STEP_S
            points = stop_hold_points(ego, route, braking_mps2=BRAKING_MPS2,
                                      horizon_s=horizon, step_s=STEP_S)
            if points[-1].speed_mps > 1e-9 or points[-1].time_s + 1e-9 < stop_time:
                raise RuntimeError("backup does not reach full stop")
            ego_radius = 0.5 * math.hypot(ego.length_m + 2 * margin,
                                          ego.width_m + 2 * margin)
            flags = {}
            required = None
            for radius in RADII_M:
                result = conditional_empty_disk_clearance(
                    source_center_xy_m=tuple(source["ego_xy_m"]),
                    source_radius_m=radius, observation_age_s=AGE_S,
                    ego_initial_xy_m=(ego.x_m, ego.y_m),
                    ego_future_points=points,
                    maximum_ego_speed_mps=route.speed_limit_mps,
                    maximum_actor_speed_mps=prior.MAX_ACTOR_SPEED_MPS,
                    ego_body_radius_m=ego_radius, actor_body_radius_m=actor_radius,
                    localization_error_m=prior.LOCALIZATION_ERROR_M)
                flags[str(radius)] = result.clear
                this_required = radius - result.minimum_slack_m
                if required is None:
                    required = this_required
                elif abs(required - this_required) > 1e-8:
                    raise RuntimeError("radius linearity failed")
            rows.append({"key": k, "seed": seed, "ego_speed_mps": ego.speed_mps,
                         "stop_time_s": stop_time, "checked_horizon_s": horizon,
                         "stopping_distance_m": ego.speed_mps**2 / (-2 * BRAKING_MPS2),
                         "minimum_required_radius_m": required,
                         "clear": flags})
    if len(rows) != len(old_rows) or {row["key"] for row in rows} != set(old_rows):
        raise RuntimeError("backup row set differs from original")
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    results = {}
    for radius in RADII_M:
        r = str(radius)
        results[r] = {"states": len(rows),
                      "clear_states": sum(row["clear"][r] for row in rows),
                      "by_seed": {str(seed): {"states": len(group),
                                             "clear_states": sum(row["clear"][r] for row in group)}
                                  for seed, group in sorted(by_seed.items())}}
    required = [row["minimum_required_radius_m"] for row in rows]
    speeds = [row["ego_speed_mps"] for row in rows]
    stop_times = [row["stop_time_s"] for row in rows]
    stop_distances = [row["stopping_distance_m"] for row in rows]
    summary = {"minimum_required_radius_m": {"min": min(required),
                                               "median": statistics.median(required),
                                               "max": max(required)},
               "ego_speed_mps": {"min": min(speeds), "max": max(speeds)},
               "stop_time_s": {"min": min(stop_times), "max": max(stop_times)},
               "stopping_distance_m": {"min": min(stop_distances), "max": max(stop_distances)}}
    OUTPUT.mkdir(parents=True)
    protocol = {"status": "EMPTY_VIEW_BACKUP_STOP_DEV_V1",
                "plan_sha256": digest(PLAN), "script_sha256": digest(Path(__file__)),
                "original_reachability_report_sha256": ORIGINAL_SHA256,
                "baseline_report_sha256": prior.BASE_REPORT_SHA256,
                "registration_sha256": prior.REGISTRATION_SHA256,
                "geometry_sha256": prior.GEOMETRY_SHA256,
                "age_s": AGE_S, "radii_m": list(RADII_M),
                "braking_mps2": BRAKING_MPS2,
                "independent_confirmation": False, "real_source_certified": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"], "protocol_sha256": digest(OUTPUT / "protocol.json"),
              "results": results, "summary": summary, "rows": rows,
              "scope": "conditional_full_stop_backup_geometry_only",
              "independent_confirmation": False, "real_source_certified": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "summary": summary,
                      "report_sha256": digest(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
