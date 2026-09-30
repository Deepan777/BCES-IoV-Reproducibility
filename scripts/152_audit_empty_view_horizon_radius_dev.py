#!/usr/bin/env python3
"""Opened-development path-prefix/radius sensitivity for conditional clearance."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.empty_view_reachability import conditional_empty_disk_clearance
from bces.oracle.planner import PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, route_for

prior = importlib.import_module("scripts.122_audit_empty_disk_reachability_dev")
PLAN = ROOT / "docs/STUDY_B_EMPTY_VIEW_HORIZON_RADIUS_DEV_V1_PLAN.md"
PREVIOUS = ROOT / "outputs/study_b/empty_disk_reachability_development_v1/report.json"
PREVIOUS_SHA256 = "5ec3dbea32a51255a544b7ce34de58cda7cbfefe33a7513fe061c8ee88104f33"
OUTPUT = ROOT / "outputs/study_b/empty_view_horizon_radius_dev_v1"
HORIZONS_S = (0.2, 0.4, 0.8, 1.0, 2.0, 3.0)
RADII_M = (20, 40, 60, 80, 100, 120)
AGE_S = 0.2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(seed, timing, condition, timestamp):
    return f"{seed}/{timing}/{condition}/{timestamp}"


def main():
    if OUTPUT.exists():
        raise RuntimeError("output exists; preserve first run")
    for path, expected in ((prior.BASE / "report.json", prior.BASE_REPORT_SHA256),
                           (prior.REGISTRATION, prior.REGISTRATION_SHA256),
                           (prior.GEOMETRY, prior.GEOMETRY_SHA256),
                           (PREVIOUS, PREVIOUS_SHA256)):
        if digest(path) != expected:
            raise RuntimeError("frozen input changed: " + str(path))
    baseline = json.loads((prior.BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(prior.REGISTRATION.read_text(encoding="utf-8"))
    for name, expected in baseline["artifact_sha256"].items():
        if digest(prior.BASE / name) != expected:
            raise RuntimeError("development branch changed: " + name)
    for name, expected in registration["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("registered source changed: " + name)
    original = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    old_rows = {}
    for row in original["rows"]:
        if row["age_s"] != AGE_S:
            continue
        row_key = key(row["seed"], row["timing"], row["condition"], row["timestamp_ms"])
        if row_key in old_rows:
            raise RuntimeError("duplicate original row")
        old_rows[row_key] = row
    if len(old_rows) != 600:
        raise RuntimeError("old 0.2-s denominator changed")
    planner = ControlledDecisionPlanner(
        PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    margin = planner.config.collision_margin_m
    actor_radius = 0.5 * math.hypot(prior.ACTOR_LENGTH_M + 2 * margin,
                                    prior.ACTOR_WIDTH_M + 2 * margin)
    branch_names = sorted(name for name in baseline["artifact_sha256"]
                          if name.endswith("_absent_negative_evidence_periodic_payload.json.gz"))
    if len(branch_names) != 32:
        raise RuntimeError("expected 32 actor-absent periodic branches")
    rows = []
    for name in branch_names:
        with gzip.open(prior.BASE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        if branch["initial_contract"]["parameters"]["actor_presence"] != "absent":
            raise RuntimeError("actor-present branch encountered")
        seed = int(name.split("_")[0])
        timing = "near" if "_near_" in name else "timing_control"
        condition = "nominal" if "_nominal_" in name else "impaired"
        visibility = {item["timestamp_ms"]: item for item in branch["visibility_audit"]}
        for sample in branch["rows"]:
            if "receiver" not in sample:
                continue
            ego = ego_from_sample(sample["receiver"])
            if ego.y_m >= prior.CROSSING_Y_M:
                continue
            timestamp = sample["timestamp_ms"]
            reference_ms = timestamp - 200
            source = visibility.get(reference_ms)
            if reference_ms < 3000 or source is None:
                continue
            if source["target_xy_m"] is not None:
                raise RuntimeError("expected empty source reference")
            row_key = key(seed, timing, condition, timestamp)
            if row_key not in old_rows:
                raise RuntimeError("new row absent from original audit")
            route = route_for(ego, "keep", planner)
            try:
                candidate = planner.plan((), (), ego, route)
            except RuntimeError as exc:
                if str(exc) != "no feasible trajectory candidate":
                    raise
                candidate = None
            if candidate is None:
                raise RuntimeError("original 0.2-s audit had no absent candidate")
            ego_radius = 0.5 * math.hypot(ego.length_m + 2 * margin,
                                          ego.width_m + 2 * margin)
            radius_requirements = {}
            clear_by_horizon = {}
            for horizon in HORIZONS_S:
                prefix = tuple(point for point in candidate.points
                               if point.time_s <= horizon + 1e-9)
                if not prefix or abs(prefix[-1].time_s - horizon) > 1e-8:
                    raise RuntimeError("planner prefix does not end at requested horizon")
                clear_by_radius = {}
                required = None
                for radius in RADII_M:
                    result = conditional_empty_disk_clearance(
                        source_center_xy_m=tuple(source["ego_xy_m"]),
                        source_radius_m=radius, observation_age_s=AGE_S,
                        ego_initial_xy_m=(ego.x_m, ego.y_m),
                        ego_future_points=prefix,
                        maximum_ego_speed_mps=route.speed_limit_mps,
                        maximum_actor_speed_mps=prior.MAX_ACTOR_SPEED_MPS,
                        ego_body_radius_m=ego_radius,
                        actor_body_radius_m=actor_radius,
                        localization_error_m=prior.LOCALIZATION_ERROR_M)
                    clear_by_radius[str(radius)] = result.clear
                    this_required = radius - result.minimum_slack_m
                    if required is None:
                        required = this_required
                    elif abs(required - this_required) > 1e-8:
                        raise RuntimeError("radius rescaling identity failed")
                radius_requirements[str(horizon)] = required
                clear_by_horizon[str(horizon)] = clear_by_radius
                if horizon == 3.0:
                    old = old_rows[row_key]
                    if (clear_by_radius["120"] != old["continuous_conditional_clearance"]
                            or abs(120 - required - old["minimum_slack_m"]) > 1e-8):
                        raise RuntimeError("old 3-s/120-m row not reproduced")
            rows.append({"key": row_key, "seed": seed, "timing": timing,
                         "condition": condition, "timestamp_ms": timestamp,
                         "required_radius_m": radius_requirements,
                         "clear": clear_by_horizon})
    if len(rows) != len(old_rows) or len({row["key"] for row in rows}) != len(old_rows):
        raise RuntimeError("development row set not reproduced")
    results = {}
    for horizon in HORIZONS_S:
        h = str(horizon)
        results[h] = {}
        required = [row["required_radius_m"][h] for row in rows]
        for radius in RADII_M:
            r = str(radius)
            by_seed = defaultdict(list)
            for row in rows:
                by_seed[row["seed"]].append(row["clear"][h][r])
            count = sum(sum(flags) for flags in by_seed.values())
            results[h][r] = {"states": len(rows), "clear_states": count,
                             "clear_fraction": count / len(rows),
                             "by_seed": {str(seed): {"states": len(flags),
                                                    "clear_states": sum(flags)}
                                         for seed, flags in sorted(by_seed.items())},
                             "required_radius_m": {"min": min(required),
                                                    "median": statistics.median(required),
                                                    "max": max(required)}}
    if results["3.0"]["120"]["clear_states"] != 600 or results["3.0"]["100"]["clear_states"] != 0:
        raise RuntimeError("previous radius sensitivity not reproduced")
    OUTPUT.mkdir(parents=True)
    protocol = {"status": "EMPTY_VIEW_HORIZON_RADIUS_DEV_V1",
                "plan_sha256": digest(PLAN), "script_sha256": digest(Path(__file__)),
                "prior_report_sha256": PREVIOUS_SHA256,
                "baseline_report_sha256": prior.BASE_REPORT_SHA256,
                "registration_sha256": prior.REGISTRATION_SHA256,
                "geometry_sha256": prior.GEOMETRY_SHA256,
                "age_s": AGE_S, "horizons_s": list(HORIZONS_S),
                "radii_m": list(RADII_M), "independent_confirmation": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"], "protocol_sha256": digest(OUTPUT / "protocol.json"),
              "results": results, "rows": rows,
              "scope": "ideal_source_collision_prefix_only_not_decision_regret",
              "independent_confirmation": False, "real_source_view_certified": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {h: {r: item["clear_states"] for r, item in cells.items()}
                                 for h, cells in results.items()},
                      "report_sha256": digest(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
