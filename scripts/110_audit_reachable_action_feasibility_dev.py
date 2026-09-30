#!/usr/bin/env python3
"""Development-only sampled robust-action feasibility on common trajectories."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.reachable_shadow import ReachabilityAssumptions, reachable_shadow
from bces.oracle.planner import PlannerConfig
from bces.oracle.world import GroundTruthWorld, PlannedTrajectory, CostVector, WorldObject
from bces.simulation.controlled_contract import center_from_front, ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, route_for
from bces.simulation.scenario_builder import scenario_spec
from bces.simulation.sumo_adapter import heading_rad
from bces.utils.hashing import sha256_file


BASE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
BASE_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
SHADOW_SOURCE = ROOT / "bces/geometry/reachable_shadow.py"
SHADOW_SHA256 = "8cc749044e94392726ae1b9a9bfd1f8d4942a4c29ba8240211e0a44795f0694f"
OUTPUT = ROOT / "outputs/study_b/reachable_action_feasibility_development_v1"
ACCEL_BOUNDS_MPS2 = (1.0, 2.0, 4.0)
POSITION_ERROR_M = 0.5
VELOCITY_ERROR_MPS = 1.0
FIRST_MS, LAST_MS = 4200, 6000


def actor_at_reference(snapshot):
    item = snapshot["vehicles"]["cross"]
    heading = heading_rad(item["heading"])
    x, y = center_from_front(item["position"], heading, item["length"])
    return WorldObject("vehicle:cross", x, y,
                       item["speed"] * math.cos(heading),
                       item["speed"] * math.sin(heading),
                       heading, item["length"], item["width"],
                       source="development_ideal_sender_prefix")


def safe_candidate_counts(planner, ego, route, shadow):
    candidate_count = 0
    no_sampled_overlap_count = 0
    no_sampled_overlap_accelerations = []
    for maneuver in planner.config.lateral_maneuvers:
        for acceleration in planner.config.longitudinal_accelerations_mps2:
            points = planner._candidate(ego, route, acceleration, maneuver)
            if points is None:
                continue
            candidate_count += 1
            placeholder = PlannedTrajectory(maneuver, acceleration, points,
                                            CostVector(0, 0, 0, 0, 0, 0, 0, None))
            cost = planner.evaluate(placeholder, GroundTruthWorld(0, (shadow,)), ego, route)
            if cost.collision == 0:
                no_sampled_overlap_count += 1
                no_sampled_overlap_accelerations.append(acceleration)
    selected = planner.plan((), (shadow,), ego, route)
    return {"candidate_count": candidate_count,
            "no_sampled_overlap_count": no_sampled_overlap_count,
            "no_sampled_overlap_accelerations_mps2": no_sampled_overlap_accelerations,
            "selected_acceleration_mps2": selected.acceleration_mps2,
            "selected_perceived_collision": selected.perceived_cost.collision}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(BASE / "report.json") != BASE_SHA256:
        raise RuntimeError("baseline report changed")
    if sha256_file(SHADOW_SOURCE) != SHADOW_SHA256:
        raise RuntimeError("conditional geometry source changed")
    if sha256_file(REGISTRATION) != REGISTRATION_SHA256:
        raise RuntimeError("registered planner specification changed")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    for name, digest in baseline["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("baseline branch changed: " + name)
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    if planner.policy_hash != registration["policy_hash"]:
        raise RuntimeError("planner policy binding changed")
    behavior = scenario_spec("unprotected_crossing").intended_behavior
    rows = []
    seeds = sorted({row["seed"] for row in baseline["rows"]})
    conditions = sorted({row["condition"] for row in baseline["rows"]})
    for seed in seeds:
        for condition in conditions:
            name = f"{seed}_near_{condition}_original_periodic_payload.json.gz"
            with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            reference_actor = actor_at_reference(branch["initial_contract"]["snapshot"])
            visibility = {point["timestamp_ms"]: point for point in branch["visibility_audit"]}
            for point in branch["rows"]:
                timestamp = point["timestamp_ms"]
                if (timestamp < FIRST_MS or timestamp > LAST_MS
                        or "receiver" not in point
                        or visibility[timestamp]["target_locally_visible"]):
                    continue
                ego = ego_from_sample(point["receiver"])
                route = route_for(ego, behavior, planner)
                age = (timestamp - 4000) / 1000.0
                for acceleration_bound in ACCEL_BOUNDS_MPS2:
                    assumptions = ReachabilityAssumptions(POSITION_ERROR_M,
                                                         VELOCITY_ERROR_MPS,
                                                         acceleration_bound)
                    shadow = reachable_shadow(reference_actor, age_s=age,
                                              horizon_s=planner.config.horizon_s,
                                              assumptions=assumptions)
                    results = safe_candidate_counts(planner, ego, route, shadow)
                    rows.append({"seed": seed, "condition": condition,
                                 "timestamp_ms": timestamp,
                                 "actor_locally_visible": False,
                                 "acceleration_bound_mps2": acceleration_bound,
                                 "shadow_width_m": shadow.width_m,
                                 **results})
    by_bound = {}
    for bound in ACCEL_BOUNDS_MPS2:
        subset = [row for row in rows if row["acceleration_bound_mps2"] == bound]
        by_bound[str(bound)] = {"evaluated_states": len(subset),
                                "no_sampled_nonoverlap_action_states":
                                    sum(row["no_sampled_overlap_count"] == 0 for row in subset),
                                "states_where_selected_plan_has_sampled_overlap":
                                    sum(bool(row["selected_perceived_collision"]) for row in subset),
                                "mean_nonoverlap_candidate_count":
                                    sum(row["no_sampled_overlap_count"] for row in subset) / len(subset),
                                "mean_shadow_width_m":
                                    sum(row["shadow_width_m"] for row in subset) / len(subset),
                                "seeds_with_any_no_action_state":
                                    sorted({row["seed"] for row in subset
                                            if row["no_sampled_overlap_count"] == 0})}
    result = {"status": "REACHABLE_ACTION_FEASIBILITY_DEVELOPMENT_ONLY_V1",
              "source_report_sha256": BASE_SHA256,
              "shadow_source_sha256": SHADOW_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "setting": "near crossing; saved periodic trajectories, locally unseen actor, 4.2-6.0 s",
              "actor_source": "ideal sender state from common t=4.0-s prefix, not actual packet delivery",
              "position_error_m": POSITION_ERROR_M,
              "velocity_error_mps": VELOCITY_ERROR_MPS,
              "acceleration_bounds_mps2": list(ACCEL_BOUNDS_MPS2),
              "by_bound": by_bound,
              "rows": rows,
              "limitations": ["same eight opened development seeds and repeated states, no independent inference",
                              "the acceleration and error bounds are hypotheses, not validated limits",
                              "only 0.2-s planner trajectory samples are checked",
                              "nonoverlap under an actor footprint bound is not a full decision-regret or actuation guarantee"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_bound, indent=2))


if __name__ == "__main__":
    main()
