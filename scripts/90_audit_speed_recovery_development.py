#!/usr/bin/env python3
"""Development-only score audit; never changes registered planner or branches."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from bces.oracle.planner import PlannerConfig
from bces.oracle.world import CostVector, GroundTruthWorld, PlannedTrajectory
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, route_for
from bces.utils.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/study_b/prefix_matched_crossing_utility_development_v2"
SOURCE_REPORT_SHA256 = "63ad1bcf2c00837a311ff7db256315e3232a763722a5076401f672ec8e59452e"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
RECOVERY_PROGRESS_WEIGHT = 4.0
CLEARANCE_X_M = 8.0  # label-only audit subset; never used by the policy score


def action_scores(planner, sample, target_speed_mps):
    ego = ego_from_sample(sample)
    route = route_for(ego, "keep", planner)
    empty = GroundTruthWorld(sample["timestamp_ms"], ())
    scores = []
    for acceleration in planner.config.longitudinal_accelerations_mps2:
        points = planner._candidate(ego, route, acceleration, "keep")
        if points is None:
            continue
        placeholder = PlannedTrajectory("keep", acceleration, points,
                                        CostVector(0, 0, 0, 0, 0, 0, 0, None))
        cost = planner.evaluate(placeholder, empty, ego, route)
        distance = math.hypot(points[-1].x_m - ego.x_m, points[-1].y_m - ego.y_m)
        desired = min(route.speed_limit_mps, target_speed_mps) * planner.config.horizon_s
        recovery_progress = max(0.0, desired - distance) / max(desired, 1e-6)
        recovery_total = (cost.total
                          - planner.config.weight_map["progress"] * cost.progress
                          + RECOVERY_PROGRESS_WEIGHT * recovery_progress)
        scores.append({"acceleration_mps2": acceleration,
                       "original_total": cost.total,
                       "recovery_total": recovery_total,
                       "collision": cost.collision, "ttc": cost.ttc})
    if not scores:
        raise RuntimeError("no feasible candidates in diagnostic state")
    original = min(scores, key=lambda row: (row["collision"], row["ttc"],
                                             row["original_total"], row["acceleration_mps2"]))
    recovery = min(scores, key=lambda row: (row["collision"], row["ttc"],
                                             row["recovery_total"], row["acceleration_mps2"]))
    return original["acceleration_mps2"], recovery["acceleration_mps2"]


def main():
    if sha256_file(SOURCE / "report.json") != SOURCE_REPORT_SHA256:
        raise RuntimeError("development source report changed")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if report["feasibility_gate"] or report["scenario_clusters"] != 8:
        raise RuntimeError("wrong development source cohort")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    rows = []
    for seed in sorted({row["seed"] for row in report["rows"]}):
        filename = f"{seed}_nominal_periodic_payload.json.gz"
        if sha256_file(SOURCE / filename) != report["artifact_sha256"][filename]:
            raise RuntimeError("trace changed: " + filename)
        with gzip.open(SOURCE / filename, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        target_speed = branch["initial_contract"]["receiver_estimator"]["speed_mps"]
        observed = {item["timestamp_ms"]: item for item in branch["visibility_audit"]}
        for step in branch["rows"]:
            timestamp = step["timestamp_ms"]
            view = observed.get(timestamp)
            if not view or view["target_xy_m"] is None:
                continue
            # Only choose diagnostic states after the actor is well beyond
            # the ego's crossing line. This uses ground truth for the audit
            # subset, not for either scored candidate.
            if view["target_xy_m"][0] - step["receiver"]["x_m"] < CLEARANCE_X_M:
                continue
            original, recovery = action_scores(planner, step["receiver"], target_speed)
            rows.append({"seed": seed, "timestamp_ms": timestamp,
                         "ego_speed_mps": step["receiver"]["speed_mps"],
                         "target_speed_mps": target_speed,
                         "original_empty_world_acceleration_mps2": original,
                         "recovery_empty_world_acceleration_mps2": recovery})
    by_seed = Counter(row["seed"] for row in rows)
    result = {"status": "DEVELOPMENT_ONLY_EMPTY_WORLD_SCORE_AUDIT",
              "source_report_sha256": SOURCE_REPORT_SHA256,
              "registration_sha256": sha256_file(REGISTRATION),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "recovery_progress_weight": RECOVERY_PROGRESS_WEIGHT,
              "label_only_actor_x_clearance_m": CLEARANCE_X_M,
              "scored_perceived_world": "empty; no future actor state in score",
              "rows": rows, "clear_states": len(rows),
              "seeds_with_clear_states": len(by_seed),
              "original_positive_acceleration_states": sum(
                  row["original_empty_world_acceleration_mps2"] > 0 for row in rows),
              "recovery_positive_acceleration_states": sum(
                  row["recovery_empty_world_acceleration_mps2"] > 0 for row in rows),
              "not_closed_loop": True, "not_policy_selection": True,
              "manuscript_allowed": False}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
