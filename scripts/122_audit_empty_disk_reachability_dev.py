#!/usr/bin/env python3
"""Development-only nonvacuity audit for a conditional empty-disk bound."""

from __future__ import annotations

from collections import defaultdict
import gzip
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
from bces.utils.hashing import sha256_file

BASE = ROOT / "outputs/study_b/negative_evidence_crossing_development_v1"
BASE_REPORT_SHA256 = "c16589984c7ece771cce70884c6a4265dc65357c0067b002f80a2fb40d3530cd"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
GEOMETRY = ROOT / "bces/geometry/empty_view_reachability.py"
GEOMETRY_SHA256 = "07dca55fbc4f69f2131dc469643a7ba03045a63153a7b38cb550a71f8f221734"
OUTPUT = ROOT / "outputs/study_b/empty_disk_reachability_development_v1"
AGES_S = (0.2, 0.4, 0.8, 1.2)
SOURCE_RADIUS_M = 120.0
MAX_ACTOR_SPEED_MPS = 20.0
ACTOR_LENGTH_M, ACTOR_WIDTH_M = 5.0, 1.8
LOCALIZATION_ERROR_M = 0.5
CROSSING_Y_M = 95.2


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, digest in ((BASE / "report.json", BASE_REPORT_SHA256),
                         (REGISTRATION, REGISTRATION_SHA256),
                         (GEOMETRY, GEOMETRY_SHA256)):
        if sha256_file(path) != digest:
            raise RuntimeError("frozen input changed: " + str(path))
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if baseline["summary"]["branches"] != 256 or not baseline["summary"]["engineering_passed_all"]:
        raise RuntimeError("balanced development cohort incomplete")
    for name, digest in baseline["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("saved development branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    planner = ControlledDecisionPlanner(
        PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    margin = planner.config.collision_margin_m
    actor_radius = 0.5 * math.hypot(ACTOR_LENGTH_M + 2 * margin,
                                    ACTOR_WIDTH_M + 2 * margin)
    rows = []
    branch_names = sorted(name for name in baseline["artifact_sha256"]
                          if name.endswith("_absent_negative_evidence_periodic_payload.json.gz"))
    if len(branch_names) != 32:
        raise RuntimeError("expected all 32 actor-absent periodic branches")
    for name in branch_names:
        with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        if branch["initial_contract"]["parameters"]["actor_presence"] != "absent":
            raise RuntimeError("unexpected actor-present branch")
        fields = name.split("_")
        seed = int(fields[0])
        timing = "near" if "_near_" in name else "timing_control"
        condition = "nominal" if "_nominal_" in name else "impaired"
        visibility = {row["timestamp_ms"]: row for row in branch["visibility_audit"]}
        for sample in branch["rows"]:
            if "receiver" not in sample:
                continue
            ego = ego_from_sample(sample["receiver"])
            if ego.y_m >= CROSSING_Y_M:
                continue
            timestamp = sample["timestamp_ms"]
            route = route_for(ego, "keep", planner)
            try:
                candidate = planner.plan((), (), ego, route)
            except RuntimeError as exc:
                if str(exc) != "no feasible trajectory candidate":
                    raise
                candidate = None
            for age in AGES_S:
                reference_ms = timestamp - round(age * 1000)
                source = visibility.get(reference_ms)
                if reference_ms < 3000 or source is None:
                    continue
                if source["target_xy_m"] is not None:
                    raise RuntimeError("empty source contract has an actor at reference")
                if candidate is None:
                    clear, slack, intervals = False, None, 0
                else:
                    ego_radius = 0.5 * math.hypot(ego.length_m + 2 * margin,
                                                   ego.width_m + 2 * margin)
                    result = conditional_empty_disk_clearance(
                        source_center_xy_m=tuple(source["ego_xy_m"]),
                        source_radius_m=SOURCE_RADIUS_M,
                        observation_age_s=age,
                        ego_initial_xy_m=(ego.x_m, ego.y_m),
                        ego_future_points=candidate.points,
                        maximum_ego_speed_mps=route.speed_limit_mps,
                        maximum_actor_speed_mps=MAX_ACTOR_SPEED_MPS,
                        ego_body_radius_m=ego_radius,
                        actor_body_radius_m=actor_radius,
                        localization_error_m=LOCALIZATION_ERROR_M)
                    clear, slack, intervals = (result.clear, result.minimum_slack_m,
                                               result.checked_intervals)
                rows.append({"seed": seed, "timing": timing, "condition": condition,
                             "timestamp_ms": timestamp, "age_s": age,
                             "ego_y_m": ego.y_m, "ego_speed_mps": ego.speed_mps,
                             "candidate_available": candidate is not None,
                             "continuous_conditional_clearance": clear,
                             "minimum_slack_m": slack,
                             "checked_intervals": intervals})
    summary = {}
    for age in AGES_S:
        subset = [row for row in rows if row["age_s"] == age]
        by_seed = {}
        for seed in sorted({row["seed"] for row in subset}):
            group = [row for row in subset if row["seed"] == seed]
            by_seed[str(seed)] = {"states": len(group),
                                  "clear_states":
                                      sum(row["continuous_conditional_clearance"] for row in group),
                                  "fraction":
                                      sum(row["continuous_conditional_clearance"] for row in group)
                                      / len(group)}
        clear_count = sum(row["continuous_conditional_clearance"] for row in subset)
        summary[str(age)] = {"approach_states": len(subset),
                             "conditional_clear_states": clear_count,
                             "fraction": clear_count / len(subset),
                             "seeds_at_least_70pct_clear":
                                 sum(item["fraction"] >= 0.7 for item in by_seed.values()),
                             "median_slack_m": statistics.median(
                                 row["minimum_slack_m"] for row in subset
                                 if row["minimum_slack_m"] is not None),
                             "by_seed": by_seed}
    primary = summary["0.4"]
    gate = (primary["fraction"] >= 0.8
            and primary["seeds_at_least_70pct_clear"] >= 6
            and all(row["candidate_available"] for row in rows if row["age_s"] == 0.4))
    report = {"status": "EMPTY_DISK_CONDITIONAL_NONVACUITY_DEVELOPMENT_ONLY_V1",
              "script_sha256": sha256_file(Path(__file__)),
              "geometry_source_sha256": GEOMETRY_SHA256,
              "balanced_pilot_report_sha256": BASE_REPORT_SHA256,
              "assumptions": {"complete_empty_ego_centered_source_disk_radius_m": SOURCE_RADIUS_M,
                              "maximum_actor_speed_mps": MAX_ACTOR_SPEED_MPS,
                              "actor_length_m": ACTOR_LENGTH_M,
                              "actor_width_m": ACTOR_WIDTH_M,
                              "maximum_ego_speed_mps": planner.config.speed_limit_mps,
                              "localization_error_m": LOCALIZATION_ERROR_M,
                              "no_actor_spawn_inside_observed_disk": True},
              "primary_age_s": 0.4, "ages_s": list(AGES_S),
              "development_nonvacuity_gate": gate,
              "summary": summary, "rows": rows,
              "limitations": ["sender's complete empty ego-centered 120-m disk is ideal synthetic ground truth, not observed source coverage",
                              "reference timestamps are hypothetical causal periodic samples, not verified packet deliveries",
                              "candidate path uses the no-object controlled planner, not a newly bound BCES/TTL policy",
                              "continuous-time clearance is conditional on unverified speed, body, localization, no-spawn, and actuation bounds",
                              "near/control and network cells are repeated within eight opened development seeds",
                              "no communication-byte advantage, regret interval, real-world safety, or fresh confirmation"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
