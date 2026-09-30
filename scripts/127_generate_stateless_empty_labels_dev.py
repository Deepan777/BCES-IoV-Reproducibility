#!/usr/bin/env python3
"""Generate policy-matched synthetic labels on already-open development seeds."""

from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.simulation.branching import _populate
from bces.simulation.controlled_contract import (
    centered_world, ego_from_sample, make_message, receiver_state,
    start_controlled,
)
from bces.simulation.development_empty_decision_labels import (
    empty_decision_pair, empty_reference_contract,
)
from bces.simulation.development_empty_view_input import DevelopmentEmptyViewPremise
from bces.simulation.development_map_loop import configure_ego_lane_actuation
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner
from bces.simulation.development_visibility_map import visible_local_objects_with_map
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import canonical_json_hash, sha256_file
from scripts._negative_evidence_freshness_support import (
    BLOCKERS, SELECTED_PROTOCOL, SELECTED_PROTOCOL_SHA256,
    selected_development_seeds,
)

OUTPUT = ROOT / "outputs/study_b/stateless_empty_labels_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
PILOT_REPORT = ROOT / "outputs/study_b/stateless_empty_planner_development_v1/report.json"
PILOT_REPORT_SHA256 = "91838157e76aa427f09454640bf8a7c44584b499d55ba509fc6d1da659c2745a"
PLAN = ROOT / "docs/STUDY_B_STATELESS_EMPTY_LABEL_GENERATION_DEV_V1_PLAN.md"
AGES_S = (0.0, 0.2, 0.4, 0.8, 1.2, 2.0)
CONTINUATIONS_MPS2 = (-1.0, 0.0, 1.0)
VIEW_ID = "synthetic_ideal_ego_centered_120m_conflict_view_v1"


def _premise(message):
    if message.objects:
        return None
    return DevelopmentEmptyViewPremise(
        hashlib.sha256(message.encode()).hexdigest(), message.sender_id,
        message.timestamp_ms, VIEW_ID, True)


def _trace(*, seed, spec, speed_mps, continuation_mps2, presence, config):
    connection = start_controlled(
        network=ROOT / config["network"],
        label=f"empty_labels_dev_{seed}_{presence}_{continuation_mps2}",
        seed=seed, step_s=config["step_s"])
    frames = {}
    previous = None
    lane_jump_violations = []
    lane_prior = None
    try:
        _populate(connection, spec)
        configure_ego_lane_actuation(connection)
        if presence == "absent":
            connection.vehicle.remove(spec.actor_id)
        end_s = config["reference_time_s"] + max(AGES_S) + config["horizon_s"]
        for _ in range(round(end_s / config["step_s"])):
            now_s = connection.simulation.getTime()
            if "ego" in connection.vehicle.getIDList():
                age_s = max(0.0, now_s - config["reference_time_s"])
                speed = max(0.0, min(20.0, speed_mps + continuation_mps2 * age_s))
                connection.vehicle.setSpeed("ego", speed)
            if spec.uncontrolled_signal:
                for signal in connection.trafficlight.getIDList():
                    lights = connection.trafficlight.getRedYellowGreenState(signal)
                    connection.trafficlight.setRedYellowGreenState(signal, "G" * len(lights))
            connection.simulationStep()
            timestamp = round(connection.simulation.getTime() * 1000)
            receiver = None
            if "ego" in connection.vehicle.getIDList():
                receiver = receiver_state(connection, previous)
                previous = receiver
                lane = (connection.vehicle.getRoadID("ego"),
                        connection.vehicle.getLaneID("ego"),
                        connection.vehicle.getLanePosition("ego"),
                        connection.vehicle.getSpeed("ego"))
                if lane_prior and lane[:2] == lane_prior[:2]:
                    forward = lane[2] - lane_prior[2]
                    maximum = max(lane[3], lane_prior[3]) * config["step_s"] + 0.5
                    if forward < -0.5 or forward > maximum:
                        lane_jump_violations.append({"timestamp_ms": timestamp,
                                                     "lane_delta_m": forward,
                                                     "maximum_expected_m": maximum})
                lane_prior = lane
            frames[timestamp] = {"receiver": receiver,
                                 "objects": centered_world(connection)}
    finally:
        connection.close()
    return frames, lane_jump_violations


def _observed(frames, timestamp, config, seed):
    frame = frames[timestamp]
    if frame["receiver"] is None:
        raise ValueError("receiver_unavailable")
    ego = ego_from_sample(frame["receiver"])
    local = visible_local_objects_with_map(
        frame["objects"], ego, blockers=BLOCKERS,
        range_m=config["local_range_m"])
    cooperative = tuple(item for item in frame["objects"]
                        if math.hypot(item.x_m - ego.x_m, item.y_m - ego.y_m)
                        <= config["cooperative_range_m"])
    message = make_message(cooperative, timestamp, message_id=seed)
    return ego, local, message, _premise(message)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, digest in ((REGISTRATION, REGISTRATION_SHA256),
                         (PILOT_REPORT, PILOT_REPORT_SHA256),
                         (ROOT / SELECTED_PROTOCOL, SELECTED_PROTOCOL_SHA256)):
        if sha256_file(path) != digest:
            raise RuntimeError("frozen input changed: " + str(path))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    selected = selected_development_seeds(ROOT)
    config = dict(registration["config"])
    config["reference_time_s"] = 3.0
    config["horizon_s"] = PlannerConfig.from_yaml(ROOT / config["planner"]).horizon_s
    if config["cooperative_range_m"] < 120.0:
        raise RuntimeError("source range below the development premise")
    thresholds_yaml = yaml.safe_load((ROOT / config["validity"]).read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(**{
        key: thresholds_yaml[key] for key in (
            "max_cost_regret", "max_cached_risk", "max_trajectory_deviation_m")
    })
    scales = DriftScales(*config["drift_scales"])
    source_paths = (
        "bces/simulation/development_empty_view_input.py",
        "bces/simulation/development_empty_decision_labels.py",
        "bces/simulation/development_stateless_empty_planner.py",
        "bces/simulation/development_visibility_map.py",
        "bces/simulation/development_map_loop.py",
        "scripts/_negative_evidence_freshness_support.py",
    )
    protocol = {
        "status": "POLICY_MATCHED_EMPTY_LABELS_DEVELOPMENT_ONLY_V1",
        "script_sha256": sha256_file(Path(__file__)),
        "plan_sha256": sha256_file(PLAN),
        "source_sha256": {name: sha256_file(ROOT / name) for name in source_paths},
        "registration_sha256": REGISTRATION_SHA256,
        "pilot_report_sha256": PILOT_REPORT_SHA256,
        "selected_protocol_sha256": SELECTED_PROTOCOL_SHA256,
        "seeds": [item["seed"] for item in selected],
        "timings": ["near", "timing_control"],
        "presence": ["present", "absent"],
        "continuations_mps2": list(CONTINUATIONS_MPS2),
        "ages_s": list(AGES_S),
        "expected_traces": 96,
        "maximum_decision_pairs": 576,
        "view_id": VIEW_ID,
        "source_view_premise_is_synthetic": True,
        "model_training_authorized": False,
        "independent_confirmation": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifacts = {}
    rows = []
    stable_references = {}
    for index, item in enumerate(selected):
        seed = item["seed"]
        for timing in protocol["timings"]:
            position = item["near_position_m" if timing == "near" else "control_position_m"]
            spec = replace(scenario_spec("unprotected_crossing"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=9.0, hidden_from_local=False)
            for presence in protocol["presence"]:
                for continuation in CONTINUATIONS_MPS2:
                    frames, lane_jumps = _trace(
                        seed=seed, spec=spec, speed_mps=item["speed_mps"],
                        continuation_mps2=continuation, presence=presence,
                        config=config)
                    planner = DevelopmentStatelessEmptyPlanner(
                        PlannerConfig.from_yaml(ROOT / config["planner"]))
                    references = []
                    points = []
                    abstentions = []
                    try:
                        ref_ego, ref_local, ref_message, ref_premise = _observed(
                            frames, 3000, config, seed)
                        reference, query, cached = empty_reference_contract(
                            scenario=str(seed), split="development",
                            behavior="keep", timestamp=3000, ego=ref_ego,
                            local=ref_local, message=ref_message,
                            empty_premise=ref_premise, planner=planner,
                            scales=scales)
                        references.append(reference)
                        identity = (seed, timing, presence)
                        if identity in stable_references and stable_references[identity] != reference:
                            raise AssertionError("continuation changed the frozen reference")
                        stable_references[identity] = reference
                    except (ValueError, RuntimeError, KeyError) as exc:
                        abstentions.append({"age_s": None, "reason": str(exc)})
                        reference = None
                    if reference is not None:
                        for age_s in AGES_S:
                            timestamp = 3000 + round(age_s * 1000)
                            try:
                                ego, local, fresh, fresh_premise = _observed(
                                    frames, timestamp, config, seed)
                                point = empty_decision_pair(
                                    reference=reference, query=query,
                                    cached_reference=cached, ego=ego, local=local,
                                    fresh_message=fresh,
                                    fresh_empty_premise=fresh_premise,
                                    future_world={t: f["objects"]
                                                  for t, f in frames.items()},
                                    planner=planner, thresholds=thresholds)
                                points.append(point)
                            except (ValueError, RuntimeError, KeyError) as exc:
                                abstentions.append({"age_s": age_s,
                                                     "reason": f"{type(exc).__name__}:{exc}"})
                    name = f"{seed}_{timing}_{presence}_{continuation:+.0f}.json.gz"
                    artifact = {
                        "seed": seed, "timing": timing, "presence": presence,
                        "continuation_mps2": continuation,
                        "scenario_spec": asdict(spec),
                        "parameters": {"speed_mps": item["speed_mps"],
                                       "actor_depart_position_m": position},
                        "lane_jump_violations": lane_jumps,
                        "references": references, "points": points,
                        "abstentions": abstentions,
                        "frames": {str(t): {
                            "receiver": f["receiver"],
                            "objects": [asdict(obj) for obj in f["objects"]],
                        } for t, f in frames.items()},
                    }
                    with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                        json.dump(artifact, handle, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
                    artifacts[name] = sha256_file(OUTPUT / name)
                    rows.append({"name": name, "seed": seed,
                                 "timing": timing, "presence": presence,
                                 "continuation_mps2": continuation,
                                 "points": len(points),
                                 "abstentions": len(abstentions),
                                 "lane_jump_violations": len(lane_jumps),
                                 "cached_expired_fresh_empty": sum(
                                     p["cached_policy_selection_sidecar"]["stop_hold_fallback"]
                                     and p["fresh_policy_selection_sidecar"]["fresh_empty_used"]
                                     for p in points),
                                 "physical_payload_points": sum(
                                     p["fresh_status"] == "objects" for p in points)})
        print(json.dumps({"completed_seed_clusters": index + 1,
                          "total_seed_clusters": len(selected)}), flush=True)
    total_points = sum(row["points"] for row in rows)
    summary = {
        "traces": len(rows), "points": total_points,
        "abstentions": sum(row["abstentions"] for row in rows),
        "lane_jump_violations": sum(row["lane_jump_violations"] for row in rows),
        "absent_near_expired_vs_fresh_pairs": sum(
            row["cached_expired_fresh_empty"] for row in rows
            if row["presence"] == "absent" and row["timing"] == "near"),
        "present_physical_payload_points": sum(
            row["physical_payload_points"] for row in rows
            if row["presence"] == "present"),
        "seed_clusters": len({row["seed"] for row in rows}),
    }
    summary["engineering_screen_passed"] = bool(
        summary["traces"] == 96 and summary["seed_clusters"] == 8
        and summary["lane_jump_violations"] == 0
        and summary["absent_near_expired_vs_fresh_pairs"] > 0
        and summary["present_physical_payload_points"] > 0)
    report = {"status": protocol["status"],
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "summary": summary,
              "independent_confirmation": False,
              "model_training_authorized": False,
              "external_view_certified": False}
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
