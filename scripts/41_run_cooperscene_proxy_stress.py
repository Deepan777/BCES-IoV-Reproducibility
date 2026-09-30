#!/usr/bin/env python3
"""Frozen, descriptive CooperScene keep-path proxy; no road-safety claim."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.geometry.drift import DriftScales, KinematicState, compute_drift, wrap_angle_rad
from bces.models.bound_policy import FrozenWirePolicy
from bces.models.decision_surface import GUARD, STEP
from bces.models.reference_input import freeze_reference_input
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.oracle.world import EgoState, Route, WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundQuery
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.simulation.controlled_contract import make_message, payload_objects
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, decision_pair
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic

ROLE_AUDIT = ROOT / "outputs/study_b/cooperscene_receiver_role_audit_v1.json"
REG = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
OUT = ROOT / "outputs/study_b/cooperscene_proxy_stress_v1/result.json"
MANIFESTS = (ROOT / "data/manifests/cooperscene_annotations_v1.json",
             ROOT / "data/manifests/cooperscene_annotations_extra_agents_v1.json")
DECISION_INDICES = (15, 20, 25, 29)
REFERENCE_INDEX = 10
SENDER = "cooperscene:rsu0"


class UnknownInput(ValueError):
    pass


def verify_and_load(audit):
    frames = defaultdict(lambda: defaultdict(dict))
    for path in MANIFESTS:
        if sha256_file(path) != audit["manifest_sha256"][path.name]:
            raise RuntimeError(f"manifest changed: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "annotations_acquired_no_bces_outcome_computed":
            raise RuntimeError("source manifest status mismatch")
        for row in manifest["records"]:
            source = ROOT / row["relative_path"]
            if sha256_file(source) != row["sha256"]:
                raise RuntimeError(f"annotation changed: {source}")
            frames[row["scene"]][row["frame"]][row["agent"]] = yaml.safe_load(source.read_bytes())
    result = {}
    for scene, sequence in frames.items():
        identifiers = sorted(sequence)
        if len(identifiers) != 60 or any(b - a != 1 for a, b in zip(identifiers, identifiers[1:])):
            raise RuntimeError(f"scene window incomplete: {scene}")
        result[scene] = [sequence[frame] for frame in identifiers]
    return result


def ego_at(sequence, agent, i):
    if i < 2:
        raise UnknownInput("insufficient_past_ego_poses")
    poses = [sequence[j][agent].get("lidar_pose") for j in (i - 2, i - 1, i)]
    if any(not isinstance(p, list) or len(p) != 6 or not all(math.isfinite(float(v)) for v in p) for p in poses):
        raise UnknownInput("missing_or_invalid_ego_pose")
    speeds = [math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1 for a, b in zip(poses, poses[1:])]
    speed = speeds[-1]
    if speed > 20.0:
        raise UnknownInput("ego_speed_exceeds_frozen_proxy_limit")
    heading = math.radians(poses[-1][4])
    previous_heading = math.radians(poses[-2][4])
    acceleration = (speeds[-1] - speeds[-2]) / 0.1
    curvature = wrap_angle_rad(heading - previous_heading) / 0.1 / max(speed, 2.0)
    return EgoState(poses[-1][0], poses[-1][1], speed, heading, acceleration, curvature)


def objects_at(sequence, agent, i, ego, radius, source):
    if i < 1:
        raise UnknownInput("insufficient_past_object_track")
    now = sequence[i][agent].get("vehicles", {})
    previous = sequence[i - 1][agent].get("vehicles", {})
    if not isinstance(now, dict) or not isinstance(previous, dict):
        raise UnknownInput("invalid_object_mapping")
    objects = []
    for identifier, item in sorted(now.items()):
        try:
            location, extent, angle = item["location"], item["extent"], item["angle"]
            center = item.get("center", [0.0, 0.0, 0.0])
            yaw = math.radians(float(angle[1]))
            x = float(location[0]) + math.cos(yaw) * float(center[0]) - math.sin(yaw) * float(center[1])
            y = float(location[1]) + math.sin(yaw) * float(center[0]) + math.cos(yaw) * float(center[1])
            length, width = 2 * float(extent[0]), 2 * float(extent[1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UnknownInput("invalid_object_geometry") from exc
        distance = math.hypot(x - ego.x_m, y - ego.y_m)
        if distance < 3.0 or (radius is not None and distance > radius):
            continue
        if identifier in previous:
            old = previous[identifier]
            old_angle = math.radians(float(old["angle"][1]))
            old_center = old.get("center", [0.0, 0.0, 0.0])
            old_x = float(old["location"][0]) + math.cos(old_angle) * float(old_center[0]) - math.sin(old_angle) * float(old_center[1])
            old_y = float(old["location"][1]) + math.sin(old_angle) * float(old_center[0]) + math.cos(old_angle) * float(old_center[1])
            vx, vy = (x - old_x) / 0.1, (y - old_y) / 0.1
        elif "speed" in item:
            speed = float(item["speed"]) / 3.6
            vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)
        else:
            raise UnknownInput("object_velocity_unavailable")
        if not all(math.isfinite(v) for v in (x, y, vx, vy, yaw, length, width)) or length <= 0 or width <= 0:
            raise UnknownInput("nonfinite_or_invalid_object")
        objects.append(WorldObject(f"vehicle:{identifier}", x, y, vx, vy, yaw,
                                   length, width, source=source))
    return tuple(objects)


def observations_at(sequence, agent, i, ego):
    local = objects_at(sequence, agent, i, ego, 30.0, "cooperscene_receiver_annotation")
    remote = objects_at(sequence, 0, i, ego, 120.0, "cooperscene_rsu_annotation")
    by_local = {obj.track_id: obj for obj in local}
    for obj in remote:
        first = by_local.get(obj.track_id)
        if first and math.hypot(first.x_m - obj.x_m, first.y_m - obj.y_m) > 0.01:
            raise UnknownInput("cross_view_global_id_position_mismatch")
    return local, remote


def reference_for(sequence, scene, agent, planner, scales):
    i = REFERENCE_INDEX
    t = i * 100
    ego = ego_at(sequence, agent, i)
    local, remote = observations_at(sequence, agent, i, ego)
    route = Route(ego.x_m, ego.y_m, ego.heading_rad, 3.6, 20.0,
                  False, False, "keep", ego.curvature_inv_m)
    message = make_message(remote, t, message_id=scene, sender=SENDER)
    perceived = payload_objects(message)
    plan = planner.plan(local, perceived, ego, route)
    state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad,
                           ego.acceleration_mps2, ego.curvature_inv_m, t)
    path = tuple(PathPoint(p.x_m, p.y_m, p.speed_mps, p.heading_rad,
                           route.curvature_inv_m, p.time_s) for p in plan.points[:12])
    query = ReceiverQuery(scene, sender_id_hash(SENDER), state, path, Behavior.KEEP,
                          1, planner.policy_hash, 0, scales)
    occlusion = max(0.0, min(1.0, (len(perceived) - len(local)) / max(len(perceived), 1)))
    frozen = freeze_reference_input(message, query, generated_at_ms=t,
        query_available_at_ms=t, context_available_at_ms=t,
        query_provenance="frozen_reference_policy", occlusion_proxy=occlusion,
        estimated_delay_s=0.0, map_context_flags=0)
    reference = {"reference_id": f"cooperscene:{scene}:{agent}:{i}",
                 "scenario_id": str(scene), "split": "external_proxy", "behavior": "keep",
                 "frozen_input": frozen.to_dict(),
                 "observable_margin_features": observable_planner_features(planner, local, perceived, ego, route)}
    return reference, query, perceived, message


def future_world_for(sequence, agent, current):
    future = {}
    for index in range(current + 2, current + 31, 2):
        ego = ego_at(sequence, agent, index)
        receiver = objects_at(sequence, agent, index, ego, None, "cooperscene_proxy_world")
        roadside = objects_at(sequence, 0, index, ego, None, "cooperscene_proxy_world")
        union = {obj.track_id: obj for obj in roadside}
        for obj in receiver:
            old = union.get(obj.track_id)
            if old and math.hypot(old.x_m - obj.x_m, old.y_m - obj.y_m) > 0.01:
                raise UnknownInput("future_cross_view_position_mismatch")
            union[obj.track_id] = obj
        future[index * 100] = tuple(union[key] for key in sorted(union))
    return future


def rule_accepts(policy, bound, payload, drift):
    offsets, origin = policy.predict(bound.encode(), payload)
    if not origin:
        return False
    quantized = np.floor(np.asarray(offsets) / STEP).clip(0, 255) * STEP
    if policy.family == "surface":
        return bool(np.min(quantized - np.asarray(NORMAL_CODEBOOK_V1) @ np.asarray(drift)) >= GUARD)
    return bool(quantized[0] - drift[6] >= GUARD)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads(ROLE_AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "SCHEMA_ONLY_NO_BCES_OUTCOMES" or audit["take_count"] != 14:
        raise RuntimeError("external schema gate not complete")
    frames = verify_and_load(audit)
    reg = json.loads(REG.read_text(encoding="utf-8"))
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml"))
    if planner.policy_hash != reg["policy_hash"]:
        raise RuntimeError("frozen planner policy hash changed")
    scales = DriftScales(*reg["config"]["drift_scales"])
    raw_thresholds = yaml.safe_load((ROOT / "configs/oracle/validity_v1.yaml").read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(raw_thresholds["max_cost_regret"],
        raw_thresholds["max_cached_risk"], raw_thresholds["max_trajectory_deviation_m"])
    models = {}
    for method in ("surface", "scalar_ttl"):
        rule = reg["model_rules"][method]
        models[method] = (FrozenWirePolicy(ROOT / rule["path"],
            expected_sha256=rule["sha256"], policy_hash=planner.policy_hash), rule["shrinkage"])
    rows = []
    for take in audit["takes"]:
        scene, agent = take["scene"], take["selected_receiver_agent"]
        if agent is None:
            rows.extend({"scene": scene, "receiver_agent": None, "decision_index": i,
                         "status": "stationary_receiver_abstention"} for i in DECISION_INDICES)
            continue
        sequence = frames[scene]
        try:
            reference, query, cached_reference, message = reference_for(sequence, scene, agent, planner, scales)
            payload = message.encode()
            bounds = {method: BoundQuery(query, hashlib.sha256(payload).hexdigest(), model.sha256,
                       model.view, method, shrinkage, tuple(reference["observable_margin_features"]),
                       query.reference_state.timestamp_ms,
                       reference["frozen_input"]["batch"]["occlusion_proxy"], 0.0, 0)
                       for method, (model, shrinkage) in models.items()}
        except (UnknownInput, RuntimeError, ValueError) as exc:
            rows.extend({"scene": scene, "receiver_agent": agent, "decision_index": i,
                         "status": "reference_abstention", "reason": str(exc)} for i in DECISION_INDICES)
            continue
        for i in DECISION_INDICES:
            row = {"scene": scene, "receiver_agent": agent, "reference_index": REFERENCE_INDEX,
                   "decision_index": i, "age_s": (i - REFERENCE_INDEX) * 0.1}
            try:
                ego = ego_at(sequence, agent, i)
                local, remote = observations_at(sequence, agent, i, ego)
                fresh_message = make_message(remote, i * 100, message_id=scene, sender=SENDER)
                state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad,
                                       ego.acceleration_mps2, ego.curvature_inv_m, i * 100)
                drift = compute_drift(query.reference_state, state, scales).normalized
                row["accepted"] = {method: rule_accepts(model, bounds[method], payload, drift)
                                   for method, (model, _) in models.items()}
                world = future_world_for(sequence, agent, i)
                label = decision_pair(reference=reference, query=query,
                    cached_reference=cached_reference, ego=ego, local=local,
                    fresh_message=fresh_message, future_world=world,
                    planner=planner, thresholds=thresholds)
                row.update({"status": "proxy_label_available", "proxy_valid": label["valid"],
                            "proxy_cost_regret": label["cost_regret"],
                            "proxy_cached_risk": label["cached_risk"],
                            "proxy_trajectory_deviation_m": label["trajectory_deviation_m"],
                            "normalized_drift": list(drift)})
            except (UnknownInput, RuntimeError, ValueError, KeyError) as exc:
                row.update({"status": "unknown_or_abstention", "reason": str(exc)})
            rows.append(row)
    summary = {}
    for method in models:
        available = [r for r in rows if r["status"] == "proxy_label_available"]
        accepted = [r for r in available if r["accepted"][method]]
        summary[method] = {"labelled_slots": len(available), "accepted_slots": len(accepted),
                           "proxy_invalid_accepted": sum(not r["proxy_valid"] for r in accepted),
                           "accepted_takes": len({r["scene"] for r in accepted})}
    report = {"status": "DESCRIPTIVE_MAP_FREE_ANNOTATION_PROXY_ONLY",
              "not_full_external_validation": True,
              "registration_sha256": sha256_file(REG),
              "schema_audit_sha256": sha256_file(ROLE_AUDIT),
              "adapter_source_sha256": sha256_file(Path(__file__)),
              "take_count": 14, "candidate_slots": len(rows),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "methods": summary, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, report)
    print(json.dumps({k: report[k] for k in ("status", "take_count", "candidate_slots", "status_counts", "methods")}, indent=2))


if __name__ == "__main__":
    main()
