#!/usr/bin/env python3
"""V2X-Real per-agent annotated-position BCES stress test.

Development uses UCLA val only. Test execution requires a frozen registration.
This is an observational, zero-object-velocity proxy, not a road-safety result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.geometry.drift import DriftScales, KinematicState, wrap_angle_rad
from bces.models.bound_policy import FrozenWirePolicy, bind_reference
from bces.models.reference_input import freeze_reference_input
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.oracle.world import EgoState, Route, WorldObject
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.bound_exchange import BoundReceiver
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.protocol.receiver import DecisionState
from bces.simulation.controlled_contract import make_message, payload_objects
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, decision_pair
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic


ARCHIVES = {
    "val": ROOT / "data/raw/v2x_real_lidar64/val.zip",
    "test": ROOT / "data/raw/v2x_real_lidar64/test.zip",
}
ARCHIVE_HASHES = {
    "val": "e06e6525f31da009566ae553dce4cf34cc2c636f7a1162ed41809ee804fd9b7b",
    "test": "add21fa98208e4e487bd4c5f986cb0a58712efbb5a6cc33b996d1b0080560899",
}
SCHEMA_AUDITS = {
    "val": ROOT / "outputs/study_b/v2xreal_receiver_schema_audit_val_v1.json",
    "test": ROOT / "outputs/study_b/v2xreal_receiver_schema_audit_v1.json",
}
REGISTRATION = ROOT / "docs/STUDY_B_V2XREAL_POSITION_PROXY_REGISTRATION_V1.json"
V2_REG = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
OUT = {
    "val": ROOT / "outputs/study_b/v2xreal_position_proxy_val_v2/result.json",
    "test": ROOT / "outputs/study_b/v2xreal_position_proxy_test_v1/result.json",
}
REFERENCE_START = 20
REFERENCE_STEP = 5
OFFSETS = (5, 10, 15, 20)
FUTURE_HORIZON_FRAMES = 30
LOCAL_RADIUS_M = 30.0
REMOTE_RADIUS_M = 120.0
MOBILE_DISPLACEMENT_M = 10.0
SENDER = "v2xreal:roadside:-1:annotated_position"


class UnknownInput(ValueError):
    pass


def _pose(frame: dict) -> list[float]:
    values = frame.get("true_ego_pose")
    if not isinstance(values, list) or len(values) < 6 or not all(math.isfinite(float(v)) for v in values[:6]):
        raise UnknownInput("ego_pose_unavailable")
    return values


def ego_at(vehicle_frames: dict[int, dict], index: int) -> EgoState:
    if not all(i in vehicle_frames for i in (index - 2, index - 1, index)):
        raise UnknownInput("ego_history_missing")
    a, b, c = (_pose(vehicle_frames[i]) for i in (index - 2, index - 1, index))
    speed0 = math.dist(a[:2], b[:2]) / 0.1
    speed1 = math.dist(b[:2], c[:2]) / 0.1
    heading = math.radians(float(c[4]))
    prev_heading = math.radians(float(b[4]))
    acceleration = (speed1 - speed0) / 0.1
    curvature = wrap_angle_rad(heading - prev_heading) / 0.1 / max(speed1, 2.0)
    if speed1 > 35 or abs(acceleration) > 30 or abs(curvature) > 1:
        raise UnknownInput("implausible_ego_kinematics")
    return EgoState(float(c[0]), float(c[1]), speed1, heading, acceleration, curvature)


def objects_at(frame: dict, ego: EgoState, radius_m: float, source: str) -> tuple[WorldObject, ...]:
    objects = frame.get("vehicles")
    if not isinstance(objects, dict):
        raise UnknownInput("per_view_object_list_unavailable")
    result = []
    for key, record in sorted(objects.items()):
        try:
            xyz = record["location"]
            extent = record["extent"]
            angle = record["angle"]
            x, y = float(xyz[0]), float(xyz[1])
            if math.hypot(x - ego.x_m, y - ego.y_m) > radius_m:
                continue
            length, width = 2 * float(extent[0]), 2 * float(extent[1])
            yaw = math.radians(float(angle[1]))
            name = str(record.get("obj_type", ""))
            vulnerable = any(term in name.lower() for term in ("pedestrian", "bicycle", "scooter", "motorcycle"))
            result.append(WorldObject(f"source:{key}", x, y, 0.0, 0.0,
                                      yaw, length, width, int(vulnerable), 1.0, source))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UnknownInput("invalid_per_view_object_geometry") from exc
    return tuple(result)


def world_at(vehicle: dict, roadside: dict, ego: EgoState) -> tuple[WorldObject, ...]:
    remote = objects_at(roadside, ego, REMOTE_RADIUS_M, "roadside_annotation_label")
    local = objects_at(vehicle, ego, REMOTE_RADIUS_M, "vehicle_annotation_label")
    merged = {o.track_id: o for o in remote}
    merged.update({o.track_id: o for o in local})
    return tuple(merged[k] for k in sorted(merged))


def reference_for(base: str, index: int, vehicle: dict, roadside: dict, planner, scales):
    ego = ego_at(vehicle, index)
    local = objects_at(vehicle[index], ego, LOCAL_RADIUS_M, "vehicle_annotation_input")
    remote = objects_at(roadside[index], ego, REMOTE_RADIUS_M, "roadside_annotation_input")
    if not remote:
        raise UnknownInput("empty_roadside_reference_view")
    identity = f"v2xreal:{base}:1:-1:{index}"
    message_id = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    message = make_message(remote, index * 100, message_id=message_id, sender=SENDER)
    perceived = payload_objects(message)
    route = Route(ego.x_m, ego.y_m, ego.heading_rad, planner.config.lane_width_m,
                  planner.config.speed_limit_mps, False, False, "keep", ego.curvature_inv_m)
    plan = planner.plan(local, perceived, ego, route)
    state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad,
                           ego.acceleration_mps2, ego.curvature_inv_m, index * 100)
    path = tuple(PathPoint(p.x_m, p.y_m, p.speed_mps, p.heading_rad,
                           route.curvature_inv_m, p.time_s) for p in plan.points[:12])
    query = ReceiverQuery(digest32((identity + ":query").encode()), sender_id_hash(SENDER),
                          state, path, Behavior.KEEP, 1, planner.policy_hash, 0, scales)
    remote_ids = {o.track_id for o in remote}
    local_ids = {o.track_id for o in local}
    occlusion_proxy = len(remote_ids - local_ids) / max(len(remote_ids), 1)
    frozen = freeze_reference_input(message, query, generated_at_ms=index * 100,
        query_available_at_ms=index * 100, context_available_at_ms=index * 100,
        query_provenance="frozen_reference_policy", occlusion_proxy=occlusion_proxy,
        estimated_delay_s=0.0, map_context_flags=0)
    features = observable_planner_features(planner, local, perceived, ego, route)
    reference = {"reference_id": identity, "scenario_id": base, "split": "external_position_proxy",
                 "behavior": "keep", "frozen_input": frozen.to_dict(),
                 "observable_margin_features": features}
    return reference, query, perceived, message


def _registered_test_check() -> dict:
    if not REGISTRATION.is_file():
        raise RuntimeError("test registration must be frozen before test outcomes")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if registration.get("status") != "FROZEN_BEFORE_TEST_OUTCOMES":
        raise RuntimeError("invalid test registration status")
    for name, path in (("runner", Path(__file__)), ("plan", ROOT / registration["plan_path"]),
                       ("analysis", ROOT / "scripts/52_analyze_v2xreal_position_proxy.py"),
                       ("archive", ARCHIVES["test"]), ("schema_audit", SCHEMA_AUDITS["test"]),
                       ("v2_registration", V2_REG)):
        if sha256_file(path) != registration["sha256"][name]:
            raise RuntimeError(f"test-registered source changed: {name}")
    return registration


def run(split: str) -> None:
    if OUT[split].exists():
        raise FileExistsError(OUT[split])
    if sha256_file(ARCHIVES[split]) != ARCHIVE_HASHES[split]:
        raise RuntimeError("archive hash mismatch")
    schema = json.loads(SCHEMA_AUDITS[split].read_text(encoding="utf-8"))
    if schema["source_sha256"]["archive"] != ARCHIVE_HASHES[split]:
        raise RuntimeError("source schema audit does not match archive")
    registration = _registered_test_check() if split == "test" else None
    v2 = json.loads(V2_REG.read_text(encoding="utf-8"))
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml"))
    if planner.policy_hash != v2["policy_hash"]:
        raise RuntimeError("frozen planner hash changed")
    scales = DriftScales(*v2["config"]["drift_scales"])
    limits = yaml.safe_load((ROOT / "configs/oracle/validity_v1.yaml").read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(limits["max_cost_regret"], limits["max_cached_risk"], limits["max_trajectory_deviation_m"])
    models = {}
    for name in ("surface", "scalar_ttl"):
        rule = v2["model_rules"][name]
        models[name] = (FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                         policy_hash=planner.policy_hash), rule["shrinkage"])

    rows = []
    with zipfile.ZipFile(ARCHIVES[split]) as archive:
        paths: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
        for name in archive.namelist():
            if name.endswith(".yaml"):
                parts = name.split("/")
                paths[parts[1]][parts[2]][int(parts[3][:-5])] = name
        for entry in schema["scenes"]:
            base = entry["base_scene"]
            if entry["paired_vehicle_roadside_frames"] == 0:
                continue
            paired = set(paths[base]["1"]) & set(paths[base]["-1"])
            if not paired:
                continue
            frames = {
                role: {i: yaml.safe_load(archive.read(path)) for i, path in paths[base][role].items() if i in paired}
                for role in ("1", "-1")
            }
            vehicle, roadside = frames["1"], frames["-1"]
            end = max(paired)
            mobile = (entry.get("ego_net_displacement_m") or 0) >= MOBILE_DISPLACEMENT_M
            for ref_index in range(REFERENCE_START, end - FUTURE_HORIZON_FRAMES - max(OFFSETS) + 1, REFERENCE_STEP):
                if not mobile:
                    rows.extend({"scene": base, "reference": ref_index, "decision": ref_index + offset,
                                 "status": "stationary_scene_excluded"} for offset in OFFSETS)
                    continue
                if not all(i in paired for i in (ref_index - 2, ref_index - 1, ref_index)):
                    rows.extend({"scene": base, "reference": ref_index, "decision": ref_index + offset,
                                 "status": "reference_frame_gap"} for offset in OFFSETS)
                    continue
                try:
                    reference, query, cached, message = reference_for(base, ref_index, vehicle, roadside, planner, scales)
                    payload = message.encode()
                    receivers = {}
                    wire_bytes = {}
                    for method, (model, shrinkage) in models.items():
                        bound = bind_reference(reference, query, payload, model_sha256=model.sha256,
                                               view=model.view, family=method, shrinkage=shrinkage)
                        wire = bound.encode()
                        receiver = BoundReceiver()
                        receiver.register(wire)
                        response = model.issue(wire, payload)
                        receiver.install(response, payload)
                        receivers[method] = receiver
                        wire_bytes[method] = len(wire) + len(response)
                except (UnknownInput, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    rows.extend({"scene": base, "reference": ref_index, "decision": ref_index + offset,
                                 "status": "reference_abstention", "reason": str(exc)} for offset in OFFSETS)
                    continue
                for offset in OFFSETS:
                    index = ref_index + offset
                    row = {"scene": base, "reference": ref_index, "decision": index,
                           "age_s": offset / 10, "payload_bytes": len(payload),
                           "query_response_bytes": wire_bytes}
                    try:
                        if not all(i in paired for i in (index - 2, index - 1, index)):
                            raise UnknownInput("decision_frame_gap")
                        ego = ego_at(vehicle, index)
                        local = objects_at(vehicle[index], ego, LOCAL_RADIUS_M, "vehicle_annotation_input")
                        remote = objects_at(roadside[index], ego, REMOTE_RADIUS_M, "roadside_annotation_input")
                        fresh_id = int.from_bytes(hashlib.sha256(f"{base}:{index}".encode()).digest()[:8], "big")
                        fresh = make_message(remote, index * 100, message_id=fresh_id, sender=SENDER)
                        current_state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps,
                            ego.heading_rad, ego.acceleration_mps2, ego.curvature_inv_m, index * 100)
                        decisions = {
                            method: receiver.evaluate(current_state, Behavior.KEEP, planner.policy_hash, SENDER)
                            for method, receiver in receivers.items()
                        }
                        row["accepted"] = {method: decision.state == DecisionState.ACCEPT_REUSE
                                           for method, decision in decisions.items()}
                        row["receiver_slack"] = {method: decision.minimum_slack
                                                 for method, decision in decisions.items()}
                        row["receiver_reason"] = {method: decision.reason
                                                  for method, decision in decisions.items()}
                        future = {}
                        for j in range(index + 2, index + FUTURE_HORIZON_FRAMES + 1, 2):
                            if j not in paired:
                                raise UnknownInput("future_annotation_frame_missing")
                            future_ego = ego_at(vehicle, j)
                            future[j * 100] = world_at(vehicle[j], roadside[j], future_ego)
                        label = decision_pair(reference=reference, query=query,
                            cached_reference=cached, ego=ego, local=local,
                            fresh_message=fresh, future_world=future,
                            planner=planner, thresholds=thresholds)
                        row.update({"status": "position_proxy_label_available",
                                    "proxy_valid": bool(label["valid"]),
                                    "proxy_cost_regret": label["cost_regret"],
                                    "proxy_cached_risk": label["cached_risk"],
                                    "proxy_fresh_risk": max(label["fresh_cost"]["collision"],
                                                            label["fresh_cost"]["ttc"]),
                                    "proxy_trajectory_deviation_m": label["trajectory_deviation_m"],
                                    "violations": label["violations"]})
                    except (UnknownInput, ValueError, RuntimeError, KeyError, TypeError) as exc:
                        row.update({"status": "unknown_or_abstention", "reason": str(exc)})
                    rows.append(row)
            print(f"processed {base}: {len(rows)} total candidates", flush=True)
    available = [r for r in rows if r["status"] == "position_proxy_label_available"]
    methods = {}
    for method in models:
        accepted = [r for r in available if r["accepted"][method]]
        methods[method] = {"labelled_slots": len(available), "accepted_slots": len(accepted),
                           "proxy_invalid_accepted": sum(not r["proxy_valid"] for r in accepted),
                           "accepted_scenes": len({r["scene"] for r in accepted})}
    result = {"status": "INDEPENDENT_V2XREAL_POSITION_ONLY_PROXY_NOT_CLOSED_LOOP",
              "split": split, "not_full_sensor_or_route_validation": True,
              "test_registration_sha256": sha256_file(REGISTRATION) if registration else None,
              "source_archive_sha256": ARCHIVE_HASHES[split],
              "schema_audit_sha256": sha256_file(SCHEMA_AUDITS[split]),
              "runner_sha256": sha256_file(Path(__file__)),
              "v2_registration_sha256": sha256_file(V2_REG),
              "candidate_slots": len(rows),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "reason_counts": dict(Counter(r.get("reason") for r in rows if r.get("reason"))),
              "methods": methods, "rows": rows}
    OUT[split].parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT[split], result)
    print(json.dumps({k: result[k] for k in ("candidate_slots", "status_counts", "reason_counts", "methods")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(ARCHIVES), required=True)
    run(parser.parse_args().split)
