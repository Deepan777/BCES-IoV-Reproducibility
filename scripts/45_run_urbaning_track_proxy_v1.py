#!/usr/bin/env python3
"""Frozen UrbanIng labeled-track geometry proxy; not source-separated V2X validation."""
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
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.bound_exchange import BoundQuery
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.simulation.controlled_contract import make_message, payload_objects
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, decision_pair
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic

AUDIT = ROOT / "outputs/study_b/urbaning_schema_audit_v1.json"
MANIFEST = ROOT / "data/manifests/urbaning_labels_v1.json"
REG = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
OUT = ROOT / "outputs/study_b/urbaning_track_proxy_v2/result.json"
REFS = (20, 60, 100, 140)
OFFSETS = (5, 10, 15, 20)
SENDER = "urbaning:synthetic_global_annotation"


class UnknownInput(ValueError):
    pass


def frames_for(path: Path) -> tuple[list[dict], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    start = float(data["timestamp"])
    frames = [dict() for _ in range(200)]
    for tr in data["tracks"]:
        n = len(tr["timestamps"])
        for j in range(n):
            frame = round((float(tr["timestamps"][j]) - start) * 10)
            if not 0 <= frame < 200:
                continue
            if abs(float(tr["timestamps"][j]) - (start + frame * 0.1)) > 0.021:
                raise UnknownInput("off_grid_annotation_timestamp")
            dimensions = tr["dimensions"][0 if len(tr["dimensions"]) == 1 else j]
            frames[frame][tr["track_id"]] = (tr["positions"][j], float(tr["orientations"][j]), dimensions, tr["object_type"])
    return frames, len(data["tracks"])


def ego_at(frames: list[dict], tid: int, i: int) -> EgoState:
    if i < 2 or any(tid not in frames[j] for j in (i - 2, i - 1, i)):
        raise UnknownInput("ego_history_missing")
    a, b, c = (frames[j][tid] for j in (i - 2, i - 1, i))
    speed0 = math.dist(a[0][:2], b[0][:2]) / 0.1
    speed1 = math.dist(b[0][:2], c[0][:2]) / 0.1
    heading = c[1]
    accel = (speed1 - speed0) / 0.1
    curvature = wrap_angle_rad(c[1] - b[1]) / 0.1 / max(speed1, 2.0)
    values = (c[0][0], c[0][1], speed1, heading, accel, curvature)
    if not all(math.isfinite(x) for x in values):
        raise UnknownInput("nonfinite_ego_kinematics")
    return EgoState(*values)


def objects_at(frames: list[dict], ego_tid: int, i: int, ego: EgoState, radius: float, source: str) -> tuple[WorldObject, ...]:
    if i < 1:
        raise UnknownInput("object_history_missing")
    objects = []
    for tid, value in sorted(frames[i].items()):
        if tid == ego_tid:
            continue
        position, yaw, dims, _ = value
        x, y = float(position[0]), float(position[1])
        if math.hypot(x - ego.x_m, y - ego.y_m) > radius:
            continue
        if tid not in frames[i - 1]:
            raise UnknownInput("object_velocity_unavailable")
        old = frames[i - 1][tid][0]
        vx, vy = (x - float(old[0])) / 0.1, (y - float(old[1])) / 0.1
        length, width = float(dims[0]), float(dims[1])
        if not all(math.isfinite(v) for v in (x, y, vx, vy, yaw, length, width)) or length <= 0 or width <= 0:
            raise UnknownInput("invalid_object_geometry")
        objects.append(WorldObject(f"track:{tid}", x, y, vx, vy, yaw, length, width, source=source))
    return tuple(objects)


def reference_for(frames, sequence, role, tid, i, planner, scales):
    ego = ego_at(frames, tid, i)
    local = objects_at(frames, tid, i, ego, 30.0, "urbaning_synthetic_local")
    remote = objects_at(frames, tid, i, ego, 120.0, "urbaning_synthetic_remote")
    if not remote:
        raise UnknownInput("no_remote_range_object")
    route = Route(ego.x_m, ego.y_m, ego.heading_rad, planner.config.lane_width_m,
                  planner.config.speed_limit_mps, False, False, "keep", ego.curvature_inv_m)
    identity = f"urbaning:{sequence}:{role}:{i}"
    message_id = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    message = make_message(remote, i * 100, message_id=message_id, sender=SENDER)
    perceived = payload_objects(message)
    plan = planner.plan(local, perceived, ego, route)
    state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad,
                           ego.acceleration_mps2, ego.curvature_inv_m, i * 100)
    path = tuple(PathPoint(p.x_m, p.y_m, p.speed_mps, p.heading_rad,
                           route.curvature_inv_m, p.time_s) for p in plan.points[:12])
    query = ReceiverQuery(digest32(identity.encode()), sender_id_hash(SENDER), state,
                          path, Behavior.KEEP, 1, planner.policy_hash, 0, scales)
    occlusion = max(0.0, min(1.0, (len(perceived) - len(local)) / len(perceived)))
    frozen = freeze_reference_input(message, query, generated_at_ms=i * 100,
        query_available_at_ms=i * 100, context_available_at_ms=i * 100,
        query_provenance="frozen_reference_policy", occlusion_proxy=occlusion,
        estimated_delay_s=0.0, map_context_flags=0)
    features = observable_planner_features(planner, local, perceived, ego, route)
    reference = {"reference_id": identity, "scenario_id": sequence, "split": "external_track_proxy",
                 "behavior": "keep", "frozen_input": frozen.to_dict(),
                 "observable_margin_features": features}
    return reference, query, perceived, message


def future_world_for(frames, tid, i):
    world = {}
    for j in range(i + 2, i + 31, 2):
        ego = ego_at(frames, tid, j)
        world[j * 100] = objects_at(frames, tid, j, ego, 120.0, "urbaning_label_world")
    return world


def accepts(model, bound, payload, drift):
    offsets, origin = model.predict(bound.encode(), payload)
    if not origin:
        return False
    quantized = np.floor(np.asarray(offsets) / STEP).clip(0, 255) * STEP
    if model.family == "surface":
        return bool(np.min(quantized - np.asarray(NORMAL_CODEBOOK_V1) @ np.asarray(drift)) >= GUARD)
    return bool(quantized[0] - drift[6] >= GUARD)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "SCHEMA_AND_MOBILITY_ONLY_NO_BCES_OUTCOMES" or audit["sequence_count"] != 34:
        raise RuntimeError("outcome-blind schema audit incomplete")
    if audit["manifest_sha256"] != sha256_file(MANIFEST):
        raise RuntimeError("label manifest changed since audit")
    receipt = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {Path(x["relative_path"]).stem: ROOT / x["relative_path"] for x in receipt["records"]}
    hashes = {Path(x["relative_path"]).stem: x["sha256"] for x in receipt["records"]}
    reg = json.loads(REG.read_text(encoding="utf-8"))
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml"))
    if planner.policy_hash != reg["policy_hash"]:
        raise RuntimeError("frozen planner hash changed")
    scales = DriftScales(*reg["config"]["drift_scales"])
    raw_limits = yaml.safe_load((ROOT / "configs/oracle/validity_v1.yaml").read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(raw_limits["max_cost_regret"], raw_limits["max_cached_risk"], raw_limits["max_trajectory_deviation_m"])
    models = {}
    for method in ("surface", "scalar_ttl"):
        rule = reg["model_rules"][method]
        models[method] = (FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"], policy_hash=planner.policy_hash), rule["shrinkage"])
    rows = []
    for entry in audit["sequences"]:
        name = entry["sequence"]
        if sha256_file(paths[name]) != hashes[name]:
            raise RuntimeError(f"source label changed: {name}")
        frames, _ = frames_for(paths[name])
        for receiver in entry["receivers"]:
            role, tid = receiver["role"], receiver["track_id"]
            for ref_idx in REFS:
                if receiver["status"] != "mobile_eligible":
                    rows.extend({"sequence": name, "receiver": role, "reference_index": ref_idx,
                                 "decision_index": ref_idx + offset, "status": receiver["status"]} for offset in OFFSETS)
                    continue
                try:
                    reference, query, cached, message = reference_for(frames, name, role, tid, ref_idx, planner, scales)
                    payload = message.encode()
                    bounds = {method: BoundQuery(query, hashlib.sha256(payload).hexdigest(), model.sha256,
                        model.view, method, shrink, tuple(reference["observable_margin_features"]),
                        query.reference_state.timestamp_ms,
                        reference["frozen_input"]["batch"]["occlusion_proxy"], 0.0, 0)
                        for method, (model, shrink) in models.items()}
                except (UnknownInput, RuntimeError, ValueError, KeyError) as exc:
                    rows.extend({"sequence": name, "receiver": role, "reference_index": ref_idx,
                                 "decision_index": ref_idx + offset, "status": "reference_abstention",
                                 "reason": str(exc)} for offset in OFFSETS)
                    continue
                for offset in OFFSETS:
                    idx = ref_idx + offset
                    row = {"sequence": name, "receiver": role, "reference_index": ref_idx,
                           "decision_index": idx, "age_s": offset / 10}
                    try:
                        ego = ego_at(frames, tid, idx)
                        local = objects_at(frames, tid, idx, ego, 30.0, "urbaning_synthetic_local")
                        remote = objects_at(frames, tid, idx, ego, 120.0, "urbaning_synthetic_remote")
                        label_id = f"{name}:{role}:{idx}"
                        message_id = int.from_bytes(hashlib.sha256(label_id.encode()).digest()[:8], "big")
                        fresh = make_message(remote, idx * 100, message_id=message_id, sender=SENDER)
                        state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad,
                                               ego.acceleration_mps2, ego.curvature_inv_m, idx * 100)
                        drift = compute_drift(query.reference_state, state, scales).normalized
                        row["accepted"] = {method: accepts(model, bounds[method], payload, drift)
                                           for method, (model, _) in models.items()}
                        label = decision_pair(reference=reference, query=query, cached_reference=cached,
                            ego=ego, local=local, fresh_message=fresh,
                            future_world=future_world_for(frames, tid, idx),
                            planner=planner, thresholds=thresholds)
                        row.update({"status": "proxy_label_available", "proxy_valid": label["valid"],
                                    "proxy_cost_regret": label["cost_regret"],
                                    "proxy_cached_risk": label["cached_risk"],
                                    "proxy_trajectory_deviation_m": label["trajectory_deviation_m"],
                                    "normalized_drift": list(drift)})
                    except (UnknownInput, RuntimeError, ValueError, KeyError) as exc:
                        row.update({"status": "unknown_or_abstention", "reason": str(exc)})
                    rows.append(row)
        print(f"{len(rows)}/1088 candidates processed through {name}", flush=True)
    available = [r for r in rows if r["status"] == "proxy_label_available"]
    summary = {}
    for method in models:
        accepted = [r for r in available if r["accepted"][method]]
        summary[method] = {"labelled_slots": len(available), "accepted_slots": len(accepted),
                           "proxy_invalid_accepted": sum(not r["proxy_valid"] for r in accepted),
                           "accepted_sequences": len({r["sequence"] for r in accepted})}
    report = {"status": "DESCRIPTIVE_SYNTHETIC_VISIBILITY_TRACK_PROXY_ONLY",
              "not_source_separated_v2x_validation": True, "not_closed_loop": True,
              "schema_audit_sha256": sha256_file(AUDIT), "manifest_sha256": sha256_file(MANIFEST),
              "adapter_source_sha256": sha256_file(Path(__file__)), "v2_registration_sha256": sha256_file(REG),
              "sequence_count": 34, "candidate_slots": len(rows),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "reason_counts": dict(Counter(r.get("reason") for r in rows if r.get("reason"))),
              "methods": summary, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, report)
    print(json.dumps({k: report[k] for k in ("candidate_slots", "status_counts", "reason_counts", "methods")}, indent=2))


if __name__ == "__main__":
    main()
