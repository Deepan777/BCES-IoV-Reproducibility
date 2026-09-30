"""Leakage-controlled V2X-Traj reference queries and observational oracles."""

from __future__ import annotations

import bisect
import hashlib
import math
from pathlib import Path
from typing import Any

from bces.data.objects import ObjectMessage, ObjectSource, PerceivedObject
from bces.geometry.drift import (
    DriftScales,
    KinematicState,
    compute_drift,
    wrap_angle_rad,
)
from bces.models.reference_input import freeze_reference_input
from bces.oracle.boundary import OraclePoint
from bces.oracle.observational import (
    BEHAVIOR_IDS,
    _ego,
    _objects,
    _snapshots,
    _target_row,
)
from bces.oracle.planner import KinematicPlanner, trajectory_deviation
from bces.oracle.validity import (
    ValidityObservation,
    ValidityThresholds,
    evaluate_validity,
)
from bces.oracle.world import EgoState, GroundTruthWorld, Route, WorldObject
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.utils.hashing import canonical_json_hash

BEHAVIORS = tuple(BEHAVIOR_IDS)
DERIVATIVE_HALF_WINDOW_FRAMES = 5
MAX_ABS_ACCELERATION_MPS2 = 8.0
MAX_ABS_CURVATURE_INV_M = 0.3
MINIMUM_CURVATURE_SPEED_MPS = 2.0


def _focal_rows(snapshots: dict[int, tuple[dict[str, str], ...]], focal_id: str) -> dict[int, dict[str, str]]:
    return {
        timestamp: _target_row(rows, focal_id)
        for timestamp, rows in snapshots.items()
        if any(str(row.get("id", "")).strip() == focal_id for row in rows)
    }


def kinematic_states(rows: dict[int, dict[str, str]]) -> dict[int, KinematicState]:
    """Estimate acceleration and curvature from observed receiver history."""
    times = sorted(rows)
    if len(times) < 2:
        raise ValueError("at least two focal states are required")
    states: dict[int, KinematicState] = {}
    for index, timestamp in enumerate(times):
        before = max(0, index - DERIVATIVE_HALF_WINDOW_FRAMES)
        after = min(len(times) - 1, index + DERIVATIVE_HALF_WINDOW_FRAMES)
        first, second = _ego(rows[times[before]]), _ego(rows[times[after]])
        current = _ego(rows[timestamp])
        seconds = max((times[after] - times[before]) / 1000.0, 1e-6)
        acceleration = max(
            -MAX_ABS_ACCELERATION_MPS2,
            min(MAX_ABS_ACCELERATION_MPS2, (second.speed_mps - first.speed_mps) / seconds),
        )
        yaw_rate = wrap_angle_rad(second.heading_rad - first.heading_rad) / seconds
        curvature = max(
            -MAX_ABS_CURVATURE_INV_M,
            min(MAX_ABS_CURVATURE_INV_M, yaw_rate / max(current.speed_mps, MINIMUM_CURVATURE_SPEED_MPS)),
        )
        states[timestamp] = KinematicState(
            current.x_m, current.y_m, current.speed_mps, current.heading_rad,
            acceleration, curvature, timestamp,
        )
    return states


def _ego_from_state(state: KinematicState, template: EgoState) -> EgoState:
    return EgoState(
        state.x_m, state.y_m, state.speed_mps, state.heading_rad,
        state.acceleration_mps2, state.curvature_inv_m,
        template.length_m, template.width_m,
    )


def route_for(ego: EgoState, behavior: str, planner: KinematicPlanner) -> Route:
    return Route(
        ego.x_m, ego.y_m, ego.heading_rad,
        planner.config.lane_width_m, planner.config.speed_limit_mps,
        allow_left=behavior == "left", allow_right=behavior == "right",
        intended_behavior=behavior, curvature_inv_m=ego.curvature_inv_m,
    )


def _message(objects: tuple[WorldObject, ...], sender: str, timestamp: int, message_id: int) -> ObjectMessage:
    mapped = tuple(
        PerceivedObject(
            item.track_id, item.class_id, item.x_m, item.y_m, item.vx_mps,
            item.vy_mps, item.length_m, item.width_m, item.confidence,
            ObjectSource.INFRASTRUCTURE,
        )
        for item in objects
    )
    return ObjectMessage(message_id, sender, timestamp, mapped)


def _world_at(
    snapshots: dict[int, tuple[dict[str, str], ...]], timestamp: int,
    ego: EgoState, focal_id: str, maximum: int, source: str,
) -> tuple[WorldObject, ...]:
    return _objects(
        snapshots.get(timestamp, ()), source=source, ego=ego,
        exclude_id=focal_id, maximum=maximum,
    )


def build_reference(
    *,
    scenario_id: str,
    intersection_id: str,
    split: str,
    behavior: str,
    focal_id: str,
    reference_timestamp: int,
    states: dict[int, KinematicState],
    ego_snapshots: dict[int, tuple[dict[str, str], ...]],
    vehicle_snapshots: dict[int, tuple[dict[str, str], ...]],
    infrastructure_snapshots: dict[int, tuple[dict[str, str], ...]],
    scales: DriftScales,
    planner: KinematicPlanner,
    maximum_objects: int,
) -> tuple[dict[str, Any], ObjectMessage, ReceiverQuery, tuple[WorldObject, ...], EgoState]:
    state = states[reference_timestamp]
    template = _ego(_target_row(ego_snapshots[reference_timestamp], focal_id))
    ego = _ego_from_state(state, template)
    local = _world_at(vehicle_snapshots, reference_timestamp, ego, focal_id, maximum_objects, "vehicle")
    cooperative = _world_at(infrastructure_snapshots, reference_timestamp, ego, focal_id, maximum_objects, "infrastructure")
    route = route_for(ego, behavior, planner)
    plan = planner.plan(local, cooperative, ego, route)
    path = tuple(
        PathPoint(p.x_m, p.y_m, p.speed_mps, p.heading_rad, route.curvature_inv_m, p.time_s)
        for p in plan.points[:12]
    )
    sender = f"rsu:{intersection_id}"
    reference_id = f"{scenario_id}:{behavior}:{reference_timestamp}"
    message = _message(cooperative, sender, reference_timestamp, digest32(reference_id.encode()))
    query = ReceiverQuery(
        query_id=digest32((reference_id + ":query").encode()),
        expected_sender_id_hash=sender_id_hash(sender),
        reference_state=state,
        reference_path=path,
        behavior=Behavior(BEHAVIOR_IDS[behavior]), risk_class=1,
        policy_hash=planner.policy_hash, calibration_id=0, drift_scales=scales,
    )
    frozen = freeze_reference_input(
        message, query, generated_at_ms=reference_timestamp,
        query_available_at_ms=reference_timestamp,
        context_available_at_ms=reference_timestamp,
        query_provenance="frozen_reference_policy",
        occlusion_proxy=max(0.0, min(1.0, (len(cooperative) - len(local)) / max(len(cooperative), 1))),
        estimated_delay_s=0.0,
    )
    record = {
        "reference_id": reference_id, "scenario_id": scenario_id,
        "intersection_id": intersection_id, "split": split, "behavior": behavior,
        "behavior_id": BEHAVIOR_IDS[behavior], "label_source": "observational",
        "policy_hash": planner.policy_hash, "reference_timestamp_ms": reference_timestamp,
        "input_contract": "exact_reference_payload_and_frozen_policy_query_v2",
        "frozen_input": frozen.to_dict(),
    }
    return record, message, query, cooperative, template


def observed_validity_point(
    *, reference: dict[str, Any], query: ReceiverQuery,
    cached_reference: tuple[WorldObject, ...], template: EgoState,
    current_timestamp: int, states: dict[int, KinematicState], focal_id: str,
    ego_snapshots: dict[int, tuple[dict[str, str], ...]],
    vehicle_snapshots: dict[int, tuple[dict[str, str], ...]],
    infrastructure_snapshots: dict[int, tuple[dict[str, str], ...]],
    planner: KinematicPlanner, thresholds: ValidityThresholds, maximum_objects: int,
) -> dict[str, Any] | None:
    if current_timestamp not in states:
        return None
    state = states[current_timestamp]
    ego = _ego_from_state(state, template)
    age = (current_timestamp - query.reference_state.timestamp_ms) / 1000.0
    local = _world_at(vehicle_snapshots, current_timestamp, ego, focal_id, maximum_objects, "vehicle")
    fresh = _world_at(infrastructure_snapshots, current_timestamp, ego, focal_id, maximum_objects, "infrastructure")
    cached = tuple(item.propagated(age) for item in cached_reference)
    truth = _world_at(ego_snapshots, current_timestamp, ego, focal_id, maximum_objects * 2, "ground_truth")
    route = route_for(ego, reference["behavior"], planner)
    try:
        cached_plan, fresh_plan = planner.plan(local, cached, ego, route), planner.plan(local, fresh, ego, route)
    except RuntimeError:
        return None
    cached_cost = planner.evaluate(cached_plan, GroundTruthWorld(current_timestamp, truth), ego, route)
    fresh_cost = planner.evaluate(fresh_plan, GroundTruthWorld(current_timestamp, truth), ego, route)
    deviation = trajectory_deviation(cached_plan, fresh_plan)
    observation = ValidityObservation(cached_cost.total, fresh_cost.total, cached_cost.risk, deviation)
    drift = compute_drift(query.reference_state, state, query.drift_scales)
    return {
        "point_id": f"{reference['reference_id']}:{current_timestamp}",
        "reference_id": reference["reference_id"], "scenario_id": reference["scenario_id"],
        "intersection_id": reference["intersection_id"], "split": reference["split"],
        "behavior": reference["behavior"], "label_source": "observational",
        "current_timestamp_ms": current_timestamp, "cache_age_s": age,
        "normalized_drift": drift.normalized, "valid": evaluate_validity(observation, thresholds),
        "cached_cost": cached_cost.to_dict(), "fresh_cost": fresh_cost.to_dict(),
        "cost_regret": observation.cost_regret, "cached_risk": observation.cached_risk,
        "trajectory_deviation_m": deviation, "cached_maneuver": cached_plan.maneuver,
        "fresh_maneuver": fresh_plan.maneuver, "policy_hash": planner.policy_hash,
    }


def state_from_drift(reference: KinematicState, drift: tuple[float, ...], scales: DriftScales) -> KinematicState | None:
    raw = tuple(value * scale for value, scale in zip(drift, scales.as_tuple()))
    elapsed = raw[6]
    speed = reference.speed_mps + raw[2]
    acceleration = reference.acceleration_mps2 + raw[4]
    curvature = reference.curvature_inv_m + raw[5]
    if (elapsed < 0 or speed < 0 or speed > 20
            or abs(acceleration) > MAX_ABS_ACCELERATION_MPS2
            or abs(curvature) > MAX_ABS_CURVATURE_INV_M):
        return None
    cosine, sine = math.cos(reference.heading_rad), math.sin(reference.heading_rad)
    x = reference.x_m + cosine * raw[0] - sine * raw[1]
    y = reference.y_m + sine * raw[0] + cosine * raw[1]
    return KinematicState(
        x, y, speed, reference.heading_rad + raw[3], acceleration, curvature,
        reference.timestamp_ms + round(elapsed * 1000),
    )


def make_directional_oracle(
    *, reference: dict[str, Any], query: ReceiverQuery,
    cached_reference: tuple[WorldObject, ...], template: EgoState,
    states: dict[int, KinematicState], focal_id: str,
    ego_snapshots: dict[int, tuple[dict[str, str], ...]],
    vehicle_snapshots: dict[int, tuple[dict[str, str], ...]],
    infrastructure_snapshots: dict[int, tuple[dict[str, str], ...]],
    planner: KinematicPlanner, thresholds: ValidityThresholds, maximum_objects: int,
):
    available = sorted(set(states) & set(ego_snapshots) & set(vehicle_snapshots) & set(infrastructure_snapshots))

    def oracle(drift: tuple[float, ...]) -> OraclePoint:
        hypothetical = state_from_drift(query.reference_state, drift, query.drift_scales)
        evidence_base = {"reference_id": reference["reference_id"], "drift": drift}
        if hypothetical is None:
            return OraclePoint(None, False, canonical_json_hash({**evidence_base, "status": "kinematic_infeasible"}))
        position = bisect.bisect_left(available, hypothetical.timestamp_ms)
        candidates = available[max(0, position - 1):min(len(available), position + 1)]
        if not candidates:
            return OraclePoint(None, True, canonical_json_hash({**evidence_base, "status": "missing_timestamp"}))
        timestamp = min(candidates, key=lambda value: abs(value - hypothetical.timestamp_ms))
        if abs(timestamp - hypothetical.timestamp_ms) > 55:
            return OraclePoint(None, True, canonical_json_hash({**evidence_base, "status": "missing_timestamp"}))
        ego = _ego_from_state(hypothetical, template)
        age = (hypothetical.timestamp_ms - query.reference_state.timestamp_ms) / 1000.0
        local = _world_at(vehicle_snapshots, timestamp, ego, focal_id, maximum_objects, "vehicle")
        fresh = _world_at(infrastructure_snapshots, timestamp, ego, focal_id, maximum_objects, "infrastructure")
        cached = tuple(item.propagated(age) for item in cached_reference)
        truth = _world_at(ego_snapshots, timestamp, ego, focal_id, maximum_objects * 2, "ground_truth")
        route = route_for(ego, reference["behavior"], planner)
        try:
            cached_plan = planner.plan(local, cached, ego, route)
            fresh_plan = planner.plan(local, fresh, ego, route)
        except RuntimeError:
            return OraclePoint(None, False, canonical_json_hash({**evidence_base, "status": "planner_infeasible"}))
        world = GroundTruthWorld(timestamp, truth)
        cached_cost, fresh_cost = planner.evaluate(cached_plan, world, ego, route), planner.evaluate(fresh_plan, world, ego, route)
        observation = ValidityObservation(
            cached_cost.total, fresh_cost.total, cached_cost.risk,
            trajectory_deviation(cached_plan, fresh_plan),
        )
        valid = evaluate_validity(observation, thresholds)
        evidence_id = canonical_json_hash({
            **evidence_base, "timestamp_ms": timestamp, "valid": valid,
            "cached_cost": cached_cost.to_dict(), "fresh_cost": fresh_cost.to_dict(),
            "trajectory_deviation_m": observation.trajectory_deviation_m,
        })
        return OraclePoint(valid, True, evidence_id)

    return oracle


def load_scene(record: dict[str, Any], raw_root):
    source = Path(record["source_file"])
    ego = _snapshots(raw_root / source)
    vehicle = _snapshots(raw_root / "vehicle" / source.parts[1] / source.name)
    infrastructure = _snapshots(raw_root / "infrastructure" / source.parts[1] / source.name)
    focal_id = str(record["focal_actor_id"])
    rows = _focal_rows(ego, focal_id)
    return ego, vehicle, infrastructure, focal_id, kinematic_states(rows)


def reference_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_hash(record).encode()).hexdigest()
