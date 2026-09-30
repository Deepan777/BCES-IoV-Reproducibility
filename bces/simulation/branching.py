"""Same-world cached/fresh SUMO branch execution and labels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from bces.data.objects import ObjectMessage, ObjectSource, PerceivedObject
from bces.geometry.drift import DriftScales, KinematicState, compute_drift
from bces.models.reference_input import freeze_reference_input
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.planner import KinematicPlanner
from bces.oracle.validity import (
    ValidityObservation,
    ValidityThresholds,
    evaluate_validity,
)
from bces.oracle.world import Route, WorldObject
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery

from .scenario_builder import ScenarioSpec
from .sumo_adapter import ego_state, start_connection, state_hash, world_objects


def _populate(connection: Any, spec: ScenarioSpec) -> None:
    connection.vehicletype.copy("DEFAULT_VEHTYPE", "ego_type")
    connection.vehicletype.setLength("ego_type", 4.8)
    connection.vehicletype.setWidth("ego_type", 1.9)
    connection.vehicletype.setMaxSpeed("ego_type", 20.0)
    connection.vehicletype.setAccel("ego_type", 2.0)
    connection.vehicletype.setDecel("ego_type", 4.0)
    connection.vehicletype.setEmergencyDecel("ego_type", 4.0)
    connection.vehicletype.setImperfection("ego_type", 0.0)
    connection.route.add("ego_route", spec.ego_route)
    connection.vehicle.add(
        "ego", "ego_route", typeID="ego_type", depart="0", departLane="0",
        departPos="5", departSpeed="10",
    )
    connection.vehicle.setSpeedMode("ego", 0)
    if spec.pedestrian:
        connection.person.add(
            spec.actor_id, spec.actor_route[0], spec.actor_depart_position_m, depart=0,
        )
        connection.person.appendWalkingStage(
            spec.actor_id, spec.actor_route, arrivalPos=70, speed=spec.actor_depart_speed_mps,
        )
    else:
        connection.route.add("actor_route", spec.actor_route)
        connection.vehicle.add(
            spec.actor_id, "actor_route", depart="0", departLane=str(spec.actor_lane),
            departPos=str(spec.actor_depart_position_m), departSpeed=str(spec.actor_depart_speed_mps),
        )


def _local(objects: tuple[WorldObject, ...], ego_x: float, ego_y: float, spec: ScenarioSpec) -> tuple[WorldObject, ...]:
    return tuple(
        item
        for item in objects
        if math.hypot(item.x_m - ego_x, item.y_m - ego_y) <= 30
        and not (spec.hidden_from_local and item.track_id == spec.actor_id)
    )


def _minimum_ttc(ego_x: float, ego_y: float, ego_speed: float, heading: float, objects: tuple[WorldObject, ...]) -> float | None:
    minimum = math.inf
    for item in objects:
        dx, dy = item.x_m - ego_x, item.y_m - ego_y
        along = math.cos(heading) * dx + math.sin(heading) * dy
        lateral = abs(-math.sin(heading) * dx + math.cos(heading) * dy)
        object_speed = math.cos(heading) * item.vx_mps + math.sin(heading) * item.vy_mps
        closing = ego_speed - object_speed
        if along > 0 and closing > 1e-6 and lateral < 3.0:
            minimum = min(minimum, along / closing)
    return None if math.isinf(minimum) else minimum


def _reference_contract(
    *, seed: int, spec: ScenarioSpec, ego: Any,
    local: tuple[WorldObject, ...], cooperative: tuple[WorldObject, ...],
    timestamp_ms: int, planner: KinematicPlanner, scales: DriftScales,
) -> tuple[dict[str, Any], KinematicState]:
    reference_state = KinematicState(
        ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad,
        ego.acceleration_mps2, ego.curvature_inv_m, timestamp_ms,
    )
    route = Route(
        ego.x_m, ego.y_m, ego.heading_rad, planner.config.lane_width_m,
        planner.config.speed_limit_mps, allow_left=spec.intended_behavior == "left",
        allow_right=spec.intended_behavior == "right",
        intended_behavior=spec.intended_behavior, curvature_inv_m=ego.curvature_inv_m,
    )
    plan = planner.plan(local, cooperative, ego, route)
    path = tuple(
        PathPoint(
            point.x_m, point.y_m, point.speed_mps, point.heading_rad,
            route.curvature_inv_m, point.time_s,
        )
        for point in plan.points[:12]
    )
    sender = "sumo:registered_grid:rsu"
    reference_id = f"sumo:{spec.family}:{seed}"
    message = ObjectMessage(
        digest32(reference_id.encode()), sender, timestamp_ms,
        tuple(
            PerceivedObject(
                item.track_id, item.class_id, item.x_m, item.y_m,
                item.vx_mps, item.vy_mps, item.length_m, item.width_m,
                item.confidence, ObjectSource.INFRASTRUCTURE,
            )
            for item in cooperative
        ),
    )
    query = ReceiverQuery(
        query_id=digest32((reference_id + ":query").encode()),
        expected_sender_id_hash=sender_id_hash(sender),
        reference_state=reference_state, reference_path=path,
        behavior=Behavior(BEHAVIOR_IDS[spec.intended_behavior]), risk_class=1,
        policy_hash=planner.policy_hash, calibration_id=0, drift_scales=scales,
    )
    frozen = freeze_reference_input(
        message, query, generated_at_ms=timestamp_ms,
        query_available_at_ms=timestamp_ms, context_available_at_ms=timestamp_ms,
        query_provenance="frozen_reference_policy",
        occlusion_proxy=max(0.0, min(1.0, (len(cooperative) - len(local)) / max(len(cooperative), 1))),
        estimated_delay_s=0.0, map_context_flags=1,
    )
    return {
        "reference_id": reference_id,
        "reference_timestamp_ms": timestamp_ms,
        "behavior": spec.intended_behavior,
        "behavior_id": BEHAVIOR_IDS[spec.intended_behavior],
        "input_contract": "exact_reference_payload_and_frozen_policy_query_v2",
        "frozen_input": frozen.to_dict(),
    }, reference_state


def _branch(
    *,
    network: Path,
    state: Path,
    label: str,
    seed: int,
    step_s: float,
    horizon_s: float,
    spec: ScenarioSpec,
    mode: str,
    planner: KinematicPlanner,
    cached_origin: tuple[WorldObject, ...],
    reference_state: KinematicState,
    drift_scales: DriftScales,
) -> dict[str, Any]:
    connection = start_connection(network=network, label=label, seed=seed, step_s=step_s, state=state)
    initial_hash = state_hash(connection)
    if spec.uncontrolled_signal:
        for signal_id in connection.trafficlight.getIDList():
            current = connection.trafficlight.getRedYellowGreenState(signal_id)
            connection.trafficlight.setRedYellowGreenState(signal_id, "G" * len(current))
    if spec.braking_event and spec.actor_id in connection.vehicle.getIDList():
        connection.vehicle.slowDown(spec.actor_id, 0.0, 1.0)
    ticks = []
    collision = False
    minimum_ttc = math.inf
    prior_acceleration = None
    total_abs_jerk = 0.0
    communication_bytes = 0
    start_distance = connection.vehicle.getDistance("ego")
    steps = round(horizon_s / step_s)
    for index in range(steps):
        ego = ego_state(connection)
        actual = world_objects(connection)
        local = _local(actual, ego.x_m, ego.y_m, spec)
        cooperative = actual if mode == "fresh" else tuple(item.propagated(index * step_s) for item in cached_origin)
        route = Route(
            ego.x_m, ego.y_m, ego.heading_rad, planner.config.lane_width_m,
            planner.config.speed_limit_mps,
            allow_left=spec.intended_behavior == "left",
            allow_right=spec.intended_behavior == "right",
            intended_behavior=spec.intended_behavior,
            curvature_inv_m=ego.curvature_inv_m,
        )
        try:
            plan = planner.plan(local, cooperative, ego, route)
        except RuntimeError as exc:
            connection.close()
            raise RuntimeError(
                f"planner infeasible for {spec.family}/{mode} at tick {index}: "
                f"speed={ego.speed_mps:.6f}"
            ) from exc
        maneuver = plan.maneuver
        connection.vehicle.setSpeed("ego", plan.points[0].speed_mps)
        lane_index = connection.vehicle.getLaneIndex("ego")
        road_id = connection.vehicle.getRoadID("ego")
        lane_count = connection.edge.getLaneNumber(road_id) if not road_id.startswith(":") else 1
        target_lane = lane_index + (1 if plan.maneuver == "left" else -1 if plan.maneuver == "right" else 0)
        if 0 <= target_lane < lane_count and target_lane != lane_index:
            connection.vehicle.changeLane("ego", target_lane, step_s)
        communication_bytes += 48 + 32
        if mode == "fresh" or index == 0:
            communication_bytes += 64 + 48 * len(cooperative)
        connection.simulationStep()
        updated = ego_state(connection)
        objects = world_objects(connection)
        ttc = _minimum_ttc(updated.x_m, updated.y_m, updated.speed_mps, updated.heading_rad, objects)
        if ttc is not None:
            minimum_ttc = min(minimum_ttc, ttc)
        collision_now = "ego" in connection.simulation.getCollidingVehiclesIDList()
        collision |= collision_now
        if prior_acceleration is not None:
            total_abs_jerk += abs(updated.acceleration_mps2 - prior_acceleration) / step_s
        prior_acceleration = updated.acceleration_mps2
        ticks.append(
            {
                "tick": index,
                "time_s": round(connection.simulation.getTime(), 6),
                "ego_x_m": updated.x_m,
                "ego_y_m": updated.y_m,
                "ego_speed_mps": updated.speed_mps,
                "ego_acceleration_mps2": updated.acceleration_mps2,
                "ego_heading_rad": updated.heading_rad,
                "ego_curvature_inv_m": updated.curvature_inv_m,
                "normalized_drift": compute_drift(
                    reference_state,
                    KinematicState(
                        updated.x_m, updated.y_m, updated.speed_mps,
                        updated.heading_rad, updated.acceleration_mps2,
                        updated.curvature_inv_m,
                        round(connection.simulation.getTime() * 1000),
                    ),
                    drift_scales,
                ).normalized,
                "maneuver": maneuver,
                "minimum_ttc_s": ttc,
                "collision": collision_now,
                "object_count": len(objects),
            }
        )
    route_progress = max(0.0, connection.vehicle.getDistance("ego") - start_distance)
    connection.close()
    minimum_ttc_value = None if math.isinf(minimum_ttc) else minimum_ttc
    risk = 1.0 if collision else (0.0 if minimum_ttc_value is None else math.exp(-max(0.0, minimum_ttc_value) / 2.0))
    cost = 1000.0 * float(collision) + 100.0 * risk - route_progress + 0.01 * total_abs_jerk
    return {
        "mode": mode,
        "pre_action_non_ego_state_hash": initial_hash,
        "collision": collision,
        "minimum_ttc_s": minimum_ttc_value,
        "risk": risk,
        "route_progress_m": route_progress,
        "total_abs_jerk": total_abs_jerk,
        "communication_bytes": communication_bytes,
        "cost": cost,
        "ticks": ticks,
    }


def recompute_pair_label(pair: dict[str, Any], thresholds: ValidityThresholds) -> bool:
    cached, fresh = pair["cached"], pair["fresh"]
    deviation = max(
        math.hypot(a["ego_x_m"] - b["ego_x_m"], a["ego_y_m"] - b["ego_y_m"])
        for a, b in zip(cached["ticks"], fresh["ticks"])
    )
    observation = ValidityObservation(cached["cost"], fresh["cost"], cached["risk"], deviation)
    return evaluate_validity(observation, thresholds)


def run_same_world_pair(
    *,
    network: Path,
    state_path: Path,
    seed: int,
    step_s: float,
    reference_time_s: float,
    horizon_s: float,
    spec: ScenarioSpec,
    planner: KinematicPlanner,
    thresholds: ValidityThresholds,
    drift_scales: DriftScales,
    split: str,
) -> dict[str, Any]:
    initial_label = f"initial_{seed}"
    connection = start_connection(network=network, label=initial_label, seed=seed, step_s=step_s)
    _populate(connection, spec)
    for _ in range(round(reference_time_s / step_s)):
        connection.simulationStep()
    if "ego" not in connection.vehicle.getIDList():
        connection.close()
        raise RuntimeError("ego left the network before the branch point")
    cached_origin = world_objects(connection)
    reference_ego = ego_state(connection)
    reference_local = _local(cached_origin, reference_ego.x_m, reference_ego.y_m, spec)
    reference_timestamp_ms = round(connection.simulation.getTime() * 1000)
    reference, reference_state = _reference_contract(
        seed=seed, spec=spec, ego=reference_ego, local=reference_local,
        cooperative=cached_origin, timestamp_ms=reference_timestamp_ms,
        planner=planner, scales=drift_scales,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.unlink(missing_ok=True)
    connection.simulation.saveState(str(state_path))
    connection.close()
    fresh = _branch(
        network=network, state=state_path, label=f"fresh_{seed}", seed=seed,
        step_s=step_s, horizon_s=horizon_s, spec=spec, mode="fresh",
        planner=planner, cached_origin=cached_origin,
        reference_state=reference_state, drift_scales=drift_scales,
    )
    cached = _branch(
        network=network, state=state_path, label=f"cached_{seed}", seed=seed,
        step_s=step_s, horizon_s=horizon_s, spec=spec, mode="cached",
        planner=planner, cached_origin=cached_origin,
        reference_state=reference_state, drift_scales=drift_scales,
    )
    state_path.unlink(missing_ok=True)
    pair = {
        "pair_id": f"sumo:{spec.family}:{seed}",
        "label_source": "sumo_same_world",
        "family": spec.family,
        "split": split,
        "seed": seed,
        "policy_hash": planner.policy_hash,
        "reference": reference,
        "fresh": fresh,
        "cached": cached,
        "state_equivalent": fresh["pre_action_non_ego_state_hash"] == cached["pre_action_non_ego_state_hash"],
    }
    pair["valid"] = recompute_pair_label(pair, thresholds) if pair["state_equivalent"] else False
    return pair
