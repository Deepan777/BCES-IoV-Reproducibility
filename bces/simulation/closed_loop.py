"""Matched closed-loop SUMO branches with refresh and network accounting."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from bces.geometry.drift import DriftScales, KinematicState, compute_drift
from bces.network.events import NetworkCondition, Packet, simulate_packets
from bces.oracle.planner import KinematicPlanner
from bces.oracle.world import Route, WorldObject

from .branching import _local, _minimum_ttc, _populate, _reference_contract
from .scenario_builder import ScenarioSpec
from .sumo_adapter import ego_state, start_connection, state_hash, world_objects


class ReusePolicy(Protocol):
    extension_bytes: int

    def encode_reference(self, reference: dict[str, Any]) -> Any: ...

    def accepts(self, token: Any, drift: tuple[float, ...]) -> bool: ...


def impaired_drift(
    drift: tuple[float, ...], scales: DriftScales, condition: NetworkCondition
) -> tuple[float, ...]:
    values = list(map(float, drift))
    values[0] += condition.pose_error_m / scales.longitudinal_m
    values[3] += math.radians(condition.yaw_error_deg) / scales.heading_rad
    values[6] += (condition.clock_offset_ms / 1000.0) / scales.elapsed_s
    return tuple(values)


def exchange_application_bytes(
    object_count: int, extension_bytes: int, retry: bool, sizes: dict[str, int]
) -> dict[str, int]:
    return {
        "query": int(sizes["query"]),
        "object_header": int(sizes["object_header"]),
        "object_records": int(object_count) * int(sizes["object_record"]),
        "method_extension": int(extension_bytes),
        "security_envelope": 2 * int(sizes["security_envelope"]),
        "retry_control": int(sizes["retry_control"]) if retry else 0,
    }


def _exchange(
    *, object_count: int, extension_bytes: int, retry: bool,
    sizes: dict[str, int], condition: NetworkCondition, seed: int, deadline_ms: float,
) -> tuple[bool, dict]:
    components = exchange_application_bytes(object_count, extension_bytes, retry, sizes)
    request_size = components["query"] + int(sizes["security_envelope"]) + components["retry_control"]
    response_size = components["object_header"] + components["object_records"] + components["method_extension"] + int(sizes["security_envelope"])
    packets = (Packet("query", 0.0, request_size, "query"), Packet("response", 0.0, response_size, "payload"))
    trace = simulate_packets(packets, condition, seed=seed)
    arrivals = {row["packet_id"]: row["delivered_ms"] for row in trace["deliveries"] if not row["duplicate"]}
    success = arrivals.get("query", float("inf")) <= deadline_ms and arrivals.get("response", float("inf")) <= deadline_ms
    return success, {
        "components": components, "generated_bytes": trace["generated_bytes"],
        "delivered_original_bytes": trace["delivered_original_bytes"], "dropped_bytes": trace["dropped_bytes"],
        "duplicate_bytes": trace["duplicate_bytes"], "generated_packets": len(packets),
        "delivered_packets": trace["delivered_packets"], "reordered_packets": trace["reordered_packets"],
        "byte_conservation_ok": trace["byte_conservation_ok"], "success": success,
    }


def prepare_matched_state(
    *, network: Path, state_path: Path, seed: int, step_s: float,
    reference_time_s: float, spec: ScenarioSpec, planner: KinematicPlanner,
    drift_scales: DriftScales,
) -> tuple[dict[str, Any], KinematicState, tuple[WorldObject, ...]]:
    connection = start_connection(network=network, label=f"phase9_initial_{seed}", seed=seed, step_s=step_s)
    _populate(connection, spec)
    for _ in range(round(reference_time_s / step_s)):
        connection.simulationStep()
    if "ego" not in connection.vehicle.getIDList():
        connection.close()
        raise RuntimeError("ego left before Phase-9 branch point")
    actual = world_objects(connection)
    ego = ego_state(connection)
    local = _local(actual, ego.x_m, ego.y_m, spec)
    timestamp_ms = round(connection.simulation.getTime() * 1000)
    reference, reference_state = _reference_contract(
        seed=seed, spec=spec, ego=ego, local=local, cooperative=actual,
        timestamp_ms=timestamp_ms, planner=planner, scales=drift_scales,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.unlink(missing_ok=True)
    connection.simulation.saveState(str(state_path))
    connection.close()
    return reference, reference_state, actual


def run_closed_loop_branch(
    *, network: Path, state_path: Path, seed: int, step_s: float, horizon_s: float,
    spec: ScenarioSpec, method: str, planner: KinematicPlanner,
    drift_scales: DriftScales, initial_reference: dict[str, Any],
    initial_reference_state: KinematicState, initial_cache: tuple[WorldObject, ...],
    condition: NetworkCondition, sizes: dict[str, int], reuse_policy: ReusePolicy | None,
    critical_ttc_s: float,
) -> dict[str, Any]:
    allowed = {"surface_every_tick", "surface_receipt_only", "learned_scalar_ttl", "always_fresh", "local_only"}
    if method not in allowed:
        raise ValueError("unknown closed-loop method")
    if method in {"surface_every_tick", "surface_receipt_only", "learned_scalar_ttl"} and reuse_policy is None:
        raise ValueError("reuse method needs a policy")
    connection = start_connection(network=network, label=f"phase9_{seed}_{method}_{abs(hash(str(condition))) % 100000}", seed=seed, step_s=step_s, state=state_path)
    initial_hash = state_hash(connection)
    if spec.uncontrolled_signal:
        for signal_id in connection.trafficlight.getIDList():
            current = connection.trafficlight.getRedYellowGreenState(signal_id)
            connection.trafficlight.setRedYellowGreenState(signal_id, "G" * len(current))
    if spec.braking_event and spec.actor_id in connection.vehicle.getIDList():
        connection.vehicle.slowDown(spec.actor_id, 0.0, 1.0)
    cache, reference, reference_state = initial_cache, initial_reference, initial_reference_state
    token = reuse_policy.encode_reference(reference) if reuse_policy is not None else None
    counters: Counter[str] = Counter()
    components: Counter[str] = Counter()
    prior_failure = False
    cache_available = method == "local_only"
    receipt_accept = False
    if reuse_policy is not None:
        success, accounting = _exchange(
            object_count=len(cache), extension_bytes=reuse_policy.extension_bytes, retry=False,
            sizes=sizes, condition=condition, seed=seed * 10000 + 1, deadline_ms=step_s * 1000,
        )
        cache_available = success
        counters.update({key: int(accounting[key]) for key in ("generated_bytes", "delivered_original_bytes", "dropped_bytes", "duplicate_bytes", "generated_packets", "delivered_packets", "reordered_packets")})
        components.update(accounting["components"])
        counters["byte_conservation_failures"] += int(not accounting["byte_conservation_ok"])
        counters["network_failures"] += int(not success)
        prior_failure = not success
        if method == "surface_receipt_only" and success:
            receipt_accept = reuse_policy.accepts(token, impaired_drift((0.0,) * 7, drift_scales, condition))
    ticks = []
    collision = False
    minimum_ttc = math.inf
    prior_acceleration = None
    total_abs_jerk = 0.0
    start_distance = connection.vehicle.getDistance("ego")
    for index in range(round(horizon_s / step_s)):
        ego = ego_state(connection)
        actual = world_objects(connection)
        local = _local(actual, ego.x_m, ego.y_m, spec)
        current_state = KinematicState(ego.x_m, ego.y_m, ego.speed_mps, ego.heading_rad, ego.acceleration_mps2, ego.curvature_inv_m, round(connection.simulation.getTime() * 1000))
        drift = compute_drift(reference_state, current_state, drift_scales).normalized
        decision = "local_only"
        cooperative: tuple[WorldObject, ...] | None = None
        if method == "always_fresh":
            success, accounting = _exchange(object_count=len(actual), extension_bytes=0, retry=prior_failure, sizes=sizes, condition=condition, seed=seed * 10000 + 100 + index, deadline_ms=step_s * 1000)
            counters.update({key: int(accounting[key]) for key in ("generated_bytes", "delivered_original_bytes", "dropped_bytes", "duplicate_bytes", "generated_packets", "delivered_packets", "reordered_packets")})
            components.update(accounting["components"]), counters.update({"byte_conservation_failures": int(not accounting["byte_conservation_ok"]), "network_failures": int(not success)})
            prior_failure = not success
            if success:
                cooperative, decision = actual, "fresh"
                counters["fresh_decisions"] += 1
            else:
                counters["local_fallback_decisions"] += 1
        elif method == "local_only":
            counters["local_fallback_decisions"] += 1
        else:
            assert reuse_policy is not None
            reuse = cache_available and (receipt_accept if method == "surface_receipt_only" else reuse_policy.accepts(token, impaired_drift(drift, drift_scales, condition)))
            if reuse:
                age = (current_state.timestamp_ms - reference_state.timestamp_ms) / 1000.0
                cooperative, decision = tuple(item.propagated(age) for item in cache), "reuse"
                counters["accepted_reuse_decisions"] += 1
            else:
                success, accounting = _exchange(object_count=len(actual), extension_bytes=reuse_policy.extension_bytes, retry=prior_failure, sizes=sizes, condition=condition, seed=seed * 10000 + 1000 + index, deadline_ms=step_s * 1000)
                counters.update({key: int(accounting[key]) for key in ("generated_bytes", "delivered_original_bytes", "dropped_bytes", "duplicate_bytes", "generated_packets", "delivered_packets", "reordered_packets")})
                components.update(accounting["components"]), counters.update({"byte_conservation_failures": int(not accounting["byte_conservation_ok"]), "network_failures": int(not success)})
                prior_failure = not success
                if success:
                    cooperative, decision = actual, "refresh"
                    counters["refresh_decisions"] += 1
                    cache = actual
                    reference, reference_state = _reference_contract(seed=seed, spec=spec, ego=ego, local=local, cooperative=actual, timestamp_ms=current_state.timestamp_ms, planner=planner, scales=drift_scales)
                    token = reuse_policy.encode_reference(reference)
                    cache_available = True
                    if method == "surface_receipt_only":
                        receipt_accept = reuse_policy.accepts(
                            token, impaired_drift((0.0,) * 7, drift_scales, condition)
                        )
                else:
                    counters["local_fallback_decisions"] += 1
        route = Route(ego.x_m, ego.y_m, ego.heading_rad, planner.config.lane_width_m, planner.config.speed_limit_mps, allow_left=spec.intended_behavior == "left", allow_right=spec.intended_behavior == "right", intended_behavior=spec.intended_behavior, curvature_inv_m=ego.curvature_inv_m)
        plan = planner.plan(local, cooperative, ego, route)
        connection.vehicle.setSpeed("ego", plan.points[0].speed_mps)
        lane_index, road_id = connection.vehicle.getLaneIndex("ego"), connection.vehicle.getRoadID("ego")
        lane_count = connection.edge.getLaneNumber(road_id) if not road_id.startswith(":") else 1
        target_lane = lane_index + (1 if plan.maneuver == "left" else -1 if plan.maneuver == "right" else 0)
        if 0 <= target_lane < lane_count and target_lane != lane_index:
            connection.vehicle.changeLane("ego", target_lane, step_s)
        connection.simulationStep()
        updated, objects = ego_state(connection), world_objects(connection)
        ttc = _minimum_ttc(updated.x_m, updated.y_m, updated.speed_mps, updated.heading_rad, objects)
        if ttc is not None:
            minimum_ttc = min(minimum_ttc, ttc)
        collision_now = "ego" in connection.simulation.getCollidingVehiclesIDList()
        collision |= collision_now
        if prior_acceleration is not None:
            total_abs_jerk += abs(updated.acceleration_mps2 - prior_acceleration) / step_s
        prior_acceleration = updated.acceleration_mps2
        ticks.append({"tick": index, "time_s": round(connection.simulation.getTime(), 6), "decision": decision, "ego_x_m": updated.x_m, "ego_y_m": updated.y_m, "ego_speed_mps": updated.speed_mps, "ego_acceleration_mps2": updated.acceleration_mps2, "minimum_ttc_s": ttc, "collision": collision_now})
    route_progress = max(0.0, connection.vehicle.getDistance("ego") - start_distance)
    connection.close()
    minimum_ttc_value = None if math.isinf(minimum_ttc) else minimum_ttc
    counters["total_decisions"] = len(ticks)
    for key in (
        "generated_bytes", "delivered_original_bytes", "dropped_bytes",
        "duplicate_bytes", "generated_packets", "delivered_packets",
        "reordered_packets", "byte_conservation_failures", "network_failures",
        "accepted_reuse_decisions", "refresh_decisions", "fresh_decisions",
        "local_fallback_decisions",
    ):
        counters.setdefault(key, 0)
    return {
        "method": method, "pre_action_non_ego_state_hash": initial_hash,
        "collision": collision, "minimum_ttc_s": minimum_ttc_value,
        "critical_event": collision or (minimum_ttc_value is not None and minimum_ttc_value < critical_ttc_s),
        "route_progress_m": route_progress, "total_abs_jerk": total_abs_jerk,
        "communication": {**dict(counters), "component_bytes": dict(sorted(components.items()))},
        "ticks": ticks,
    }
