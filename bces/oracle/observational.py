"""Observational cached-versus-fresh labels from official V2X-Traj trajectories."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bces.geometry.drift import DriftScales, KinematicState, compute_drift
from bces.oracle.planner import KinematicPlanner, trajectory_deviation
from bces.oracle.validity import (
    ValidityObservation,
    ValidityThresholds,
    evaluate_validity,
)
from bces.oracle.world import EgoState, GroundTruthWorld, Route, WorldObject

BEHAVIOR_IDS = {"keep": 0, "brake": 1, "accelerate": 2, "left": 3, "right": 4}
CLASS_IDS = {"VEHICLE": 0, "PEDESTRIAN": 1, "CYCLIST": 2}


@dataclass(frozen=True)
class ObservationalConfig:
    thresholds: ValidityThresholds
    cache_ages_s: tuple[float, ...]
    drift_scales: DriftScales
    maximum_objects_per_source: int = 16

    def __post_init__(self) -> None:
        if not self.cache_ages_s or any(age <= 0 for age in self.cache_ages_s):
            raise ValueError("cache ages must be positive")
        if tuple(sorted(set(self.cache_ages_s))) != self.cache_ages_s:
            raise ValueError("cache ages must be unique and increasing")
        if self.maximum_objects_per_source <= 0:
            raise ValueError("maximum_objects_per_source must be positive")


def _timestamp_ms(value: str) -> int:
    numeric = float(value)
    return round(numeric if numeric >= 1e11 else numeric * 1000)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _snapshots(path: Path) -> dict[int, tuple[dict[str, str], ...]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = _timestamp_ms(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            grouped[timestamp].append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _heading(row: dict[str, str]) -> float:
    theta = _float(row, "theta", math.nan)
    if math.isfinite(theta):
        return theta
    vx, vy = _float(row, "v_x"), _float(row, "v_y")
    return math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-6 else 0.0


def _ego(row: dict[str, str]) -> EgoState:
    vx, vy = _float(row, "v_x"), _float(row, "v_y")
    return EgoState(
        x_m=_float(row, "x"),
        y_m=_float(row, "y"),
        speed_mps=math.hypot(vx, vy),
        heading_rad=_heading(row),
        length_m=max(0.5, _float(row, "length", 4.8)),
        width_m=max(0.3, _float(row, "width", 1.9)),
    )


def _world_object(row: dict[str, str], source: str) -> WorldObject | None:
    track_id = str(row.get("id", "")).strip()
    if not track_id:
        return None
    return WorldObject(
        track_id=track_id,
        x_m=_float(row, "x"),
        y_m=_float(row, "y"),
        vx_mps=_float(row, "v_x"),
        vy_mps=_float(row, "v_y"),
        heading_rad=_heading(row),
        length_m=max(0.3, _float(row, "length", 4.5)),
        width_m=max(0.2, _float(row, "width", 1.8)),
        class_id=CLASS_IDS.get(str(row.get("type", "")).upper(), 3),
        confidence=1.0,
        source=source,
    )


def _target_row(rows: Iterable[dict[str, str]], focal_actor_id: str) -> dict[str, str]:
    matches = [row for row in rows if str(row.get("id", "")).strip() == focal_actor_id]
    if len(matches) != 1:
        raise ValueError("target actor is not unique at requested timestamp")
    return matches[0]


def _objects(
    rows: Iterable[dict[str, str]],
    *,
    source: str,
    ego: EgoState,
    exclude_id: str,
    maximum: int,
) -> tuple[WorldObject, ...]:
    objects = []
    for row in rows:
        if str(row.get("id", "")).strip() == exclude_id:
            continue
        item = _world_object(row, source)
        if item is not None:
            objects.append(item)
    objects.sort(key=lambda item: (math.hypot(item.x_m - ego.x_m, item.y_m - ego.y_m), item.track_id))
    return tuple(objects[:maximum])


def _state(row: dict[str, str], timestamp_ms: int) -> KinematicState:
    ego = _ego(row)
    return KinematicState(
        x_m=ego.x_m,
        y_m=ego.y_m,
        speed_mps=ego.speed_mps,
        heading_rad=ego.heading_rad,
        acceleration_mps2=0.0,
        curvature_inv_m=0.0,
        timestamp_ms=timestamp_ms,
    )


def _aggregate(objects: tuple[WorldObject, ...], ego: EgoState) -> tuple[float, ...]:
    if not objects:
        return (0.0,) * 8
    dx = [item.x_m - ego.x_m for item in objects]
    dy = [item.y_m - ego.y_m for item in objects]
    speeds = [math.hypot(item.vx_mps, item.vy_mps) for item in objects]
    distances = [math.hypot(x, y) for x, y in zip(dx, dy)]
    return (
        sum(dx) / len(dx), sum(dy) / len(dy), sum(speeds) / len(speeds),
        min(distances), max(distances),
        sum(item.length_m for item in objects) / len(objects),
        sum(item.width_m for item in objects) / len(objects),
        sum(item.confidence for item in objects) / len(objects),
    )


def inference_features(
    ego: EgoState,
    local_objects: tuple[WorldObject, ...],
    cached_objects: tuple[WorldObject, ...],
    behavior: str,
    cache_age_s: float,
) -> tuple[float, ...]:
    behavior_one_hot = tuple(1.0 if index == BEHAVIOR_IDS[behavior] else 0.0 for index in range(5))
    local = _aggregate(local_objects, ego)
    cooperative = _aggregate(cached_objects, ego)
    relative_close = sum(
        math.hypot(item.x_m - ego.x_m, item.y_m - ego.y_m) <= 20.0
        for item in cached_objects
    )
    return (
        ego.speed_mps / 20.0,
        math.sin(ego.heading_rad),
        math.cos(ego.heading_rad),
        len(local_objects) / 16.0,
        len(cached_objects) / 16.0,
        cache_age_s / 4.0,
        *(value / scale for value, scale in zip(local, (50, 50, 20, 50, 80, 6, 3, 1))),
        *(value / scale for value, scale in zip(cooperative, (50, 50, 20, 50, 80, 6, 3, 1))),
        relative_close / 16.0,
        *behavior_one_hot,
        1.0,
        0.0,
        0.0,
    )


def label_pair(
    *,
    scenario_id: str,
    intersection_id: str,
    split: str,
    behavior: str,
    focal_actor_id: str,
    current_timestamp_ms: int,
    cache_age_s: float,
    ego_snapshots: dict[int, tuple[dict[str, str], ...]],
    vehicle_snapshots: dict[int, tuple[dict[str, str], ...]],
    infrastructure_snapshots: dict[int, tuple[dict[str, str], ...]],
    planner: KinematicPlanner,
    config: ObservationalConfig,
) -> dict[str, Any]:
    reference_timestamp_ms = current_timestamp_ms - round(cache_age_s * 1000)
    current_ego_row = _target_row(ego_snapshots[current_timestamp_ms], focal_actor_id)
    reference_ego_row = _target_row(ego_snapshots[reference_timestamp_ms], focal_actor_id)
    ego = _ego(current_ego_row)
    route = Route(ego.x_m, ego.y_m, ego.heading_rad, planner.config.lane_width_m, planner.config.speed_limit_mps)
    local = _objects(
        vehicle_snapshots.get(current_timestamp_ms, ()), source="vehicle", ego=ego,
        exclude_id=focal_actor_id, maximum=config.maximum_objects_per_source,
    )
    fresh = _objects(
        infrastructure_snapshots.get(current_timestamp_ms, ()), source="infrastructure",
        ego=ego, exclude_id=focal_actor_id, maximum=config.maximum_objects_per_source,
    )
    cached_reference = _objects(
        infrastructure_snapshots.get(reference_timestamp_ms, ()), source="infrastructure",
        ego=ego, exclude_id=focal_actor_id, maximum=config.maximum_objects_per_source,
    )
    cached = tuple(item.propagated(cache_age_s) for item in cached_reference)
    truth = _objects(
        ego_snapshots[current_timestamp_ms], source="ground_truth", ego=ego,
        exclude_id=focal_actor_id, maximum=config.maximum_objects_per_source * 2,
    )
    world = GroundTruthWorld(current_timestamp_ms, truth)
    cached_plan = planner.plan(local, cached, ego, route)
    fresh_plan = planner.plan(local, fresh, ego, route)
    cached_cost = planner.evaluate(cached_plan, world, ego, route)
    fresh_cost = planner.evaluate(fresh_plan, world, ego, route)
    deviation = trajectory_deviation(cached_plan, fresh_plan)
    observation = ValidityObservation(
        cached_cost=cached_cost.total,
        fresh_cost=fresh_cost.total,
        cached_risk=cached_cost.risk,
        trajectory_deviation_m=deviation,
    )
    valid = evaluate_validity(observation, config.thresholds)
    drift = compute_drift(
        _state(reference_ego_row, reference_timestamp_ms),
        _state(current_ego_row, current_timestamp_ms),
        config.drift_scales,
    )
    features = inference_features(ego, local, cached_reference, behavior, cache_age_s)
    return {
        "pair_id": f"{scenario_id}:{current_timestamp_ms}:{round(cache_age_s * 1000)}",
        "scenario_id": scenario_id,
        "intersection_id": intersection_id,
        "split": split,
        "behavior": behavior,
        "behavior_id": BEHAVIOR_IDS[behavior],
        "label_source": "observational",
        "reference_timestamp_ms": reference_timestamp_ms,
        "current_timestamp_ms": current_timestamp_ms,
        "cache_age_s": cache_age_s,
        # This legacy adapter has no sender reference query or directional
        # oracle. Its aggregates are receiver-time diagnostics, not SurfaceNet
        # inputs. Do not synthesize boundary targets from the validity label.
        "diagnostic_receiver_features": features,
        "training_eligible": False,
        "boundary_target_status": "not_measured",
        "input_contract_status": "missing_frozen_sender_query",
        "normalized_drift": drift.normalized,
        "cached_cost": cached_cost.to_dict(),
        "fresh_cost": fresh_cost.to_dict(),
        "cost_regret": observation.cost_regret,
        "cached_risk": observation.cached_risk,
        "trajectory_deviation_m": deviation,
        "cached_maneuver": cached_plan.maneuver,
        "fresh_maneuver": fresh_plan.maneuver,
        "valid": valid,
        "policy_hash": planner.policy_hash,
    }


def generate_scene_labels(
    review_record: dict[str, Any],
    *,
    raw_root: Path,
    planner: KinematicPlanner,
    config: ObservationalConfig,
) -> list[dict[str, Any]]:
    source = Path(review_record["source_file"])
    ego_path = raw_root / source
    vehicle_path = raw_root / "vehicle" / source.parts[1] / source.name
    infrastructure_path = raw_root / "infrastructure" / source.parts[1] / source.name
    ego_snapshots = _snapshots(ego_path)
    vehicle_snapshots = _snapshots(vehicle_path)
    infrastructure_snapshots = _snapshots(infrastructure_path)
    focal_actor_id = str(review_record["focal_actor_id"])
    focal_times = sorted(
        timestamp
        for timestamp, rows in ego_snapshots.items()
        if any(str(row.get("id", "")).strip() == focal_actor_id for row in rows)
    )
    current_timestamp = focal_times[79]
    behavior = str(review_record["behavior_families"][0])
    return [
        label_pair(
            scenario_id=str(review_record["scenario_id"]),
            intersection_id=str(review_record["intersection_id"]),
            split=str(review_record["split"]),
            behavior=behavior,
            focal_actor_id=focal_actor_id,
            current_timestamp_ms=current_timestamp,
            cache_age_s=age,
            ego_snapshots=ego_snapshots,
            vehicle_snapshots=vehicle_snapshots,
            infrastructure_snapshots=infrastructure_snapshots,
            planner=planner,
            config=config,
        )
        for age in config.cache_ages_s
    ]


def label_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = tuple(rows)
    cells: dict[str, Counter[str]] = defaultdict(Counter)
    for row in materialized:
        cells[str(row["split"])][str(row["behavior"])] += 1
    valid = sum(bool(row["valid"]) for row in materialized)
    return {
        "decisions": len(materialized),
        "valid": valid,
        "invalid": len(materialized) - valid,
        "valid_rate": valid / len(materialized) if materialized else None,
        "decisions_by_split_behavior": {
            split: dict(sorted(counts.items())) for split, counts in sorted(cells.items())
        },
        "scenario_count": len({row["scenario_id"] for row in materialized}),
        "pair_sha256": hashlib.sha256(
            "\n".join(str(row["pair_id"]) for row in materialized).encode()
        ).hexdigest(),
    }
