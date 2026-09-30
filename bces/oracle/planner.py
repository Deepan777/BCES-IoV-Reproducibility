"""Frozen deterministic kinematic candidate planner with auditable cost vectors."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import yaml  # type: ignore[import-untyped]

from bces.utils.hashing import canonical_json_hash

from .world import (
    CostVector,
    EgoState,
    GroundTruthWorld,
    PlannedTrajectory,
    Route,
    TrajectoryPoint,
    WorldObject,
)


@dataclass(frozen=True)
class PlannerConfig:
    horizon_s: float = 3.0
    step_s: float = 0.2
    longitudinal_accelerations_mps2: tuple[float, ...] = (-4.0, -2.0, 0.0, 1.0, 2.0)
    lateral_maneuvers: tuple[str, ...] = ("keep", "left", "right")
    lane_width_m: float = 3.6
    speed_limit_mps: float = 20.0
    collision_margin_m: float = 0.5
    weights: tuple[tuple[str, float], ...] = (
        ("collision", 1000.0), ("ttc", 100.0), ("route", 10.0),
        ("lane", 20.0), ("comfort", 1.0), ("progress", 2.0),
    )
    policy_name: str = "bces_kinematic_policy_a_v1"

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)

    @property
    def sha256(self) -> str:
        return canonical_json_hash(asdict(self))

    @property
    def policy_hash(self) -> int:
        definition = {
            "config_sha256": self.sha256,
            "planner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_library_version": "kinematic_lattice_v2_behavior_bound",
            "cost_function_version": "six_component_v2_jerk_curvature",
        }
        return int(canonical_json_hash(definition)[:8], 16)

    @classmethod
    def from_yaml(cls, path: Path) -> PlannerConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            horizon_s=float(payload["horizon_s"]),
            step_s=float(payload["step_s"]),
            longitudinal_accelerations_mps2=tuple(map(float, payload["longitudinal_accelerations_mps2"])),
            lateral_maneuvers=tuple(payload["lateral_maneuvers"]),
            lane_width_m=float(payload["lane_width_m"]),
            speed_limit_mps=float(payload["speed_limit_mps"]),
            collision_margin_m=float(payload["collision_margin_m"]),
            weights=tuple((key, float(value)) for key, value in sorted(payload["weights"].items())),
            policy_name=str(payload["policy_name"]),
        )


class PlannerAdapter(Protocol):
    policy_hash: int

    def plan(
        self,
        local_objects: tuple[WorldObject, ...],
        cooperative_objects: tuple[WorldObject, ...] | None,
        ego: EgoState,
        route: Route,
    ) -> PlannedTrajectory: ...

    def evaluate(
        self, trajectory: PlannedTrajectory, world: GroundTruthWorld, ego: EgoState, route: Route
    ) -> CostVector: ...


def _smoothstep5(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def _axes(heading: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((math.cos(heading), math.sin(heading)), (-math.sin(heading), math.cos(heading)))


def _rect_overlap(
    ax: float, ay: float, ah: float, al: float, aw: float,
    bx: float, by: float, bh: float, bl: float, bw: float,
    margin: float,
) -> bool:
    delta = (bx - ax, by - ay)
    a_axes, b_axes = _axes(ah), _axes(bh)
    for ux, uy in (*a_axes, *b_axes):
        distance = abs(delta[0] * ux + delta[1] * uy)
        a_radius = (al / 2 + margin) * abs(a_axes[0][0] * ux + a_axes[0][1] * uy) + (aw / 2 + margin) * abs(a_axes[1][0] * ux + a_axes[1][1] * uy)
        b_radius = (bl / 2 + margin) * abs(b_axes[0][0] * ux + b_axes[0][1] * uy) + (bw / 2 + margin) * abs(b_axes[1][0] * ux + b_axes[1][1] * uy)
        if distance > a_radius + b_radius:
            return False
    return True


class KinematicPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.policy_hash = self.config.policy_hash

    def _candidate(self, ego: EgoState, route: Route, acceleration: float, maneuver: str) -> tuple[TrajectoryPoint, ...] | None:
        required_maneuver = {
            "keep": "keep", "brake": "keep", "accelerate": "keep",
            "left": "left", "right": "right",
        }[route.intended_behavior]
        if maneuver != required_maneuver:
            return None
        if route.intended_behavior == "brake" and acceleration >= 0:
            return None
        if route.intended_behavior == "accelerate" and acceleration <= 0:
            return None
        if maneuver == "left" and not route.allow_left:
            return None
        if maneuver == "right" and not route.allow_right:
            return None
        lateral_target = {"keep": 0.0, "left": route.lane_width_m, "right": -route.lane_width_m}[maneuver]
        points = []
        steps = round(self.config.horizon_s / self.config.step_s)
        for index in range(1, steps + 1):
            time_s = index * self.config.step_s
            speed = ego.speed_mps + acceleration * time_s
            if speed < -1e-9 or speed > route.speed_limit_mps + 1e-9:
                return None
            speed = max(0.0, speed)
            longitudinal = ego.speed_mps * time_s + 0.5 * acceleration * time_s**2
            lateral = lateral_target * _smoothstep5(time_s / self.config.horizon_s)
            curvature = route.curvature_inv_m
            if abs(curvature) < 1e-9:
                forward, curve_lateral = longitudinal, 0.0
                path_heading = route.heading_rad
            else:
                forward = math.sin(curvature * longitudinal) / curvature
                curve_lateral = (1.0 - math.cos(curvature * longitudinal)) / curvature
                path_heading = route.heading_rad + curvature * longitudinal
            cosine, sine = math.cos(route.heading_rad), math.sin(route.heading_rad)
            points.append(
                TrajectoryPoint(
                    time_s=time_s,
                    x_m=ego.x_m + cosine * forward - sine * (curve_lateral + lateral),
                    y_m=ego.y_m + sine * forward + cosine * (curve_lateral + lateral),
                    speed_mps=speed,
                    heading_rad=path_heading,
                    acceleration_mps2=acceleration,
                    lateral_offset_m=lateral,
                )
            )
        return tuple(points)

    def evaluate(self, trajectory: PlannedTrajectory, world: GroundTruthWorld, ego: EgoState, route: Route) -> CostVector:
        collision = 0.0
        minimum_ttc = math.inf
        for point in trajectory.points:
            for item in world.objects:
                predicted = item.propagated(point.time_s)
                if _rect_overlap(
                    point.x_m, point.y_m, point.heading_rad, ego.length_m, ego.width_m,
                    predicted.x_m, predicted.y_m, predicted.heading_rad, predicted.length_m,
                    predicted.width_m, self.config.collision_margin_m,
                ):
                    collision = 1.0
                dx, dy = predicted.x_m - point.x_m, predicted.y_m - point.y_m
                along = math.cos(point.heading_rad) * dx + math.sin(point.heading_rad) * dy
                lateral = abs(-math.sin(point.heading_rad) * dx + math.cos(point.heading_rad) * dy)
                object_along_speed = math.cos(point.heading_rad) * predicted.vx_mps + math.sin(point.heading_rad) * predicted.vy_mps
                closing = point.speed_mps - object_along_speed
                if along > 0 and closing > 1e-6 and lateral <= (ego.width_m + predicted.width_m) / 2 + self.config.collision_margin_m:
                    minimum_ttc = min(minimum_ttc, along / closing)
        ttc_cost = 0.0 if math.isinf(minimum_ttc) else math.exp(-max(0.0, minimum_ttc) / 2.0)
        final = trajectory.points[-1]
        route_cost = abs(final.lateral_offset_m) / route.lane_width_m if trajectory.maneuver == "keep" else 0.0
        lane_cost = max(0.0, abs(final.lateral_offset_m) / (1.5 * route.lane_width_m) - 1.0)
        comfort = abs(trajectory.acceleration_mps2 - ego.acceleration_mps2) / 4.0 + (0.25 if trajectory.maneuver != "keep" else 0.0)
        desired = min(route.speed_limit_mps, max(ego.speed_mps, 1.0)) * self.config.horizon_s
        progress = max(0.0, desired - math.hypot(final.x_m - ego.x_m, final.y_m - ego.y_m)) / max(desired, 1e-6)
        components = {
            "collision": collision, "ttc": ttc_cost, "route": route_cost,
            "lane": lane_cost, "comfort": comfort, "progress": progress,
        }
        total = sum(self.config.weight_map[key] * value for key, value in components.items())
        return CostVector(
            **components,
            total=total,
            minimum_ttc_s=None if math.isinf(minimum_ttc) else minimum_ttc,
        )

    def plan(
        self,
        local_objects: tuple[WorldObject, ...],
        cooperative_objects: tuple[WorldObject, ...] | None,
        ego: EgoState,
        route: Route,
    ) -> PlannedTrajectory:
        merged = {item.track_id: item for item in cooperative_objects or ()}
        merged.update({item.track_id: item for item in local_objects})
        perceived_world = GroundTruthWorld(0, tuple(merged[key] for key in sorted(merged)))
        candidates: list[PlannedTrajectory] = []
        for maneuver in self.config.lateral_maneuvers:
            for acceleration in self.config.longitudinal_accelerations_mps2:
                points = self._candidate(ego, route, acceleration, maneuver)
                if points is None:
                    continue
                placeholder = PlannedTrajectory(
                    maneuver=maneuver,
                    acceleration_mps2=acceleration,
                    points=points,
                    perceived_cost=CostVector(0, 0, 0, 0, 0, 0, 0, None),
                )
                cost = self.evaluate(placeholder, perceived_world, ego, route)
                candidates.append(
                    PlannedTrajectory(maneuver, acceleration, points, cost)
                )
        if not candidates:
            raise RuntimeError("no feasible trajectory candidate")
        return min(
            candidates,
            key=lambda item: (
                item.perceived_cost.collision,
                item.perceived_cost.ttc,
                item.perceived_cost.total,
                item.maneuver,
                item.acceleration_mps2,
            ),
        )


def trajectory_deviation(first: PlannedTrajectory, second: PlannedTrajectory) -> float:
    if len(first.points) != len(second.points):
        raise ValueError("trajectory lengths differ")
    return max(
        math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
        for a, b in zip(first.points, second.points)
    )
