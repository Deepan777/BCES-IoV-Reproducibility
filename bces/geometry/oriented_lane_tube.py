"""Conditional sampled collision overbound for a mapped straight actor lane.

This assumes the actor center remains near the map centerline and its body
heading remains within a stated angular deviation. Those premises require
independent validation before any certification claim.
"""

from __future__ import annotations

import math

from bces.geometry.lane_reachable_tube import PolylinePath, longitudinal_reach_m
from bces.oracle.world import TrajectoryPoint


def _maximum_projection(longitudinal_half: float, lateral_half: float,
                        max_heading_error_rad: float, *, axis: str) -> float:
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    if not 0 <= max_heading_error_rad <= math.pi / 2:
        raise ValueError("heading-error bound must lie within a right angle")
    def value(theta: float) -> float:
        if axis == "x":
            return longitudinal_half * math.cos(theta) + lateral_half * math.sin(theta)
        return longitudinal_half * math.sin(theta) + lateral_half * math.cos(theta)
    turning_point = (math.atan2(lateral_half, longitudinal_half) if axis == "x"
                     else math.atan2(longitudinal_half, lateral_half))
    return max(value(0.0), value(max_heading_error_rad),
               value(turning_point) if turning_point <= max_heading_error_rad else 0.0)


def straight_lane_possible_overlap(
    path: PolylinePath, ego_point: TrajectoryPoint, *,
    ego_length_m: float, ego_width_m: float,
    actor_length_m: float, actor_width_m: float,
    actor_speed_mps: float, elapsed_s: float,
    position_error_m: float, velocity_error_mps: float,
    acceleration_bound_mps2: float, lane_center_deviation_m: float,
    collision_margin_m: float, heading_error_bound_rad: float,
) -> bool:
    """Overapproximate rectangle collision for an eastbound straight lane.

    If the reach interval extends beyond the supplied map, fall back to an
    orientation-independent ball for that entire interval. The caller must
    separately cover every other legal path.
    """
    inputs = (ego_length_m, ego_width_m, actor_length_m, actor_width_m,
              actor_speed_mps, elapsed_s, position_error_m, velocity_error_mps,
              acceleration_bound_mps2, lane_center_deviation_m, collision_margin_m)
    if any(not math.isfinite(value) or value < 0 for value in inputs):
        raise ValueError("dimensions and reach inputs must be nonnegative and finite")
    if min(ego_length_m, ego_width_m, actor_length_m, actor_width_m) <= 0:
        raise ValueError("vehicle dimensions must be positive")
    if any(abs(point[1] - path.points[0][1]) > 1e-8 for point in path.points):
        raise ValueError("mapped path is not a horizontal straight lane")
    if any(right[0] < left[0] - 1e-8 for left, right in zip(path.points, path.points[1:])):
        raise ValueError("mapped path must progress eastward")
    lo, hi = longitudinal_reach_m(
        speed_mps=actor_speed_mps, elapsed_s=elapsed_s,
        position_error_m=position_error_m, velocity_error_mps=velocity_error_mps,
        acceleration_bound_mps2=acceleration_bound_mps2)
    ego_long = ego_length_m / 2 + collision_margin_m
    ego_lat = ego_width_m / 2 + collision_margin_m
    actor_long = actor_length_m / 2 + collision_margin_m
    actor_lat = actor_width_m / 2 + collision_margin_m
    if hi > path.length_m:
        radius = (math.hypot(ego_long, ego_lat) + math.hypot(actor_long, actor_lat)
                  + lane_center_deviation_m + position_error_m)
        return path.min_distance_over_interval((ego_point.x_m, ego_point.y_m), lo, hi) <= radius
    start_x = path.position_at(lo)[0]
    end_x = path.position_at(hi)[0]
    if end_x < start_x:
        raise RuntimeError("straight path interval reversed")
    c, s = abs(math.cos(ego_point.heading_rad)), abs(math.sin(ego_point.heading_rad))
    ego_x = ego_long * c + ego_lat * s
    ego_y = ego_long * s + ego_lat * c
    actor_x = _maximum_projection(actor_long, actor_lat, heading_error_bound_rad, axis="x")
    actor_y = _maximum_projection(actor_long, actor_lat, heading_error_bound_rad, axis="y")
    longitudinal_overlap = start_x - actor_x - ego_x <= ego_point.x_m <= end_x + actor_x + ego_x
    lateral_overlap = (abs(ego_point.y_m - path.points[0][1])
                       <= ego_y + actor_y + lane_center_deviation_m + position_error_m)
    return longitudinal_overlap and lateral_overlap
