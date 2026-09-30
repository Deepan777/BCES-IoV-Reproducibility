"""Conditional continuous-time clearance from a completely observed empty disk.

The geometry is a sufficient condition *only if* the source really covered
the disk at the reference time, omitted no relevant actor, all actors obey
the stated speed/body bounds, and no actor spawns inside afterward. It is
not an authentication check or a decision-regret certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bces.oracle.world import TrajectoryPoint


@dataclass(frozen=True)
class EmptyDiskClearance:
    clear: bool
    minimum_slack_m: float
    checked_intervals: int


def conditional_empty_disk_clearance(
    *,
    source_center_xy_m: tuple[float, float],
    source_radius_m: float,
    observation_age_s: float,
    ego_initial_xy_m: tuple[float, float],
    ego_future_points: tuple[TrajectoryPoint, ...],
    maximum_ego_speed_mps: float,
    maximum_actor_speed_mps: float,
    ego_body_radius_m: float,
    actor_body_radius_m: float,
    localization_error_m: float = 0.0,
) -> EmptyDiskClearance:
    """Bound possible contact with any actor initially outside the empty disk.

    For interval [t_i, t_{i+1}], ego position is within
    ``maximum_ego_speed_mps * (t_{i+1}-t_i)`` of its known start position.
    An outside actor's center cannot penetrate farther than
    ``maximum_actor_speed_mps * (age+t_{i+1})`` from the disk boundary.
    Positive slack at *every* interval therefore excludes contact over
    the entire piecewise-continuous candidate horizon, not just samples.
    """
    values = (*source_center_xy_m, source_radius_m, observation_age_s,
              *ego_initial_xy_m, maximum_ego_speed_mps,
              maximum_actor_speed_mps, ego_body_radius_m,
              actor_body_radius_m, localization_error_m)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("clearance inputs must be finite")
    if (source_radius_m <= 0 or observation_age_s < 0
            or maximum_ego_speed_mps < 0 or maximum_actor_speed_mps < 0
            or ego_body_radius_m <= 0 or actor_body_radius_m < 0
            or localization_error_m < 0 or not ego_future_points):
        raise ValueError("invalid disk, speed, body, or trajectory")
    previous_time = 0.0
    previous_xy = ego_initial_xy_m
    minimum_slack = math.inf
    for point in ego_future_points:
        if (not math.isfinite(point.time_s) or point.time_s <= previous_time
                or not math.isfinite(point.x_m) or not math.isfinite(point.y_m)
                or not math.isfinite(point.speed_mps) or point.speed_mps < 0
                or point.speed_mps > maximum_ego_speed_mps + 1e-9):
            raise ValueError("nonmonotone, nonfinite, or over-speed trajectory")
        interval_s = point.time_s - previous_time
        if math.dist(previous_xy, (point.x_m, point.y_m)) > maximum_ego_speed_mps * interval_s + 1e-8:
            raise ValueError("candidate jumps farther than the assumed continuous speed bound")
        required = (math.dist(previous_xy, source_center_xy_m)
                    + maximum_ego_speed_mps * interval_s
                    + maximum_actor_speed_mps * (observation_age_s + point.time_s)
                    + ego_body_radius_m + actor_body_radius_m
                    + localization_error_m)
        minimum_slack = min(minimum_slack, source_radius_m - required)
        previous_time = point.time_s
        previous_xy = point.x_m, point.y_m
    return EmptyDiskClearance(minimum_slack > 0.0, minimum_slack,
                              len(ego_future_points))
