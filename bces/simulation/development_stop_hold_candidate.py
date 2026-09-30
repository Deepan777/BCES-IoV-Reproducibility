"""Development-only stopped-vehicle trajectory candidate.

This is deliberately outside the registered planner. A new closed-loop
policy would require its own identity, training, wire binding, and tests.
"""

from __future__ import annotations

import math

from bces.oracle.world import EgoState, Route, TrajectoryPoint


def stop_hold_points(
    ego: EgoState, route: Route, *, braking_mps2: float,
    horizon_s: float, step_s: float,
) -> tuple[TrajectoryPoint, ...]:
    """Brake at a constant rate until zero speed, then remain stationary."""
    if (not math.isfinite(braking_mps2) or braking_mps2 >= 0
            or not math.isfinite(horizon_s) or horizon_s <= 0
            or not math.isfinite(step_s) or step_s <= 0
            or not math.isclose(horizon_s / step_s, round(horizon_s / step_s), abs_tol=1e-9)):
        raise ValueError("invalid stop-hold discretization or braking rate")
    if ego.speed_mps < 0 or ego.speed_mps > route.speed_limit_mps:
        raise ValueError("initial speed is outside route limits")
    stop_time = ego.speed_mps / -braking_mps2
    points = []
    for index in range(1, round(horizon_s / step_s) + 1):
        time_s = index * step_s
        moving_time = min(time_s, stop_time)
        speed = max(0.0, ego.speed_mps + braking_mps2 * moving_time)
        longitudinal = ego.speed_mps * moving_time + 0.5 * braking_mps2 * moving_time**2
        curvature = route.curvature_inv_m
        if abs(curvature) < 1e-9:
            forward, lateral = longitudinal, 0.0
            heading = route.heading_rad
        else:
            forward = math.sin(curvature * longitudinal) / curvature
            lateral = (1.0 - math.cos(curvature * longitudinal)) / curvature
            heading = route.heading_rad + curvature * longitudinal
        cosine, sine = math.cos(route.heading_rad), math.sin(route.heading_rad)
        points.append(TrajectoryPoint(
            time_s=time_s,
            x_m=ego.x_m + cosine * forward - sine * lateral,
            y_m=ego.y_m + sine * forward + cosine * lateral,
            speed_mps=speed,
            heading_rad=heading,
            acceleration_mps2=braking_mps2 if time_s < stop_time else 0.0,
            lateral_offset_m=0.0,
        ))
    return tuple(points)
