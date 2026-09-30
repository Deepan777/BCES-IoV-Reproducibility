"""Development-only 2-D visibility model for communication-value studies.

This is deterministic geometric line of sight, not a calibrated camera, radar,
lidar, or public-road sensor model. It must not be substituted into registered
v1/v2 runners or interpreted as source-faithful external validation.
"""

from __future__ import annotations

import math
from dataclasses import replace

from bces.oracle.world import EgoState, WorldObject


def _footprint_points(item: WorldObject) -> tuple[tuple[float, float], ...]:
    """Center and four corners of an oriented rectangular footprint."""
    forward = (math.cos(item.heading_rad), math.sin(item.heading_rad))
    left = (-forward[1], forward[0])
    points = [(item.x_m, item.y_m)]
    for along in (-0.5 * item.length_m, 0.5 * item.length_m):
        for across in (-0.5 * item.width_m, 0.5 * item.width_m):
            points.append((item.x_m + along * forward[0] + across * left[0],
                           item.y_m + along * forward[1] + across * left[1]))
    return tuple(points)


def _segment_hits_footprint(origin: tuple[float, float],
                            destination: tuple[float, float],
                            blocker: WorldObject) -> bool:
    """Whether a ray reaches a blocker rectangle before its destination."""
    cosine = math.cos(blocker.heading_rad)
    sine = math.sin(blocker.heading_rad)

    def local(point: tuple[float, float]) -> tuple[float, float]:
        dx, dy = point[0] - blocker.x_m, point[1] - blocker.y_m
        return (cosine * dx + sine * dy, -sine * dx + cosine * dy)

    start = local(origin)
    end = local(destination)
    delta = (end[0] - start[0], end[1] - start[1])
    enter, leave = 0.0, 1.0
    for coordinate, direction, half_extent in (
        (start[0], delta[0], blocker.length_m / 2),
        (start[1], delta[1], blocker.width_m / 2),
    ):
        if abs(direction) < 1e-12:
            if abs(coordinate) > half_extent:
                return False
            continue
        first = (-half_extent - coordinate) / direction
        second = (half_extent - coordinate) / direction
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
        if enter > leave:
            return False
    return enter >= 0.0 and enter < 1.0 - 1e-9 and leave >= 0.0


def visible_local_objects(objects: tuple[WorldObject, ...], ego: EgoState, *,
                          range_m: float = 30.0, field_of_view_rad: float = math.pi,
                          sensor_height_m: float = 1.5) -> tuple[WorldObject, ...]:
    """Return objects with at least one unblocked footprint ray in range/FOV.

    Vehicle rectangles are opaque only in 2-D. ``sensor_height_m`` is retained
    as an explicit contract parameter but does not alter rays: height-aware
    occlusion requires measured object heights and is deliberately unsupported.
    """
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError("range_m must be positive and finite")
    if not math.isfinite(field_of_view_rad) or not 0 < field_of_view_rad <= 2 * math.pi:
        raise ValueError("field_of_view_rad must be in (0, 2*pi]")
    if not math.isfinite(sensor_height_m) or sensor_height_m <= 0:
        raise ValueError("sensor_height_m must be positive and finite")
    if len({item.track_id for item in objects}) != len(objects):
        raise ValueError("track IDs must be unique")

    origin = (ego.x_m, ego.y_m)
    result = []
    for target in objects:
        if math.hypot(target.x_m - ego.x_m, target.y_m - ego.y_m) > range_m:
            continue
        visible = False
        for point in _footprint_points(target):
            bearing = math.atan2(point[1] - ego.y_m, point[0] - ego.x_m)
            angle = math.atan2(math.sin(bearing - ego.heading_rad),
                               math.cos(bearing - ego.heading_rad))
            if abs(angle) > field_of_view_rad / 2:
                continue
            if not any(blocker.track_id != target.track_id and blocker.class_id == 0
                       and _segment_hits_footprint(origin, point, blocker)
                       for blocker in objects):
                visible = True
                break
        if visible:
            result.append(replace(target, source="development_2d_local_sensor"))
    return tuple(result)
