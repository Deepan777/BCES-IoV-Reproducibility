"""Development-only geometric local view with off-road static obstructions.

Static rectangles represent declared map obstructions, not SUMO vehicles and
not calibrated buildings, cameras, lidar, or radar. They never appear in the
planner's object list. This module must not change any registered v1/v2 study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from bces.oracle.world import EgoState, WorldObject
from bces.simulation.development_visibility import (
    _footprint_points,
    _segment_hits_footprint,
)


@dataclass(frozen=True)
class StaticMapObstruction:
    obstruction_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if not self.obstruction_id:
            raise ValueError("obstruction_id is required")
        if not all(math.isfinite(value) for value in (
            self.x_m, self.y_m, self.heading_rad, self.length_m, self.width_m
        )) or self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("invalid static obstruction geometry")

    def footprint(self) -> WorldObject:
        return WorldObject("map:" + self.obstruction_id, self.x_m, self.y_m,
                           0.0, 0.0, self.heading_rad, self.length_m,
                           self.width_m, source="development_map_obstruction")


def visible_local_objects_with_map(
    objects: tuple[WorldObject, ...],
    ego: EgoState,
    *,
    blockers: tuple[StaticMapObstruction, ...],
    range_m: float = 30.0,
    field_of_view_rad: float = math.pi,
) -> tuple[WorldObject, ...]:
    """Show a target only if one footprint ray avoids *all* opaque blockers.

    Dynamic vehicles and static map rectangles both block rays. Pedestrians
    are not treated as opaque. Range applies to target centers. This is an
    ideal deterministic 2-D visibility rule, not a physical sensor model.
    """
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError("range_m must be positive and finite")
    if not math.isfinite(field_of_view_rad) or not 0 < field_of_view_rad <= 2 * math.pi:
        raise ValueError("field_of_view_rad must be in (0, 2*pi]")
    if len({item.track_id for item in objects}) != len(objects):
        raise ValueError("track IDs must be unique")
    if len({item.obstruction_id for item in blockers}) != len(blockers):
        raise ValueError("obstruction IDs must be unique")

    opaque = tuple(item for item in objects if item.class_id == 0) + tuple(
        blocker.footprint() for blocker in blockers
    )
    origin = (ego.x_m, ego.y_m)
    visible = []
    for target in objects:
        if math.hypot(target.x_m - ego.x_m, target.y_m - ego.y_m) > range_m:
            continue
        for point in _footprint_points(target):
            bearing = math.atan2(point[1] - ego.y_m, point[0] - ego.x_m)
            angle = math.atan2(math.sin(bearing - ego.heading_rad),
                               math.cos(bearing - ego.heading_rad))
            if abs(angle) > field_of_view_rad / 2:
                continue
            if all(blocker.track_id == target.track_id or
                   not _segment_hits_footprint(origin, point, blocker)
                   for blocker in opaque):
                visible.append(replace(target, source="development_2d_map_sensor"))
                break
    return tuple(visible)
