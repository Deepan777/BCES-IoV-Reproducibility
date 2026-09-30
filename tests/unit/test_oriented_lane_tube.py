import math

import pytest

from bces.geometry.lane_reachable_tube import PolylinePath
from bces.geometry.oriented_lane_tube import straight_lane_possible_overlap
from bces.oracle.world import TrajectoryPoint


def _possible(x=10.0, y=0.0, heading=math.pi / 2):
    path = PolylinePath(((0.0, 0.0), (30.0, 0.0)))
    point = TrajectoryPoint(0.2, x, y, 8.0, heading, 0.0, 0.0)
    return straight_lane_possible_overlap(
        path, point, ego_length_m=4.8, ego_width_m=1.9,
        actor_length_m=5.0, actor_width_m=1.8,
        actor_speed_mps=10.0, elapsed_s=1.0,
        position_error_m=0.5, velocity_error_mps=1.0,
        acceleration_bound_mps2=0.0, lane_center_deviation_m=2.5,
        collision_margin_m=0.5, heading_error_bound_rad=0.35)


def test_straight_lane_overlap_is_directional():
    assert _possible(y=0.0)
    assert not _possible(y=20.0)
    assert not _possible(x=-20.0)


def test_invalid_nonstraight_map_rejected():
    path = PolylinePath(((0.0, 0.0), (30.0, 2.0)))
    point = TrajectoryPoint(0.2, 10.0, 0.0, 8.0, math.pi / 2, 0.0, 0.0)
    with pytest.raises(ValueError):
        straight_lane_possible_overlap(
            path, point, ego_length_m=4.8, ego_width_m=1.9,
            actor_length_m=5.0, actor_width_m=1.8,
            actor_speed_mps=10.0, elapsed_s=1.0,
            position_error_m=0.5, velocity_error_mps=1.0,
            acceleration_bound_mps2=0.0, lane_center_deviation_m=2.5,
            collision_margin_m=0.5, heading_error_bound_rad=0.35)
