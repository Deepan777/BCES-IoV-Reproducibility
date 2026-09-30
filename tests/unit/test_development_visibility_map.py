import math

import pytest

from bces.oracle.world import EgoState, WorldObject
from bces.simulation.development_visibility_map import (
    StaticMapObstruction,
    visible_local_objects_with_map,
)


def target(x, y):
    return WorldObject("vehicle:cross", x, y, 8.0, 0.0, 0.0, 4.8, 1.8)


def test_corner_obstruction_temporarily_hides_approaching_cross_actor():
    ego = EgoState(104.8, 82.0, 14.0, math.pi / 2)
    corner = StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 6.0, 5.0)
    hidden = visible_local_objects_with_map((target(78.0, 95.2),), ego,
                                            blockers=(corner,), range_m=35.0)
    revealed = visible_local_objects_with_map((target(100.0, 95.2),), ego,
                                              blockers=(corner,), range_m=35.0)
    assert hidden == ()
    assert [item.track_id for item in revealed] == ["vehicle:cross"]


def test_map_obstruction_is_not_a_planner_object():
    ego = EgoState(104.8, 82.0, 14.0, math.pi / 2)
    corner = StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 6.0, 5.0)
    assert visible_local_objects_with_map((), ego, blockers=(corner,)) == ()


def test_invalid_and_duplicate_obstructions_rejected():
    with pytest.raises(ValueError):
        StaticMapObstruction("", 0, 0, 0, 2, 2)
    ego = EgoState(0, 0, 0, 0)
    a = StaticMapObstruction("same", 1, 1, 0, 2, 2)
    with pytest.raises(ValueError):
        visible_local_objects_with_map((), ego, blockers=(a, a))
