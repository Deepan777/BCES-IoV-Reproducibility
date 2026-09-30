import math

import pytest

from bces.oracle.world import EgoState, WorldObject
from bces.simulation.development_visibility import visible_local_objects


def vehicle(track_id, x, y, *, length=4.0, width=2.0):
    return WorldObject(track_id, x, y, 0.0, 0.0, 0.0, length, width)


def ids(objects):
    return {item.track_id for item in objects}


def test_vehicle_blocker_hides_lead_but_is_locally_visible():
    ego = EgoState(0, 0, 10, 0)
    blocker = vehicle("truck", 10, 0, length=6, width=4)
    lead = vehicle("lead", 20, 0)
    assert ids(visible_local_objects((blocker, lead), ego)) == {"truck"}


def test_lateral_reveal_is_not_permanent_invisibility():
    ego = EgoState(0, 0, 10, 0)
    blocker = vehicle("truck", 10, 0, length=6, width=4)
    lead = vehicle("lead", 20, 8)
    assert ids(visible_local_objects((blocker, lead), ego)) == {"truck", "lead"}


def test_range_and_field_of_view_are_applied():
    ego = EgoState(0, 0, 10, 0)
    objects = (vehicle("near", 20, 0), vehicle("far", 40, 0),
               vehicle("behind", -10, 0))
    assert ids(visible_local_objects(objects, ego, range_m=30,
                                     field_of_view_rad=math.pi)) == {"near"}


def test_invalid_contract_and_duplicate_ids_rejected():
    ego = EgoState(0, 0, 10, 0)
    with pytest.raises(ValueError):
        visible_local_objects((vehicle("a", 2, 0),), ego, range_m=0)
    with pytest.raises(ValueError):
        visible_local_objects((vehicle("a", 2, 0), vehicle("a", 5, 0)), ego)
