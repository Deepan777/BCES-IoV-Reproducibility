import math

import pytest

from bces.geometry.lane_reachable_tube import PolylinePath, longitudinal_reach_m


def test_reach_interval_contains_bounded_constant_acceleration():
    low, high = longitudinal_reach_m(speed_mps=10, elapsed_s=2,
                                     position_error_m=0.5,
                                     velocity_error_mps=1,
                                     acceleration_bound_mps2=4)
    assert (low, high) == pytest.approx((9.5, 30.5))
    for start_error in (-0.5, 0.5):
        for velocity_error in (-1, 1):
            for acceleration in (-4, 4):
                distance = 20 + start_error + velocity_error * 2 + 0.5 * acceleration * 4
                assert low <= distance <= high


def test_polyline_projection_slice_and_distance_to_turn():
    path = PolylinePath(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))
    distance, progress = path.nearest_progress((4.0, 1.0))
    assert distance == pytest.approx(1.0)
    assert progress == pytest.approx(4.0)
    sliced = path.remaining_from(progress)
    assert sliced.points == ((4.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    assert sliced.min_distance_over_interval((10.0, 6.0), 8.0, 14.0) == pytest.approx(0.0)
    assert sliced.min_distance_over_interval((10.0, 6.0), 0.0, 4.0) == pytest.approx(math.sqrt(40.0))


def test_after_map_end_uses_conservative_residual_ball():
    path = PolylinePath(((0.0, 0.0), (10.0, 0.0)))
    assert path.min_distance_over_interval((14.0, 0.0), 11.0, 13.0) == pytest.approx(1.0)
    assert path.min_distance_over_interval((12.0, 0.0), 9.0, 13.0) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_rejects_invalid_motion_and_intervals(bad):
    with pytest.raises(ValueError):
        longitudinal_reach_m(speed_mps=bad, elapsed_s=1.0,
                             position_error_m=0.0, velocity_error_mps=0.0,
                             acceleration_bound_mps2=0.0)
    path = PolylinePath(((0.0, 0.0), (1.0, 0.0)))
    with pytest.raises(ValueError):
        path.min_distance_over_interval((0.0, 0.0), bad, 1.0)
