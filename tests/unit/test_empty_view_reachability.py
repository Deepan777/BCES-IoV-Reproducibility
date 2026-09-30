import pytest

from bces.geometry.empty_view_reachability import conditional_empty_disk_clearance
from bces.oracle.world import TrajectoryPoint


def _points():
    return (
        TrajectoryPoint(0.2, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0),
        TrajectoryPoint(0.4, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0),
    )


def _clearance(radius=20.0, age=0.2):
    return conditional_empty_disk_clearance(
        source_center_xy_m=(0.0, 0.0), source_radius_m=radius,
        observation_age_s=age, ego_initial_xy_m=(0.0, 0.0),
        ego_future_points=_points(), maximum_ego_speed_mps=5.0,
        maximum_actor_speed_mps=10.0, ego_body_radius_m=2.0,
        actor_body_radius_m=1.0, localization_error_m=0.5)


def test_clearance_is_positive_when_entire_reachable_tube_stays_inside():
    result = _clearance()
    assert result.clear
    assert result.minimum_slack_m > 0
    assert result.checked_intervals == 2


def test_older_or_smaller_empty_region_abstains():
    assert not _clearance(radius=7.0).clear
    assert not _clearance(age=2.0).clear


def test_invalid_speed_or_time_is_rejected():
    with pytest.raises(ValueError):
        conditional_empty_disk_clearance(
            source_center_xy_m=(0.0, 0.0), source_radius_m=20.0,
            observation_age_s=-0.1, ego_initial_xy_m=(0.0, 0.0),
            ego_future_points=_points(), maximum_ego_speed_mps=5.0,
            maximum_actor_speed_mps=10.0, ego_body_radius_m=2.0,
            actor_body_radius_m=1.0)
