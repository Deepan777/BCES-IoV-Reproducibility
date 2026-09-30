from __future__ import annotations

import math
from pathlib import Path

import pytest

from bces.oracle.planner import KinematicPlanner, PlannerConfig, trajectory_deviation
from bces.oracle.validity import (
    ValidityObservation,
    ValidityThresholds,
    evaluate_validity,
)
from bces.oracle.world import EgoState, GroundTruthWorld, Route, WorldObject

pytestmark = pytest.mark.phase4

ROOT = Path(__file__).resolve().parents[2]


def _object(track_id: str, x: float, speed: float = 0.0) -> WorldObject:
    return WorldObject(track_id, x, 0.0, speed, 0.0, 0.0, 4.5, 1.8)


def test_planner_is_byte_structurally_deterministic() -> None:
    planner = KinematicPlanner()
    ego = EgoState(0, 0, 10, 0)
    route = Route(0, 0, 0)
    objects = (_object("lead", 22, 4),)
    first = planner.plan((), objects, ego, route)
    second = planner.plan((), objects, ego, route)
    assert first == second
    assert planner.policy_hash == PlannerConfig().policy_hash
    assert first.perceived_cost.to_dict() == second.perceived_cost.to_dict()


def test_registered_second_planner_is_distinct_and_batch_compatible() -> None:
    first = PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml")
    second = PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic_policy_b.yaml")
    assert first.policy_hash != second.policy_hash
    assert first.policy_name != second.policy_name
    assert round(second.horizon_s / second.step_s) == 12
    assert set(first.longitudinal_accelerations_mps2) < set(second.longitudinal_accelerations_mps2)


def test_cost_vector_records_collision_ttc_and_all_components() -> None:
    planner = KinematicPlanner()
    ego = EgoState(0, 0, 10, 0)
    route = Route(0, 0, 0)
    trajectory = planner.plan((), (), ego, route)
    cost = planner.evaluate(trajectory, GroundTruthWorld(0, (_object("stopped", 12),)), ego, route)
    assert cost.collision == 1.0
    assert cost.minimum_ttc_s is not None
    assert 0 < cost.ttc <= 1
    assert set(cost.to_dict()) == {
        "collision", "ttc", "route", "lane", "comfort", "progress", "total", "minimum_ttc_s"
    }


def test_hand_checked_cached_fresh_validity_cases() -> None:
    planner = KinematicPlanner()
    ego = EgoState(0, 0, 10, 0)
    route = Route(0, 0, 0)
    safe_truth = GroundTruthWorld(0, (_object("distant", 100, 10),))
    cached_equal = planner.plan((), safe_truth.objects, ego, route)
    fresh_equal = planner.plan((), safe_truth.objects, ego, route)
    cached_equal_cost = planner.evaluate(cached_equal, safe_truth, ego, route)
    fresh_equal_cost = planner.evaluate(fresh_equal, safe_truth, ego, route)
    thresholds = ValidityThresholds(5.0, 0.4, 2.0)
    assert evaluate_validity(
        ValidityObservation(
            cached_equal_cost.total,
            fresh_equal_cost.total,
            cached_equal_cost.risk,
            trajectory_deviation(cached_equal, fresh_equal),
        ),
        thresholds,
    )
    hazardous_truth = GroundTruthWorld(0, (_object("lead", 15, 0),))
    fresh = planner.plan((), hazardous_truth.objects, ego, route)
    fresh_cost = planner.evaluate(fresh, hazardous_truth, ego, route)
    cached_missed = planner.plan((), (), ego, route)
    missed_cost = planner.evaluate(cached_missed, hazardous_truth, ego, route)
    assert not evaluate_validity(
        ValidityObservation(
            missed_cost.total,
            fresh_cost.total,
            missed_cost.risk,
            trajectory_deviation(cached_missed, fresh),
        ),
        thresholds,
    )


def test_oriented_rectangles_do_not_report_distant_lateral_collision() -> None:
    planner = KinematicPlanner()
    ego = EgoState(0, 0, 5, 0)
    route = Route(0, 0, 0)
    trajectory = planner.plan((), (), ego, route)
    world = GroundTruthWorld(0, (WorldObject("side", 5, 10, 0, 0, math.pi / 2, 4.5, 1.8),))
    assert planner.evaluate(trajectory, world, ego, route).collision == 0.0


@pytest.mark.parametrize(
    ("behavior", "expected_maneuver", "acceleration_sign"),
    [("keep", "keep", 0), ("brake", "keep", -1),
     ("accelerate", "keep", 1), ("left", "left", 0), ("right", "right", 0)],
)
def test_intended_behavior_constrains_candidate_library(
    behavior: str, expected_maneuver: str, acceleration_sign: int
) -> None:
    planner = KinematicPlanner()
    ego = EgoState(0, 0, 10, 0)
    route = Route(0, 0, 0, intended_behavior=behavior,
                  allow_left=behavior == "left", allow_right=behavior == "right")
    plan = planner.plan((), (), ego, route)
    assert plan.maneuver == expected_maneuver
    if acceleration_sign:
        assert math.copysign(1, plan.acceleration_mps2) == acceleration_sign


def test_acceleration_and_curvature_change_policy_cost_and_path() -> None:
    planner = KinematicPlanner()
    still = EgoState(0, 0, 10, 0, acceleration_mps2=0, curvature_inv_m=0)
    changing = EgoState(0, 0, 10, 0, acceleration_mps2=2, curvature_inv_m=0.03)
    straight = planner.plan((), (), still, Route(0, 0, 0, curvature_inv_m=0))
    curved = planner.plan((), (), changing, Route(0, 0, 0, curvature_inv_m=0.03))
    assert straight.points[-1].y_m == 0
    assert curved.points[-1].y_m != 0
    assert straight.acceleration_mps2 != curved.acceleration_mps2
