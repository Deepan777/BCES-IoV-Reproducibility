"""Focused checks for the separately bound development recovery objective."""

from bces.oracle.planner import PlannerConfig
from bces.oracle.world import EgoState, Route
from bces.simulation.controlled_decisions import ControlledDecisionPlanner
from bces.simulation.development_recovery_planner import DevelopmentCruiseRecoveryPlanner


def route(ego: EgoState) -> Route:
    return Route(ego.x_m, ego.y_m, ego.heading_rad, intended_behavior="keep")


def test_reference_speed_is_causal_fixed_and_policy_hash_differs():
    config = PlannerConfig()
    original = ControlledDecisionPlanner(config)
    recovery = DevelopmentCruiseRecoveryPlanner(config)
    assert recovery.policy_hash != original.policy_hash

    first = EgoState(0.0, 0.0, 11.0, 0.0)
    recovery.plan((), (), first, route(first))
    assert recovery.reference_speed_mps == 11.0

    slowed = EgoState(40.0, 0.0, 5.8, 0.0)
    original_plan = original.plan((), (), slowed, route(slowed))
    recovery_plan = recovery.plan((), (), slowed, route(slowed))
    assert original_plan.acceleration_mps2 == 0.0
    assert recovery_plan.acceleration_mps2 > 0.0
    assert recovery.reference_speed_mps == 11.0


def test_collision_and_ttc_components_are_unchanged_for_same_candidate():
    config = PlannerConfig()
    original = ControlledDecisionPlanner(config)
    recovery = DevelopmentCruiseRecoveryPlanner(config)
    first = EgoState(0.0, 0.0, 11.0, 0.0)
    recovery.plan((), (), first, route(first))

    slowed = EgoState(40.0, 0.0, 5.8, 0.0)
    original_plan = original.plan((), (), slowed, route(slowed))
    from bces.oracle.world import GroundTruthWorld

    world = GroundTruthWorld(0, ())
    original_cost = original.evaluate(original_plan, world, slowed, route(slowed))
    recovery_cost = recovery.evaluate(original_plan, world, slowed, route(slowed))
    assert recovery_cost.collision == original_cost.collision
    assert recovery_cost.ttc == original_cost.ttc
