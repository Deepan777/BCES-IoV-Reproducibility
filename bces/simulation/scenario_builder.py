"""Registered compact SUMO scenario families."""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES = (
    "straight_lead_braking",
    "hidden_adjacent_lane_change",
    "on_ramp_merge",
    "signalized_turn_crossing",
    "unprotected_crossing",
    "pedestrian_crossing",
)


@dataclass(frozen=True)
class ScenarioSpec:
    family: str
    ego_route: tuple[str, ...]
    actor_route: tuple[str, ...]
    actor_id: str
    actor_depart_position_m: float
    actor_depart_speed_mps: float
    actor_lane: int = 0
    pedestrian: bool = False
    braking_event: bool = False
    hidden_from_local: bool = False
    uncontrolled_signal: bool = False
    intended_behavior: str = "keep"


def scenario_spec(family: str) -> ScenarioSpec:
    specs = {
        "straight_lead_braking": ScenarioSpec(
            family, ("B0B1", "B1B2"), ("B0B1", "B1B2"), "lead", 40, 8,
            braking_event=True, hidden_from_local=True, intended_behavior="keep",
        ),
        "hidden_adjacent_lane_change": ScenarioSpec(
            family, ("B0B1", "B1B2"), ("B0B1", "B1B2"), "hidden", 18, 9,
            actor_lane=1, hidden_from_local=True, intended_behavior="left",
        ),
        "on_ramp_merge": ScenarioSpec(
            family, ("B0B1", "B1B2"), ("A0B0", "B0B1", "B1B2"), "merge", 55, 9,
            hidden_from_local=True,
        ),
        "signalized_turn_crossing": ScenarioSpec(
            family, ("B0B1", "B1C1"), ("A1B1", "B1C1"), "cross", 28, 8,
            intended_behavior="left",
        ),
        "unprotected_crossing": ScenarioSpec(
            family, ("B0B1", "B1B2"), ("A1B1", "B1C1"), "cross", 30, 9,
            uncontrolled_signal=True,
        ),
        "pedestrian_crossing": ScenarioSpec(
            family, ("B0B1", "B1B2"), ("A1B1", "B1C1"), "pedestrian", 60, 1.4,
            pedestrian=True, hidden_from_local=True, intended_behavior="keep",
        ),
    }
    if family not in specs:
        raise ValueError(f"unknown scenario family: {family}")
    return specs[family]
