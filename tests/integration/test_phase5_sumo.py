from __future__ import annotations

import json
from pathlib import Path

import pytest

from bces.geometry.drift import DriftScales
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.simulation.branching import recompute_pair_label, run_same_world_pair
from bces.simulation.scenario_builder import FAMILIES, scenario_spec
from bces.simulation.sumo_adapter import sumo_version

pytestmark = pytest.mark.phase5
ROOT = Path(__file__).resolve().parents[2]
SCALES = DriftScales(*json.loads((ROOT / "data" / "manifests" / "drift_scales_v2.json").read_text(encoding="utf-8"))["scales"])


def _run(tmp_path: Path, family: str, seed: int = 9001) -> dict:
    return run_same_world_pair(
        network=ROOT / "sumo" / "registered_grid_v1.net.xml",
        state_path=tmp_path / f"{family}.xml.gz",
        seed=seed,
        step_s=0.2,
        reference_time_s=5.0,
        horizon_s=3.0,
        spec=scenario_spec(family),
        planner=KinematicPlanner(PlannerConfig.from_yaml(ROOT / "configs" / "planner" / "kinematic.yaml")),
        thresholds=ValidityThresholds(5.0, 0.4, 2.0),
        drift_scales=SCALES,
        split="train",
    )


def test_pinned_sumo_version_is_available() -> None:
    assert sumo_version() == "1.27.1"


@pytest.mark.parametrize("family", FAMILIES)
def test_saved_state_branches_match_and_label_recomputes(tmp_path: Path, family: str) -> None:
    pair = _run(tmp_path, family, seed=9001 + FAMILIES.index(family))
    assert pair["state_equivalent"]
    assert pair["fresh"]["pre_action_non_ego_state_hash"] == pair["cached"]["pre_action_non_ego_state_hash"]
    assert pair["valid"] == recompute_pair_label(pair, ValidityThresholds(5.0, 0.4, 2.0))
    assert pair["reference"]["input_contract"] == "exact_reference_payload_and_frozen_policy_query_v2"
    assert pair["reference"]["behavior"] == scenario_spec(family).intended_behavior
    assert len(pair["reference"]["frozen_input"]["batch"]["object_features"]) == 32
    assert len(pair["reference"]["frozen_input"]["batch"]["reference_path"]) == 12
    assert all(len(tick["normalized_drift"]) == 7 for tick in pair["cached"]["ticks"])
    assert not list(tmp_path.glob("*.xml.gz"))


def test_all_six_scenario_families_are_registered() -> None:
    assert len(FAMILIES) == 6
    assert {scenario_spec(name).family for name in FAMILIES} == set(FAMILIES)


def test_sumo_ego_dynamics_remain_inside_frozen_planner_bounds(tmp_path: Path) -> None:
    pair = _run(tmp_path, "straight_lead_braking", seed=5124)
    accelerations = [
        tick["ego_acceleration_mps2"]
        for branch in (pair["cached"], pair["fresh"])
        for tick in branch["ticks"]
    ]
    assert min(accelerations) >= -4.0 - 1e-6
    assert max(accelerations) <= 2.0 + 1e-6
