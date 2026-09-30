from __future__ import annotations

import json
from pathlib import Path

from bces.geometry.drift import DriftScales
from bces.network.events import NetworkCondition
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.simulation.closed_loop import prepare_matched_state, run_closed_loop_branch
from bces.simulation.scenario_builder import scenario_spec


ROOT = Path(__file__).resolve().parents[2]


def test_phase9_local_only_branch_replays_saved_state_without_communication(tmp_path) -> None:
    payload = json.loads((ROOT / "data/manifests/drift_scales_v2.json").read_text(encoding="utf-8"))
    scales = DriftScales(*map(float, payload["scales"]))
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml"))
    spec = scenario_spec("straight_lead_braking")
    state = tmp_path / "phase9_state.xml.gz"
    reference, reference_state, cache = prepare_matched_state(
        network=ROOT / "sumo/registered_grid_v1.net.xml", state_path=state,
        seed=19991, step_s=.2, reference_time_s=5.0, spec=spec,
        planner=planner, drift_scales=scales,
    )
    result = run_closed_loop_branch(
        network=ROOT / "sumo/registered_grid_v1.net.xml", state_path=state,
        seed=19991, step_s=.2, horizon_s=.4, spec=spec, method="local_only",
        planner=planner, drift_scales=scales, initial_reference=reference,
        initial_reference_state=reference_state, initial_cache=cache,
        condition=NetworkCondition(0, 0, 4),
        sizes={"query": 32, "object_header": 64, "object_record": 48,
               "security_envelope": 64, "retry_control": 16},
        reuse_policy=None, critical_ttc_s=2.0,
    )
    assert len(result["ticks"]) == 2
    assert result["communication"]["generated_bytes"] == 0
    assert result["pre_action_non_ego_state_hash"]

