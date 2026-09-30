from __future__ import annotations

import json

import numpy as np

from bces.network.events import NetworkCondition, simulate_packets
from bces.network.study import application_workload, delivered_decisions, grid_conditions


SIZES = {
    "query": 32, "object_header": 64, "object_record": 48,
    "bces_extension": 48, "scalar_extension": 2,
    "security_envelope": 64, "refresh_request": 32, "retry_control": 16,
}


def test_authoritative_grid_has_9600_unique_conditions() -> None:
    grid = {
        "latency_ms": [0, 50, 100, 200, 300, 500],
        "random_loss": [0, .05, .1, .2, .4],
        "bandwidth_mbps": [.25, .5, 1, 2, 4],
        "pose_error_m": [0, .2, .5, 1],
        "yaw_error_deg": [0, .5, 1, 2],
        "clock_offset_ms": [0, 10, 25, 50],
    }
    conditions = grid_conditions(grid)
    assert len(conditions) == len({tuple(row.values()) for row in conditions}) == 9600


def test_workload_accounts_every_application_component() -> None:
    packets, components, requirements = application_workload(
        "surface", (True, False), (2, 3), SIZES,
        scenario_index=0, scenario_period_ms=4000, decision_step_ms=200,
    )
    assert components == {
        "bces_extension": 96, "object_header": 128, "object_records": 240,
        "query": 32, "refresh_request": 32, "retry_control": 0,
        "security_envelope": 256,
    }
    assert sum(packet.size_bytes for packet in packets) == sum(components.values())
    trace = simulate_packets(packets, NetworkCondition(0, 0, 4), seed=1)
    assert delivered_decisions(trace, requirements) == 2


def test_always_fresh_has_no_extension_or_refresh_bytes() -> None:
    _, components, _ = application_workload(
        "always_fresh", (False, False), (1, 1), SIZES,
        scenario_index=0, scenario_period_ms=4000, decision_step_ms=200,
    )
    assert "bces_extension" not in components
    assert "scalar_extension" not in components
    assert "refresh_request" not in components


def test_numpy_decision_count_is_normalized_before_json_output() -> None:
    flags = (np.bool_(True), np.bool_(False), np.bool_(True))
    payload = {"accepted_reuse_decisions": int(sum(flags))}
    assert json.loads(json.dumps(payload)) == {"accepted_reuse_decisions": 2}
