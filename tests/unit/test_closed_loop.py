from __future__ import annotations

import pytest

from bces.geometry.drift import DriftScales
from bces.network.events import NetworkCondition
from bces.simulation.closed_loop import exchange_application_bytes, impaired_drift


def test_locked_impairments_map_to_registered_drift_axes() -> None:
    scales = DriftScales(10, 2, 5, 0.5, 3, 0.1, 4)
    condition = NetworkCondition(100, .05, 1, clock_offset_ms=20, pose_error_m=1, yaw_error_deg=1)
    value = impaired_drift((0,) * 7, scales, condition)
    assert value[0] == pytest.approx(.1)
    assert value[3] == pytest.approx(0.034906585)
    assert value[6] == pytest.approx(.005)


def test_exchange_accounting_includes_security_extension_and_retry() -> None:
    sizes = {"query": 32, "object_header": 64, "object_record": 48, "security_envelope": 64, "retry_control": 16}
    value = exchange_application_bytes(3, 48, True, sizes)
    assert value == {"query": 32, "object_header": 64, "object_records": 144, "method_extension": 48, "security_envelope": 128, "retry_control": 16}

