from __future__ import annotations

import math

import pytest

from bces.geometry.drift import DriftScales, KinematicState
from bces.oracle.observational_v2 import state_from_drift

pytestmark = pytest.mark.phase4
SCALES = DriftScales(10, 4, 5, 0.5, 2, 0.1, 4)
REFERENCE = KinematicState(5, 7, 10, math.pi / 2, 1, 0.02, 1000)


def test_normalized_drift_reconstructs_receiver_state_in_reference_frame():
    result = state_from_drift(REFERENCE, (1, 0.5, -0.2, 0.2, 0.5, -0.1, 0.25), SCALES)
    assert result is not None
    assert result.x_m == pytest.approx(3)
    assert result.y_m == pytest.approx(17)
    assert result.speed_mps == pytest.approx(9)
    assert result.acceleration_mps2 == pytest.approx(2)
    assert result.curvature_inv_m == pytest.approx(0.01)
    assert result.timestamp_ms == 2000


@pytest.mark.parametrize("drift", [
    (0, 0, 0, 0, 0, 0, -0.1),
    (0, 0, -3, 0, 0, 0, 0),
    (0, 0, 0, 0, 5, 0, 0),
    (0, 0, 0, 0, 0, 5, 0),
])
def test_impossible_receiver_state_is_rejected(drift):
    assert state_from_drift(REFERENCE, drift, SCALES) is None
