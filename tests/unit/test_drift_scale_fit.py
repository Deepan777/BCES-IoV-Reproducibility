from __future__ import annotations

import pytest

from bces.geometry.drift import KinematicState
from bces.oracle.scales import MINIMUM_SCALES, fit_drift_scales

pytestmark = pytest.mark.phase4


def state(x, timestamp):
    return KinematicState(x, 0, 10 + x, 0.01 * x, x / 10, x / 1000, timestamp)


def test_scale_fit_uses_absolute_training_drift_and_registered_floors():
    scales, report = fit_drift_scales([(state(0, 0), (state(1, 1000), state(2, 2000)))])
    assert report["sample_count"] == 2
    assert report["method"].endswith("_v2")
    assert all(value >= floor for value, floor in zip(scales.as_tuple(), MINIMUM_SCALES))


def test_scale_fit_rejects_empty_or_unregistered_quantile():
    with pytest.raises(ValueError):
        fit_drift_scales([])
    with pytest.raises(ValueError):
        fit_drift_scales([(state(0, 0), (state(1, 1000),))], quantile=1.0)
