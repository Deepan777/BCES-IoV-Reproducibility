from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bces.models.calibration import (
    CalibrationTarget,
    calibration_curve,
    decision_counts,
    quantize_numpy,
    select_shrinkage,
)
from bces.models.dataset import SurfaceDataset
from bces.protocol.codec import quantize_offsets

pytestmark = pytest.mark.phase7


def test_numpy_quantization_matches_protocol_codec() -> None:
    values = np.linspace(0.0, 1.4, 16, dtype=np.float64)[None, :]
    assert quantize_numpy(values)[0] == pytest.approx(
        quantize_offsets(values[0]).dequantized_offsets, abs=1e-12
    )


def test_shrinkage_never_increases_coverage() -> None:
    offsets = np.full((4, 16), 0.6)
    drift = np.zeros((4, 7))
    drift[:, 0] = (0.1, 0.3, 0.5, 0.7)
    valid = np.array((True, True, False, False))
    first = decision_counts(offsets, drift, valid, shrinkage=0.0, guard_band=0.05)
    second = decision_counts(offsets, drift, valid, shrinkage=0.2, guard_band=0.05)
    assert second["coverage"] <= first["coverage"]


def test_cluster_bootstrap_curve_is_seed_reproducible() -> None:
    offsets = np.full((6, 16), 0.4)
    drift = np.zeros((6, 7))
    drift[:, 0] = (0.1, 0.2, 0.3, 0.5, 0.6, 0.7)
    valid = np.array((True, True, False, False, False, True))
    clusters = ("a", "a", "b", "b", "c", "c")
    target = CalibrationTarget(bootstrap_replicates=100, minimum_accepted=1)
    first = calibration_curve(offsets, drift, valid, clusters, (0.0, 0.2), target, seed=17)
    second = calibration_curve(offsets, drift, valid, clusters, (0.0, 0.2), target, seed=17)
    assert first == second


def test_calibration_supports_frozen_ablation_codebooks() -> None:
    offsets = np.full((3, 8), 0.4)
    drift = np.zeros((3, 7))
    valid = np.array((True, False, True))
    normals = np.eye(7, dtype=np.float64)[:4]
    normals = np.concatenate((normals, -normals), axis=0)
    result = decision_counts(
        offsets,
        drift,
        valid,
        shrinkage=0.0,
        guard_band=0.05,
        normals=normals,
    )
    assert result["accepted"] == 3


def test_selection_reports_target_met_or_explicit_fallback() -> None:
    target = CalibrationTarget(unsafe_accept_upper=0.05, minimum_accepted=10)
    selected = select_shrinkage(
        [
            {"shrinkage": 0.0, "accepted": 20, "coverage": 0.4, "unsafe_accept_upper": 0.1},
            {"shrinkage": 0.2, "accepted": 12, "coverage": 0.24, "unsafe_accept_upper": 0.04},
        ],
        target,
    )
    assert selected["target_met"] and selected["shrinkage"] == 0.2


def test_phase7_cannot_open_test_partition(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="locked"):
        SurfaceDataset(tmp_path, "test")
