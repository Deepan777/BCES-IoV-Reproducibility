from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bces.geometry.codebooks import NORMAL_CODEBOOK_V1, codebook_sha256
from bces.geometry.drift import (
    DriftScales,
    KinematicState,
    compute_drift,
    wrap_angle_rad,
)
from bces.geometry.surfaces import ExpirySurface

pytestmark = pytest.mark.phase1


def state(**overrides: float) -> KinematicState:
    values: dict[str, float | int] = {
        "x_m": 0.0,
        "y_m": 0.0,
        "speed_mps": 10.0,
        "heading_rad": 0.0,
        "acceleration_mps2": 0.0,
        "curvature_inv_m": 0.0,
        "timestamp_ms": 1000,
    }
    values.update(overrides)
    return KinematicState(**values)  # type: ignore[arg-type]


def scales() -> DriftScales:
    return DriftScales(10.0, 2.0, 5.0, 0.5, 2.0, 0.1, 1.0)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(math.pi, -math.pi), (-math.pi, -math.pi), (3 * math.pi, -math.pi), (0.0, 0.0)],
)
def test_angle_wrapping(angle: float, expected: float) -> None:
    assert wrap_angle_rad(angle) == pytest.approx(expected)


def test_path_frame_projection_and_normalization() -> None:
    reference = state(heading_rad=math.pi / 2)
    current = state(
        x_m=-2.0,
        y_m=10.0,
        speed_mps=15.0,
        heading_rad=-math.pi + 0.1,
        acceleration_mps2=2.0,
        curvature_inv_m=0.1,
        timestamp_ms=2000,
    )
    drift = compute_drift(reference, current, scales())
    assert drift.raw[0] == pytest.approx(10.0)
    assert drift.raw[1] == pytest.approx(2.0)
    assert drift.normalized[0] == pytest.approx(1.0)
    assert drift.normalized[1] == pytest.approx(1.0)
    assert drift.normalized[2] == pytest.approx(1.0)
    assert drift.normalized[4:] == pytest.approx((1.0, 1.0, 1.0))


def test_negative_elapsed_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes"):
        compute_drift(state(timestamp_ms=2000), state(timestamp_ms=1999), scales())


def test_scales_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError):
        DriftScales(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        state(x_m=float("nan"))


def test_codebook_has_sixteen_unit_normals_and_registered_structure() -> None:
    assert len(NORMAL_CODEBOOK_V1) == 16
    assert all(len(row) == 7 for row in NORMAL_CODEBOOK_V1)
    assert all(
        math.sqrt(sum(value * value for value in row)) == pytest.approx(1.0)
        for row in NORMAL_CODEBOOK_V1
    )
    assert NORMAL_CODEBOOK_V1[0] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert NORMAL_CODEBOOK_V1[1] == (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert NORMAL_CODEBOOK_V1[15] == tuple(-value for value in NORMAL_CODEBOOK_V1[14])
    assert (
        codebook_sha256()
        == "2ba57e3d277ecda2ea947fe54e7c6c139b5a117c714a2f54aa56d4c7338106b1"
    )


def test_surface_inclusion_and_minimum_slack() -> None:
    surface = ExpirySurface(offsets=(1.0,) * 16, calibration_id=3)
    assert surface.contains((0.0,) * 7)
    assert surface.minimum_slack((0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx(
        0.5
    )
    assert not surface.contains((1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


@given(
    offsets=st.lists(
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=16,
        max_size=16,
    ),
    drift=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=7,
        max_size=7,
    ),
    delta=st.floats(
        min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
)
def test_conservative_shrinkage_never_increases_acceptance(
    offsets: list[float], drift: list[float], delta: float
) -> None:
    original = ExpirySurface(offsets=tuple(offsets))
    shrunken = original.shrink(delta)
    if shrunken.contains(drift):
        assert original.contains(drift)
