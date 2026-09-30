"""Fixed-normal polytope inclusion, slack, and conservative shrinkage."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from .codebooks import get_normal_codebook
from .drift import validate_normalized_drift


@dataclass(frozen=True)
class ExpirySurface:
    offsets: tuple[float, ...]
    codebook_id: int = 1
    calibration_id: int = 0

    def __post_init__(self) -> None:
        codebook = get_normal_codebook(self.codebook_id)
        offsets = tuple(float(value) for value in self.offsets)
        if len(offsets) != len(codebook):
            raise ValueError(f"surface requires {len(codebook)} offsets")
        if any(not math.isfinite(value) or value < 0.0 for value in offsets):
            raise ValueError("surface offsets must be finite and non-negative")
        if (
            not isinstance(self.calibration_id, int)
            or not 0 <= self.calibration_id <= 255
        ):
            raise ValueError("calibration_id must be uint8")
        object.__setattr__(self, "offsets", offsets)

    def slacks(self, normalized_drift: Iterable[float]) -> tuple[float, ...]:
        drift = validate_normalized_drift(normalized_drift)
        codebook = get_normal_codebook(self.codebook_id)
        return tuple(
            offset - sum(normal * value for normal, value in zip(row, drift))
            for row, offset in zip(codebook, self.offsets)
        )

    def minimum_slack(self, normalized_drift: Iterable[float]) -> float:
        return min(self.slacks(normalized_drift))

    def contains(
        self, normalized_drift: Iterable[float], *, tolerance: float = 0.0
    ) -> bool:
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        return self.minimum_slack(normalized_drift) >= -tolerance

    def shrink(self, delta: Real | Iterable[float]) -> ExpirySurface:
        if isinstance(delta, Real):
            deltas = (float(delta),) * len(self.offsets)
        else:
            deltas = tuple(float(value) for value in delta)
        if len(deltas) != len(self.offsets):
            raise ValueError("shrinkage must be scalar or match the offset count")
        if any(not math.isfinite(value) or value < 0.0 for value in deltas):
            raise ValueError("shrinkage values must be finite and non-negative")
        return ExpirySurface(
            offsets=tuple(
                max(0.0, offset - amount)
                for offset, amount in zip(self.offsets, deltas)
            ),
            codebook_id=self.codebook_id,
            calibration_id=self.calibration_id,
        )
