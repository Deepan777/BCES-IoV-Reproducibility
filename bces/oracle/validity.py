"""Registered conjunction defining cached-versus-fresh decision validity."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ValidityThresholds:
    max_cost_regret: float
    max_cached_risk: float
    max_trajectory_deviation_m: float

    def __post_init__(self) -> None:
        for name in (
            "max_cost_regret",
            "max_cached_risk",
            "max_trajectory_deviation_m",
        ):
            value = _finite(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ValidityObservation:
    cached_cost: float
    fresh_cost: float
    cached_risk: float
    trajectory_deviation_m: float

    def __post_init__(self) -> None:
        for name in (
            "cached_cost",
            "fresh_cost",
            "cached_risk",
            "trajectory_deviation_m",
        ):
            value = _finite(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if self.cached_risk < 0.0 or self.trajectory_deviation_m < 0.0:
            raise ValueError("risk and trajectory deviation must be non-negative")

    @property
    def cost_regret(self) -> float:
        return self.cached_cost - self.fresh_cost


def evaluate_validity(
    observation: ValidityObservation, thresholds: ValidityThresholds
) -> bool:
    def within(value: float, limit: float) -> bool:
        return value <= limit or math.isclose(
            value, limit, rel_tol=1e-12, abs_tol=1e-12
        )

    return (
        within(observation.cost_regret, thresholds.max_cost_regret)
        and within(observation.cached_risk, thresholds.max_cached_risk)
        and within(
            observation.trajectory_deviation_m,
            thresholds.max_trajectory_deviation_m,
        )
    )
