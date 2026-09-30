"""Oracle-driven directional searches; finite sampling is not a safety proof.

A direction yields a measured valid radius and, when found, an invalid bracket.
Missing evidence, an infeasible state, and an invalid origin are different results.
No radius is synthesized from a binary label. World construction belongs to a
separately registered adapter, which must return the evidence for every call.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

from bces.geometry.drift import DRIFT_DIMENSION


@dataclass(frozen=True)
class OraclePoint:
    valid: bool | None
    feasible: bool
    evidence_id: str

    def __post_init__(self) -> None:
        if type(self.feasible) is not bool or (self.valid is not None and type(self.valid) is not bool):
            raise ValueError("oracle outcomes must be booleans or an explicit unknown")
        if not self.evidence_id:
            raise ValueError("each oracle call requires a traceable evidence ID")
        if not self.feasible and self.valid is not None:
            raise ValueError("an infeasible point cannot have a validity label")


@dataclass(frozen=True)
class BoundaryConfig:
    expansion_step: float = 0.1
    tolerance: float = 0.001
    max_radius: float = 2.0
    max_queries: int = 10000

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (self.expansion_step, self.tolerance, self.max_radius)):
            raise ValueError("search scales must be finite and positive")
        if self.tolerance >= self.expansion_step or self.max_queries < 2:
            raise ValueError("search tolerance or query budget is invalid")


@dataclass(frozen=True)
class BoundaryProbe:
    radius: float
    normalized_drift: tuple[float, ...]
    result: OraclePoint


@dataclass(frozen=True)
class DirectionalBoundary:
    direction: tuple[float, ...]
    status: str
    valid_radius: float | None
    invalid_radius: float | None
    censored: bool
    probes: tuple[BoundaryProbe, ...]

    @property
    def has_regression_target(self) -> bool:
        return self.status == "transition" and not self.censored

    def to_dict(self) -> dict:
        return asdict(self)


def search_boundary(
    oracle: Callable[[tuple[float, ...]], OraclePoint],
    direction: tuple[float, ...],
    config: BoundaryConfig,
) -> DirectionalBoundary:
    """Bracket the first *sampled* transition, then bisect to the tolerance.

    Uniform expansion bounds the unsampled gap by expansion_step. Narrow
    disconnected regions may still be missed and require the geometry audit.
    Unknown evidence is never treated as valid or as ordinary right censoring.
    """
    direction = tuple(float(v) for v in direction)
    if len(direction) != DRIFT_DIMENSION or not all(math.isfinite(v) for v in direction):
        raise ValueError("direction must have seven finite coordinates")
    if not math.isclose(sum(v * v for v in direction), 1.0, abs_tol=1e-10):
        raise ValueError("direction must be a unit vector from the registered codebook")
    probes: list[BoundaryProbe] = []

    def query(radius: float) -> OraclePoint:
        if len(probes) >= config.max_queries:
            raise RuntimeError("oracle query budget exhausted; boundary is incomplete")
        drift = tuple(radius * v for v in direction)
        result = oracle(drift)
        if not isinstance(result, OraclePoint):
            raise TypeError("oracle must return an OraclePoint with evidence")
        probes.append(BoundaryProbe(radius, drift, result))
        return result

    def finish(status: str, lower: float | None, upper: float | None = None) -> DirectionalBoundary:
        return DirectionalBoundary(direction, status, lower, upper,
                                   status in {"radius_limit", "feasibility_limit"}, tuple(probes))

    origin = query(0.0)
    if not origin.feasible:
        return finish("origin_infeasible", None)
    if origin.valid is None:
        return finish("missing_evidence", None)
    if not origin.valid:
        return finish("origin_invalid", None)
    lower = 0.0
    for index in range(1, math.ceil(config.max_radius / config.expansion_step) + 1):
        upper = min(index * config.expansion_step, config.max_radius)
        outcome = query(upper)
        if outcome.feasible and outcome.valid is None:
            return finish("missing_evidence", lower)
        if outcome.feasible and outcome.valid:
            lower = upper
            continue
        # Both a validity transition and a feasibility boundary are refined.
        invalid_upper = upper if outcome.feasible else None
        while upper - lower > config.tolerance:
            middle = (lower + upper) / 2.0
            outcome = query(middle)
            if outcome.feasible and outcome.valid is None:
                return finish("missing_evidence", lower, invalid_upper)
            if not outcome.feasible:
                upper = middle
            elif outcome.valid:
                lower = middle
            else:
                upper = middle
                invalid_upper = middle
        if invalid_upper is not None and invalid_upper - lower <= config.tolerance:
            return finish("transition", lower, invalid_upper)
        return finish("feasibility_limit", lower)
    return finish("radius_limit", lower)


def audit_boundary(boundary: DirectionalBoundary, oracle: Callable[[tuple[float, ...]], OraclePoint]) -> dict:
    """Replay every recorded point against an independently constructed oracle."""
    mismatches = []
    for index, probe in enumerate(boundary.probes):
        if oracle(probe.normalized_drift) != probe.result:
            mismatches.append(index)
    return {"probe_count": len(boundary.probes), "mismatches": mismatches,
            "passed": bool(boundary.probes) and not mismatches}


def directional_boundary_from_dict(payload: dict) -> DirectionalBoundary:
    probes = tuple(
        BoundaryProbe(
            float(item["radius"]), tuple(map(float, item["normalized_drift"])),
            OraclePoint(**item["result"]),
        )
        for item in payload["probes"]
    )
    return DirectionalBoundary(
        tuple(map(float, payload["direction"])), str(payload["status"]),
        None if payload["valid_radius"] is None else float(payload["valid_radius"]),
        None if payload["invalid_radius"] is None else float(payload["invalid_radius"]),
        bool(payload["censored"]), probes,
    )
