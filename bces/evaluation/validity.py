"""Metric denominators for accepted reuse and oracle validity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bces.protocol.receiver import DecisionState


@dataclass(frozen=True)
class ValidityMetrics:
    total_decisions: int
    accepted_count: int
    accepted_oracle_invalid_count: int
    refresh_count: int
    invalid_count: int
    coverage: float
    unsafe_accept_rate: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "total_decisions": self.total_decisions,
            "accepted_count": self.accepted_count,
            "accepted_oracle_invalid_count": self.accepted_oracle_invalid_count,
            "refresh_count": self.refresh_count,
            "invalid_count": self.invalid_count,
            "coverage": self.coverage,
            "unsafe_accept_rate": self.unsafe_accept_rate,
        }


def compute_validity_metrics(
    outcomes: Iterable[tuple[DecisionState, bool]],
) -> ValidityMetrics:
    rows = tuple(outcomes)
    if not rows:
        raise ValueError("at least one decision is required")
    accepted = sum(state is DecisionState.ACCEPT_REUSE for state, _ in rows)
    unsafe = sum(
        state is DecisionState.ACCEPT_REUSE and not oracle_valid
        for state, oracle_valid in rows
    )
    refresh = sum(state is DecisionState.REFRESH for state, _ in rows)
    invalid = sum(state is DecisionState.INVALID for state, _ in rows)
    return ValidityMetrics(
        total_decisions=len(rows),
        accepted_count=accepted,
        accepted_oracle_invalid_count=unsafe,
        refresh_count=refresh,
        invalid_count=invalid,
        coverage=accepted / len(rows),
        unsafe_accept_rate=unsafe / accepted if accepted else None,
    )
