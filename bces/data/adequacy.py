"""Minimum confirmatory evidence-adequacy classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .splits import Partition

MIN_VALID_WINDOWS = 500
MIN_SCENARIOS = 30
MIN_INTERSECTIONS = 8
MIN_BEHAVIORS = 3
MIN_TEST_INTERSECTIONS = 2
MIN_VALIDITY_DECISIONS_PER_SPLIT_BEHAVIOR = 50


class AdequacyStatus(str, Enum):
    CONFIRMATORY = "confirmatory"
    PILOT = "pilot"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class AdequacyReport:
    status: AdequacyStatus
    passes: bool
    counts: dict[str, int]
    validity_decisions_by_split_behavior: dict[str, dict[str, int]]
    unmet: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status.value,
            "passes": self.passes,
            "counts": self.counts,
            "validity_decisions_by_split_behavior": self.validity_decisions_by_split_behavior,
            "unmet": list(self.unmet),
            "blockers": list(self.blockers),
        }


def evaluate_adequacy(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    split_assignments: dict[str, str],
    validity_decisions_by_split_behavior: dict[str, dict[str, int]] | None = None,
    blockers: tuple[str, ...] = (),
) -> AdequacyReport:
    evidence_records = [
        record for record in records if str(record.get("source_role", "")) != "map"
    ]
    validity_counts = {
        split: dict(counts)
        for split, counts in (validity_decisions_by_split_behavior or {}).items()
    }
    scenarios = {str(record["scenario_id"]) for record in evidence_records}
    intersections = {str(record["intersection_id"]) for record in evidence_records}
    behaviors = {
        str(behavior)
        for record in evidence_records
        for behavior in record.get("behavior_families", ())
    }
    valid_windows_by_scenario: dict[str, int] = {}
    for record in evidence_records:
        scenario_id = str(record["scenario_id"])
        valid_windows_by_scenario[scenario_id] = max(
            valid_windows_by_scenario.get(scenario_id, 0),
            int(record.get("valid_windows", 0)),
        )
    test_intersections = {
        key
        for key, partition in split_assignments.items()
        if partition == Partition.TEST.value
    }
    counts = {
        "valid_windows": sum(valid_windows_by_scenario.values()),
        "scenarios": len(scenarios),
        "intersections": len(intersections),
        "behaviors": len(behaviors),
        "test_intersections": len(test_intersections),
    }
    unmet = []
    if counts["valid_windows"] < MIN_VALID_WINDOWS:
        unmet.append("valid_windows")
    if counts["scenarios"] < MIN_SCENARIOS:
        unmet.append("scenarios")
    if counts["intersections"] < MIN_INTERSECTIONS:
        unmet.append("intersections")
    if counts["behaviors"] < MIN_BEHAVIORS:
        unmet.append("behavior_families")
    if counts["test_intersections"] < MIN_TEST_INTERSECTIONS:
        unmet.append("held_out_test_intersections")
    if not validity_counts or any(
        validity_counts.get(partition.value, {}).get(behavior, 0)
        < MIN_VALIDITY_DECISIONS_PER_SPLIT_BEHAVIOR
        for partition in Partition
        for behavior in behaviors
    ):
        unmet.append("validity_decisions_per_split_behavior")
    passes = not unmet and not blockers
    status = (
        AdequacyStatus.CONFIRMATORY
        if passes
        else AdequacyStatus.BLOCKED
        if blockers
        else AdequacyStatus.PILOT
    )
    return AdequacyReport(
        status=status,
        passes=passes,
        counts=counts,
        validity_decisions_by_split_behavior=validity_counts,
        unmet=tuple(unmet),
        blockers=tuple(blockers),
    )
