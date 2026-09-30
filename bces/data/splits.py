"""Intersection-disjoint partitioning and final-test access protection."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any

from bces.utils.hashing import canonical_json_hash


class Partition(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    TEST = "test"


class AccessPurpose(str, Enum):
    TRAINING = "training"
    MODEL_SELECTION = "model_selection"
    THRESHOLD_TUNING = "threshold_tuning"
    EARLY_STOPPING = "early_stopping"
    CALIBRATION = "calibration"
    FINAL_EVALUATION = "final_evaluation"


PROHIBITED_TEST_PURPOSES = frozenset(
    {
        AccessPurpose.TRAINING,
        AccessPurpose.MODEL_SELECTION,
        AccessPurpose.THRESHOLD_TUNING,
        AccessPurpose.EARLY_STOPPING,
        AccessPurpose.CALIBRATION,
    }
)


class PartitionAccessError(PermissionError):
    pass


@dataclass(frozen=True)
class SplitAssignment:
    salt: str
    by_intersection: dict[str, Partition]
    method: str = "sha256_ranked_whole_intersection_v1"
    capacity_basis: dict[str, int] | None = None
    capacity_objective: dict[str, Any] | None = None

    @property
    def sha256(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "method": self.method,
            "salt": self.salt,
            "assignments": {
                key: value.value for key, value in sorted(self.by_intersection.items())
            },
        }
        if self.capacity_basis is not None:
            payload["capacity_basis"] = dict(sorted(self.capacity_basis.items()))
        if self.capacity_objective is not None:
            payload["capacity_objective"] = self.capacity_objective
        if include_hash:
            payload["split_sha256"] = self.sha256
        return payload


def assign_intersections(
    intersection_ids: set[str] | list[str] | tuple[str, ...], *, salt: str
) -> SplitAssignment:
    unique = sorted(set(intersection_ids))
    if not unique:
        return SplitAssignment(salt=salt, by_intersection={})
    if not salt:
        raise ValueError("split salt must be non-empty")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    count = len(ranked)
    test_count = min(count, max(2 if count >= 8 else 1, round(0.15 * count)))
    calibration_count = min(
        count - test_count, max(1 if count >= 4 else 0, round(0.15 * count))
    )
    validation_count = min(
        count - test_count - calibration_count,
        max(1 if count >= 4 else 0, round(0.10 * count)),
    )
    train_count = count - test_count - calibration_count - validation_count
    assignments: dict[str, Partition] = {}
    cursor = 0
    for partition, size in (
        (Partition.TRAIN, train_count),
        (Partition.CALIBRATION, calibration_count),
        (Partition.VALIDATION, validation_count),
        (Partition.TEST, test_count),
    ):
        for intersection_id in ranked[cursor : cursor + size]:
            assignments[intersection_id] = partition
        cursor += size
    return SplitAssignment(salt=salt, by_intersection=assignments)


def assign_intersections_by_capacity(
    intersection_capacities: dict[str, int], *, salt: str
) -> SplitAssignment:
    """Freeze whole-intersection splits using metadata-only scenario capacity.

    Partition intersection counts stay identical to ``assign_intersections``.  Of
    those admissible assignments, this selects the closest scenario-capacity mix
    to 60/15/10/15.  The salt breaks exact objective ties only.
    """

    capacities = {str(key): int(value) for key, value in intersection_capacities.items()}
    if len(capacities) < 4 or any(value <= 0 for value in capacities.values()):
        raise ValueError("capacity-aware splitting requires four positive capacities")
    if not salt:
        raise ValueError("split salt must be non-empty")
    ids = sorted(capacities)
    count = len(ids)
    test_count = min(count, max(2 if count >= 8 else 1, round(0.15 * count)))
    calibration_count = min(
        count - test_count, max(1 if count >= 4 else 0, round(0.15 * count))
    )
    validation_count = min(
        count - test_count - calibration_count,
        max(1 if count >= 4 else 0, round(0.10 * count)),
    )
    ratios = {
        Partition.TRAIN: 0.60,
        Partition.CALIBRATION: 0.15,
        Partition.VALIDATION: 0.10,
        Partition.TEST: 0.15,
    }
    total_capacity = sum(capacities.values())
    best: tuple[float, str, dict[str, Partition], dict[str, int]] | None = None
    for test_ids in itertools.combinations(ids, test_count):
        after_test = [value for value in ids if value not in test_ids]
        for calibration_ids in itertools.combinations(after_test, calibration_count):
            after_calibration = [
                value for value in after_test if value not in calibration_ids
            ]
            for validation_ids in itertools.combinations(
                after_calibration, validation_count
            ):
                assignment = {
                    value: (
                        Partition.TEST
                        if value in test_ids
                        else Partition.CALIBRATION
                        if value in calibration_ids
                        else Partition.VALIDATION
                        if value in validation_ids
                        else Partition.TRAIN
                    )
                    for value in ids
                }
                partition_capacities = {
                    partition.value: sum(
                        capacities[key]
                        for key, assigned in assignment.items()
                        if assigned is partition
                    )
                    for partition in Partition
                }
                score = sum(
                    (
                        partition_capacities[partition.value] / total_capacity
                        - target
                    )
                    ** 2
                    / target
                    for partition, target in ratios.items()
                )
                signature = ",".join(
                    f"{key}={assignment[key].value}" for key in sorted(assignment)
                )
                tie_break = hashlib.sha256(f"{salt}:{signature}".encode()).hexdigest()
                candidate = (score, tie_break, assignment, partition_capacities)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
    assert best is not None
    score, _, assignment, partition_capacities = best
    return SplitAssignment(
        salt=salt,
        by_intersection=assignment,
        method="capacity_balanced_whole_intersection_v1",
        capacity_basis=capacities,
        capacity_objective={
            "name": "minimum_chi_square_distance_to_60_15_10_15",
            "score": score,
            "target_fractions": {
                partition.value: target for partition, target in ratios.items()
            },
            "partition_capacities": partition_capacities,
            "selection_inputs": "official scenario counts only; no behavior, model, oracle, or outcome labels",
        },
    )


def assert_scenario_integrity(
    scenario_intersections: list[tuple[str, str]], assignment: SplitAssignment
) -> None:
    seen: dict[str, str] = {}
    for scenario_id, intersection_id in scenario_intersections:
        previous = seen.setdefault(scenario_id, intersection_id)
        if previous != intersection_id:
            raise ValueError("one scenario crosses multiple intersections")
        if intersection_id not in assignment.by_intersection:
            raise ValueError("scenario references an unassigned intersection")


def require_partition_access(partition: Partition, purpose: AccessPurpose) -> None:
    partition = Partition(partition)
    purpose = AccessPurpose(purpose)
    if partition is Partition.TEST and purpose in PROHIBITED_TEST_PURPOSES:
        raise PartitionAccessError(
            f"final-test partition cannot be opened for {purpose.value}"
        )
