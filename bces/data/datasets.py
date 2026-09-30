"""Frozen dataset roles and schema-adapter boundaries for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceRole(str, Enum):
    PRIMARY = "primary"
    EXTERNAL_VALIDATION = "external_validation"
    SENSITIVITY = "sensitivity"
    OPTIONAL_REPLICATION = "optional_replication"


@dataclass(frozen=True)
class DatasetProfile:
    dataset_id: str
    evidence_role: EvidenceRole
    required_scene_roles: frozenset[str]
    requires_map: bool
    adapter_id: str
    may_unlock_phase4: bool


DATASET_PROFILES = {
    "v2xtraj_primary": DatasetProfile(
        "v2xtraj_primary",
        EvidenceRole.PRIMARY,
        frozenset({"ego", "vehicle", "infrastructure", "traffic_light"}),
        True,
        "v2xtraj",
        True,
    ),
    "tumtraf_v2x_external": DatasetProfile(
        "tumtraf_v2x_external",
        EvidenceRole.EXTERNAL_VALIDATION,
        frozenset({"tracks"}),
        True,
        "tumtraf_openlabel",
        False,
    ),
    "interaction_sensitivity": DatasetProfile(
        "interaction_sensitivity",
        EvidenceRole.SENSITIVITY,
        frozenset({"tracks"}),
        True,
        "interaction_csv",
        False,
    ),
    "v2xseq_tfd_compact": DatasetProfile(
        "v2xseq_tfd_compact",
        EvidenceRole.OPTIONAL_REPLICATION,
        frozenset({"cooperative", "vehicle", "infrastructure", "traffic_light"}),
        True,
        "v2xseq_tfd",
        False,
    ),
}


def dataset_profile(dataset_id: str) -> DatasetProfile:
    try:
        return DATASET_PROFILES[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset_id: {dataset_id}") from exc
