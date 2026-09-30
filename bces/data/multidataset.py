"""Multi-dataset plan validation and primary-only Phase-4 gating."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from bces.utils.hashing import canonical_json_hash

from .adequacy import AdequacyReport
from .datasets import EvidenceRole, dataset_profile


@dataclass(frozen=True)
class PlannedDataset:
    dataset_id: str
    evidence_role: EvidenceRole
    catalog: str
    purpose: str


@dataclass(frozen=True)
class MultiDatasetPlan:
    amendment_id: str
    network_impairments: str
    datasets: tuple[PlannedDataset, ...]

    def __post_init__(self) -> None:
        if not self.amendment_id:
            raise ValueError("amendment_id is required")
        if self.network_impairments != "replayed_or_simulated_not_observed_packet_logs":
            raise ValueError("network impairment provenance must remain explicit")
        ids = [item.dataset_id for item in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("planned dataset IDs must be unique")
        primary = [item for item in self.datasets if item.evidence_role is EvidenceRole.PRIMARY]
        if len(primary) != 1:
            raise ValueError("the plan must contain exactly one primary dataset")
        for item in self.datasets:
            profile = dataset_profile(item.dataset_id)
            if item.evidence_role is not profile.evidence_role:
                raise ValueError(f"evidence role mismatch for {item.dataset_id}")

    @property
    def primary_dataset_id(self) -> str:
        return next(
            item.dataset_id
            for item in self.datasets
            if item.evidence_role is EvidenceRole.PRIMARY
        )

    @property
    def sha256(self) -> str:
        return canonical_json_hash(
            {
                "amendment_id": self.amendment_id,
                "network_impairments": self.network_impairments,
                "datasets": [
                    {**asdict(item), "evidence_role": item.evidence_role.value}
                    for item in self.datasets
                ],
            }
        )


def load_multidataset_plan(path: Path) -> MultiDatasetPlan:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return MultiDatasetPlan(
        amendment_id=str(payload.get("amendment_id", "")),
        network_impairments=str(payload.get("network_impairments", "")),
        datasets=tuple(
            PlannedDataset(
                dataset_id=str(item["dataset_id"]),
                evidence_role=EvidenceRole(item["evidence_role"]),
                catalog=str(item["catalog"]),
                purpose=str(item["purpose"]),
            )
            for item in payload.get("datasets", ())
        ),
    )


def aggregate_adequacy(
    plan: MultiDatasetPlan, reports: dict[str, AdequacyReport]
) -> dict[str, Any]:
    missing = sorted(set(item.dataset_id for item in plan.datasets) - set(reports))
    if missing:
        raise ValueError(f"missing dataset adequacy reports: {missing}")
    primary = reports[plan.primary_dataset_id]
    datasets = {}
    for item in plan.datasets:
        report = reports[item.dataset_id]
        datasets[item.dataset_id] = {
            "evidence_role": item.evidence_role.value,
            "may_unlock_phase4": dataset_profile(item.dataset_id).may_unlock_phase4,
            "status": report.status.value,
            "passes": report.passes,
            "unmet": list(report.unmet),
            "blockers": list(report.blockers),
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "amendment_id": plan.amendment_id,
        "plan_sha256": plan.sha256,
        "primary_dataset_id": plan.primary_dataset_id,
        "primary_adequacy_status": primary.status.value,
        "phase4_permitted": primary.passes,
        "gate_rule": "primary_dataset_only",
        "network_impairments": plan.network_impairments,
        "datasets": datasets,
    }
    payload["report_sha256"] = canonical_json_hash(payload)
    return payload
