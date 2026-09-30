"""Deterministic manifest, split, and adequacy workflows for Phase 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import write_json_atomic

from .adequacy import AdequacyReport, evaluate_adequacy
from .manifest import Catalog, ManifestRecord, manifest_payload
from .splits import (
    SplitAssignment,
    assert_scenario_integrity,
    assign_intersections,
    assign_intersections_by_capacity,
)
from .adapters import validate_dataset_file

SPLIT_SALT = "bces-iov-v1-preregistered"


@dataclass(frozen=True)
class DataState:
    dataset_id: str
    evidence_role: str
    records: tuple[ManifestRecord, ...]
    split: SplitAssignment
    adequacy: AdequacyReport


def blocking_reasons(catalog_path: Path) -> tuple[str, ...]:
    payload = yaml.safe_load(Path(catalog_path).read_text(encoding="utf-8")) or {}
    return tuple(str(value) for value in payload.get("blocking_reasons", ()))


def load_verified_intersection_capacities(path: Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = payload.pop("inventory_sha256")
    if canonical_json_hash(payload) != expected_hash:
        raise ValueError("intersection inventory hash does not verify")
    return {
        str(key): int(value)
        for key, value in payload.get("intersection_scenario_counts", {}).items()
    }


def _manifest_record(payload: dict[str, Any]) -> ManifestRecord:
    return ManifestRecord(
        release=str(payload["release"]),
        official_url=str(payload["official_url"]),
        license_url=str(payload["license_url"]),
        intersection_id=str(payload["intersection_id"]),
        scenario_id=str(payload["scenario_id"]),
        source_file=str(payload["source_file"]),
        source_role=payload["source_role"],
        source_bytes=int(payload["source_bytes"]),
        sha256=str(payload["sha256"]),
        start_timestamp_ms=int(payload["start_timestamp_ms"]),
        end_timestamp_ms=int(payload["end_timestamp_ms"]),
        split=str(payload["split"]),
        selection_reason=str(payload["selection_reason"]),
        row_count=int(payload.get("row_count", 0)),
        valid_windows=int(payload.get("valid_windows", 0)),
        behavior_families=tuple(payload.get("behavior_families", ())),
        dropped_rows=tuple(payload.get("dropped_rows", ())),
        continuity_gaps=tuple(payload.get("continuity_gaps", ())),
        archive_member=payload.get("archive_member"),
    )


def load_verified_manifest(path: Path) -> tuple[ManifestRecord, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = payload.pop("manifest_sha256")
    if canonical_json_hash(payload) != expected_hash:
        raise ValueError("dataset manifest hash does not verify")
    return tuple(_manifest_record(item) for item in payload.get("records", ()))


def build_data_state(
    catalog: Catalog,
    *,
    raw_root: Path,
    blockers: tuple[str, ...] = (),
    split_salt: str = SPLIT_SALT,
    split_capacities: dict[str, int] | None = None,
) -> DataState:
    raw_root = Path(raw_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    catalog_by_path = {entry.destination: entry for entry in catalog.entries}
    actual_files = {
        path.relative_to(raw_root).as_posix(): path
        for path in raw_root.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    }
    unexpected = sorted(set(actual_files) - set(catalog_by_path))
    if unexpected:
        raise PermissionError(f"raw files are not allowlisted: {unexpected}")

    present_entries = [
        entry for entry in catalog.entries if entry.destination in actual_files
    ]
    present_intersections = {entry.intersection_id for entry in present_entries}
    if split_capacities is not None and present_intersections:
        missing_capacities = present_intersections - set(split_capacities)
        if missing_capacities:
            raise ValueError(f"split capacities missing intersections: {sorted(missing_capacities)}")
        split = assign_intersections_by_capacity(
            {key: split_capacities[key] for key in present_intersections},
            salt=split_salt,
        )
    else:
        split = assign_intersections(present_intersections, salt=split_salt)
    assert_scenario_integrity(
        [(entry.scene_id, entry.intersection_id) for entry in present_entries], split
    )
    records = []
    for entry in present_entries:
        path = actual_files[entry.destination]
        measured_bytes = path.stat().st_size
        digest = sha256_file(path)
        if entry.expected_bytes is not None and entry.expected_bytes != measured_bytes:
            raise ValueError(f"size mismatch for {entry.entry_id}")
        if entry.expected_sha256 is not None and entry.expected_sha256 != digest:
            raise ValueError(f"SHA-256 mismatch for {entry.entry_id}")
        validation = validate_dataset_file(path, entry, dataset_id=catalog.dataset_id)
        duration_ms = validation.timestamp_end_ms - validation.timestamp_start_ms
        records.append(
            ManifestRecord(
                release=entry.release,
                official_url=entry.url,
                license_url=entry.license_url,
                intersection_id=entry.intersection_id,
                scenario_id=entry.scene_id,
                source_file=entry.destination,
                source_role=entry.role,
                source_bytes=measured_bytes,
                sha256=digest,
                start_timestamp_ms=validation.timestamp_start_ms,
                end_timestamp_ms=validation.timestamp_end_ms,
                split=split.by_intersection[entry.intersection_id].value,
                selection_reason=entry.selection_reason,
                row_count=validation.valid_row_count,
                valid_windows=(
                    entry.expected_valid_windows
                    if entry.expected_valid_windows is not None
                    else max(0, (duration_ms - 8_000) // 4_000 + 1)
                ),
                behavior_families=entry.behavior_families,
                dropped_rows=validation.dropped_rows,
                continuity_gaps=validation.continuity_gaps,
                archive_member=entry.archive_member,
            )
        )
    serialized = []
    for record in records:
        item = record.__dict__.copy()
        item["source_role"] = record.source_role.value
        serialized.append(item)
    adequacy = evaluate_adequacy(
        serialized,
        split_assignments={
            key: value.value for key, value in split.by_intersection.items()
        },
        blockers=blockers,
    )
    return DataState(catalog.dataset_id, str(catalog.evidence_role), tuple(records), split, adequacy)


def write_data_state(
    state: DataState,
    *,
    catalog: Catalog,
    manifest_path: Path,
    split_path: Path,
    adequacy_path: Path,
) -> dict[str, Any]:
    status = state.adequacy.status.value
    manifest = manifest_payload(
        state.records,
        dataset_id=catalog.dataset_id,
        evidence_role=str(catalog.evidence_role),
        catalog_sha256=catalog.sha256,
        status=status,
    )
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(split_path, state.split.to_dict())
    write_json_atomic(adequacy_path, state.adequacy.to_dict())
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "split_sha256": state.split.sha256,
        "adequacy": state.adequacy.to_dict(),
    }
