"""Allowlist and immutable raw-data manifest models for Phase 3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml  # type: ignore[import-untyped]

from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import write_json_atomic

LEGACY_DATASET_ID = "v2xseq_tfd_compact"
ALLOWED_OFFICIAL_HOSTS = frozenset(
    {
        "drive.google.com",
        "drive.usercontent.google.com",
        "github.com",
        "raw.githubusercontent.com",
        "thudair.baai.ac.cn",
        "tum-traffic-dataset.github.io",
        "interaction-dataset.com",
    }
)
ALLOWED_DATA_SUFFIXES = frozenset({".csv", ".json"})
FORBIDDEN_MODALITY_TOKENS = frozenset(
    {
        "camera",
        "image",
        "lidar",
        "pointcloud",
        "point-cloud",
        "v2v4real",
        "v2xverse",
        "carla",
        "checkpoint",
    }
)


class DataRole(str, Enum):
    COOPERATIVE = "cooperative"
    VEHICLE = "vehicle"
    INFRASTRUCTURE = "infrastructure"
    TRAFFIC_LIGHT = "traffic_light"
    MAP = "map"
    EGO = "ego"
    TRACKS = "tracks"


@dataclass(frozen=True)
class CatalogEntry:
    entry_id: str
    url: str
    destination: str
    role: DataRole
    scene_id: str
    intersection_id: str
    selection_reason: str
    release: str
    license_url: str
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    expected_valid_windows: int | None = None
    behavior_families: tuple[str, ...] = ()
    archive_member: str | None = None

    def __post_init__(self) -> None:
        if not self.entry_id or not self.scene_id or not self.intersection_id:
            raise ValueError("catalog identifiers must be non-empty")
        if not self.selection_reason or not self.release or not self.license_url:
            raise ValueError("selection reason, release, and license URL must be non-empty")
        parsed = urlparse(self.url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_OFFICIAL_HOSTS:
            raise ValueError("catalog URL is not an approved official HTTPS host")
        if urlparse(self.license_url).scheme != "https":
            raise ValueError("license URL must use HTTPS")
        lowered_url = self.url.lower()
        if any(token in lowered_url for token in FORBIDDEN_MODALITY_TOKENS):
            raise ValueError("catalog URL references a forbidden modality")
        destination = PurePosixPath(self.destination)
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError("catalog destination must be a contained relative path")
        if destination.suffix.lower() not in ALLOWED_DATA_SUFFIXES:
            raise ValueError("only selective CSV/JSON data files may be allowlisted")
        if self.expected_bytes is not None and self.expected_bytes <= 0:
            raise ValueError("expected_bytes must be positive when present")
        if self.expected_valid_windows is not None and self.expected_valid_windows < 0:
            raise ValueError("expected_valid_windows cannot be negative")
        if self.expected_sha256 is not None:
            digest = self.expected_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
            object.__setattr__(self, "expected_sha256", digest)
        object.__setattr__(self, "role", DataRole(self.role))
        object.__setattr__(self, "behavior_families", tuple(self.behavior_families))
        if self.archive_member is not None:
            member = PurePosixPath(self.archive_member)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("archive member must be a contained relative path")


@dataclass(frozen=True)
class Catalog:
    dataset_id: str
    status: str
    official_sources: tuple[dict[str, Any], ...]
    entries: tuple[CatalogEntry, ...]
    evidence_role: str | None = None

    def __post_init__(self) -> None:
        from .datasets import dataset_profile

        profile = dataset_profile(self.dataset_id)
        supplied_role = self.evidence_role or profile.evidence_role.value
        if supplied_role != profile.evidence_role.value:
            raise ValueError("catalog evidence_role does not match the frozen dataset profile")
        object.__setattr__(self, "evidence_role", supplied_role)
        ids = [entry.entry_id for entry in self.entries]
        destinations = [entry.destination for entry in self.entries]
        if len(ids) != len(set(ids)) or len(destinations) != len(set(destinations)):
            raise ValueError("catalog entry IDs and destinations must be unique")

    @property
    def sha256(self) -> str:
        return canonical_json_hash(
            {
                "dataset_id": self.dataset_id,
                "status": self.status,
                "evidence_role": self.evidence_role,
                "official_sources": self.official_sources,
                "entries": [asdict(entry) for entry in self.entries],
            }
        )

    def require_url(self, url: str) -> CatalogEntry:
        matches = [entry for entry in self.entries if entry.url == url]
        if len(matches) != 1:
            raise PermissionError("URL is not uniquely present in the acquisition allowlist")
        return matches[0]

    def validate_complete_bundles(self) -> None:
        """Require matching trajectory, signal, and map roles before any transfer."""
        if not self.entries:
            return
        from .datasets import dataset_profile

        profile = dataset_profile(self.dataset_id)
        required_scene_roles = {DataRole(value) for value in profile.required_scene_roles}
        roles_by_scene: dict[str, set[DataRole]] = {}
        intersections_by_scene: dict[str, set[str]] = {}
        mapped_intersections = set()
        for entry in self.entries:
            if entry.role is DataRole.MAP:
                mapped_intersections.add(entry.intersection_id)
                continue
            roles_by_scene.setdefault(entry.scene_id, set()).add(entry.role)
            intersections_by_scene.setdefault(entry.scene_id, set()).add(
                entry.intersection_id
            )
        for scene_id, roles in roles_by_scene.items():
            if roles != required_scene_roles:
                missing = sorted(role.value for role in required_scene_roles - roles)
                raise ValueError(f"scene {scene_id} is missing matching roles: {missing}")
            intersections = intersections_by_scene[scene_id]
            if len(intersections) != 1:
                raise ValueError(f"scene {scene_id} crosses intersections")
            if profile.requires_map and not intersections <= mapped_intersections:
                raise ValueError(f"scene {scene_id} has no matching map")


@dataclass(frozen=True)
class ManifestRecord:
    release: str
    official_url: str
    license_url: str
    intersection_id: str
    scenario_id: str
    source_file: str
    source_role: DataRole
    source_bytes: int
    sha256: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    split: str
    selection_reason: str
    row_count: int
    valid_windows: int
    behavior_families: tuple[str, ...]
    dropped_rows: tuple[dict[str, Any], ...] = ()
    continuity_gaps: tuple[dict[str, Any], ...] = ()
    archive_member: str | None = None

    def __post_init__(self) -> None:
        CatalogEntry(
            entry_id="manifest-validation",
            url=self.official_url,
            destination=self.source_file,
            role=self.source_role,
            scene_id=self.scenario_id,
            intersection_id=self.intersection_id,
            selection_reason=self.selection_reason,
            release=self.release,
            license_url=self.license_url,
            expected_bytes=self.source_bytes,
            expected_sha256=self.sha256,
            behavior_families=self.behavior_families,
            archive_member=self.archive_member,
        )
        if self.start_timestamp_ms < 0 or self.end_timestamp_ms < self.start_timestamp_ms:
            raise ValueError("invalid manifest timestamp interval")
        if self.split not in {"train", "calibration", "validation", "test"}:
            raise ValueError("manifest split is invalid")
        if self.row_count < 0 or self.valid_windows < 0:
            raise ValueError("manifest counts must be non-negative")
        object.__setattr__(self, "source_role", DataRole(self.source_role))
        object.__setattr__(self, "behavior_families", tuple(self.behavior_families))
        object.__setattr__(self, "dropped_rows", tuple(self.dropped_rows))
        object.__setattr__(self, "continuity_gaps", tuple(self.continuity_gaps))


def load_catalog(path: Path) -> Catalog:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = tuple(
        CatalogEntry(
            entry_id=item["entry_id"],
            url=item["url"],
            destination=item["destination"],
            role=DataRole(item["role"]),
            scene_id=str(item["scene_id"]),
            intersection_id=str(item["intersection_id"]),
            selection_reason=item["selection_reason"],
            release=item["release"],
            license_url=item["license_url"],
            expected_bytes=item.get("expected_bytes"),
            expected_sha256=item.get("expected_sha256"),
            expected_valid_windows=item.get("expected_valid_windows"),
            behavior_families=tuple(item.get("behavior_families", ())),
            archive_member=item.get("archive_member"),
        )
        for item in payload.get("files", ())
    )
    return Catalog(
        dataset_id=payload.get("dataset_id", ""),
        status=payload.get("status", "unspecified"),
        official_sources=tuple(payload.get("official_sources", ())),
        entries=entries,
        evidence_role=payload.get("evidence_role"),
    )


def manifest_payload(
    records: tuple[ManifestRecord, ...], *, dataset_id: str = LEGACY_DATASET_ID,
    evidence_role: str | None = None, catalog_sha256: str, status: str
) -> dict[str, Any]:
    serialized = []
    for record in records:
        item = asdict(record)
        item["source_role"] = record.source_role.value
        serialized.append(item)
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "evidence_role": evidence_role,
        "status": status,
        "catalog_sha256": catalog_sha256,
        "records": serialized,
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    return payload


def write_manifest(
    path: Path,
    records: tuple[ManifestRecord, ...],
    *,
    catalog_sha256: str,
    status: str,
    dataset_id: str = LEGACY_DATASET_ID,
    evidence_role: str | None = None,
) -> dict[str, Any]:
    payload = manifest_payload(
        records,
        dataset_id=dataset_id,
        evidence_role=evidence_role,
        catalog_sha256=catalog_sha256,
        status=status,
    )
    write_json_atomic(path, payload)
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
