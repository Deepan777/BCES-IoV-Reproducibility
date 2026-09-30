"""Allowlist-only, transactional one-tranche data acquisition."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from bces.utils.budget import Budget, BudgetExceeded, require_headroom
from bces.utils.hashing import sha256_file

from .adequacy import AdequacyReport, evaluate_adequacy
from .manifest import Catalog, CatalogEntry
from .splits import assign_intersections
from .adapters import validate_dataset_file
from .tfd import ValidationResult

CHUNK_BYTES = 4 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


class DownloadDecision(str, Enum):
    STOP_ADEQUATE = "STOP_ADEQUATE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED_NO_USEFUL_FILE = "BLOCKED_NO_USEFUL_FILE"


@dataclass(frozen=True)
class PreflightInfo:
    content_length: int
    final_url: str
    content_type: str | None = None

    def __post_init__(self) -> None:
        if self.content_length <= 0:
            raise DownloadError("official source did not provide a positive content length")


@dataclass(frozen=True)
class DownloadedFile:
    entry: CatalogEntry
    path: Path
    measured_bytes: int
    sha256: str
    validation: ValidationResult


@dataclass(frozen=True)
class FetchResult:
    decision: DownloadDecision
    adequacy_before: AdequacyReport
    adequacy_after: AdequacyReport
    downloaded: tuple[DownloadedFile, ...] = ()
    records: tuple[dict[str, Any], ...] = ()


class DownloadTransport(Protocol):
    def preflight(self, url: str) -> PreflightInfo: ...

    def stream(self, url: str, chunk_bytes: int) -> Iterator[bytes]: ...


class RequestsTransport:
    def __init__(self, *, timeout_s: float = 30.0) -> None:
        self.timeout_s = timeout_s

    def preflight(self, url: str) -> PreflightInfo:
        response = requests.head(url, allow_redirects=True, timeout=self.timeout_s)
        response.raise_for_status()
        length_text = response.headers.get("content-length")
        if length_text is None:
            raise DownloadError("official source HEAD response omitted Content-Length")
        return PreflightInfo(
            content_length=int(length_text),
            final_url=response.url,
            content_type=response.headers.get("content-type"),
        )

    def stream(self, url: str, chunk_bytes: int) -> Iterator[bytes]:
        with requests.get(
            url, stream=True, allow_redirects=True, timeout=self.timeout_s
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    yield chunk


def _contained_destination(raw_root: Path, relative: str) -> Path:
    root = Path(raw_root).resolve()
    destination = (root / Path(relative)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise DownloadError("destination escapes data/raw") from exc
    return destination


def _validate_final_url(entry: CatalogEntry, final_url: str) -> None:
    original_host = urlparse(entry.url).hostname
    final_host = urlparse(final_url).hostname
    allowed_redirects = {
        original_host,
        "drive.usercontent.google.com" if original_host == "drive.google.com" else None,
    }
    if final_host not in allowed_redirects:
        raise DownloadError("official download redirected to an unapproved host")


def fetch_transactionally(
    entries: Iterable[CatalogEntry],
    *,
    budget: Budget,
    transport: DownloadTransport,
    dataset_id: str = "v2xseq_tfd_compact",
    raw_root: Path | None = None,
) -> tuple[DownloadedFile, ...]:
    selected = tuple(entries)
    if not selected:
        return ()
    preflights = []
    for entry in selected:
        metadata = transport.preflight(entry.url)
        _validate_final_url(entry, metadata.final_url)
        if entry.expected_bytes is not None and metadata.content_length != entry.expected_bytes:
            raise DownloadError("preflight size does not match the allowlist")
        preflights.append((entry, metadata))
    incoming = sum(metadata.content_length for _, metadata in preflights)
    if incoming > budget.max_download_tranche_bytes:
        raise BudgetExceeded("selected files exceed the one-tranche byte cap")
    require_headroom(budget, incoming, preserve_reserve=True)
    downloaded = []
    for entry, metadata in preflights:
        destination = _contained_destination(
            raw_root or budget.data_root / "raw", entry.destination
        )
        if destination.exists():
            raise DownloadError("immutable raw destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        part_name = hashlib.sha256(entry.url.encode()).hexdigest() + ".part"
        part_path = budget.data_root / "tmp" / part_name
        part_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with part_path.open("xb") as handle:
                for chunk in transport.stream(entry.url, CHUNK_BYTES):
                    if written + len(chunk) > metadata.content_length:
                        raise DownloadError("stream exceeds preflight Content-Length")
                    require_headroom(
                        budget,
                        len(chunk),
                        preserve_reserve=True,
                        enforce_tranche=False,
                    )
                    handle.write(chunk)
                    written += len(chunk)
            if written != metadata.content_length:
                raise DownloadError("stream length does not match preflight")
            digest = sha256_file(part_path)
            if entry.expected_sha256 is not None and digest != entry.expected_sha256:
                raise DownloadError("downloaded SHA-256 does not match allowlist")
            validation = validate_dataset_file(part_path, entry, dataset_id=dataset_id)
            os.replace(part_path, destination)
            downloaded.append(
                DownloadedFile(
                    entry=entry,
                    path=destination,
                    measured_bytes=written,
                    sha256=digest,
                    validation=validation,
                )
            )
        except Exception:
            part_path.unlink(missing_ok=True)
            raise
    return tuple(downloaded)


def rank_lexicographically(
    entries: Iterable[CatalogEntry],
    *,
    unmet: tuple[str, ...],
    existing_intersections: set[str],
    existing_behaviors: set[str],
    existing_scenarios: set[str],
) -> tuple[CatalogEntry, ...]:
    def key(entry: CatalogEntry) -> tuple[float | str, ...]:
        intersection_gain = int(entry.intersection_id not in existing_intersections)
        behavior_gain = len(set(entry.behavior_families) - existing_behaviors)
        scenario_gain = int(entry.scene_id not in existing_scenarios)
        efficiency = (
            entry.expected_valid_windows / entry.expected_bytes
            if entry.expected_bytes and entry.expected_valid_windows is not None
            else 0.0
        )
        return (
            -intersection_gain if "intersections" in unmet else 0,
            -behavior_gain if "behavior_families" in unmet else 0,
            -scenario_gain if "scenarios" in unmet else 0,
            -efficiency,
            entry.entry_id,
        )

    return tuple(sorted(entries, key=key))


def take_until_bytes(
    ranked: Iterable[CatalogEntry], maximum_bytes: int
) -> tuple[CatalogEntry, ...]:
    selected = []
    total = 0
    for entry in ranked:
        if entry.expected_bytes is None:
            continue
        if total + entry.expected_bytes > maximum_bytes:
            continue
        selected.append(entry)
        total += entry.expected_bytes
    return tuple(selected)


def fetch_next_data(
    catalog: Catalog,
    budget: Budget,
    *,
    records: list[dict[str, Any]] | None = None,
    split_assignments: dict[str, str] | None = None,
    blockers: tuple[str, ...] = (),
    transport: DownloadTransport | None = None,
    split_salt: str = "bces-iov-v1-preregistered",
    raw_root: Path | None = None,
) -> FetchResult:
    records = records or []
    split_assignments = split_assignments or {}
    catalog.validate_complete_bundles()
    before = evaluate_adequacy(
        records,
        split_assignments=split_assignments,
        blockers=blockers,
    )
    if before.passes:
        return FetchResult(
            DownloadDecision.STOP_ADEQUATE, before, before, records=tuple(records)
        )
    if not catalog.entries:
        return FetchResult(
            DownloadDecision.BLOCKED_NO_USEFUL_FILE,
            before,
            before,
            records=tuple(records),
        )
    existing_intersections = {str(item["intersection_id"]) for item in records}
    existing_behaviors = {
        str(behavior)
        for item in records
        for behavior in item.get("behavior_families", ())
    }
    existing_scenarios = {str(item["scenario_id"]) for item in records}
    acquired_paths = {str(item["source_file"]) for item in records}
    candidates = tuple(
        entry for entry in catalog.entries if entry.destination not in acquired_paths
    )
    ranked = rank_lexicographically(
        candidates,
        unmet=before.unmet,
        existing_intersections=existing_intersections,
        existing_behaviors=existing_behaviors,
        existing_scenarios=existing_scenarios,
    )
    tranche = take_until_bytes(ranked, budget.max_download_tranche_bytes)
    if not tranche:
        return FetchResult(
            DownloadDecision.BLOCKED_NO_USEFUL_FILE,
            before,
            before,
            records=tuple(records),
        )
    downloaded = fetch_transactionally(
        tranche,
        budget=budget,
        transport=transport or RequestsTransport(),
        dataset_id=catalog.dataset_id,
        raw_root=raw_root,
    )
    updated_records = [*records]
    for item in downloaded:
        duration_ms = (
            item.validation.timestamp_end_ms - item.validation.timestamp_start_ms
        )
        updated_records.append(
            {
                "official_url": item.entry.url,
                "release": item.entry.release,
                "license_url": item.entry.license_url,
                "source_role": item.entry.role.value,
                "source_file": item.entry.destination,
                "source_bytes": item.measured_bytes,
                "sha256": item.sha256,
                "scenario_id": item.entry.scene_id,
                "intersection_id": item.entry.intersection_id,
                "selection_reason": item.entry.selection_reason,
                "start_timestamp_ms": item.validation.timestamp_start_ms,
                "end_timestamp_ms": item.validation.timestamp_end_ms,
                "row_count": item.validation.valid_row_count,
                "valid_windows": max(0, (duration_ms - 8_000) // 4_000 + 1),
                "behavior_families": list(item.entry.behavior_families),
                "dropped_rows": list(item.validation.dropped_rows),
                "continuity_gaps": list(item.validation.continuity_gaps),
            }
        )
    updated_split = assign_intersections(
        {str(item["intersection_id"]) for item in updated_records}, salt=split_salt
    )
    after = evaluate_adequacy(
        updated_records,
        split_assignments={
            key: partition.value for key, partition in updated_split.by_intersection.items()
        },
        blockers=blockers,
    )
    decision = (
        DownloadDecision.STOP_ADEQUATE
        if after.passes
        else DownloadDecision.REVIEW_REQUIRED
    )
    return FetchResult(
        decision, before, after, downloaded, records=tuple(updated_records)
    )
