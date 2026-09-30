"""Streaming schema validation for permitted V2X-Seq TFD CSV/JSON files."""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import CatalogEntry, DataRole

TRAJECTORY_COLUMNS = frozenset(
    {
        "city",
        "timestamp",
        "id",
        "type",
        "sub_type",
        "tag",
        "x",
        "y",
        "z",
        "length",
        "width",
        "height",
        "theta",
        "v_x",
        "v_y",
        "intersect_id",
    }
)
COOPERATIVE_EXTRA_COLUMNS = frozenset(
    {"vic_tag", "from_side", "car_side_id", "road_side_id"}
)
TRAFFIC_LIGHT_COLUMNS = frozenset(
    {
        "city",
        "timestamp",
        "x",
        "y",
        "direction",
        "lane_id",
        "color_1",
        "remain_1",
        "color_2",
        "remain_2",
        "color_3",
        "remain_3",
        "intersect_id",
    }
)
TRAJECTORY_NUMERIC_COLUMNS = (
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "theta",
    "v_x",
    "v_y",
)
TRAFFIC_NUMERIC_COLUMNS = (
    "x",
    "y",
    "remain_1",
    "remain_2",
    "remain_3",
)


@dataclass(frozen=True)
class ValidationResult:
    row_count: int
    valid_row_count: int
    timestamp_start_ms: int
    timestamp_end_ms: int
    dropped_rows: tuple[dict[str, Any], ...]
    continuity_gaps: tuple[dict[str, Any], ...]


def _timestamp_ms(value: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError("timestamp must be finite and non-negative")
    return round(numeric if numeric >= 1e11 else numeric * 1000.0)


def _finite_columns(row: dict[str, str], names: tuple[str, ...]) -> None:
    for name in names:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")


def validate_tfd_csv(path: Path, entry: CatalogEntry) -> ValidationResult:
    if entry.role is DataRole.MAP:
        raise ValueError("map entries require JSON validation")
    required = (
        TRAFFIC_LIGHT_COLUMNS
        if entry.role is DataRole.TRAFFIC_LIGHT
        else TRAJECTORY_COLUMNS
        | (COOPERATIVE_EXTRA_COLUMNS if entry.role is DataRole.COOPERATIVE else set())
    )
    timestamps = []
    dropped = []
    seen = set()
    by_actor: dict[str, list[int]] = {}
    row_count = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = required - columns
        if missing:
            raise ValueError(f"missing TFD columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            try:
                timestamp_ms = _timestamp_ms(row["timestamp"])
                _finite_columns(
                    row,
                    TRAFFIC_NUMERIC_COLUMNS
                    if entry.role is DataRole.TRAFFIC_LIGHT
                    else TRAJECTORY_NUMERIC_COLUMNS,
                )
                intersection_id = str(row["intersect_id"]).strip()
                if intersection_id != entry.intersection_id:
                    raise ValueError("intersection ID does not match allowlist")
                actor_id = (
                    str(row["lane_id"]).strip()
                    if entry.role is DataRole.TRAFFIC_LIGHT
                    else str(row["id"]).strip()
                )
                if not actor_id:
                    raise ValueError("actor/lane ID is empty")
                duplicate_key = (timestamp_ms, actor_id)
                if duplicate_key in seen:
                    raise ValueError("duplicate timestamp and actor/lane ID")
                seen.add(duplicate_key)
            except (KeyError, TypeError, ValueError) as exc:
                dropped.append({"line": line_number, "reason": str(exc)})
                continue
            timestamps.append(timestamp_ms)
            by_actor.setdefault(actor_id, []).append(timestamp_ms)
    if not timestamps:
        raise ValueError("TFD file contains no valid rows")
    continuity_gaps = []
    for actor_id, actor_timestamps in sorted(by_actor.items()):
        ordered = sorted(actor_timestamps)
        positive_steps = [
            current - previous
            for previous, current in itertools.pairwise(ordered)
            if current > previous
        ]
        if not positive_steps:
            continue
        nominal_step = min(positive_steps)
        for previous, current in itertools.pairwise(ordered):
            if current - previous > nominal_step * 2:
                continuity_gaps.append(
                    {
                        "actor_id": actor_id,
                        "previous_timestamp_ms": previous,
                        "current_timestamp_ms": current,
                    }
                )
    return ValidationResult(
        row_count=row_count,
        valid_row_count=len(timestamps),
        timestamp_start_ms=min(timestamps),
        timestamp_end_ms=max(timestamps),
        dropped_rows=tuple(dropped),
        continuity_gaps=tuple(continuity_gaps),
    )


def validate_tfd_map(path: Path) -> ValidationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("LANE", "STOPLINE", "CROSSWALK"):
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"map JSON is missing object {key}")
    for lane_id, lane in payload["LANE"].items():
        if not lane_id or not isinstance(lane, dict):
            raise ValueError("map lane entries must be keyed objects")
        if "centerline" not in lane or not isinstance(lane["centerline"], list):
            raise ValueError("map lane is missing a centerline")
    return ValidationResult(
        row_count=len(payload["LANE"]),
        valid_row_count=len(payload["LANE"]),
        timestamp_start_ms=0,
        timestamp_end_ms=0,
        dropped_rows=(),
        continuity_gaps=(),
    )


def validate_downloaded_file(path: Path, entry: CatalogEntry) -> ValidationResult:
    suffix = Path(entry.destination).suffix.lower()
    if entry.role is DataRole.MAP and suffix == ".json":
        return validate_tfd_map(path)
    if suffix == ".csv":
        return validate_tfd_csv(path, entry)
    raise ValueError("downloaded file suffix does not match its TFD role")
