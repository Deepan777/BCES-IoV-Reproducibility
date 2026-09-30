"""Auditable V2X-Traj scene-window and behavior-proxy review.

This module reviews data availability only.  It does not create cached-versus-fresh
oracle labels and therefore cannot populate Phase-4 validity-decision counts.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class WindowReviewConfig:
    window_ms: int = 8_000
    stride_ms: int = 4_000
    nominal_step_ms: int = 100
    max_step_ms: int = 150
    local_relevance_radius_m: float = 50.0
    turn_threshold_rad: float = math.radians(30.0)
    lane_change_threshold_m: float = 3.0
    speed_change_threshold_mps: float = 1.0
    endpoint_samples: int = 10
    focal_tag: str = "TARGET_AGENT"

    def __post_init__(self) -> None:
        if self.window_ms <= 0 or self.stride_ms <= 0:
            raise ValueError("window and stride must be positive")
        if self.nominal_step_ms <= 0 or self.max_step_ms < self.nominal_step_ms:
            raise ValueError("invalid sampling-step limits")
        for name in (
            "local_relevance_radius_m",
            "turn_threshold_rad",
            "lane_change_threshold_m",
            "speed_change_threshold_mps",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.endpoint_samples < 2:
            raise ValueError("endpoint_samples must be at least two")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["behavior_precedence"] = ["left/right", "brake/accelerate", "keep"]
        payload["interpretation"] = {
            "focal_actor": "official TARGET_AGENT trajectory used as an intended-behavior proxy",
            "relevance": "at least one other observed actor within the official model's 50 m local radius at a shared timestamp",
            "duration": "inclusive sampled duration: last timestamp minus first plus nominal sample step",
        }
        return payload


@dataclass(frozen=True)
class TrackPoint:
    timestamp_ms: int
    actor_id: str
    tag: str
    x_m: float
    y_m: float
    speed_mps: float | None


def _timestamp_ms(value: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("timestamp must be finite and non-negative")
    return round(numeric if numeric >= 1e11 else numeric * 1000.0)


def _optional_speed(row: dict[str, str]) -> float | None:
    try:
        vx = float(row["v_x"])
        vy = float(row["v_y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(vx) or not math.isfinite(vy):
        return None
    return math.hypot(vx, vy)


def read_track_points(path: Path) -> tuple[TrackPoint, ...]:
    points = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "id", "tag", "x", "y", "v_x", "v_y"}
        if required - set(reader.fieldnames or ()):
            raise ValueError("trajectory CSV lacks window-review columns")
        for row in reader:
            try:
                timestamp_ms = _timestamp_ms(row["timestamp"])
                x_m = float(row["x"])
                y_m = float(row["y"])
                actor_id = str(row["id"]).strip()
                if not actor_id or not math.isfinite(x_m) or not math.isfinite(y_m):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            points.append(
                TrackPoint(
                    timestamp_ms=timestamp_ms,
                    actor_id=actor_id,
                    tag=str(row.get("tag", "")).strip(),
                    x_m=x_m,
                    y_m=y_m,
                    speed_mps=_optional_speed(row),
                )
            )
    return tuple(points)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _mean_xy(points: list[TrackPoint]) -> tuple[float, float]:
    return (
        statistics.fmean(point.x_m for point in points),
        statistics.fmean(point.y_m for point in points),
    )


def _segment_heading(
    points: list[TrackPoint], sample_count: int, *, end: bool
) -> float | None:
    width = min(sample_count, max(2, len(points) // 2))
    segment = points[-width:] if end else points[:width]
    half = max(1, len(segment) // 2)
    x0, y0 = _mean_xy(segment[:half])
    x1, y1 = _mean_xy(segment[-half:])
    if math.hypot(x1 - x0, y1 - y0) < 0.25:
        return None
    return math.atan2(y1 - y0, x1 - x0)


def classify_behavior(
    focal: Iterable[TrackPoint], config: WindowReviewConfig
) -> tuple[str, dict[str, float | str]]:
    points = sorted(focal, key=lambda point: point.timestamp_ms)
    if len(points) < config.endpoint_samples * 2:
        raise ValueError("focal actor has too few samples for behavior review")
    start_heading = _segment_heading(points, config.endpoint_samples, end=False)
    end_heading = _segment_heading(points, config.endpoint_samples, end=True)
    heading_available = start_heading is not None and end_heading is not None
    heading_change = (
        _wrap_angle(end_heading - start_heading) if heading_available else 0.0
    )

    origin_x, origin_y = _mean_xy(points[: config.endpoint_samples])
    end_x, end_y = _mean_xy(points[-config.endpoint_samples :])
    dx, dy = end_x - origin_x, end_y - origin_y
    lateral_displacement = (
        -math.sin(start_heading) * dx + math.cos(start_heading) * dy
        if start_heading is not None
        else 0.0
    )

    start_speeds = [
        point.speed_mps
        for point in points[: config.endpoint_samples]
        if point.speed_mps is not None
    ]
    end_speeds = [
        point.speed_mps
        for point in points[-config.endpoint_samples :]
        if point.speed_mps is not None
    ]
    if not start_speeds or not end_speeds:
        raise ValueError("focal actor lacks finite velocity for behavior review")
    start_speed = statistics.fmean(start_speeds)
    end_speed = statistics.fmean(end_speeds)
    speed_change = end_speed - start_speed

    if heading_change >= config.turn_threshold_rad:
        behavior, trigger = "left", "heading_change"
    elif heading_change <= -config.turn_threshold_rad:
        behavior, trigger = "right", "heading_change"
    elif lateral_displacement >= config.lane_change_threshold_m:
        behavior, trigger = "left", "lateral_displacement"
    elif lateral_displacement <= -config.lane_change_threshold_m:
        behavior, trigger = "right", "lateral_displacement"
    elif speed_change <= -config.speed_change_threshold_mps:
        behavior, trigger = "brake", "speed_change"
    elif speed_change >= config.speed_change_threshold_mps:
        behavior, trigger = "accelerate", "speed_change"
    else:
        behavior, trigger = "keep", "within_thresholds"
    return behavior, {
        "trigger": trigger,
        "heading_available": 1.0 if heading_available else 0.0,
        "heading_change_rad": heading_change,
        "lateral_displacement_m": lateral_displacement,
        "start_speed_mps": start_speed,
        "end_speed_mps": end_speed,
        "speed_change_mps": speed_change,
    }


def review_scene(
    path: Path,
    *,
    scenario_id: str,
    intersection_id: str,
    split: str,
    config: WindowReviewConfig | None = None,
) -> dict[str, Any]:
    config = config or WindowReviewConfig()
    points = read_track_points(path)
    focal_ids = sorted(
        {point.actor_id for point in points if point.tag == config.focal_tag}
    )
    reasons = []
    if len(focal_ids) != 1:
        reasons.append("requires_exactly_one_target_agent")
        return {
            "scenario_id": scenario_id,
            "intersection_id": intersection_id,
            "split": split,
            "valid_windows": 0,
            "behavior_families": [],
            "review_reasons": reasons,
        }

    focal_id = focal_ids[0]
    focal = sorted(
        (point for point in points if point.actor_id == focal_id),
        key=lambda point: point.timestamp_ms,
    )
    timestamps = [point.timestamp_ms for point in focal]
    unique_timestamps = sorted(set(timestamps))
    steps = [b - a for a, b in zip(unique_timestamps, unique_timestamps[1:])]
    sample_step_ms = round(statistics.median(steps)) if steps else 0
    inclusive_duration_ms = (
        unique_timestamps[-1] - unique_timestamps[0] + sample_step_ms
        if unique_timestamps
        else 0
    )
    if len(timestamps) != len(unique_timestamps):
        reasons.append("duplicate_target_timestamp")
    if sample_step_ms != config.nominal_step_ms:
        reasons.append("unexpected_sample_step")
    if any(step > config.max_step_ms for step in steps):
        reasons.append("target_continuity_gap")
    if inclusive_duration_ms < config.window_ms:
        reasons.append("shorter_than_eight_seconds")

    focal_by_time = {point.timestamp_ms: point for point in focal}
    min_distance = math.inf
    relevant_actor_ids = set()
    for point in points:
        if point.actor_id == focal_id or point.timestamp_ms not in focal_by_time:
            continue
        reference = focal_by_time[point.timestamp_ms]
        distance = math.hypot(point.x_m - reference.x_m, point.y_m - reference.y_m)
        min_distance = min(min_distance, distance)
        if distance <= config.local_relevance_radius_m:
            relevant_actor_ids.add(point.actor_id)
    if not relevant_actor_ids:
        reasons.append("no_actor_within_local_relevance_radius")

    behavior = None
    behavior_metrics: dict[str, float | str] = {}
    try:
        behavior, behavior_metrics = classify_behavior(focal, config)
    except ValueError as exc:
        reasons.append(str(exc).replace(" ", "_"))

    valid = not reasons and behavior is not None
    possible_windows = (
        max(0, (inclusive_duration_ms - config.window_ms) // config.stride_ms + 1)
        if valid
        else 0
    )
    return {
        "scenario_id": str(scenario_id),
        "intersection_id": str(intersection_id),
        "split": split,
        "source_file": Path(path).as_posix(),
        "focal_actor_id": focal_id,
        "focal_actor_tag": config.focal_tag,
        "focal_samples": len(focal),
        "sample_step_ms": sample_step_ms,
        "inclusive_duration_ms": inclusive_duration_ms,
        "relevant_actor_count": len(relevant_actor_ids),
        "minimum_actor_distance_m": None if math.isinf(min_distance) else min_distance,
        "valid_windows": possible_windows,
        "behavior_families": [behavior] if valid else [],
        "behavior_metrics": behavior_metrics,
        "review_reasons": reasons,
        "oracle_validity_decisions": 0,
    }
