from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from bces.data.windows import TrackPoint, WindowReviewConfig, classify_behavior, review_scene

pytestmark = pytest.mark.phase3


def _track(
    *, speed0: float = 5.0, speed1: float = 5.0, heading_rate: float = 0.0
) -> list[TrackPoint]:
    points = []
    x = y = 0.0
    for index in range(80):
        fraction = index / 79
        speed = speed0 + (speed1 - speed0) * fraction
        heading = heading_rate * fraction
        x += speed * 0.1 * math.cos(heading)
        y += speed * 0.1 * math.sin(heading)
        points.append(TrackPoint(index * 100, "target", "TARGET_AGENT", x, y, speed))
    return points


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "keep"),
        ({"speed0": 6.0, "speed1": 3.0}, "brake"),
        ({"speed0": 3.0, "speed1": 6.0}, "accelerate"),
        ({"heading_rate": 1.0}, "left"),
        ({"heading_rate": -1.0}, "right"),
    ],
)
def test_behavior_proxy_classification(kwargs: dict[str, float], expected: str) -> None:
    behavior, metrics = classify_behavior(_track(**kwargs), WindowReviewConfig())
    assert behavior == expected
    assert metrics["trigger"]


def _write_scene(path: Path, *, include_neighbor: bool) -> None:
    fields = ["timestamp", "id", "tag", "x", "y", "v_x", "v_y"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(80):
            timestamp = index / 10
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "id": "target",
                    "tag": "TARGET_AGENT",
                    "x": index / 2,
                    "y": 0,
                    "v_x": 5,
                    "v_y": 0,
                }
            )
            if include_neighbor:
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "id": "neighbor",
                        "tag": "OTHERS",
                        "x": index / 2 + 10,
                        "y": 2,
                        "v_x": 5,
                        "v_y": 0,
                    }
                )


def test_review_scene_requires_relevant_actor(tmp_path: Path) -> None:
    path = tmp_path / "scene.csv"
    _write_scene(path, include_neighbor=False)
    record = review_scene(path, scenario_id="1", intersection_id="i", split="train")
    assert record["valid_windows"] == 0
    assert "no_actor_within_local_relevance_radius" in record["review_reasons"]


def test_review_scene_accepts_inclusive_eight_second_window(tmp_path: Path) -> None:
    path = tmp_path / "scene.csv"
    _write_scene(path, include_neighbor=True)
    record = review_scene(path, scenario_id="1", intersection_id="i", split="train")
    assert record["inclusive_duration_ms"] == 8_000
    assert record["valid_windows"] == 1
    assert record["behavior_families"] == ["keep"]
    assert record["oracle_validity_decisions"] == 0
