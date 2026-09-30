#!/usr/bin/env python3
"""Validation-only audit of source static attributes against next-frame motion.

This is a receiver-input feasibility check. It computes no BCES decisions,
validity labels, or model outcomes, and never reads test/train frames.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/v2x_real_lidar64/val.zip"
OUTPUT = ROOT / "outputs/study_b/v2xreal_static_attribute_audit_val_v1.json"


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    measurements: dict[str, list[float]] = defaultdict(list)
    next_attributes: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    per_scene: dict[str, Counter[str]] = defaultdict(Counter)
    with zipfile.ZipFile(ARCHIVE) as archive:
        paths: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
        for name in archive.namelist():
            if name.endswith(".yaml"):
                _, scene, role, filename = name.split("/")
                if role in ("1", "-1"):
                    paths[(scene, role)][int(filename[:-5])] = name
        for (scene, role), frames in sorted(paths.items()):
            previous = None
            previous_index = None
            for index, name in sorted(frames.items()):
                now = yaml.safe_load(archive.read(name))
                current_objects = now.get("vehicles", {})
                for object_row in current_objects.values():
                    attribute = str(object_row.get("attribute", "<missing>"))
                    objects[attribute] += 1
                    per_scene[scene][attribute] += 1
                if previous is not None and index == previous_index + 1:
                    for track_id, old in previous.get("vehicles", {}).items():
                        if track_id not in current_objects:
                            continue
                        new = current_objects[track_id]
                        if not (len(old.get("location", [])) >= 2 and len(new.get("location", [])) >= 2):
                            continue
                        attribute = str(old.get("attribute", "<missing>"))
                        speed = math.dist(old["location"][:2], new["location"][:2]) / 0.1
                        measurements[attribute].append(speed)
                        next_attributes[f"{attribute}->{new.get('attribute', '<missing>')}"] += 1
                previous, previous_index = now, index
            print(f"checked {scene}/{role}", flush=True)
    report = {
        "status": "VALIDATION_ATTRIBUTE_INPUT_AUDIT_NO_BCES_OUTCOMES",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "frame_interval_s": 0.1,
        "object_attribute_counts": dict(objects),
        "per_scene_attribute_counts": {key: dict(value) for key, value in sorted(per_scene.items())},
        "matched_track_motion_by_old_attribute": {
            attribute: {
                "pairs": len(values),
                "median_speed_mps": quantile(values, 0.5),
                "p95_speed_mps": quantile(values, 0.95),
                "p99_speed_mps": quantile(values, 0.99),
                "over_0_5_mps": sum(value > 0.5 for value in values),
                "over_1_mps": sum(value > 1 for value in values),
                "over_2_mps": sum(value > 2 for value in values),
            }
            for attribute, values in sorted(measurements.items())
        },
        "next_frame_attribute_transitions": dict(next_attributes),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "object_attribute_counts": report["object_attribute_counts"],
        "matched_track_motion_by_old_attribute": report["matched_track_motion_by_old_attribute"],
    }, indent=2))


if __name__ == "__main__":
    main()
