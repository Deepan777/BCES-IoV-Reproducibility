#!/usr/bin/env python3
"""Descriptive roadside/vehicle annotation-ID discordance on existing YAMLs."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/STUDY_B_COOPERSCENE_VIEW_DISCORDANCE_DEV_V1_PLAN.md"
MANIFESTS = (
    ROOT / "data/manifests/cooperscene_annotations_v1.json",
    ROOT / "data/manifests/cooperscene_annotations_extra_agents_v1.json",
)
OUTPUT = ROOT / "outputs/study_b/cooperscene_view_discordance_dev_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xy(values):
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("missing XY coordinates")
    result = (float(values[0]), float(values[1]))
    if not all(map(math.isfinite, result)):
        raise ValueError("nonfinite XY coordinates")
    return result


def load_frames():
    records = []
    manifest_hashes = {}
    for path in MANIFESTS:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["file_count"] != len(manifest["records"]):
            raise RuntimeError("manifest count mismatch")
        manifest_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        records.extend(manifest["records"])
    if len(records) != 3360:
        raise RuntimeError("unexpected annotation tranche size")
    frames = defaultdict(dict)
    for record in records:
        key = (int(record["scene"]), int(record["frame"]))
        agent = int(record["agent"])
        if agent in frames[key]:
            raise RuntimeError("duplicate take/frame/agent")
        path = ROOT / record["relative_path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError("source annotation hash mismatch: " + str(path))
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        pose = xy(data["lidar_pose"])
        objects = {}
        for object_id, obj in data["vehicles"].items():
            identifier = int(object_id)
            if identifier in objects:
                raise RuntimeError("duplicate object ID")
            objects[identifier] = xy(obj["location"])
        frames[key][agent] = {"pose": pose, "objects": objects}
    takes = defaultdict(int)
    for (take, _frame), agents in frames.items():
        if set(agents) != {0, 1, 2, 3}:
            raise RuntimeError("incomplete four-agent frame")
        takes[take] += 1
    if len(takes) != 14 or set(takes.values()) != {60}:
        raise RuntimeError("expected 14 takes x 60 frames")
    return frames, manifest_hashes


def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def bin_name(radius):
    if radius < 20:
        return "0_20m"
    if radius < 40:
        return "20_40m"
    if radius < 80:
        return "40_80m"
    return "80_120m"


def main():
    if OUTPUT.exists():
        raise RuntimeError("output exists; preserve first run")
    frames, manifest_hashes = load_frames()
    overall = Counter()
    take_counts = defaultdict(Counter)
    examples = []
    for (take, frame), agents in sorted(frames.items()):
        roadside = agents[0]
        source_ids = set(roadside["objects"])
        union = {}
        for agent in (1, 2, 3):
            for identifier, position in agents[agent]["objects"].items():
                if identifier in union and distance(union[identifier], position) > 0.01:
                    raise RuntimeError(f"cross-vehicle coordinate mismatch {take}/{frame}/{identifier}")
                union[identifier] = position
        for identifier in source_ids.intersection(union):
            if distance(roadside["objects"][identifier], union[identifier]) > 0.01:
                raise RuntimeError(f"source/vehicle coordinate mismatch {take}/{frame}/{identifier}")
        sample = Counter(frames=1, roadside_ids=len(source_ids), vehicle_union_ids=len(union))
        eligible = {}
        for identifier, position in union.items():
            radius = distance(position, roadside["pose"])
            if radius > 120:
                continue
            if any(distance(position, agents[a]["pose"]) < 3 for a in (1, 2, 3)):
                sample["excluded_possible_self"] += 1
                continue
            eligible[identifier] = radius
            sample["vehicle_ids_eligible_120m"] += 1
            bucket = bin_name(radius)
            if identifier in source_ids:
                sample["matched_120m"] += 1
                sample["matched_" + bucket] += 1
            else:
                sample["discordant_120m"] += 1
                sample["discordant_" + bucket] += 1
                if len(examples) < 20:
                    examples.append({"take": take, "frame": frame,
                                     "id": identifier, "source_distance_m": round(radius, 3)})
        if not source_ids and eligible:
            sample["source_empty_with_vehicle_ids_120m"] += 1
        if sample["vehicle_ids_eligible_120m"] != sample["matched_120m"] + sample["discordant_120m"]:
            raise RuntimeError("ID accounting mismatch")
        overall.update(sample)
        take_counts[take].update(sample)
    OUTPUT.mkdir(parents=True)
    protocol = {"status": "COOPERSCENE_VIEW_DISCORDANCE_DEV_V1",
                "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
                "manifest_sha256": manifest_hashes, "frame_count": len(frames),
                "source_agent": 0, "comparison_agents": [1, 2, 3],
                "range_m": 120, "possible_self_exclusion_m": 3,
                "independent_confirmation": False, "complete_view_certified": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"], "protocol_sha256": sha256(OUTPUT / "protocol.json"),
              "overall": dict(sorted(overall.items())),
              "by_take": {str(take): dict(sorted(counts.items()))
                          for take, counts in sorted(take_counts.items())},
              "first_20_discordance_examples": examples,
              "interpretation": "cross_view_annotation_discordance_not_physical_false_negative",
              "independent_confirmation": False, "complete_view_certified": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"],
                      "report_sha256": sha256(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
