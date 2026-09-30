#!/usr/bin/env python3
"""Outcome-blind V2X-Real/VIPS role, mobility, and map compatibility audit.

No BCES model, validity label, regret, or baseline result is evaluated here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/raw/vips_v2xreal_assets/meta/spd_infos_temporal_test.pkl"
MAP = ROOT / "data/raw/vips_v2xreal_assets/meta/maps_final/v2x_real_map.json"
ARCHIVES = {
    "test": (ROOT / "data/raw/v2x_real_lidar64/test.zip", "add21fa98208e4e487bd4c5f986cb0a58712efbb5a6cc33b996d1b0080560899"),
    "val": (ROOT / "data/raw/v2x_real_lidar64/val.zip", "e06e6525f31da009566ae553dce4cf34cc2c636f7a1162ed41809ee804fd9b7b"),
}
EXPECTED_META_SHA256 = "a13992f0bf47f73af0ef91fe4ed38ea5f3ccb6507b9232da51e33d2359291cd4"
EXPECTED_MAP_SHA256 = "7bf358ef38f32c68508256074f4d7c3715402a6f7e54ec70bf52a234713d45e5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_pose(value: object) -> bool:
    return isinstance(value, list) and len(value) >= 6 and all(
        isinstance(x, (int, float)) and math.isfinite(float(x)) for x in value[:6]
    )


def xy_distance(a: list, b: list) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(ARCHIVES), default="test")
    args = parser.parse_args()
    archive_path, expected_archive_sha256 = ARCHIVES[args.split]
    output_path = ROOT / f"outputs/study_b/v2xreal_receiver_schema_audit_{args.split}_v1.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    for path, expected in (
        (archive_path, expected_archive_sha256),
        (META, EXPECTED_META_SHA256),
        (MAP, EXPECTED_MAP_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Source changed: {path}")

    # This published pickle was separately scanned for construction opcodes.
    with META.open("rb") as stream:
        infos = pickle.load(stream)["infos"]
    mapped_bases = {info["scene_token"].split("_folder_")[0] for info in infos}
    map_data = json.loads(MAP.read_text(encoding="utf-8"))
    lane_rows = list(map_data["LANE"].values())
    map_audit = {
        "lanes": len(lane_rows),
        "lanes_with_centerline": sum(bool(v["centerline"]) for v in lane_rows),
        "lanes_with_both_boundaries": sum(bool(v["left_boundary"]) and bool(v["right_boundary"]) for v in lane_rows),
        "lanes_with_topology": sum(bool(v["predecessors"]) or bool(v["successors"]) for v in lane_rows),
        "lanes_with_turn_label": sum(v["turn_direction"] != "NONE" for v in lane_rows),
        "junctions": len(map_data["JUNCTION"]),
        "crosswalks": len(map_data["CROSSWALK"]),
    }

    with zipfile.ZipFile(archive_path) as archive:
        index: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
        for name in archive.namelist():
            if not name.endswith(".yaml"):
                continue
            parts = name.split("/")
            if len(parts) != 4:
                raise RuntimeError(f"Unexpected YAML path: {name}")
            base, agent, stem = parts[1], parts[2], parts[3][:-5]
            if not stem.isdigit():
                raise RuntimeError(f"Non-numeric frame index: {name}")
            index[base][agent][int(stem)] = name

        scenes = []
        for base in sorted(index):
            agents = index[base]
            common = sorted(set(agents.get("1", {})) & set(agents.get("-1", {})))
            row: dict[str, object] = {
                "base_scene": base,
                "vips_metadata_overlap": base in mapped_bases,
                "agents": sorted(agents),
                "frames_by_agent": {a: len(frames) for a, frames in sorted(agents.items())},
                "paired_vehicle_roadside_frames": len(common),
            }
            if not common:
                scenes.append(row)
                continue
            # Audit all aligned frame metadata, but do not construct messages or score policies.
            positions = []
            ego_speed_values = []
            view_differences = 0
            both_view_objects = 0
            same_id_distances = []
            malformed = Counter()
            for frame in common:
                vehicle = yaml.safe_load(archive.read(agents["1"][frame]))
                roadside = yaml.safe_load(archive.read(agents["-1"][frame]))
                if not valid_pose(vehicle.get("true_ego_pose")) or not valid_pose(roadside.get("true_ego_pose")):
                    malformed["invalid_agent_pose"] += 1
                    continue
                if vehicle.get("infra") is not False or roadside.get("infra") is not True:
                    malformed["role_flag_mismatch"] += 1
                positions.append((frame, vehicle["true_ego_pose"][:2]))
                speed = vehicle.get("ego_speed")
                if isinstance(speed, (int, float)) and math.isfinite(float(speed)):
                    ego_speed_values.append(float(speed))
                v_objects = vehicle.get("vehicles")
                r_objects = roadside.get("vehicles")
                if not isinstance(v_objects, dict) or not isinstance(r_objects, dict):
                    malformed["invalid_object_dictionary"] += 1
                    continue
                both_view_objects += min(len(v_objects), len(r_objects))
                if set(v_objects) != set(r_objects):
                    view_differences += 1
                for object_id in set(v_objects) & set(r_objects):
                    a = v_objects[object_id].get("location")
                    b = r_objects[object_id].get("location")
                    if isinstance(a, list) and isinstance(b, list) and len(a) >= 2 and len(b) >= 2:
                        same_id_distances.append(xy_distance(a, b))
            steps = [
                (b[0] - a[0], xy_distance(a[1], b[1]))
                for a, b in zip(positions, positions[1:])
            ]
            one_step = [distance for delta, distance in steps if delta == 1]
            speed_estimates = [distance / 0.1 for distance in one_step]
            row.update({
                "valid_paired_poses": len(positions),
                "first_frame": common[0],
                "last_frame": common[-1],
                "frame_gaps": sum(delta != 1 for delta, _ in steps),
                "ego_net_displacement_m": xy_distance(positions[0][1], positions[-1][1]) if positions else None,
                "ego_cumulative_step_motion_m": sum(one_step),
                "ego_step_speed_mps_median": statistics.median(speed_estimates) if speed_estimates else None,
                "ego_step_speed_mps_p90": sorted(speed_estimates)[int(0.9 * (len(speed_estimates) - 1))] if speed_estimates else None,
                "reported_ego_speed_nonzero_frames": sum(abs(v) > 0.01 for v in ego_speed_values),
                "different_object_id_sets_frames": view_differences,
                "min_view_object_count_sum": both_view_objects,
                "same_id_cross_view_position_median": statistics.median(same_id_distances) if same_id_distances else None,
                "same_id_cross_view_position_p95": sorted(same_id_distances)[int(0.95 * (len(same_id_distances) - 1))] if same_id_distances else None,
                "malformed": dict(malformed),
            })
            scenes.append(row)
            print(f"audited {base}: {len(common)} paired frames", flush=True)

    report = {
        "status": "OUTCOME_BLIND_SOURCE_CONTRACT_AUDIT_NO_BCES_RESULTS",
        "source_sha256": {"archive": expected_archive_sha256, "vips_meta": EXPECTED_META_SHA256, "vips_map": EXPECTED_MAP_SHA256},
        "split": args.split,
        "map": map_audit,
        "raw_scene_count": len(scenes),
        "mapped_scene_base_count": len(mapped_bases),
        "raw_mapped_overlap_count": sum(bool(s["vips_metadata_overlap"]) for s in scenes),
        "scenes": scenes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw_scene_count", "raw_mapped_overlap_count", "map")}, indent=2))


if __name__ == "__main__":
    main()
