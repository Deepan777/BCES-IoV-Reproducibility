#!/usr/bin/env python3
"""Schema-only audit of V2XPnP sample trajectories and map, no BCES scores.

Untrusted pickle globals are opcode-scanned and restricted during loading.
Map classes are replaced with inert local containers; publisher code is not run.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import pickletools
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/v2xpnp_sample_v1"


class Passive:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_pickle_state"] = state


class MapStub(Passive):
    pass


class LaneStub(Passive):
    pass


class MapPointStub(Passive):
    pass


ALLOWED = {
    "collections OrderedDict": OrderedDict,
    "numpy.core.multiarray _reconstruct": np.core.multiarray._reconstruct,
    "numpy ndarray": np.ndarray,
    "numpy dtype": np.dtype,
    "opencood.data_utils.datasets.map.map_types Map": MapStub,
    "opencood.data_utils.datasets.map.map_types Lane": LaneStub,
    "opencood.data_utils.datasets.map.map_types MapPoint": MapPointStub,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = f"{module} {name}"
        if key not in ALLOWED:
            raise pickle.UnpicklingError(f"disallowed global: {key}")
        return ALLOWED[key]


def load_restricted(blob: bytes, expected_globals: set[str]):
    operations = list(pickletools.genops(blob))
    globals_seen = {str(arg) for op, arg, _ in operations if op.name == "GLOBAL"}
    if any(op.name in {"STACK_GLOBAL", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}
           for op, _, _ in operations):
        raise RuntimeError("unsupported pickle opcode")
    if globals_seen != expected_globals:
        raise RuntimeError(f"pickle globals changed: {sorted(globals_seen)}")
    return RestrictedUnpickler(io.BytesIO(blob)).load(), Counter(op.name for op, _, _ in operations)


def shape(value):
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {"type": "dict", "len": len(value), "keys_sample": list(map(str, list(value)[:8]))}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "len": len(value),
                "first_type": type(value[0]).__name__ if value else None}
    if isinstance(value, Passive):
        return {"type": type(value).__name__, "fields": sorted(vars(value))}
    return {"type": type(value).__name__, "value": value if isinstance(value, (int, float, str, bool)) else None}


def describe_record(record):
    if isinstance(record, dict):
        return {str(key): shape(value) for key, value in record.items()}
    return shape(record)


def main() -> None:
    trajectory_blob = (RAW / "trajectory_database_sample.pkl").read_bytes()
    trajectory, trajectory_opcodes = load_restricted(trajectory_blob, {
        "collections OrderedDict", "numpy.core.multiarray _reconstruct",
        "numpy ndarray", "numpy dtype",
    })
    if not isinstance(trajectory, dict):
        raise RuntimeError("unexpected trajectory root type")
    scenario_sample = {}
    for scenario, actors in list(trajectory.items())[:7]:
        if not isinstance(actors, dict):
            raise RuntimeError("unexpected scenario actor structure")
        sample_actor = next(iter(actors.items()), None)
        scenario_sample[str(scenario)] = {
            "actor_count": len(actors),
            "actor_id_sample": list(map(str, list(actors)[:8])),
            "first_actor": describe_record(sample_actor[1]) if sample_actor else None,
        }
    maps = {}
    map_alignment = {}
    with zipfile.ZipFile(RAW / "map.zip") as archive:
        for name in archive.namelist():
            if not name.endswith(".pkl"):
                continue
            map_obj, map_opcodes = load_restricted(archive.read(name), {
                "opencood.data_utils.datasets.map.map_types Map",
                "opencood.data_utils.datasets.map.map_types Lane",
                "opencood.data_utils.datasets.map.map_types MapPoint",
            })
            map_fields = vars(map_obj) if isinstance(map_obj, Passive) else map_obj
            result = {str(key): shape(value) for key, value in map_fields.items()}
            for key, value in map_fields.items():
                if isinstance(value, (dict, list, tuple)) and value:
                    first = next(iter(value.values())) if isinstance(value, dict) else value[0]
                    if isinstance(first, Passive):
                        result[str(key)]["first_item_fields"] = {
                            str(field): shape(item) for field, item in vars(first).items()
                        }
            maps[name] = {"fields": result, "pickle_opcode_counts": dict(map_opcodes)}
            if name.endswith("v2x_intersection_vector_map.pkl"):
                centerlines = np.array([[point.x, point.y] for feature in map_obj.map_features
                                        if feature.type == 1 for point in feature.polyline], dtype=float)
                if len(centerlines) == 0:
                    raise RuntimeError("no driving-lane centerlines")
                tree = cKDTree(centerlines)
                scene_points = defaultdict(list)
                with zipfile.ZipFile(ROOT / "data/raw/v2x_real_lidar64/val.zip") as val_archive:
                    for yaml_name in val_archive.namelist():
                        parts = PurePosixPath(yaml_name).parts
                        if len(parts) == 4 and parts[2] == "1" and yaml_name.endswith(".yaml") \
                                and int(PurePosixPath(yaml_name).stem) % 10 == 0:
                            frame = yaml.safe_load(val_archive.read(yaml_name))
                            pose = frame.get("true_ego_pose")
                            if isinstance(pose, list) and len(pose) >= 2:
                                scene_points[parts[1]].append([float(pose[0]), float(pose[1])])
                for scene, positions in sorted(scene_points.items()):
                    distances, _ = tree.query(np.array(positions, dtype=float))
                    map_alignment[scene] = {"sampled_ego_poses": len(positions),
                                            "nearest_driving_centerline_median_m": float(np.median(distances)),
                                            "nearest_driving_centerline_p95_m": float(np.percentile(distances, 95))}
    print(json.dumps({
        "status": "SAMPLE_SCHEMA_ONLY_NO_BCES_OUTCOME",
        "sample_scenarios": len(trajectory),
        "trajectory_scenario_sample": scenario_sample,
        "trajectory_pickle_opcode_counts": dict(trajectory_opcodes),
        "maps": maps,
        "val_map_alignment_no_bces_outcome": map_alignment,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
