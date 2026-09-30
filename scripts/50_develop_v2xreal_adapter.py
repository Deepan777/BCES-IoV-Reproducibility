#!/usr/bin/env python3
"""Outcome-blind V2X-Real validation-split adapter coverage diagnostic.

Inspects whether fixed candidate windows have causal ego/object histories and
future annotated frames. Does not load BCES models or calculate validity labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/v2x_real_lidar64/val.zip"
AUDIT = ROOT / "outputs/study_b/v2xreal_receiver_schema_audit_val_v1.json"
OFFSETS = (5, 10, 15, 20)
REFERENCE_START = 20
REFERENCE_STEP = 5
FUTURE_HORIZON_FRAMES = 30
LOCAL_RADIUS_M = 30.0


def history_counts(now: dict, before: list[dict], ego_xy: list, radius_m: float) -> tuple[int, int]:
    current = now.get("vehicles", {})
    previous = set().union(*(frame.get("vehicles", {}) for frame in before))
    considered = [
        key for key, object_row in current.items()
        if isinstance(object_row, dict)
        and isinstance(object_row.get("location"), list)
        and len(object_row["location"]) >= 2
        and math.dist(object_row["location"][:2], ego_xy) <= radius_m
    ]
    return len(considered), sum(key not in previous for key in considered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-radius-m", type=float, default=120.0)
    parser.add_argument("--history-lookback-frames", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--future-label-bidirectional", action="store_true",
                        help="Use future adjacent annotations for label-world velocity only, never receiver inputs")
    parser.add_argument("--future-horizon-frames", type=int, choices=(10, 20, 30), default=30,
                        help="Development-only feasibility sensitivity; default preserves the original 3 s horizon")
    args = parser.parse_args()
    if args.remote_radius_m <= 0 or args.remote_radius_m > 120 or not args.remote_radius_m.is_integer():
        raise ValueError("remote radius must be an integer in (0, 120]")
    remote_radius_m = args.remote_radius_m
    history_suffix = "" if args.history_lookback_frames == 1 else f"_h{args.history_lookback_frames}"
    label_suffix = "_labelbi" if args.future_label_bidirectional else ""
    horizon_suffix = "" if args.future_horizon_frames == FUTURE_HORIZON_FRAMES else f"_f{args.future_horizon_frames}"
    out = ROOT / f"outputs/study_b/v2xreal_adapter_coverage_val_r{int(remote_radius_m)}{history_suffix}{label_suffix}{horizon_suffix}_v1.json"
    if out.exists():
        raise FileExistsError(out)
    schema = json.loads(AUDIT.read_text(encoding="utf-8"))
    if schema["status"] != "OUTCOME_BLIND_SOURCE_CONTRACT_AUDIT_NO_BCES_RESULTS":
        raise RuntimeError("validation-split schema audit missing")
    scenes = {x["base_scene"]: x for x in schema["scenes"]}
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        paths: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
        for name in archive.namelist():
            if name.endswith(".yaml"):
                parts = name.split("/")
                paths[parts[1]][parts[2]][int(parts[3][:-5])] = name
        for base in sorted(paths):
            scene = scenes[base]
            if scene["paired_vehicle_roadside_frames"] == 0:
                continue
            paired = set(paths[base]["1"]) & set(paths[base]["-1"])
            if not paired:
                continue
            frames = {
                role: {index: yaml.safe_load(archive.read(name)) for index, name in paths[base][role].items() if index in paired}
                for role in ("1", "-1")
            }
            end = max(paired)
            vehicle, roadside = frames["1"], frames["-1"]
            def past(role: dict[int, dict], index: int) -> list[dict]:
                return [role[j] for j in range(index - args.history_lookback_frames, index) if j in role]
            def label_neighbors(role: dict[int, dict], index: int) -> list[dict]:
                neighbors = past(role, index)
                if args.future_label_bidirectional:
                    neighbors += [role[j] for j in range(index + 1, index + 4) if j in role]
                return neighbors
            for reference in range(REFERENCE_START, end - args.future_horizon_frames - max(OFFSETS) + 1, REFERENCE_STEP):
                for offset in OFFSETS:
                    decision = reference + offset
                    row = {"scene": base, "reference": reference, "decision": decision,
                           "offset_frames": offset, "mobile": (scene.get("ego_net_displacement_m") or 0) >= 10.0}
                    needed = (reference - 2, reference - 1, reference, decision - 2, decision - 1, decision)
                    if not all(index in paired for index in needed):
                        row["status"] = "ego_history_missing"
                        rows.append(row)
                        continue
                    ego_reference = vehicle[reference]["true_ego_pose"][:2]
                    ego_decision = vehicle[decision]["true_ego_pose"][:2]
                    n_local, missing_local = history_counts(vehicle[reference], past(vehicle, reference), ego_reference, LOCAL_RADIUS_M)
                    n_remote, missing_remote = history_counts(roadside[reference], past(roadside, reference), ego_reference, remote_radius_m)
                    row.update({"reference_local_objects": n_local, "reference_remote_objects": n_remote,
                                "reference_local_missing_velocity": missing_local,
                                "reference_remote_missing_velocity": missing_remote})
                    if n_remote == 0:
                        row["status"] = "no_remote_reference_objects"
                    elif missing_local or missing_remote:
                        row["status"] = "reference_object_history_missing"
                    else:
                        n_local, missing_local = history_counts(vehicle[decision], past(vehicle, decision), ego_decision, LOCAL_RADIUS_M)
                        n_remote, missing_remote = history_counts(roadside[decision], past(roadside, decision), ego_decision, remote_radius_m)
                        row.update({"decision_local_objects": n_local, "decision_remote_objects": n_remote,
                                    "decision_local_missing_velocity": missing_local,
                                    "decision_remote_missing_velocity": missing_remote})
                        future_indices = range(decision + 2, decision + args.future_horizon_frames + 1, 2)
                        if not all(index in paired and index - 1 in paired for index in future_indices):
                            row["status"] = "future_pair_frames_missing"
                        elif missing_local or missing_remote:
                            row["status"] = "decision_object_history_missing"
                        else:
                            future_missing = sum(
                                sum(history_counts(vehicle[index], label_neighbors(vehicle, index), vehicle[index]["true_ego_pose"][:2], remote_radius_m)[1:])
                                + sum(history_counts(roadside[index], label_neighbors(roadside, index), vehicle[index]["true_ego_pose"][:2], remote_radius_m)[1:])
                                for index in future_indices
                            )
                            row["future_missing_velocity_count"] = future_missing
                            row["status"] = "complete_causal_history" if future_missing == 0 else "future_object_history_missing"
                    rows.append(row)
            print(f"checked {base}", flush=True)
    report = {"status": "VALIDATION_INPUT_COVERAGE_ONLY_NO_BCES_OUTCOMES",
              "source_archive_sha256": schema["source_sha256"]["archive"],
              "remote_radius_m": remote_radius_m,
              "history_lookback_frames": args.history_lookback_frames,
              "future_label_bidirectional": args.future_label_bidirectional,
              "future_horizon_frames": args.future_horizon_frames,
              "candidate_count": len(rows),
              "mobile_candidate_count": sum(r["mobile"] for r in rows),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "mobile_status_counts": dict(Counter(r["status"] for r in rows if r["mobile"])),
              "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidate_count", "mobile_candidate_count", "mobile_status_counts")}, indent=2))


if __name__ == "__main__":
    main()
