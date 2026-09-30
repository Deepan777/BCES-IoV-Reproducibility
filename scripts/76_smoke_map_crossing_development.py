#!/usr/bin/env python3
"""Paired engineering smoke of an off-path crossing-view obstruction."""

from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.development_map_loop import run_branch
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/map_crossing_visibility_engineering_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)

    seed = 8200000
    spec = replace(scenario_spec("unprotected_crossing"),
                   actor_depart_position_m=30.0,
                   actor_depart_speed_mps=9.0,
                   hidden_from_local=False)
    parameters = {
        "benchmark_stratum": "off_path_map_crossing_engineering_only",
        "ego_initial_speed_mps": 14.0,
        "ego_post_reference_acceleration_mps2": 0.0,
        "braking_event_delay_s": 0.2,
    }
    blockers = (StaticMapObstruction("southwest_corner", 93.0, 89.5,
                                      0.0, 6.0, 5.0),)
    protocol = {
        "status": "ENGINEERING_VISIBILITY_SMOKE_ONLY_NOT_CONFIRMATION",
        "registration_sha256": sha256_file(registration_path),
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "seed": seed,
        "spec": asdict(spec),
        "parameters": parameters,
        "blockers": [asdict(item) for item in blockers],
        "network_condition": "nominal",
        "methods": ["periodic_payload", "local_only"],
        "purpose": "show a map-attributable finite local-view gap and reveal in a paired crossing branch",
        "success_rule": "equal prefix, causal input, byte and actuation checks, and a map-blocked in-range frame followed by a locally visible frame",
        "sender_limit": "ideal centered-world cooperative range; RSU occlusion is not modeled",
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    condition = NetworkCondition(**registration["conditions"]["nominal"])
    factory = lambda _seed, _config: (spec, parameters)
    branches = {}
    artifacts = {}
    for method in protocol["methods"]:
        branch = run_branch(seed=seed, method=method, config=registration["config"],
                            condition=condition, network_seed=seed + 910000,
                            horizon_s=registration["horizon_s"],
                            scenario_factory=factory, map_blockers=blockers)
        branches[method] = branch
        name = method + ".json.gz"
        with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
            json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        artifacts[name] = sha256_file(OUTPUT / name)

    periodic, local = branches["periodic_payload"], branches["local_only"]
    prefix = compare_snapshots(periodic["initial_contract"], local["initial_contract"],
                               tolerance=1e-6)["equal_within_tolerance"]
    visibility = periodic["visibility_audit"]
    map_blocked = [row["timestamp_ms"] for row in visibility
                   if row["target_within_local_range"]
                   and row["target_visible_without_map"]
                   and not row["target_locally_visible"]]
    first_reveal_after_gap = next(
        (row["timestamp_ms"] for row in visibility
         if map_blocked and row["timestamp_ms"] > map_blocked[0]
         and row["target_locally_visible"]), None)
    causal = all(row["used_timestamp_ms"] <= row["available_ms"]
                 for branch in branches.values() for row in branch["input_availability_audit"])
    bytes_ok = all(branch["traffic"]["byte_conservation_ok"] and
                   branch["traffic"]["component_conservation_ok"]
                   for branch in branches.values())
    actuation = all(branch["actuation_contract_passed"] for branch in branches.values())
    report = {
        "status": "ENGINEERING_VISIBILITY_SMOKE_ONLY_NOT_CONFIRMATION",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "prefix_equal": prefix,
        "causal_inputs": causal,
        "byte_conservation": bytes_ok,
        "actuation_passed": actuation,
        "map_blocked_in_range_frames": len(map_blocked),
        "first_map_blocked_ms": map_blocked[0] if map_blocked else None,
        "first_reveal_after_gap_ms": first_reveal_after_gap,
        "visibility_engineering_gate": bool(prefix and causal and bytes_ok and actuation
                                            and map_blocked and first_reveal_after_gap is not None),
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
