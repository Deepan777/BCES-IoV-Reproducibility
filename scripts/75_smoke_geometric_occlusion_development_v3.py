#!/usr/bin/env python3
"""One paired engineering smoke of a physical occluder; never confirmation."""

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
from bces.simulation.development_critical_loop import OccludingVehicle, run_branch
from bces.simulation.scenario_builder import scenario_spec
from bces.simulation.controlled_contract import compare_snapshots
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/geometric_occlusion_engineering_smoke_v3"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)

    seed = 8100002
    spec = replace(scenario_spec("straight_lead_braking"),
                   actor_depart_position_m=80.0,
                   actor_depart_speed_mps=4.0,
                   hidden_from_local=False)
    parameters = {
        "benchmark_stratum": "geometric_occlusion_engineering_smoke_only",
        "ego_initial_speed_mps": 14.0,
        "ego_post_reference_acceleration_mps2": 0.0,
        "braking_event_delay_s": 0.2,
    }
    occluder = OccludingVehicle(position_m=40.0, lane=0, speed_mps=12.0,
                                 length_m=7.0, width_m=2.5)
    protocol = {
        "status": "ENGINEERING_SMOKE_ONLY_NOT_CONFIRMATION",
        "registration_sha256": sha256_file(registration_path),
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_critical_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "seed": seed,
        "spec": asdict(spec),
        "parameters": parameters,
        "occluder": asdict(occluder),
        "network_condition": "nominal",
        "methods": ["periodic_payload", "local_only"],
        "purpose": "verify a paired physical blocker and a measured local-visibility gap; no policy selection",
        "success_rule": "paired complete prefix, causal timestamps, byte conservation, legal tracked actuation, and at least one target-in-range occluded frame",
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
                            scenario_factory=factory, occluder=occluder)
        branches[method] = branch
        name = method + ".json.gz"
        with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
            json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        artifacts[name] = sha256_file(OUTPUT / name)

    periodic, local = branches["periodic_payload"], branches["local_only"]
    prefix = compare_snapshots(periodic["initial_contract"], local["initial_contract"],
                               tolerance=1e-6)["equal_within_tolerance"]
    visible = periodic["visibility_audit"]
    in_range_occluded = sum(row["target_within_local_range"] and
                            not row["target_locally_visible"] for row in visible)
    in_range_revealed = sum(row["target_within_local_range"] and
                            row["target_locally_visible"] for row in visible)
    causal = all(row["used_timestamp_ms"] <= row["available_ms"]
                 for branch in branches.values() for row in branch["input_availability_audit"])
    byte_conservation = all(branch["traffic"]["byte_conservation_ok"] and
                            branch["traffic"]["component_conservation_ok"]
                            for branch in branches.values())
    actuation = all(branch["actuation_contract_passed"] for branch in branches.values())
    report = {
        "status": "ENGINEERING_SMOKE_ONLY_NOT_CONFIRMATION",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "prefix_equal": prefix,
        "causal_inputs": causal,
        "byte_conservation": byte_conservation,
        "actuation_passed": actuation,
        "in_range_occluded_frames": in_range_occluded,
        "in_range_revealed_frames": in_range_revealed,
        "occlusion_engineering_gate": bool(prefix and causal and byte_conservation and actuation
                                           and in_range_occluded > 0),
        "temporary_reveal_observed": in_range_revealed > 0,
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


