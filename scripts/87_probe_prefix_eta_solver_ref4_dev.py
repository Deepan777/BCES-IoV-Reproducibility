#!/usr/bin/env python3
"""Fresh prefix-only solver audit at 4 s, with a 0--40 m actor bracket."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/prefix_eta_solver_development_v2"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
HELPER = ROOT / "scripts/86_probe_prefix_eta_solver_dev.py"
HELPER_SHA256 = "a903738895091ce24e21d3af92d57f3801b680583e164a61f54e9724b51f4f9a"
SPEEDS = (11.0, 12.0, 13.0, 14.0)
REPLICATES = (0, 1)
TARGET_DELTA_S = 0.20
POSITION_LOW_M = 0.0
POSITION_HIGH_M = 40.0
REFERENCE_TIME_S = 4.0
TOLERANCE_S = 0.10


def load_helper():
    if sha256_file(HELPER) != HELPER_SHA256:
        raise RuntimeError("frozen prefix helper changed")
    spec = importlib.util.spec_from_file_location("prefix_eta_helper_v1", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen prefix helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    helper = load_helper()
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    config = dict(registration["config"])
    config["reference_time_s"] = REFERENCE_TIME_S
    condition = NetworkCondition(**registration["conditions"]["nominal"])
    seeds = [8600400 + 2 * i + replicate
             for i, _ in enumerate(SPEEDS) for replicate in REPLICATES]
    protocol = {
        "status": "FROZEN_PREFIX_ONLY_DEVELOPMENT_ENGINEERING_V2",
        "registration_sha256": sha256_file(REGISTRATION),
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "helper_sha256": HELPER_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "seeds": seeds, "speeds_mps": list(SPEEDS),
        "replicates": list(REPLICATES),
        "reference_time_s": REFERENCE_TIME_S,
        "position_bracket_m": [POSITION_LOW_M, POSITION_HIGH_M],
        "target_signed_eta_difference_s": TARGET_DELTA_S,
        "solver": "exactly two fixed endpoint prefix probes, one bounded linear interpolation if monotone/bracketed, then exactly one final prefix; no branch-outcome selection or retries",
        "success_rule": "at least 6/8 final prefixes on A1B1 with absolute signed ETA error <=0.10 s, spanning at least 3 ego speeds",
        "failed_brackets_and_wrong_lanes_retained": True,
        "no_policy_outcomes": True, "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows, artifacts = [], {}
    for speed_index, speed in enumerate(SPEEDS):
        for replicate in REPLICATES:
            seed = 8600400 + 2 * speed_index + replicate
            probes = []
            for index, position in enumerate((POSITION_LOW_M, POSITION_HIGH_M)):
                row, initial = helper.probe(seed, speed, position, index, config, condition)
                probes.append(row)
                name = f"{seed}_prefix_{index}.json"
                (OUTPUT / name).write_text(json.dumps(initial, sort_keys=True,
                                                     separators=(",", ":")) + "\n", encoding="utf-8")
                artifacts[name] = sha256_file(OUTPUT / name)
            low, high = probes
            low_delta, high_delta = (item.get("signed_eta_difference_s") for item in probes)
            selected = (helper.linear_position(POSITION_LOW_M, low_delta,
                                               POSITION_HIGH_M, high_delta,
                                               TARGET_DELTA_S)
                        if low_delta is not None and high_delta is not None else None)
            if selected is None:
                status = "unbracketed_or_missing_prefix"
            else:
                final, initial = helper.probe(seed, speed, selected, 2, config, condition)
                probes.append(final)
                name = f"{seed}_prefix_2.json"
                (OUTPUT / name).write_text(json.dumps(initial, sort_keys=True,
                                                     separators=(",", ":")) + "\n", encoding="utf-8")
                artifacts[name] = sha256_file(OUTPUT / name)
                delta = final.get("signed_eta_difference_s")
                status = ("target_hit_on_approach" if delta is not None
                          and abs(delta - TARGET_DELTA_S) <= TOLERANCE_S
                          and str(final.get("actor_lane", "")).startswith("A1B1")
                          and str(final.get("ego_lane", "")).startswith("B0B1")
                          else "target_or_approach_failed")
            rows.append({"seed": seed, "ego_speed_setting_mps": speed,
                         "replicate": replicate, "selected_position_m": selected,
                         "status": status, "probes": probes})
            print(json.dumps({"completed": len(rows), "total": 8,
                              "status": status}), flush=True)
    hits = [row for row in rows if row["status"] == "target_hit_on_approach"]
    report = {"status": "PREFIX_ONLY_DEVELOPMENT_ENGINEERING_V2",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "target_hits": len(hits),
              "hit_speeds": sorted({row["ego_speed_setting_mps"] for row in hits}),
              "solver_feasible": len(hits) >= 6 and len({row["ego_speed_setting_mps"] for row in hits}) >= 3,
              "no_policy_outcomes": True, "not_confirmatory": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("target_hits", "hit_speeds", "solver_feasible")}, indent=2))


if __name__ == "__main__":
    main()
