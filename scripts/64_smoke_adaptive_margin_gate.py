#!/usr/bin/env python3
"""Actual packet-bound adaptive-gate smoke on two old development scenarios.

This is an engineering test only. It is not a powered validation or a new
confirmation and cannot establish a safety/communication benefit.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.bound_policy import FrozenWirePolicy
from bces.network.adaptive_margin_policy import AdaptiveMarginWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.utils.hashing import sha256_file

PILOT = ROOT / "outputs/study_b/controlled_closed_loop_repaired_pilot_v2"
OUTPUT = ROOT / "outputs/study_b/adaptive_margin_gate_smoke_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    frozen = {
        name: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                               policy_hash=registration["policy_hash"])
        for name, rule in registration["model_rules"].items()
    }
    pilot_protocol = json.loads((PILOT / "pilot_protocol.json").read_text(encoding="utf-8"))
    seeds = [next(case["seed"] for case in pilot_protocol["cases"] if case["family"] == family)
             for family in ("straight_lead_braking", "unprotected_crossing")]
    protocol = {"status": "DEVELOPMENT_ENGINEERING_SMOKE_ONLY", "seeds": seeds,
                "conditions": list(registration["condition_order"]),
                "registration_sha256": sha256_file(registration_path),
                "pilot_protocol_sha256": sha256_file(PILOT / "pilot_protocol.json"),
                "adaptive_source_sha256": sha256_file(ROOT / "bces/network/adaptive_margin_policy.py"),
                "script_sha256": sha256_file(Path(__file__)),
                "not_confirmatory": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = []
    artifacts = {}
    import torch
    torch.set_num_threads(2)
    for seed in seeds:
        for condition_name in registration["condition_order"]:
            condition = NetworkCondition(**registration["conditions"][condition_name])
            policy = AdaptiveMarginWirePolicy(
                surface=frozen["surface"], scalar_ttl=frozen["scalar_ttl"],
                shrinkages={name: rule["shrinkage"] for name, rule in registration["model_rules"].items()})
            with patch.object(loop, "bind_reference", policy.bind):
                branch = loop.run_branch(
                    seed=seed, method="surface", config=registration["config"],
                    condition=condition, network_seed=seed + 910000,
                    model=policy, rule=registration["model_rules"]["surface"],
                    horizon_s=registration["horizon_s"])
            branch["method"] = "adaptive_margin_gate"
            branch["adaptive_selections"] = policy.selections
            if (not branch["actuation_contract_passed"]
                    or not branch["traffic"]["byte_conservation_ok"]
                    or not branch["traffic"]["component_conservation_ok"]
                    or any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"])):
                raise RuntimeError("adaptive branch engineering gate failed")
            if not policy.selections:
                raise RuntimeError("adaptive branch never selected a bound")
            periodic_path = PILOT / f"{seed}_{condition_name}_periodic_payload.json.gz"
            with gzip.open(periodic_path, "rt", encoding="utf-8") as handle:
                periodic = json.load(handle)
            if not compare_snapshots(periodic["initial_contract"], branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]:
                raise RuntimeError("prefix mismatch")
            name = f"{seed}_{condition_name}_adaptive_margin_gate.json.gz"
            with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            artifacts[name] = sha256_file(OUTPUT / name)
            rows.append({"seed": seed, "condition": condition_name, "family": branch["family"],
                         "selections": [item["family"] for item in policy.selections],
                         "generated_bytes": branch["traffic"]["generated_bytes"],
                         "reduction_vs_periodic": 1 - branch["traffic"]["generated_bytes"] / periodic["traffic"]["generated_bytes"],
                         "outcome": branch["outcome"], "not_confirmatory": True})
    report = {"status": "DEVELOPMENT_ENGINEERING_SMOKE_ONLY",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "claim": "packet-binding and run feasibility only; not efficacy"}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
