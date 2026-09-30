#!/usr/bin/env python3
"""Outcome-blind receiver-feature gate on existing *development* branches.

This recombines complete branch outcomes only because the gate is fixed before
the first reference query. It is a screening proxy, not an implemented hybrid
wire policy or independent confirmation.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.simulation.controlled_contract import compare_snapshots
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/controlled_development_v2"
PILOT = ROOT / "outputs/study_b/controlled_closed_loop_repaired_pilot_v2"
OUTPUT = ROOT / "outputs/study_b/margin_gate_proxy_development_v1"
METHODS = ("surface", "scalar_ttl", "periodic_payload", "local_only")
CONDITIONS = ("nominal", "impaired")
FRONT_MIN_M, FRONT_MAX_M = 5.0, 100.0
LATERAL_MAX_M = 2.25
FIRST_COMPETITOR_TOTAL_GAP_MIN = 0.20


def choose_method(reference: dict) -> tuple[str, dict]:
    """Use only exact reference-state, payload, and planner-margin fields."""
    batch = reference["frozen_input"]["batch"]
    state = batch["reference_state"]
    heading = float(state[3])
    cosine, sine = math.cos(heading), math.sin(heading)
    lead_like = False
    for row, present in zip(batch["object_features"], batch["object_mask"]):
        if not present or int(row[6]) != 0:  # class 0 is vehicle in this schema
            continue
        dx, dy = float(row[0]), float(row[1])
        along = cosine * dx + sine * dy
        lateral = abs(-sine * dx + cosine * dy)
        if FRONT_MIN_M <= along <= FRONT_MAX_M and lateral <= LATERAL_MAX_M:
            lead_like = True
            break
    gap = float(reference["observable_margin_features"][12])
    chosen = "scalar_ttl" if lead_like and gap >= FIRST_COMPETITOR_TOTAL_GAP_MIN else "surface"
    return chosen, {"lead_like": lead_like, "first_competitor_total_cost_gap": gap}


def load_scene(seed: int, expected_sha256: str) -> dict:
    path = SOURCE / f"train_{seed}.json.gz"
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("development source hash mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        scene = json.load(handle)
    if scene["split"] != "train" or str(scene["scenario_id"]) != str(seed):
        raise RuntimeError("nontraining scene or wrong identity")
    return scene


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protocol_path = PILOT / "pilot_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if set(protocol["methods"]) != set(METHODS) or set(protocol["conditions"]) != set(CONDITIONS):
        raise RuntimeError("incomplete development pilot")
    if len(protocol["cases"]) != 80:
        raise RuntimeError("expected 80 development scenarios")
    by_family: dict[str, list[dict]] = defaultdict(list)
    for case in protocol["cases"]:
        by_family[case["family"]].append(case)
    if set(by_family) != {"straight_lead_braking", "on_ramp_merge",
                          "unprotected_crossing", "pedestrian_crossing"} or any(len(v) != 20 for v in by_family.values()):
        raise RuntimeError("unexpected family distribution")
    rows = []
    artifact_hashes = {}
    for family, cases in sorted(by_family.items()):
        for ordinal, case in enumerate(sorted(cases, key=lambda item: item["seed"])):
            seed = case["seed"]
            scene = load_scene(seed, case["source_sha256"])
            if scene["family"] != family:
                raise RuntimeError("pilot/source family mismatch")
            refs = [ref for ref in scene["references"] if ref["behavior"] == "keep"]
            if len(refs) != 1:
                raise RuntimeError("expected one keep reference")
            chosen, evidence = choose_method(refs[0])
            for condition in CONDITIONS:
                branches = {}
                for method in METHODS:
                    path = PILOT / f"{seed}_{condition}_{method}.json.gz"
                    artifact_hashes[path.name] = sha256_file(path)
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        branch = json.load(handle)
                    if branch["family"] != family or str(branch["scenario_id"]) != str(seed) or branch["method"] != method:
                        raise RuntimeError("branch identity mismatch")
                    if not branch["actuation_contract_passed"] or not branch["traffic"]["byte_conservation_ok"] or not branch["traffic"]["component_conservation_ok"]:
                        raise RuntimeError("branch engineering gate failed")
                    if any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"]):
                        raise RuntimeError("branch used future input")
                    branches[method] = branch
                prefix = branches["periodic_payload"]["initial_contract"]
                if any(not compare_snapshots(prefix, branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
                       for branch in branches.values()):
                    raise RuntimeError("unpaired development prefix")
                periodic = branches["periodic_payload"]
                result = {method: {"bytes": branches[method]["traffic"]["generated_bytes"],
                                   "severe": _severe(branches[method]),
                                   "progress_deficit_m": periodic["route_distance_lower_bound_m"] - branches[method]["route_distance_lower_bound_m"]}
                          for method in METHODS}
                rows.append({"seed": seed, "family": family, "condition": condition,
                             "outcome_fold": "design" if ordinal < 10 else "development_validation",
                             "chosen_method": chosen, "gate_evidence": evidence,
                             "methods": result,
                             "hybrid_proxy": result[chosen],
                             "not_an_implemented_wire_policy": True})
    summary = {}
    for fold in ("design", "development_validation"):
        group = [row for row in rows if row["outcome_fold"] == fold]
        summary[fold] = {"scenarios": len({row["seed"] for row in group}),
                         "branches": len(group),
                         "ttl_chosen": sum(row["chosen_method"] == "scalar_ttl" for row in group),
                         "family": {family: {"branches": len(frows),
                                            "ttl_chosen": sum(row["chosen_method"] == "scalar_ttl" for row in frows),
                                            "mean_reduction_hybrid_proxy": float(np.mean([1 - row["hybrid_proxy"]["bytes"] / row["methods"]["periodic_payload"]["bytes"] for row in frows])),
                                            "mean_reduction_surface": float(np.mean([1 - row["methods"]["surface"]["bytes"] / row["methods"]["periodic_payload"]["bytes"] for row in frows])),
                                            "mean_reduction_ttl": float(np.mean([1 - row["methods"]["scalar_ttl"]["bytes"] / row["methods"]["periodic_payload"]["bytes"] for row in frows])),
                                            "hybrid_severe": sum(row["hybrid_proxy"]["severe"] for row in frows)}
                                    for family in sorted(by_family)
                                    for frows in [[row for row in group if row["family"] == family]]}}
    report = {"status": "DEVELOPMENT_BRANCH_SELECTION_PROXY_ONLY",
              "source_protocol_sha256": sha256_file(protocol_path),
              "source_development_manifest_sha256": sha256_file(SOURCE / "run_manifest.json"),
              "script_sha256": sha256_file(Path(__file__)),
              "gate": {"front_m": [FRONT_MIN_M, FRONT_MAX_M], "lateral_max_m": LATERAL_MAX_M,
                       "first_competitor_total_gap_min": FIRST_COMPETITOR_TOTAL_GAP_MIN,
                       "all_fields_reference_time_only": True},
              "artifact_sha256": artifact_hashes, "rows": rows, "summary": summary,
              "old_confirmation_outcomes_accessed": False, "fresh_confirmation_accessed": False,
              "limitations": ["entire-branch selection proxy, not implemented in an adaptive bound session",
                              "small development-only folds, no powered risk comparison",
                              "family-specific SUMO scenarios may make front-lead classification too easy"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
