#!/usr/bin/env python3
"""Frozen, scene-aware descriptive analysis of V2X-Real position proxy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic

REGISTRATION = ROOT / "docs/STUDY_B_V2XREAL_POSITION_PROXY_REGISTRATION_V1.json"
INPUTS = {
    "val": ROOT / "outputs/study_b/v2xreal_position_proxy_val_v2/result.json",
    "test": ROOT / "outputs/study_b/v2xreal_position_proxy_test_v1/result.json",
}
OUTPUTS = {
    "val": ROOT / "outputs/study_b/v2xreal_position_proxy_val_v2/analysis.json",
    "test": ROOT / "outputs/study_b/v2xreal_position_proxy_test_v1/analysis.json",
}
METHODS = ("surface", "scalar_ttl")


def endpoint(row: dict, kind: str) -> bool:
    if kind == "composite_invalid":
        return not row["proxy_valid"]
    if kind == "avoidable_regret_or_deviation":
        return bool(row["violations"]["regret"] or row["violations"]["deviation"])
    if kind == "absolute_risk_violation":
        return bool(row["violations"]["absolute_risk"])
    if kind == "fresh_also_high_risk":
        return float(row["proxy_fresh_risk"]) > 0.40
    raise KeyError(kind)


def counts(rows: list[dict]) -> dict:
    return {kind: sum(endpoint(row, kind) for row in rows) for kind in (
        "composite_invalid", "avoidable_regret_or_deviation",
        "absolute_risk_violation", "fresh_also_high_risk")}


def accepted_rows(rows: list[dict], method: str) -> list[dict]:
    return [row for row in rows if row["accepted"][method]]


def slack_order(row: dict, method: str) -> tuple:
    slack = row["receiver_slack"][method]
    if slack is None:
        raise RuntimeError("accepted slot lacks receiver minimum slack")
    return (-float(slack), int(row["reference"]), int(row["decision"]))


def analyze(split: str) -> None:
    source = INPUTS[split]
    output = OUTPUTS[split]
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw["split"] != split:
        raise RuntimeError("input split mismatch")
    if split == "test":
        registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
        if registration["status"] != "FROZEN_BEFORE_TEST_OUTCOMES":
            raise RuntimeError("registration status mismatch")
        if sha256_file(Path(__file__)) != registration["sha256"]["analysis"]:
            raise RuntimeError("analysis script changed since registration")
        if raw["test_registration_sha256"] != sha256_file(REGISTRATION):
            raise RuntimeError("input did not use current test registration")
        for key, source_key in (("runner", "runner_sha256"),
                                ("archive", "source_archive_sha256"),
                                ("schema_audit", "schema_audit_sha256"),
                                ("v2_registration", "v2_registration_sha256")):
            if raw[source_key] != registration["sha256"][key]:
                raise RuntimeError(f"input {key} hash mismatch")
    rows = raw["rows"]
    available = [row for row in rows if row["status"] == "position_proxy_label_available"]
    by_scene = defaultdict(list)
    for row in available:
        by_scene[row["scene"]].append(row)
    scene_results = {}
    for scene, scene_rows in sorted(by_scene.items()):
        own = {method: accepted_rows(scene_rows, method) for method in METHODS}
        k = min(len(group) for group in own.values())
        matched = {method: sorted(own[method], key=lambda row: slack_order(row, method))[:k]
                   for method in METHODS}
        scene_results[scene] = {
            "evaluable_slots": len(scene_rows),
            "all_evaluable": counts(scene_rows),
            "matched_k": k,
            "methods": {method: {
                "accepted": len(own[method]), "accepted_counts": counts(own[method]),
                "matched_accepted": len(matched[method]),
                "matched_counts": counts(matched[method]),
                "payload_bytes_on_accepted_slots": sum(row["payload_bytes"] for row in own[method]),
                "query_response_bytes_on_accepted_slots": sum(
                    row["query_response_bytes"][method] for row in own[method]),
            } for method in METHODS},
        }
    aggregate = {
        "evaluable_slots": len(available),
        "evaluable_acquisition_scenes": len(by_scene),
        "all_evaluable": counts(available),
        "matched_slots_per_method": sum(item["matched_k"] for item in scene_results.values()),
        "methods": {},
    }
    for method in METHODS:
        accepted = sum(item["methods"][method]["accepted"] for item in scene_results.values())
        aggregate["methods"][method] = {
            "accepted": accepted,
            "acceptance_fraction": accepted / len(available) if available else None,
            "accepted_counts": {kind: sum(item["methods"][method]["accepted_counts"][kind]
                                     for item in scene_results.values()) for kind in counts([])},
            "matched_accepted": aggregate["matched_slots_per_method"],
            "matched_counts": {kind: sum(item["methods"][method]["matched_counts"][kind]
                                    for item in scene_results.values()) for kind in counts([])},
            "payload_bytes_on_accepted_slots": sum(
                item["methods"][method]["payload_bytes_on_accepted_slots"]
                for item in scene_results.values()),
            "query_response_bytes_on_accepted_slots": sum(
                item["methods"][method]["query_response_bytes_on_accepted_slots"]
                for item in scene_results.values()),
        }
    result = {
        "status": "DESCRIPTIVE_POSITION_ONLY_PROXY_NOT_Q1_EXTERNAL_CONFIRMATION",
        "split": split,
        "source_sha256": sha256_file(source),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "test_registration_sha256": sha256_file(REGISTRATION) if split == "test" else None,
        "candidate_slots": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "scene_results": scene_results,
        "aggregate": aggregate,
        "inference_limitation": "Descriptive scene-level report only; correlated slots are not independent trials.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result)
    print(json.dumps({"candidate_slots": result["candidate_slots"],
                      "status_counts": result["status_counts"],
                      "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=sorted(INPUTS))
    analyze(parser.parse_args().split)
