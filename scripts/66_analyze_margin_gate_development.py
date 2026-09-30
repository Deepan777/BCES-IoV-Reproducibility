#!/usr/bin/env python3
"""Scenario-clustered descriptive analysis of the adaptive development run.

All 400 branch hashes and the exact frozen development protocol are verified
before any outcome summary. This analysis is not confirmatory inference.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/study_b/adaptive_margin_gate_development_v1"
METHODS = ("periodic_payload", "local_only", "surface", "scalar_ttl", "adaptive_margin_gate")
CONDITIONS = ("nominal", "impaired")
BOOTSTRAP_SEED = 20260923
REPLICATES = 20000


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = BASE / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    protocol_path, report_path = BASE / "protocol.json", BASE / "report.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (protocol["status"] != "FROZEN_DEVELOPMENT_ONLY"
            or report["status"] != "COMPLETE_DEVELOPMENT_ONLY"
            or report["protocol_sha256"] != sha256(protocol_path)
            or protocol["methods"] != list(METHODS)
            or protocol["conditions"] != list(CONDITIONS)):
        raise RuntimeError("incomplete or changed development protocol")
    expected = {f"branch_{case['seed']}_{condition}_{method}.json.gz"
                for case in protocol["selected"] for condition in CONDITIONS for method in METHODS}
    if set(report["artifact_sha256"]) != expected or len(expected) != 400:
        raise RuntimeError("missing or extra development branch artifact")
    for name, digest in report["artifact_sha256"].items():
        if sha256(BASE / name) != digest:
            raise RuntimeError("branch artifact changed: " + name)
    rows = report["rows"]
    if len(rows) != 400:
        raise RuntimeError("incomplete development summary")
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] not in METHODS or row["condition"] not in CONDITIONS:
            raise RuntimeError("unknown method/condition")
        by_seed[row["seed"]].append(row)
    if set(by_seed) != {case["seed"] for case in protocol["selected"]}:
        raise RuntimeError("scenario identity mismatch")
    scenario = []
    for seed, group in sorted(by_seed.items()):
        if {(row["condition"], row["method"]) for row in group} != {(c, m) for c in CONDITIONS for m in METHODS}:
            raise RuntimeError("incomplete scenario pairing")
        family = group[0]["family"]
        if any(row["family"] != family for row in group):
            raise RuntimeError("family mismatch")
        values = {method: [row for row in group if row["method"] == method] for method in METHODS}
        scenario.append({"seed": seed, "family": family,
                         "mean_reduction": {method: float(np.mean([row["byte_reduction_vs_periodic"] for row in values[method]]))
                                            for method in METHODS},
                         "adverse": {method: any(row["adverse_vs_periodic"] for row in values[method]) for method in METHODS},
                         "severe": {method: any(row["severe"] for row in values[method]) for method in METHODS},
                         "adaptive_bound_families": [selected for row in values["adaptive_margin_gate"]
                                                     for selected in row["adaptive_selected_families"]]})
    matrix = np.asarray([[row["mean_reduction"][method] for method in METHODS] for row in scenario])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(scenario), size=(REPLICATES, len(scenario)))
    adaptive_index = METHODS.index("adaptive_margin_gate")
    contrasts = {}
    for comparator in ("surface", "scalar_ttl"):
        difference = matrix[:, adaptive_index] - matrix[:, METHODS.index(comparator)]
        boot = difference[draws].mean(axis=1)
        contrasts["adaptive_minus_" + comparator] = {
            "mean_percentage_points": float(100 * difference.mean()),
            "descriptive_scenario_bootstrap_95pct_ci_points": (100 * np.quantile(boot, [.025, .975])).tolist(),
            "positive_scenarios": int((difference > 0).sum()),
            "negative_scenarios": int((difference < 0).sum()),
            "tied_scenarios": int((difference == 0).sum()),
        }
    family = {}
    for name in sorted({row["family"] for row in scenario}):
        group = [row for row in scenario if row["family"] == name]
        family[name] = {"scenarios": len(group),
                        "mean_byte_reduction_vs_periodic": {method: float(np.mean([row["mean_reduction"][method] for row in group])) for method in METHODS},
                        "adverse_scenarios": {method: sum(row["adverse"][method] for row in group) for method in METHODS},
                        "severe_scenarios": {method: sum(row["severe"][method] for row in group) for method in METHODS},
                        "adaptive_bound_families": {method: sum(row["adaptive_bound_families"].count(method) for row in group)
                                                    for method in ("surface", "scalar_ttl")}}
    result = {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY",
              "protocol_sha256": sha256(protocol_path), "report_sha256": sha256(report_path),
              "script_sha256": sha256(Path(__file__)), "scenarios": len(scenario),
              "branch_artifacts_verified": len(expected), "bootstrap_unit": "whole scenario",
              "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": REPLICATES,
              "mean_byte_reduction_vs_periodic": {method: float(matrix[:, i].mean()) for i, method in enumerate(METHODS)},
              "contrasts": contrasts, "family": family,
              "local_only_severe_scenarios": sum(row["severe"]["local_only"] for row in scenario),
              "no_confirmatory_claim": True,
              "limitations": ["40 validation-partition scenarios only; model training/selection reused the underlying development source",
                              "simulation branch outcomes, not independent external trajectories",
                              "zero-event results cannot establish relative driving safety",
                              "new gate must be frozen and run on truly fresh independent seeds after development"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
