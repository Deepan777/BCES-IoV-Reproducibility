#!/usr/bin/env python3
"""Rebuild all Phase-10 intervals, paired tests, corrections, and taxonomy."""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]
from scipy.stats import wilcoxon  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.statistics import clustered_uar_difference_ci, holm_adjust, paired_bootstrap_mean_ci, paired_permutation_test, standardized_paired_effect
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/statistics_v1.yaml"


def _gzip_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _top(score: np.ndarray, fraction: float) -> np.ndarray:
    accepted = np.zeros(len(score), dtype=bool)
    accepted[np.argsort(score, kind="stable")[:max(1, int(np.floor(len(score) * fraction)))]] = True
    return accepted


def _scenario_contribution(accepted: np.ndarray, invalid: np.ndarray, clusters: list[str]) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(clusters):
        grouped[name].append(index)
    return {name: float((accepted[indices] & invalid[indices]).sum() / len(indices)) for name, indices in grouped.items()}


def _paired(values: list[float], config: dict, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    permutation = paired_permutation_test(array, replicates=int(config["permutation_replicates"]), seed=seed)
    if np.all(array == 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(array, zero_method="wilcox", alternative="two-sided").pvalue)
    return {
        "bootstrap_ci": paired_bootstrap_mean_ci(array, confidence=float(config["confidence"]), replicates=int(config["bootstrap_replicates"]), seed=seed + 1),
        "permutation": permutation, "wilcoxon_p_value": wilcoxon_p,
        "standardized_paired_effect_dz": standardized_paired_effect(array),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("statistics_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("statistics require a clean committed revision")
    phase8, phase9 = json.loads((ROOT / config["phase8_gate"]).read_text(encoding="utf-8")), json.loads((ROOT / config["phase9_gate"]).read_text(encoding="utf-8"))
    if phase8["status"] != "PASS" or phase9["status"] != "PASS":
        raise RuntimeError("Phase 8 and 9 gates must pass")
    offline_manifest = json.loads((ROOT / config["offline_manifest"]).read_text(encoding="utf-8"))
    offline_path = ROOT / config["offline_decisions"]
    sumo_manifest = json.loads((ROOT / config["sumo_manifest"]).read_text(encoding="utf-8"))
    sumo_path = ROOT / config["sumo_routes"]
    if sha256_file(offline_path) != offline_manifest["raw_sha256"] or sha256_file(sumo_path) != sumo_manifest["raw_sha256"]:
        raise ValueError("raw evidence hash mismatch")
    offline = _gzip_rows(offline_path)
    valid = np.asarray([row["valid"] for row in offline], dtype=bool)
    invalid = ~valid
    clusters = [str(row["scenario_id"]) for row in offline]
    fraction = float(config["matched_coverage_fraction"])
    matched = {name: _top(np.asarray([row["scores"][name] for row in offline], dtype=np.float64), fraction) for name in ("surface", "learned_scalar_ttl", "validity_mlp")}
    offline_comparisons = {}
    offline_p = {}
    for comparison_index, comparator in enumerate(("learned_scalar_ttl", "validity_mlp")):
        name = f"surface_minus_{comparator}"
        interval = clustered_uar_difference_ci(matched["surface"], matched[comparator], invalid, clusters, confidence=float(config["confidence"]), replicates=int(config["bootstrap_replicates"]), seed=int(config["seed"]) + comparison_index * 100)
        first = _scenario_contribution(matched["surface"], invalid, clusters)
        second = _scenario_contribution(matched[comparator], invalid, clusters)
        paired_values = [first[key] - second[key] for key in sorted(first)]
        paired = _paired(paired_values, config, int(config["seed"]) + 1000 + comparison_index * 100)
        offline_comparisons[name] = {"clustered_uar_difference": interval, "paired_unsafe_contribution": paired}
        offline_p[name] = paired["permutation"]["p_value"]

    sumo = _gzip_rows(sumo_path)
    indexed = {(row["seed"], row["condition"], row["method"]): row for row in sumo if row["status"] == "PASS"}
    seeds = sorted({row["seed"] for row in sumo})
    sumo_comparisons, communication_p, driving_p = {}, {}, {}
    pairs = (("surface_every_tick", "learned_scalar_ttl"), ("surface_every_tick", "always_fresh"), ("surface_every_tick", "local_only"))
    for condition_index, condition in enumerate(("clean", "locked_impairment")):
        sumo_comparisons[condition] = {}
        for pair_index, (first_name, second_name) in enumerate(pairs):
            name = f"{first_name}_minus_{second_name}"
            endpoints = {}
            for endpoint_index, endpoint in enumerate(("critical_event", "collision", "route_progress_m", "total_abs_jerk")):
                differences = [float(indexed[(seed, condition, first_name)][endpoint]) - float(indexed[(seed, condition, second_name)][endpoint]) for seed in seeds]
                endpoints[endpoint] = _paired(differences, config, int(config["seed"]) + 3000 + condition_index * 1000 + pair_index * 100 + endpoint_index * 10)
                driving_p[f"{condition}:{name}:{endpoint}"] = endpoints[endpoint]["permutation"]["p_value"]
            byte_differences = [float(indexed[(seed, condition, first_name)]["communication"]["generated_bytes"]) - float(indexed[(seed, condition, second_name)]["communication"]["generated_bytes"]) for seed in seeds]
            endpoints["application_bytes"] = _paired(byte_differences, config, int(config["seed"]) + 7000 + condition_index * 1000 + pair_index * 100)
            communication_p[f"{condition}:{name}:application_bytes"] = endpoints["application_bytes"]["permutation"]["p_value"]
            sumo_comparisons[condition][name] = endpoints

    second = json.loads((ROOT / config["second_planner"]).read_text(encoding="utf-8"))
    uncertainty = json.loads((ROOT / config["object_uncertainty"]).read_text(encoding="utf-8"))
    geometry = json.loads((ROOT / config["geometry_audit"]).read_text(encoding="utf-8"))
    network = json.loads((ROOT / config["network_grid"]).read_text(encoding="utf-8"))
    maximum_changes = {
        method: max(float(row[f"{method}_decision_change_fraction"]) for row in uncertainty["conditions"])
        for method in ("surface", "validity_mlp")
    }
    surface_scalar_ci = offline_comparisons["surface_minus_learned_scalar_ttl"]["clustered_uar_difference"]
    clean_comm_ci = sumo_comparisons["clean"]["surface_every_tick_minus_always_fresh"]["application_bytes"]["bootstrap_ci"]
    conclusions = {
        "surface_uar_improvement_vs_scalar_ci_below_zero": surface_scalar_ci["upper"] < 0,
        "surface_communication_reduction_vs_fresh_ci_below_zero": clean_comm_ci["upper"] < 0,
        "closed_loop_safety_difference_identifiable": any(indexed[(seed, "clean", "surface_every_tick")]["critical_event"] != indexed[(seed, "clean", "learned_scalar_ttl")]["critical_event"] for seed in seeds),
        "locked_impairment_supports_primary_claim": False,
        "quantization_does_not_erase_benefit": False,
        "q1_strength_conditions_met": False,
    }
    report = {
        "schema_version": 1, "phase": 10, "component": "statistics_and_robustness", "status": "PASS",
        "scientific_success": False, "offline_unit": config["unit_of_inference_offline"], "sumo_unit": config["unit_of_inference_sumo"],
        "offline_matched_coverage": {"fraction": fraction, "comparisons": offline_comparisons},
        "sumo_paired_comparisons": sumo_comparisons,
        "holm_adjusted_p_values": {"offline_selective_risk": holm_adjust(offline_p), "sumo_communication": holm_adjust(communication_p), "sumo_driving": holm_adjust(driving_p)},
        "sensitivity": {
            "second_planner_oracle_agreement": second["oracle_label_agreement"]["overall"],
            "second_planner_scientific_target_met": second["scientific_target_met"],
            "maximum_object_uncertainty_decision_change_fraction": maximum_changes,
            "geometry_boundary_audit": geometry["boundary_audit"],
            "network_grid_conditions": network["grid_condition_count"],
        },
        "failure_taxonomy": {
            "phase9_execution": sumo_manifest["failure_taxonomy"],
            "scientific": {"offline_safety_target_failure": 1, "coverage_collapse": 1, "zero_sumo_critical_event_discordance": 1, "communication_increase": 1, "second_planner_target_failure": 1, "sumo_source_underpowered": 1},
        },
        "conclusions": conclusions,
        "raw_sha256": {"offline": sha256_file(offline_path), "sumo": sha256_file(sumo_path)},
        "source_sha256": {key: sha256_file(ROOT / config[key]) for key in ("phase8_gate", "phase9_gate", "offline_manifest", "sumo_manifest", "second_planner", "object_uncertainty", "geometry_audit", "network_grid")},
        "config_sha256": sha256_file(CONFIG), "git": state, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "statistics.json", report)
    print(json.dumps({"status": report["status"], "conclusions": conclusions, "surface_vs_scalar": surface_scalar_ci, "surface_vs_fresh_bytes": clean_comm_ci, "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

