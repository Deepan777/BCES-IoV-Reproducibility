#!/usr/bin/env python3
"""Run the frozen 30-seed matched closed-loop SUMO confirmation."""

from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.models.calibration import quantize_numpy
from bces.models.surfacenet import ScalarTTLLite, SurfaceNetLite, normal_tensor
from bces.network.events import NetworkCondition
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.simulation.closed_loop import ReusePolicy, prepare_matched_state, run_closed_loop_branch
from bces.simulation.scenario_builder import scenario_spec
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/simulation/phase9_confirmatory_v1.yaml"


def _batch(reference: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    row = reference["frozen_input"]["batch"]
    result = {}
    for name in ("object_features", "reference_state", "drift_scales", "reference_path", "identifiers", "object_count", "occlusion_proxy", "estimated_delay_s", "map_context_flags"):
        result[name] = torch.tensor(row[name], dtype=torch.float32, device=device).unsqueeze(0)
    result["object_mask"] = torch.tensor(row["object_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    return result


class SurfacePolicy:
    extension_bytes = 48

    def __init__(self, model: SurfaceNetLite, shrinkages: dict[int, float], device: torch.device, guard: float) -> None:
        self.model, self.shrinkages, self.device, self.guard = model, shrinkages, device, guard
        self.normals = normal_tensor().numpy().astype(np.float64)

    def encode_reference(self, reference: dict[str, Any]) -> np.ndarray:
        behavior = int(reference["behavior_id"])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            offsets, _ = self.model(_batch(reference, self.device))
        return quantize_numpy(np.maximum(0.0, offsets.float().cpu().numpy().astype(np.float64) - self.shrinkages[behavior]))[0]

    def accepts(self, token: np.ndarray, drift: tuple[float, ...]) -> bool:
        return bool((token - np.asarray(drift, dtype=np.float64) @ self.normals.T).min() >= self.guard)


class ScalarPolicy:
    extension_bytes = 2

    def __init__(self, model: ScalarTTLLite, shrinkages: dict[int, float], device: torch.device, guard: float) -> None:
        self.model, self.shrinkages, self.device, self.guard = model, shrinkages, device, guard

    def encode_reference(self, reference: dict[str, Any]) -> float:
        behavior = int(reference["behavior_id"])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            ttl, _ = self.model(_batch(reference, self.device))
        return max(0.0, float(ttl.float().cpu()) - self.shrinkages[behavior])

    def accepts(self, token: float, drift: tuple[float, ...]) -> bool:
        return token - float(drift[6]) >= self.guard


def _condition(payload: dict) -> NetworkCondition:
    return NetworkCondition(**{key: float(value) for key, value in payload.items()})


def _summaries(rows: list[dict], methods: list[str], conditions: list[str]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["method"])].append(row)
    result = {}
    for condition in conditions:
        result[condition] = {}
        for method in methods:
            values = grouped[(condition, method)]
            completed = [row for row in values if row["status"] == "PASS"]
            result[condition][method] = {
                "registered_seeds": len(values), "completed_seeds": len(completed),
                "failures": len(values) - len(completed),
                "collisions": sum(row.get("collision", False) for row in completed),
                "critical_events": sum(row.get("critical_event", False) for row in completed),
                "mean_route_progress_m": float(np.mean([row["route_progress_m"] for row in completed])) if completed else None,
                "mean_total_abs_jerk": float(np.mean([row["total_abs_jerk"] for row in completed])) if completed else None,
                "total_application_bytes": sum(row["communication"]["generated_bytes"] for row in completed),
                "mean_application_bytes": float(np.mean([row["communication"]["generated_bytes"] for row in completed])) if completed else None,
                "network_failures": sum(row["communication"].get("network_failures", 0) for row in completed),
                "accepted_reuse_decisions": sum(row["communication"].get("accepted_reuse_decisions", 0) for row in completed),
                "refresh_decisions": sum(row["communication"].get("refresh_decisions", 0) for row in completed),
                "local_fallback_decisions": sum(row["communication"].get("local_fallback_decisions", 0) for row in completed),
            }
    return result


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("Phase-9 confirmatory output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Phase-9 confirmation requires a clean committed revision")
    pilot_path = ROOT / config["pilot_report"]
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot["report_sha256"] != config["pilot_report_sha256"] or pilot["power_analysis"]["recommended_confirmatory_matched_seeds"] != int(config["matched_seed_count"]):
        raise RuntimeError("confirmatory sample count does not match frozen pilot")
    if json.loads((ROOT / config["phase8_gate"]).read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Phase 8 is incomplete")
    if sumo_version() != str(config["sumo_version"]):
        raise RuntimeError("SUMO version changed")
    scale_path = ROOT / config["drift_scale_manifest"]
    scale_payload = json.loads(scale_path.read_text(encoding="utf-8"))
    scale_hash = scale_payload.pop("scales_sha256")
    if canonical_json_hash(scale_payload) != scale_hash or scale_payload["partition"] != "train" or scale_payload["test_partition_accessed"]:
        raise ValueError("drift-scale manifest is not frozen training-only evidence")
    scales = DriftScales(*map(float, scale_payload["scales"]))
    planner_path = ROOT / config["planner_config"]
    planner = KinematicPlanner(PlannerConfig.from_yaml(planner_path))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-9 model inference requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    surface_model, scalar_model = SurfaceNetLite().to(device), ScalarTTLLite().to(device)
    surface_checkpoint = ROOT / config["surface_root"] / f"seed_{config['surface_seed']}" / "best.pt"
    scalar_checkpoint = ROOT / config["scalar_root"] / f"seed_{config['scalar_seed']}" / "best.pt"
    surface_model.load_state_dict(torch.load(surface_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    scalar_model.load_state_dict(torch.load(scalar_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    surface_model.eval(), scalar_model.eval()
    surface_payload = json.loads((ROOT / config["surface_calibration"]).read_text(encoding="utf-8"))
    surface_seed = next(row for row in surface_payload["seed_results"] if int(row["seed"]) == int(config["surface_seed"]))
    surface_shrinkages = {int(key): float(value["shrinkage"]) for key, value in surface_seed["selections"].items()}
    scalar_manifest_path = ROOT / config["scalar_root"] / "run_manifest.json"
    scalar_payload = json.loads(scalar_manifest_path.read_text(encoding="utf-8"))
    scalar_seed = next(row for row in scalar_payload["seeds"] if int(row["seed"]) == int(config["scalar_seed"]))
    scalar_shrinkages = {int(key): float(value["selection"]["shrinkage"]) for key, value in scalar_seed["calibration"].items()}
    surface_policy = SurfacePolicy(surface_model, surface_shrinkages, device, 0.05)
    scalar_policy = ScalarPolicy(scalar_model, scalar_shrinkages, device, 0.05)
    policies: dict[str, ReusePolicy | None] = {"surface_every_tick": surface_policy, "surface_receipt_only": surface_policy, "learned_scalar_ttl": scalar_policy, "always_fresh": None, "local_only": None}
    methods, conditions = list(config["methods"]), list(config["conditions"])
    network = ROOT / config["network"]
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    before, started = build_budget_report(budget)["measurements"], utc_now()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / ".phase9_confirmatory.jsonl.gz.part"
    temp.unlink(missing_ok=True)
    rows, failure_types = [], Counter()
    try:
        with gzip.open(temp, "xt", encoding="utf-8", newline="\n") as handle:
            for seed_index in range(int(config["matched_seed_count"])):
                seed = int(config["seed_start"]) + seed_index
                family = config["families"][seed_index % len(config["families"])]
                spec = scenario_spec(family)
                state_path = ROOT / "sumo" / "states" / f"phase9_{seed}.xml.gz"
                try:
                    reference, reference_state, cache = prepare_matched_state(network=network, state_path=state_path, seed=seed, step_s=float(config["step_s"]), reference_time_s=float(config["reference_time_s"]), spec=spec, planner=planner, drift_scales=scales)
                    seed_rows = []
                    for condition_name in conditions:
                        condition = _condition(config["conditions"][condition_name])
                        for method in methods:
                            try:
                                branch = run_closed_loop_branch(network=network, state_path=state_path, seed=seed, step_s=float(config["step_s"]), horizon_s=float(config["horizon_s"]), spec=spec, method=method, planner=planner, drift_scales=scales, initial_reference=reference, initial_reference_state=reference_state, initial_cache=cache, condition=condition, sizes=config["network_bytes"], reuse_policy=policies[method], critical_ttc_s=float(config["critical_ttc_s"]))
                                row = {"status": "PASS", "seed": seed, "family": family, "condition": condition_name, **branch}
                            except Exception as exc:
                                category = "planner_failure" if "planner" in str(exc).lower() or "trajectory" in str(exc).lower() else "simulation_or_network_failure"
                                failure_types[category] += 1
                                row = {"status": "FAIL", "seed": seed, "family": family, "condition": condition_name, "method": method, "failure_category": category, "error_type": type(exc).__name__, "error": str(exc)}
                            seed_rows.append(row)
                    hashes = {row["pre_action_non_ego_state_hash"] for row in seed_rows if row["status"] == "PASS"}
                    if len(hashes) != 1:
                        failure_types["state_equivalence_failure"] += 1
                    for row in seed_rows:
                        row["matched_pre_action_state"] = len(hashes) == 1
                        rows.append(row)
                        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                except Exception as exc:
                    failure_types["seed_setup_failure"] += 1
                    for condition_name in conditions:
                        for method in methods:
                            row = {"status": "FAIL", "seed": seed, "family": family, "condition": condition_name, "method": method, "failure_category": "seed_setup_failure", "error_type": type(exc).__name__, "error": str(exc), "matched_pre_action_state": False}
                            rows.append(row), handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                finally:
                    state_path.unlink(missing_ok=True)
                if (seed_index + 1) % 5 == 0:
                    print(f"confirmatory SUMO seeds {seed_index + 1}/{config['matched_seed_count']}", flush=True)
        output.mkdir(parents=True, exist_ok=False)
        raw_path = output / "routes.jsonl.gz"
        os.replace(temp, raw_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    expected = int(config["matched_seed_count"]) * len(methods) * len(conditions)
    summaries = _summaries(rows, methods, conditions)
    complete = len(rows) == expected and not failure_types and all(row["status"] == "PASS" and row["matched_pre_action_state"] for row in rows)
    clean = summaries["clean"]
    report = {
        "schema_version": 1, "phase": 9, "component": "locked_confirmatory_closed_loop_sumo",
        "status": "PASS" if complete else "FAIL", "scientific_success": False,
        "matched_seed_count": int(config["matched_seed_count"]), "registered_branch_count": expected,
        "completed_branch_count": sum(row["status"] == "PASS" for row in rows), "failure_taxonomy": dict(failure_types),
        "summaries": summaries,
        "primary_directional_checks": {
            "surface_fewer_critical_events_than_scalar": clean["surface_every_tick"]["critical_events"] < clean["learned_scalar_ttl"]["critical_events"],
            "surface_fewer_bytes_than_always_fresh": clean["surface_every_tick"]["total_application_bytes"] < clean["always_fresh"]["total_application_bytes"],
            "surface_nonzero_reuse": clean["surface_every_tick"]["accepted_reuse_decisions"] > 0,
        },
        "test_partition_access": config["test_partition_access"], "rendered_frames": 0,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "config_sha256": sha256_file(CONFIG), "pilot_report_sha256": pilot["report_sha256"],
        "phase8_gate_sha256": sha256_file(ROOT / config["phase8_gate"]), "planner_config_sha256": sha256_file(planner_path),
        "drift_scales_sha256": scale_hash, "surface_checkpoint_sha256": sha256_file(surface_checkpoint),
        "surface_calibration_sha256": sha256_file(ROOT / config["surface_calibration"]), "scalar_checkpoint_sha256": sha256_file(scalar_checkpoint),
        "scalar_manifest_sha256": sha256_file(scalar_manifest_path), "raw_sha256": sha256_file(raw_path),
        "budget_before": before, "budget_after": build_budget_report(budget)["measurements"],
        "git": state, "started_utc": started, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(output / "run_manifest.json", report)
    print(json.dumps({"status": report["status"], "matched_seeds": report["matched_seed_count"], "failure_taxonomy": report["failure_taxonomy"], "primary_directional_checks": report["primary_directional_checks"], "summaries": summaries, "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

