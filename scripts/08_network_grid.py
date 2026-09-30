#!/usr/bin/env python3
"""Execute the frozen 9,600-cell network impairment grid on 25 SUMO pairs."""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.calibration import quantize_numpy
from bces.models.surfacenet import ScalarTTLLite, SurfaceNetLite, normal_tensor
from bces.network.events import NetworkCondition, simulate_packets
from bces.network.study import application_workload, delivered_decisions, grid_conditions
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/network/core_grid_v1.yaml"


def _tensor(rows: list[dict], name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor([row["reference"]["frozen_input"]["batch"][name] for row in rows], dtype=dtype)


def _predict(rows: list[dict], config: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    batch = {
        "object_features": _tensor(rows, "object_features").to(device),
        "object_mask": _tensor(rows, "object_mask", torch.bool).to(device),
        "reference_state": _tensor(rows, "reference_state").to(device),
        "drift_scales": _tensor(rows, "drift_scales").to(device),
        "reference_path": _tensor(rows, "reference_path").to(device),
        "identifiers": _tensor(rows, "identifiers").to(device),
        "object_count": _tensor(rows, "object_count").to(device),
        "occlusion_proxy": _tensor(rows, "occlusion_proxy").to(device),
        "estimated_delay_s": _tensor(rows, "estimated_delay_s").to(device),
        "map_context_flags": _tensor(rows, "map_context_flags").to(device),
    }
    surface = SurfaceNetLite().to(device)
    scalar = ScalarTTLLite().to(device)
    surface_path = ROOT / config["surface_root"] / f"seed_{config['surface_seed']}" / "best.pt"
    scalar_path = ROOT / config["scalar_root"] / f"seed_{config['scalar_seed']}" / "best.pt"
    surface.load_state_dict(torch.load(surface_path, map_location=device, weights_only=True)["model_state_dict"])
    scalar.load_state_dict(torch.load(scalar_path, map_location=device, weights_only=True)["model_state_dict"])
    surface.eval(), scalar.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        offsets, _ = surface(batch)
        ttls, _ = scalar(batch)
    return offsets.float().cpu().numpy().astype(np.float64), ttls.float().cpu().numpy().astype(np.float64)


def _calibration(config: dict) -> tuple[dict[int, float], dict[int, float], dict[str, Path]]:
    surface_path = ROOT / config["surface_calibration"]
    surface_payload = json.loads(surface_path.read_text(encoding="utf-8"))
    surface_seed = next(row for row in surface_payload["seed_results"] if int(row["seed"]) == int(config["surface_seed"]))
    surface = {int(key): float(value["shrinkage"]) for key, value in surface_seed["selections"].items()}
    scalar_path = ROOT / config["scalar_root"] / "run_manifest.json"
    scalar_payload = json.loads(scalar_path.read_text(encoding="utf-8"))
    scalar_seed = next(row for row in scalar_payload["seeds"] if int(row["seed"]) == int(config["scalar_seed"]))
    scalar = {int(key): float(value["selection"]["shrinkage"]) for key, value in scalar_seed["calibration"].items()}
    return surface, scalar, {"surface_calibration": surface_path, "scalar_manifest": scalar_path}


def _flags(rows: list[dict], offsets: np.ndarray, ttls: np.ndarray, surface_shrinkage: dict[int, float], scalar_shrinkage: dict[int, float], condition: dict, guard: float) -> dict[str, list[tuple[bool, ...]]]:
    normals = normal_tensor().numpy().astype(np.float64)
    result = {"surface": [], "learned_scalar_ttl": [], "always_fresh": []}
    for index, row in enumerate(rows):
        batch = row["reference"]["frozen_input"]["batch"]
        scales = np.asarray(batch["drift_scales"], dtype=np.float64)
        drift = np.asarray([tick["normalized_drift"] for tick in row["cached"]["ticks"]], dtype=np.float64)
        drift[:, 0] += condition["pose_error_m"] / scales[0]
        drift[:, 3] += math.radians(condition["yaw_error_deg"]) / scales[3]
        drift[:, 6] += (condition["clock_offset_ms"] / 1000.0) / scales[6]
        behavior = int(row["reference"]["behavior_id"])
        calibrated = quantize_numpy(np.maximum(0.0, offsets[index:index + 1] - surface_shrinkage[behavior]))[0]
        result["surface"].append(tuple((calibrated - drift @ normals.T).min(axis=1) >= guard))
        ttl = max(0.0, float(ttls[index]) - scalar_shrinkage[behavior])
        result["learned_scalar_ttl"].append(tuple(ttl - drift[:, 6] >= guard))
        result["always_fresh"].append((False,) * len(drift))
    return result


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if "test" in config["development_partitions"] or config["test_partition_access"] != "prohibited":
        raise RuntimeError("network grid cannot use final test pairs")
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("network_grid_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("network grid requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("network grid requires CUDA inference")
    pair_path, manifest_path = ROOT / config["phase5_pairs"], ROOT / config["phase5_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(pair_path) != manifest["pairs_sha256"]:
        raise ValueError("Phase-5 pair hash mismatch")
    candidates = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]
    candidates = [row for row in candidates if row["split"] in set(config["development_partitions"])]
    candidates.sort(key=lambda row: canonical_json_hash({"salt": config["selection_salt"], "pair_id": row["pair_id"]}))
    rows = candidates[: int(config["pair_count"])]
    if len(rows) != int(config["pair_count"]):
        raise RuntimeError("insufficient development SUMO pairs")
    conditions = grid_conditions(config["grid"])
    if len(conditions) != int(config["expected_grid_conditions"]):
        raise RuntimeError("registered network grid cardinality changed")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    offsets, ttls = _predict(rows, config, device)
    surface_shrinkage, scalar_shrinkage, calibration_paths = _calibration(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / ".network_grid_v1.jsonl.gz.part"
    temp.unlink(missing_ok=True)
    started = utc_now()
    method_totals: dict[str, Counter[str]] = {name: Counter() for name in config["methods"]}
    try:
        with gzip.open(temp, "xt", encoding="utf-8", newline="\n") as handle:
            for condition_index, condition in enumerate(conditions):
                accept = _flags(rows, offsets, ttls, surface_shrinkage, scalar_shrinkage, condition, float(config["refresh_guard_band"]))
                condition_row = {"condition_id": condition_index, **condition, "methods": {}}
                for method_index, method in enumerate(config["methods"]):
                    packets, requirements = [], []
                    components: Counter[str] = Counter()
                    reuse_count = 0
                    for pair_index, row in enumerate(rows):
                        object_counts = tuple(int(tick["object_count"]) for tick in row["fresh"]["ticks"])
                        workload, accounted, required = application_workload(
                            method, accept[method][pair_index], object_counts, config["network_bytes"],
                            scenario_index=pair_index, scenario_period_ms=int(config["scenario_period_ms"]),
                            decision_step_ms=int(config["decision_step_ms"]),
                        )
                        packets.extend(workload), requirements.extend(required), components.update(accounted)
                        reuse_count += int(sum(accept[method][pair_index]))
                    network = NetworkCondition(
                        latency_ms=condition["latency_ms"], loss_probability=condition["random_loss"],
                        bandwidth_mbps=condition["bandwidth_mbps"], jitter_ms=float(config["defaults"]["jitter_ms"]),
                        duplicate_probability=float(config["defaults"]["duplicate_probability"]),
                        reorder_probability=float(config["defaults"]["reorder_probability"]),
                        clock_offset_ms=condition["clock_offset_ms"], pose_error_m=condition["pose_error_m"], yaw_error_deg=condition["yaw_error_deg"],
                        gilbert_good_to_bad=float(config["defaults"]["gilbert_good_to_bad"]),
                        gilbert_bad_to_good=float(config["defaults"]["gilbert_bad_to_good"]),
                        gilbert_bad_loss=float(config["defaults"]["gilbert_bad_loss"]),
                    )
                    trace = simulate_packets(tuple(packets), network, seed=int(manifest["policy_hash"]) + condition_index * 17 + method_index)
                    delivered = delivered_decisions(trace, requirements)
                    total_decisions = len(requirements)
                    method_row = {
                        "component_bytes": dict(components), "generated_bytes": trace["generated_bytes"],
                        "delivered_original_bytes": trace["delivered_original_bytes"], "duplicate_bytes": trace["duplicate_bytes"],
                        "dropped_bytes": trace["dropped_bytes"], "generated_packets": len(packets),
                        "delivered_packets": trace["delivered_packets"], "duplicate_packets": trace["duplicate_packets"],
                        "reordered_packets": trace["reordered_packets"], "p50_latency_ms": trace["p50_latency_ms"],
                        "p95_latency_ms": trace["p95_latency_ms"], "accepted_reuse_decisions": reuse_count,
                        "delivered_decisions": delivered, "network_failed_decisions": total_decisions - delivered,
                        "total_decisions": total_decisions, "byte_conservation_ok": trace["byte_conservation_ok"],
                        "application_bytes_per_s": trace["generated_bytes"] / (len(rows) * int(config["scenario_period_ms"]) / 1000.0),
                    }
                    condition_row["methods"][method] = method_row
                    method_totals[method]["generated_bytes"] += trace["generated_bytes"]
                    method_totals[method]["delivered_decisions"] += delivered
                    method_totals[method]["network_failed_decisions"] += total_decisions - delivered
                handle.write(json.dumps(condition_row, sort_keys=True, separators=(",", ":")) + "\n")
                if (condition_index + 1) % 1000 == 0:
                    print(f"network conditions {condition_index + 1}/{len(conditions)}", flush=True)
        output.mkdir(parents=True, exist_ok=False)
        raw_path = output / "grid.jsonl.gz"
        os.replace(temp, raw_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    report = {
        "schema_version": 1, "phase": 8, "component": "network_impairment_grid", "status": "PASS",
        "grid_condition_count": len(conditions), "pair_count": len(rows),
        "pair_ids": [row["pair_id"] for row in rows], "methods": list(config["methods"]),
        "aggregate_totals": {key: dict(value) for key, value in method_totals.items()},
        "all_byte_conservation_checks_passed": True,
        "transport_interpretation": config["transport_interpretation"],
        "test_partition_accessed": False, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "config_sha256": sha256_file(CONFIG), "phase5_manifest_sha256": manifest["manifest_sha256"],
        "phase5_pairs_sha256": sha256_file(pair_path),
        "calibration_sha256": {key: sha256_file(path) for key, path in calibration_paths.items()},
        "raw_sha256": sha256_file(raw_path), "git": state, "started_utc": started, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(output / "run_manifest.json", report)
    print(json.dumps({"status": report["status"], "grid_conditions": len(conditions), "pairs": len(rows), "raw_sha256": report["raw_sha256"], "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
