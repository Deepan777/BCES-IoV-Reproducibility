#!/usr/bin/env python3
"""Train the mandatory scenario-level SUMO-only label-source ablation."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.ablations import AblationLossWeights, AblationSurfaceNetLite, ablation_normal_tensor, ablation_surface_loss
from bces.models.calibration import CalibrationTarget, calibration_curve, quantize_numpy, select_shrinkage
from bces.models.sumo_dataset import SumoPairDataset
from bces.models.surfacenet import SurfaceNetLite, trainable_parameter_count
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/sumo_source_ablation_v1.yaml"


def _seed(value: int) -> None:
    random.seed(value), np.random.seed(value), torch.manual_seed(value), torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _predict(model: torch.nn.Module, dataset: SumoPairDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    offsets, losses = [], []
    model.eval()
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            batch = _move(raw, device)
            predicted, auxiliary = model(batch)
            offsets.append(predicted.float().cpu().numpy().astype(np.float64))
            losses.append(float(F.binary_cross_entropy_with_logits(auxiliary.float(), batch["valid"].float())))
    return np.concatenate(offsets), np.asarray(losses)


def _metrics(offsets: np.ndarray, dataset: SumoPairDataset, shrinkage: float, guard: float) -> dict:
    normals = ablation_normal_tensor(16).numpy().astype(np.float64)
    calibrated = quantize_numpy(np.maximum(0.0, offsets - shrinkage))
    accepted = (calibrated - dataset.normalized_drift.numpy().astype(np.float64) @ normals.T).min(axis=1) >= guard
    valid = dataset.valid.numpy().astype(bool)
    unsafe = accepted & ~valid
    count = int(accepted.sum())
    return {"total": len(valid), "accepted": count, "unsafe_accepted": int(unsafe.sum()), "coverage": count / len(valid), "unsafe_accept_rate": int(unsafe.sum()) / count if count else None, "valid_coverage": int((accepted & valid).sum()) / max(int(valid.sum()), 1)}


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["test_partition_access"] != "prohibited":
        raise RuntimeError("SUMO source ablation cannot access test")
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("sumo_source_ablation_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("SUMO source ablation requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("SUMO source ablation requires CUDA")
    pair_path, manifest_path = ROOT / config["phase5_pairs"], ROOT / config["phase5_manifest"]
    phase5 = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(pair_path) != phase5["pairs_sha256"]:
        raise ValueError("Phase-5 pair hash mismatch")
    train, validation = SumoPairDataset(pair_path, "train"), SumoPairDataset(pair_path, "validation")
    device = torch.device("cuda")
    _seed(int(config["seed"])), torch.cuda.reset_peak_memory_stats(device)
    model = AblationSurfaceNetLite(16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    weights, normals = AblationLossWeights(), ablation_normal_tensor(16, device=device)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(train, batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        losses = []
        for raw in loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                offsets, auxiliary = model(batch)
                loss = ablation_surface_loss(offsets, auxiliary, batch, normals, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step(), losses.append(float(loss.detach()))
        _, validation_losses = _predict(model, validation, device)
        validation_loss = float(validation_losses.mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_bce": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration = SumoPairDataset(pair_path, "calibration")
    calibration_offsets, _ = _predict(model, calibration, device)
    grid = tuple(np.arange(float(config["calibration_grid_start"]), float(config["calibration_grid_stop"]) + float(config["calibration_grid_step"]) / 2, float(config["calibration_grid_step"])).round(10))
    target = CalibrationTarget(float(config["unsafe_accept_upper_target"]), float(config["confidence"]), int(config["bootstrap_replicates"]), float(config["refresh_guard_band"]), int(config["minimum_accepted"]))
    curve = calibration_curve(calibration_offsets, calibration.normalized_drift.numpy().astype(np.float64), calibration.valid.numpy().astype(bool), calibration.scenario_ids, grid, target, seed=int(config["bootstrap_seed"]))
    selection = select_shrinkage(curve, target)
    validation_offsets, _ = _predict(model, validation, device)
    source_metrics = _metrics(validation_offsets, validation, float(selection["shrinkage"]), float(config["refresh_guard_band"]))
    observational = SurfaceNetLite().to(device)
    observational_checkpoint = ROOT / config["observational_surface_root"] / f"seed_{config['surface_seed']}" / "best.pt"
    observational.load_state_dict(torch.load(observational_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    observational_offsets, _ = _predict(observational, validation, device)
    observational_calibration_path = ROOT / config["observational_calibration"]
    observational_payload = json.loads(observational_calibration_path.read_text(encoding="utf-8"))
    observational_seed = next(row for row in observational_payload["seed_results"] if int(row["seed"]) == int(config["surface_seed"]))
    shrinkages = {int(key): float(value["shrinkage"]) for key, value in observational_seed["selections"].items()}
    behavior = validation.reference_tensors["identifiers"][:, 0].numpy().astype(int)
    adjusted = np.maximum(0.0, observational_offsets - np.asarray([shrinkages[value] for value in behavior])[:, None])
    observational_metrics = _metrics(adjusted, validation, 0.0, float(config["refresh_guard_band"]))
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best.pt"
    torch.save({"model_state_dict": best_state, "seed": int(config["seed"]), "best_epoch": best_epoch}, checkpoint)
    write_json_atomic(output / "history.json", history)
    report = {
        "schema_version": 1, "phase": 8, "component": "label_source_ablation", "status": "PASS",
        "scientific_target_met": False, "underpowered": len(calibration) < int(config["minimum_accepted"]),
        "underpowered_reason": "scenario-level calibration partition cannot supply the registered minimum accepted count",
        "unit_of_inference": config["unit_of_inference"], "train": train.summary, "calibration": calibration.summary, "validation": validation.summary,
        "best_epoch": best_epoch, "best_validation_bce": best_loss, "parameter_count": trainable_parameter_count(model),
        "sumo_only_surface": {"calibration_selection": selection, "validation": source_metrics},
        "observational_only_surface_forced_transfer": observational_metrics,
        "test_partition_accessed": False, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_sha256": sha256_file(checkpoint), "history_sha256": sha256_file(output / "history.json"),
        "config_sha256": sha256_file(CONFIG), "phase5_manifest_sha256": phase5["manifest_sha256"],
        "observational_checkpoint_sha256": sha256_file(observational_checkpoint), "observational_calibration_sha256": sha256_file(observational_calibration_path),
        "git": state, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(output / "run_manifest.json", report)
    print(json.dumps({"status": report["status"], "underpowered": report["underpowered"], "sumo_only_validation": source_metrics, "observational_transfer": observational_metrics, "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
