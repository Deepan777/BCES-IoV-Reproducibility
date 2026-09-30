#!/usr/bin/env python3
"""Train and calibrate the equal-feature behavior-conditioned scalar-TTL baseline."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.calibration import (
    CalibrationTarget,
    clustered_upper_bound,
    select_shrinkage,
)
from bces.models.dataset import SurfaceDataset
from bces.models.scalar_ttl import ScalarLossWeights, scalar_slack, scalar_ttl_loss
from bces.models.surfacenet import ScalarTTLLite, trainable_parameter_count
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "evaluation" / "scalar_ttl_v1.yaml"


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _metrics(
    model: ScalarTTLLite, loader: DataLoader, device: torch.device,
    weights: ScalarLossWeights, use_amp: bool,
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    examples = valid_total = invalid_total = valid_accept = unsafe_accept = 0
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                ttl, auxiliary = model(batch)
                loss = scalar_ttl_loss(ttl, auxiliary, batch, weights)
            count = len(batch["valid"])
            total_loss += float(loss) * count
            accepted = scalar_slack(ttl.float(), batch["normalized_drift"].float()) >= 0
            valid = batch["valid"] > 0.5
            valid_total += int(valid.sum())
            invalid_total += int((~valid).sum())
            valid_accept += int((accepted & valid).sum())
            unsafe_accept += int((accepted & ~valid).sum())
            examples += count
    accepted_count = valid_accept + unsafe_accept
    return {
        "total_loss": total_loss / examples,
        "unsafe_accept_rate": unsafe_accept / accepted_count if accepted_count else None,
        "invalid_accept_fraction": unsafe_accept / max(invalid_total, 1),
        "valid_coverage": valid_accept / max(valid_total, 1),
        "accepted_count": float(accepted_count), "examples": float(examples),
    }


def _predict(model: ScalarTTLLite, dataset: SurfaceDataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            ttl, _ = model(_move(raw, device))
            rows.append(ttl.cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def _calibrate(
    ttl: np.ndarray, dataset: SurfaceDataset, specification: dict, *, seed: int,
) -> dict:
    target = CalibrationTarget(
        unsafe_accept_upper=float(specification["unsafe_accept_upper_target"]),
        confidence=float(specification["confidence"]),
        bootstrap_replicates=int(specification["bootstrap_replicates"]),
        refresh_guard_band=float(specification["refresh_guard_band"]),
        minimum_accepted=int(specification["minimum_accepted"]),
    )
    grid = tuple(
        np.arange(
            float(specification["grid_start"]),
            float(specification["grid_stop"]) + float(specification["grid_step"]) / 2,
            float(specification["grid_step"]),
        ).round(10)
    )
    ages = dataset.normalized_drift[:, 6].numpy().astype(np.float64)
    valid = dataset.valid.numpy().astype(bool)
    behaviors = dataset.reference_tensors["identifiers"][dataset.reference_indices, 0].numpy().astype(int)
    results = {}
    for behavior in range(5):
        indices = np.flatnonzero(behaviors == behavior)
        curve = []
        for grid_index, shrinkage in enumerate(grid):
            accepted = np.maximum(0.0, ttl[indices] - shrinkage) - ages[indices] >= target.refresh_guard_band
            unsafe = accepted & ~valid[indices]
            accepted_count = int(accepted.sum())
            unsafe_count = int(unsafe.sum())
            clusters = tuple(dataset.scenario_ids[index] for index in indices)
            curve.append(
                {
                    "shrinkage": float(shrinkage), "total": len(indices),
                    "accepted": accepted_count, "unsafe_accepted": unsafe_count,
                    "coverage": accepted_count / len(indices),
                    "unsafe_accept_rate": unsafe_count / accepted_count if accepted_count else None,
                    "unsafe_accept_upper": clustered_upper_bound(
                        accepted, unsafe, clusters, confidence=target.confidence,
                        replicates=target.bootstrap_replicates,
                        seed=seed + 100 * behavior + grid_index,
                    ),
                }
            )
        results[str(behavior)] = {
            "selection": select_shrinkage(curve, target), "curve": curve,
            "cluster_count": len({dataset.scenario_ids[index] for index in indices}),
        }
    return results


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("scalar_ttl_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("scalar baseline requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("scalar baseline training requires CUDA")
    device = torch.device("cuda")
    train = SurfaceDataset(ROOT / config["artifact_root"], "train")
    validation = SurfaceDataset(ROOT / config["artifact_root"], "validation")
    weights = ScalarLossWeights(**{key: float(value) for key, value in config["loss_weights"].items()})
    parameters = trainable_parameter_count(ScalarTTLLite())
    if parameters > int(config["max_parameters"]):
        raise RuntimeError("scalar baseline exceeds parameter budget")
    output.mkdir(parents=True, exist_ok=False)
    seed_results = []
    total_checkpoint_bytes = peak_cuda = 0
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        _seed(seed)
        model = ScalarTTLLite().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler("cuda")
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train, batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0)
        validation_loader = DataLoader(validation, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
        best_key = None
        best_state = None
        best_epoch = 0
        best_metrics = None
        patience = 0
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, int(config["max_epochs"]) + 1):
            model.train()
            for raw in train_loader:
                batch = _move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    ttl, auxiliary = model(batch)
                    loss = scalar_ttl_loss(ttl, auxiliary, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite scalar baseline loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
            metrics = _metrics(model, validation_loader, device, weights, True)
            key = (
                float(metrics["accepted_count"] < int(config["minimum_validation_accepted"])),
                metrics["unsafe_accept_rate"] if metrics["unsafe_accept_rate"] is not None else 1.0,
                -metrics["valid_coverage"], metrics["total_loss"],
            )
            if best_key is None or key < best_key:
                best_key, best_epoch, best_metrics, patience = key, epoch, metrics, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                patience += 1
            print(
                f"scalar seed={seed} epoch={epoch} uar={metrics['unsafe_accept_rate']:.4f} "
                f"coverage={metrics['valid_coverage']:.4f}", flush=True,
            )
            if patience >= int(config["early_stopping_patience"]):
                break
        if best_state is None or best_metrics is None:
            raise RuntimeError("scalar best state missing")
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        best_path, final_path = seed_dir / "best.pt", seed_dir / "final.pt"
        torch.save({"seed": seed, "epoch": best_epoch, "model_state_dict": best_state}, best_path)
        torch.save({"seed": seed, "epoch": epoch, "model_state_dict": model.state_dict()}, final_path)
        checkpoint_bytes = best_path.stat().st_size + final_path.stat().st_size
        total_checkpoint_bytes += checkpoint_bytes
        peak = torch.cuda.max_memory_allocated(device)
        peak_cuda = max(peak_cuda, peak)
        seed_results.append(
            {
                "seed": seed, "epochs": epoch, "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "best_checkpoint_sha256": sha256_file(best_path),
                "final_checkpoint_sha256": sha256_file(final_path),
                "checkpoint_bytes": checkpoint_bytes,
                "peak_cuda_allocated_bytes": peak,
            }
        )
    training_completed_utc = utc_now()
    calibration = SurfaceDataset(ROOT / config["artifact_root"], "calibration")
    for seed_index, result in enumerate(seed_results):
        seed = int(result["seed"])
        model = ScalarTTLLite().to(device)
        checkpoint = torch.load(
            output / f"seed_{seed}" / "best.pt", map_location=device, weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        torch.cuda.reset_peak_memory_stats(device)
        calibration_result = _calibrate(
            _predict(model, calibration, device), calibration,
            config["calibration"],
            seed=int(config["calibration"]["bootstrap_seed"]) + 10000 * seed_index,
        )
        result["calibration"] = calibration_result
        result["calibration_targets_met"] = sum(
            row["selection"]["target_met"] for row in calibration_result.values()
        )
        peak_cuda = max(peak_cuda, torch.cuda.max_memory_allocated(device))
    primary = min(
        seed_results,
        key=lambda row: (
            row["best_metrics"]["unsafe_accept_rate"],
            -row["best_metrics"]["valid_coverage"],
            row["best_metrics"]["total_loss"],
        ),
    )["seed"]
    manifest = {
        "schema_version": 1, "phase": 8, "component": "learned_scalar_ttl",
        "status": "PASS" if peak_cuda <= int(config["max_peak_cuda_bytes"]) and total_checkpoint_bytes <= int(config["max_checkpoint_bytes"]) else "FAIL",
        "parameter_count": parameters, "primary_seed": primary,
        "train_data": train.summary, "validation_data": validation.summary,
        "calibration_data": calibration.summary, "test_partition_accessed": False,
        "training_completed_utc": training_completed_utc,
        "calibration_opened_after_all_training": True,
        "seeds": seed_results, "checkpoint_bytes": total_checkpoint_bytes,
        "peak_cuda_allocated_bytes": peak_cuda, "config_sha256": sha256_file(CONFIG),
        "git": state, "completed_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_json_atomic(output / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "component": manifest["component"],
        "primary_seed": primary, "parameter_count": parameters,
        "checkpoint_bytes": total_checkpoint_bytes,
        "peak_cuda_allocated_bytes": peak_cuda,
        "calibration_targets_met_total": sum(
            row["calibration_targets_met"] for row in seed_results
        ),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
