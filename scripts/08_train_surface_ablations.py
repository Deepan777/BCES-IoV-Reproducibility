#!/usr/bin/env python3
"""Train and calibrate the registered lightweight SurfaceNet ablation matrix."""

from __future__ import annotations

import argparse
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

from bces.models.ablations import (
    AblationLossWeights,
    AblationSurfaceNetLite,
    ablation_normal_tensor,
    ablation_surface_loss,
)
from bces.models.calibration import (
    CalibrationTarget,
    calibration_curve,
    select_shrinkage,
)
from bces.models.dataset import SurfaceDataset
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/surface_ablation_matrix_v1.yaml"
INTERRUPTED_TRAINING_COMMIT = "58d6a78203ca6944631bc716e3a73e01d2e52789"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _model(specification: dict, device: torch.device) -> AblationSurfaceNetLite:
    return AblationSurfaceNetLite(
        int(specification["normal_count"]),
        remove_behavior=bool(specification.get("remove_behavior", False)),
        remove_objects=bool(specification.get("remove_objects", False)),
    ).to(device)


def _weights(config: dict, specification: dict) -> AblationLossWeights:
    values = {
        key: float(value) for key, value in config["loss_weights"].items()
    }
    for key, value in specification.get("loss_overrides", {}).items():
        values[key] = float(value)
    return AblationLossWeights(**values)


def _metrics(
    model: AblationSurfaceNetLite,
    loader: DataLoader,
    device: torch.device,
    weights: AblationLossWeights,
    normals: torch.Tensor,
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    examples = valid_total = invalid_total = valid_accept = unsafe_accept = 0
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                offsets, auxiliary = model(batch)
                loss = ablation_surface_loss(
                    offsets, auxiliary, batch, normals, weights
                )
            count = len(batch["valid"])
            total_loss += float(loss) * count
            minimum = (
                offsets.float()
                - batch["normalized_drift"].float() @ normals.float().T
            ).amin(dim=1)
            accepted = minimum >= 0
            valid = batch["valid"] > 0.5
            valid_accept += int((accepted & valid).sum())
            unsafe_accept += int((accepted & ~valid).sum())
            valid_total += int(valid.sum())
            invalid_total += int((~valid).sum())
            examples += count
    accepted_count = valid_accept + unsafe_accept
    return {
        "total_loss": total_loss / examples,
        "unsafe_accept_rate": (
            unsafe_accept / accepted_count if accepted_count else None
        ),
        "invalid_accept_fraction": unsafe_accept / max(invalid_total, 1),
        "valid_coverage": valid_accept / max(valid_total, 1),
        "accepted_count": float(accepted_count),
        "examples": float(examples),
    }


def _predict(
    model: AblationSurfaceNetLite,
    dataset: SurfaceDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        for raw in loader:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                offsets, _ = model(_move(raw, device))
            rows.append(offsets.float().cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def _calibrate(
    offsets: np.ndarray,
    dataset: SurfaceDataset,
    normals: np.ndarray,
    specification: dict,
    *,
    seed: int,
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
            float(specification["grid_stop"])
            + float(specification["grid_step"]) / 2,
            float(specification["grid_step"]),
        ).round(10)
    )
    drift = dataset.normalized_drift.numpy().astype(np.float64)
    valid = dataset.valid.numpy().astype(bool)
    behaviors = (
        dataset.reference_tensors["identifiers"][dataset.reference_indices, 0]
        .numpy()
        .astype(int)
    )
    results = {}
    for behavior in range(5):
        indices = np.flatnonzero(behaviors == behavior)
        clusters = tuple(dataset.scenario_ids[index] for index in indices)
        curve = calibration_curve(
            offsets[indices],
            drift[indices],
            valid[indices],
            clusters,
            grid,
            target,
            seed=seed + 1000 * behavior,
            normals=normals,
        )
        results[str(behavior)] = {
            "selection": select_shrinkage(curve, target),
            "curve": curve,
            "cluster_count": len(set(clusters)),
        }
    return results


def _train_variant(
    specification: dict,
    train: SurfaceDataset,
    validation: SurfaceDataset,
    config: dict,
    device: torch.device,
    output: Path,
) -> dict:
    seed = int(config["seed"])
    _seed(seed)
    model = _model(specification, device)
    weights = _weights(config, specification)
    normals = ablation_normal_tensor(
        int(specification["normal_count"]), device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda")
    train_loader = DataLoader(
        train,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    best_key = None
    best_state = best_metrics = None
    best_epoch = patience = 0
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for raw in train_loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                offsets, auxiliary = model(batch)
                loss = ablation_surface_loss(
                    offsets, auxiliary, batch, normals, weights
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss in {specification['id']}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
        metrics = _metrics(
            model, validation_loader, device, weights, normals
        )
        key = (
            float(
                metrics["accepted_count"]
                < int(config["minimum_validation_accepted"])
            ),
            metrics["unsafe_accept_rate"]
            if metrics["unsafe_accept_rate"] is not None
            else 1.0,
            -metrics["valid_coverage"],
            metrics["total_loss"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        print(
            f"ablation={specification['id']} epoch={epoch} "
            f"uar={metrics['unsafe_accept_rate']:.4f} "
            f"coverage={metrics['valid_coverage']:.4f}",
            flush=True,
        )
        if patience >= int(config["early_stopping_patience"]):
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError(f"no best state for {specification['id']}")
    directory = output / specification["id"]
    directory.mkdir()
    checkpoint_path = directory / "best.pt"
    torch.save(
        {
            "variant": specification,
            "seed": seed,
            "epoch": best_epoch,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    return {
        "variant": specification,
        "seed": seed,
        "epochs": epoch,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "parameter_count": trainable_parameter_count(model),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def _recover_variants(
    config: dict,
    validation: SurfaceDataset,
    device: torch.device,
    output: Path,
) -> list[dict]:
    validation_loader = DataLoader(
        validation,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    results = []
    for specification in config["variants"]:
        path = output / specification["id"] / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing interrupted-run checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["variant"] != specification:
            raise RuntimeError(f"variant contract mismatch for {specification['id']}")
        if int(checkpoint["seed"]) != int(config["seed"]):
            raise RuntimeError(f"seed mismatch for {specification['id']}")
        model = _model(specification, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        normals = ablation_normal_tensor(
            int(specification["normal_count"]), device=device
        )
        torch.cuda.reset_peak_memory_stats(device)
        metrics = _metrics(
            model,
            validation_loader,
            device,
            _weights(config, specification),
            normals,
        )
        results.append(
            {
                "variant": specification,
                "seed": int(checkpoint["seed"]),
                "epochs": None,
                "best_epoch": int(checkpoint["epoch"]),
                "best_metrics": metrics,
                "parameter_count": trainable_parameter_count(model),
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_bytes": path.stat().st_size,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
                "recovered_after_interruption": True,
            }
        )
    return results


def _training_memory_probe(
    train: SurfaceDataset, config: dict, device: torch.device
) -> int:
    specification = {"id": "memory_probe", "normal_count": 32}
    model = _model(specification, device)
    weights = _weights(config, specification)
    normals = ablation_normal_tensor(32, device=device)
    optimizer = torch.optim.AdamW(model.parameters())
    batch = _move(
        next(
            iter(
                DataLoader(
                    train,
                    batch_size=int(config["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                )
            )
        ),
        device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        offsets, auxiliary = model(batch)
        loss = ablation_surface_loss(
            offsets, auxiliary, batch, normals, weights
        )
    loss.backward()
    optimizer.step()
    return int(torch.cuda.max_memory_allocated(device))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="finish calibration from the complete frozen checkpoint set",
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists() and not args.recover_interrupted:
        raise FileExistsError("surface_ablation_matrix_v1 output is immutable")
    if args.recover_interrupted and not output.is_dir():
        raise FileNotFoundError("no interrupted ablation output to recover")
    if (output / "run_manifest.json").exists():
        raise FileExistsError("completed ablation output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("ablation training requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("ablation training requires CUDA")
    device = torch.device("cuda")
    artifact_root = ROOT / config["artifact_root"]
    train = SurfaceDataset(artifact_root, "train")
    validation = SurfaceDataset(artifact_root, "validation")
    if args.recover_interrupted:
        results = _recover_variants(config, validation, device, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        results = [
            _train_variant(
                specification, train, validation, config, device, output
            )
            for specification in config["variants"]
        ]
    training_completed_utc = utc_now()
    calibration = SurfaceDataset(artifact_root, "calibration")
    batch_size = int(config["batch_size"])
    for index, result in enumerate(results):
        specification = result["variant"]
        model = _model(specification, device)
        checkpoint = torch.load(
            output / specification["id"] / "best.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        normals = (
            ablation_normal_tensor(int(specification["normal_count"]))
            .numpy()
            .astype(np.float64)
        )
        calibration_result = _calibrate(
            _predict(model, calibration, device, batch_size),
            calibration,
            normals,
            config["calibration"],
            seed=int(config["calibration"]["bootstrap_seed"]) + 10000 * index,
        )
        result["calibration"] = calibration_result
        result["calibration_targets_met"] = sum(
            row["selection"]["target_met"]
            for row in calibration_result.values()
        )
    checkpoint_bytes = sum(row["checkpoint_bytes"] for row in results)
    peak_cuda = max(row["peak_cuda_allocated_bytes"] for row in results)
    recovery_training_probe = (
        _training_memory_probe(train, config, device)
        if args.recover_interrupted
        else None
    )
    if recovery_training_probe is not None:
        peak_cuda = max(peak_cuda, recovery_training_probe)
    parameters_ok = all(
        row["parameter_count"] <= int(config["max_parameters"])
        for row in results
    )
    status = (
        "PASS"
        if parameters_ok
        and checkpoint_bytes <= int(config["max_checkpoint_bytes"])
        and peak_cuda <= int(config["max_peak_cuda_bytes"])
        else "FAIL"
    )
    manifest = {
        "schema_version": 1,
        "phase": 8,
        "component": "surface_ablation_matrix",
        "status": status,
        "train_data": train.summary,
        "validation_data": validation.summary,
        "calibration_data": calibration.summary,
        "training_completed_utc": training_completed_utc,
        "recovered_interrupted_run": args.recover_interrupted,
        "training_git_commit": (
            INTERRUPTED_TRAINING_COMMIT
            if args.recover_interrupted
            else state["commit"]
        ),
        "calibration_opened_after_all_training": True,
        "test_partition_accessed": False,
        "variants": results,
        "checkpoint_bytes": checkpoint_bytes,
        "peak_cuda_allocated_bytes": peak_cuda,
        "peak_cuda_measurement_scope": (
            "equivalent_k32_training_step_and_recovery_inference"
            if args.recover_interrupted
            else "full_original_training"
        ),
        "recovery_training_probe_bytes": recovery_training_probe,
        "config_sha256": sha256_file(CONFIG),
        "git": state,
        "completed_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_json_atomic(output / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": status,
                "variant_count": len(results),
                "checkpoint_bytes": checkpoint_bytes,
                "peak_cuda_allocated_bytes": peak_cuda,
                "targets_met": {
                    row["variant"]["id"]: row["calibration_targets_met"]
                    for row in results
                },
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
