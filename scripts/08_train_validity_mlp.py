#!/usr/bin/env python3
"""Train and calibrate the flexible MLP validity-gate reference."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.calibration import clustered_upper_bound
from bces.models.dataset import SurfaceDataset
from bces.models.surfacenet import trainable_parameter_count
from bces.models.validity_mlp import (
    ValidityLossWeights,
    ValidityMLPLite,
    validity_mlp_loss,
)
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "evaluation" / "validity_mlp_v1.yaml"


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _predict(
    model: ValidityMLPLite,
    dataset: SurfaceDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            batch = _move(raw, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch)
            scores.append(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64))
            labels.append(batch["valid"].cpu().numpy().astype(bool))
    return np.concatenate(scores), np.concatenate(labels)


def _metrics(
    model: ValidityMLPLite,
    dataset: SurfaceDataset,
    device: torch.device,
    weights: ValidityLossWeights,
    *,
    batch_size: int,
    threshold: float,
) -> dict[str, float | int]:
    scores, valid = _predict(model, dataset, device, batch_size=batch_size)
    accepted = scores >= threshold
    unsafe = accepted & ~valid
    valid_accept = accepted & valid
    accepted_count = int(accepted.sum())
    logits = np.log(np.clip(scores, 1e-7, 1 - 1e-7) / np.clip(1 - scores, 1e-7, 1))
    labels = valid.astype(np.float64)
    sample_weights = np.where(valid, weights.positive, weights.negative)
    loss = np.mean(
        sample_weights
        * (np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits))))
    )
    return {
        "weighted_bce": float(loss),
        "auroc": float(roc_auc_score(valid, scores)),
        "auprc": float(average_precision_score(valid, scores)),
        "unsafe_accept_rate": float(unsafe.sum() / accepted_count) if accepted_count else 1.0,
        "valid_coverage": float(valid_accept.sum() / valid.sum()),
        "accepted_count": accepted_count,
        "valid_accept_count": int(valid_accept.sum()),
        "unsafe_accept_count": int(unsafe.sum()),
        "examples": len(valid),
    }


def _select_threshold(curve: list[dict], specification: dict) -> dict:
    minimum = int(specification["minimum_accepted"])
    target = float(specification["unsafe_accept_upper_target"])
    eligible = [
        row for row in curve
        if row["accepted"] >= minimum and row["unsafe_accept_upper"] <= target
    ]
    if eligible:
        selected = max(eligible, key=lambda row: (row["coverage"], row["threshold"]))
        return {**selected, "target_met": True, "fallback": False}
    nondegenerate = [row for row in curve if row["accepted"] >= minimum]
    pool = nondegenerate or curve
    selected = min(
        pool,
        key=lambda row: (row["unsafe_accept_upper"], -row["coverage"], -row["threshold"]),
    )
    return {**selected, "target_met": False, "fallback": True}


def _calibrate(
    scores: np.ndarray,
    dataset: SurfaceDataset,
    specification: dict,
    *,
    seed: int,
) -> dict:
    valid = dataset.valid.numpy().astype(bool)
    behaviors = (
        dataset.reference_tensors["identifiers"][dataset.reference_indices, 0]
        .numpy().astype(int)
    )
    thresholds = np.arange(
        float(specification["threshold_start"]),
        float(specification["threshold_stop"]) + float(specification["threshold_step"]) / 2,
        float(specification["threshold_step"]),
    ).round(10)
    results = {}
    for behavior in range(5):
        indices = np.flatnonzero(behaviors == behavior)
        clusters = tuple(dataset.scenario_ids[index] for index in indices)
        curve = []
        for threshold_index, threshold in enumerate(thresholds):
            accepted = scores[indices] >= threshold
            unsafe = accepted & ~valid[indices]
            accepted_count = int(accepted.sum())
            unsafe_count = int(unsafe.sum())
            curve.append(
                {
                    "threshold": float(threshold),
                    "total": len(indices),
                    "accepted": accepted_count,
                    "unsafe_accepted": unsafe_count,
                    "coverage": accepted_count / len(indices),
                    "unsafe_accept_rate": (
                        unsafe_count / accepted_count if accepted_count else None
                    ),
                    "unsafe_accept_upper": clustered_upper_bound(
                        accepted,
                        unsafe,
                        clusters,
                        confidence=float(specification["confidence"]),
                        replicates=int(specification["bootstrap_replicates"]),
                        seed=seed + 100 * behavior + threshold_index,
                    ),
                }
            )
        results[str(behavior)] = {
            "selection": _select_threshold(curve, specification),
            "curve": curve,
            "cluster_count": len(set(clusters)),
        }
    return results


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("validity_mlp_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("validity MLP requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("validity MLP training requires CUDA")
    device = torch.device("cuda")
    train = SurfaceDataset(ROOT / config["artifact_root"], "train")
    validation = SurfaceDataset(ROOT / config["artifact_root"], "validation")
    weights = ValidityLossWeights(
        **{key: float(value) for key, value in config["loss_weights"].items()}
    )
    parameter_count = trainable_parameter_count(ValidityMLPLite())
    if parameter_count > int(config["max_parameters"]):
        raise RuntimeError("validity MLP exceeds parameter budget")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    total_checkpoint_bytes = peak_cuda = 0
    batch_size = int(config["batch_size"])
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        _seed(seed)
        model = ValidityMLPLite().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler("cuda")
        train_loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
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
                    logits = model(batch)
                    loss = validity_mlp_loss(logits, batch["valid"], weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validity MLP loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
            metrics = _metrics(
                model,
                validation,
                device,
                weights,
                batch_size=batch_size,
                threshold=float(config["decision_threshold_for_monitoring"]),
            )
            key = (-metrics["auprc"], -metrics["auroc"], metrics["weighted_bce"])
            if best_key is None or key < best_key:
                best_key, best_epoch, best_metrics, patience = key, epoch, metrics, 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                patience += 1
            print(
                f"mlp seed={seed} epoch={epoch} auprc={metrics['auprc']:.4f} "
                f"auroc={metrics['auroc']:.4f}",
                flush=True,
            )
            if patience >= int(config["early_stopping_patience"]):
                break
        if best_state is None or best_metrics is None:
            raise RuntimeError("validity MLP best state missing")
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        best_path, final_path = seed_dir / "best.pt", seed_dir / "final.pt"
        torch.save({"seed": seed, "epoch": best_epoch, "model_state_dict": best_state}, best_path)
        torch.save({"seed": seed, "epoch": epoch, "model_state_dict": model.state_dict()}, final_path)
        checkpoint_bytes = best_path.stat().st_size + final_path.stat().st_size
        total_checkpoint_bytes += checkpoint_bytes
        peak = torch.cuda.max_memory_allocated(device)
        peak_cuda = max(peak_cuda, peak)
        results.append(
            {
                "seed": seed,
                "epochs": epoch,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "best_checkpoint_sha256": sha256_file(best_path),
                "final_checkpoint_sha256": sha256_file(final_path),
                "checkpoint_bytes": checkpoint_bytes,
                "peak_cuda_allocated_bytes": peak,
            }
        )
    training_completed_utc = utc_now()
    calibration = SurfaceDataset(ROOT / config["artifact_root"], "calibration")
    for seed_index, result in enumerate(results):
        seed = int(result["seed"])
        model = ValidityMLPLite().to(device)
        checkpoint = torch.load(
            output / f"seed_{seed}" / "best.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        scores, _ = _predict(model, calibration, device, batch_size=batch_size)
        calibration_result = _calibrate(
            scores,
            calibration,
            config["calibration"],
            seed=int(config["calibration"]["bootstrap_seed"]) + 10000 * seed_index,
        )
        result["calibration"] = calibration_result
        result["calibration_targets_met"] = sum(
            row["selection"]["target_met"] for row in calibration_result.values()
        )
    primary = max(
        results,
        key=lambda row: (
            row["best_metrics"]["auprc"],
            row["best_metrics"]["auroc"],
            -row["best_metrics"]["weighted_bce"],
        ),
    )["seed"]
    status = (
        "PASS"
        if peak_cuda <= int(config["max_peak_cuda_bytes"])
        and total_checkpoint_bytes <= int(config["max_checkpoint_bytes"])
        else "FAIL"
    )
    manifest = {
        "schema_version": 1,
        "phase": 8,
        "component": "validity_mlp",
        "status": status,
        "parameter_count": parameter_count,
        "primary_seed": primary,
        "train_data": train.summary,
        "validation_data": validation.summary,
        "calibration_data": calibration.summary,
        "training_completed_utc": training_completed_utc,
        "calibration_opened_after_all_training": True,
        "test_partition_accessed": False,
        "seeds": results,
        "checkpoint_bytes": total_checkpoint_bytes,
        "peak_cuda_allocated_bytes": peak_cuda,
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
                "primary_seed": primary,
                "parameter_count": parameter_count,
                "checkpoint_bytes": total_checkpoint_bytes,
                "peak_cuda_allocated_bytes": peak_cuda,
                "calibration_targets_met_total": sum(
                    row["calibration_targets_met"] for row in results
                ),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
