#!/usr/bin/env python3
"""Toy-overfit gate and five sequential SurfaceNet-Lite training seeds."""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.dataset import SurfaceDataset
from bces.models.losses import LossWeights, surface_slack, surfacenet_loss
from bces.models.surfacenet import SurfaceNetLite, trainable_parameter_count
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "training" / "surfacenet_lite_v2.yaml"


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _metrics(
    model: SurfaceNetLite, loader: DataLoader, device: torch.device,
    weights: LossWeights, use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    examples = valid_total = invalid_total = valid_accept = unsafe_accept = 0
    with torch.no_grad():
        for raw in loader:
            batch = _device_batch(raw, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                offsets, auxiliary = model(batch)
                loss, _ = surfacenet_loss(offsets, auxiliary, batch, weights)
            count = len(batch["valid"])
            total_loss += float(loss) * count
            minimum = surface_slack(offsets.float(), batch["normalized_drift"].float()).amin(dim=1)
            accepted = minimum >= 0
            valid = batch["valid"] > 0.5
            valid_total += int(valid.sum())
            invalid_total += int((~valid).sum())
            valid_accept += int((accepted & valid).sum())
            unsafe_accept += int((accepted & ~valid).sum())
            examples += count
    accepted = valid_accept + unsafe_accept
    return {
        "total_loss": total_loss / max(examples, 1),
        "unsafe_accept_rate": unsafe_accept / accepted if accepted else None,
        "invalid_accept_fraction": unsafe_accept / max(invalid_total, 1),
        "valid_coverage": valid_accept / max(valid_total, 1),
        "accepted_count": float(accepted),
        "accuracy": (valid_accept + invalid_total - unsafe_accept) / max(examples, 1),
        "examples": float(examples),
    }


def _toy_gate(
    dataset: SurfaceDataset, config: dict, weights: LossWeights, device: torch.device,
) -> dict:
    toy_config = config["toy"]
    count = int(toy_config["examples"])
    valid_indices = torch.where(dataset.valid > 0.5)[0][: count // 2].tolist()
    invalid_indices = torch.where(dataset.valid < 0.5)[0][: count // 2].tolist()
    subset = Subset(dataset, valid_indices + invalid_indices)
    loader = DataLoader(subset, batch_size=count, shuffle=False)
    batch = _device_batch(next(iter(loader)), device)
    _seed(int(config["seeds"][0]))
    model = SurfaceNetLite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(toy_config["learning_rate"]))
    initial = None
    model.train()
    for _ in range(int(toy_config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        offsets, auxiliary = model(batch)
        loss, _ = surfacenet_loss(offsets, auxiliary, batch, weights)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
    metrics = _metrics(model, loader, device, weights, False)
    final_loss = metrics["total_loss"]
    reduction = (float(initial) - final_loss) / max(float(initial), 1e-12)
    passed = (
        metrics["accuracy"] >= float(toy_config["minimum_accuracy"])
        and reduction >= float(toy_config["minimum_loss_reduction_fraction"])
        and torch.isfinite(torch.tensor(final_loss)).item()
    )
    return {
        "examples": count, "steps": int(toy_config["steps"]),
        "initial_loss": initial, "final_loss": final_loss,
        "loss_reduction_fraction": reduction, **metrics, "passed": bool(passed),
    }


def _train_seed(
    seed: int, train: SurfaceDataset, validation: SurfaceDataset, config: dict,
    weights: LossWeights, device: torch.device, output: Path,
) -> dict:
    _seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
    )
    model = SurfaceNetLite().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        finite = True
        for raw in train_loader:
            batch = _device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                offsets, auxiliary = model(batch)
                loss, _ = surfacenet_loss(offsets, auxiliary, batch, weights)
            if not torch.isfinite(loss):
                finite = False
                break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
        if not finite:
            raise FloatingPointError(f"non-finite training loss for seed {seed}")
        metrics = _metrics(model, validation_loader, device, weights, use_amp)
        metrics["epoch"] = epoch
        history.append(metrics)
        key = (
            float(metrics["accepted_count"] < int(config["minimum_validation_accepted"])),
            metrics["unsafe_accept_rate"] if metrics["unsafe_accept_rate"] is not None else 1.0,
            -metrics["valid_coverage"],
            metrics["total_loss"],
        )
        if best_key is None or key < best_key:
            best_key, best_epoch, patience = key, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        print(
            f"seed={seed} epoch={epoch} val_loss={metrics['total_loss']:.6f} "
            f"uar={metrics['unsafe_accept_rate']:.4f} coverage={metrics['valid_coverage']:.4f}",
            flush=True,
        )
        if patience >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("best state was not selected")
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"seed": seed, "config_sha256": sha256_file(CONFIG), "code_commit": git_state(ROOT)["commit"]}
    best_path, final_path = seed_dir / "best.pt", seed_dir / "final.pt"
    torch.save({**metadata, "epoch": best_epoch, "model_state_dict": best_state}, best_path)
    torch.save({**metadata, "epoch": history[-1]["epoch"], "model_state_dict": model.state_dict()}, final_path)
    history_path = seed_dir / "history.jsonl.gz"
    with gzip.open(history_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    peak = torch.cuda.max_memory_allocated(device) if use_amp else 0
    return {
        "seed": seed, "epochs": len(history), "best_epoch": best_epoch,
        "best_metrics": history[best_epoch - 1], "final_metrics": history[-1],
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "history_sha256": sha256_file(history_path),
        "peak_cuda_allocated_bytes": peak,
        "checkpoint_bytes": best_path.stat().st_size + final_path.stat().st_size,
        "nan_or_inf_detected": False,
    }


def main() -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("surfacenet_v2 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("full training requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("registered Phase 6 training requires CUDA")
    device = torch.device("cuda")
    artifact_root = ROOT / config["artifact_root"]
    train = SurfaceDataset(artifact_root, "train")
    validation = SurfaceDataset(artifact_root, "validation")
    weights = LossWeights(**{key: float(value) for key, value in config["loss_weights"].items()})
    parameter_count = trainable_parameter_count(SurfaceNetLite())
    if parameter_count > int(config["max_parameters"]):
        raise RuntimeError("model exceeds the registered parameter limit")
    output.mkdir(parents=True, exist_ok=False)
    toy = _toy_gate(train, config, weights, device)
    write_json_atomic(output / "toy_overfit.json", toy)
    if not toy["passed"]:
        raise RuntimeError("toy overfit gate failed")
    seed_results = [
        _train_seed(int(seed), train, validation, config, weights, device, output)
        for seed in config["seeds"]
    ]
    checkpoint_bytes = sum(row["checkpoint_bytes"] for row in seed_results)
    peak = max(row["peak_cuda_allocated_bytes"] for row in seed_results)
    phase4_manifest = json.loads((artifact_root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1, "phase": 6,
        "status": "PASS" if checkpoint_bytes <= int(config["max_checkpoint_bytes"]) and peak <= int(config["max_peak_cuda_bytes"]) else "FAIL",
        "architecture": "SurfaceNet-Lite", "parameter_count": parameter_count,
        "config_sha256": sha256_file(CONFIG),
        "phase4_manifest_sha256": phase4_manifest["manifest_sha256"],
        "phase4_artifact_sha256": phase4_manifest["artifact_sha256"],
        "train_data": train.summary, "validation_data": validation.summary,
        "calibration_partition_accessed": False, "test_partition_accessed": False,
        "toy_overfit": toy, "seeds": seed_results,
        "peak_cuda_allocated_bytes": peak, "checkpoint_bytes": checkpoint_bytes,
        "git": state, "completed_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_json_atomic(output / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
