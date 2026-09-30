"""Registered SurfaceNet boundary, safety, validity, codec, and compactness losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .surfacenet import normal_tensor


@dataclass(frozen=True)
class LossWeights:
    boundary: float = 1.0
    unsafe: float = 10.0
    validity: float = 1.0
    quantization: float = 0.1
    compactness: float = 0.01
    auxiliary: float = 0.25
    unsafe_margin: float = 0.02
    validity_temperature: float = 0.10


def torch_quantize_offsets(offsets: torch.Tensor) -> torch.Tensor:
    """Differentiable loss target matching the unsigned-Q15/uint8 codec."""
    scale_q15 = torch.ceil(offsets.amax(dim=1, keepdim=True) * 32768.0).clamp(1, 65535)
    scale = scale_q15 / 32768.0
    quantized = torch.floor(255.0 * offsets / scale + 0.5).clamp(0, 255)
    return scale * quantized / 255.0


def surface_slack(offsets: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
    normals = normal_tensor(device=offsets.device).to(offsets.dtype)
    return offsets - drift @ normals.T


def surfacenet_loss(
    offsets: torch.Tensor,
    auxiliary_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    boundary_mask = batch["boundary_mask"].bool()
    if boundary_mask.any():
        boundary = F.huber_loss(
            offsets[boundary_mask], batch["boundary_targets"][boundary_mask], reduction="mean"
        )
    else:
        boundary = offsets.sum() * 0.0
    minimum_slack = surface_slack(offsets, batch["normalized_drift"]).amin(dim=1)
    labels = batch["valid"].to(offsets.dtype)
    invalid = labels < 0.5
    unsafe = (
        F.relu(minimum_slack[invalid] + weights.unsafe_margin).mean()
        if invalid.any() else offsets.sum() * 0.0
    )
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    sample_weights = torch.where(labels > 0.5, labels.numel() / (2.0 * positive), labels.numel() / (2.0 * negative))
    validity = F.binary_cross_entropy_with_logits(
        minimum_slack / weights.validity_temperature, labels, weight=sample_weights
    )
    auxiliary = F.binary_cross_entropy_with_logits(auxiliary_logits, labels, weight=sample_weights)
    dequantized = torch_quantize_offsets(offsets.detach())
    quantization = F.mse_loss(offsets, dequantized)
    compactness = offsets.mean()
    total = (
        weights.boundary * boundary
        + weights.unsafe * unsafe
        + weights.validity * validity
        + weights.quantization * quantization
        + weights.compactness * compactness
        + weights.auxiliary * auxiliary
    )
    return total, {
        "boundary": boundary.detach(), "unsafe": unsafe.detach(),
        "validity": validity.detach(), "quantization": quantization.detach(),
        "compactness": compactness.detach(), "auxiliary": auxiliary.detach(),
        "minimum_slack": minimum_slack.detach(),
    }
