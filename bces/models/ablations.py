"""Registered lightweight SurfaceNet ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .losses import torch_quantize_offsets
from .surfacenet import MAX_PROTOCOL_OFFSET, SurfaceNetLite, normal_tensor

NORMAL_COUNTS = (8, 12, 16, 24, 32)


def ablation_normal_tensor(
    count: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Return a frozen centrally symmetric normal codebook."""
    if count not in NORMAL_COUNTS:
        raise ValueError(f"normal count must be one of {NORMAL_COUNTS}")
    if count == 16:
        return normal_tensor(device=device)
    generator = torch.Generator(device="cpu").manual_seed(7100 + count)
    half = torch.randn((count // 2, 7), generator=generator)
    half = F.normalize(half, dim=1)
    return torch.cat((half, -half), dim=0).to(device=device)


class AblationSurfaceNetLite(SurfaceNetLite):
    """SurfaceNet with a configurable output codebook and feature removals."""

    def __init__(
        self,
        normal_count: int,
        *,
        remove_behavior: bool = False,
        remove_objects: bool = False,
    ) -> None:
        super().__init__()
        if normal_count not in NORMAL_COUNTS:
            raise ValueError("unsupported normal count")
        del self.surface_head
        self.surface_head = nn.Linear(128, normal_count)
        self.normal_count = normal_count
        self.remove_behavior = remove_behavior
        self.remove_objects = remove_objects

    def _masked_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.remove_behavior and not self.remove_objects:
            return batch
        value = dict(batch)
        if self.remove_behavior:
            identifiers = batch["identifiers"].clone()
            identifiers[:, 0] = 0
            value["identifiers"] = identifiers
        if self.remove_objects:
            value["object_features"] = torch.zeros_like(batch["object_features"])
            value["object_mask"] = torch.zeros_like(batch["object_mask"])
            value["object_count"] = torch.zeros_like(batch["object_count"])
            value["occlusion_proxy"] = torch.zeros_like(batch["occlusion_proxy"])
        return value

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(self._masked_batch(batch))
        offsets = F.softplus(self.surface_head(latent)).clamp_max(MAX_PROTOCOL_OFFSET)
        return offsets, self.auxiliary_head(latent).squeeze(1)


@dataclass(frozen=True)
class AblationLossWeights:
    unsafe: float = 10.0
    validity: float = 1.0
    quantization: float = 0.1
    compactness: float = 0.01
    auxiliary: float = 0.25
    unsafe_margin: float = 0.02
    validity_temperature: float = 0.10


def ablation_surface_loss(
    offsets: torch.Tensor,
    auxiliary: torch.Tensor,
    batch: dict[str, torch.Tensor],
    normals: torch.Tensor,
    weights: AblationLossWeights,
) -> torch.Tensor:
    minimum_slack = (offsets - batch["normalized_drift"] @ normals.T).amin(dim=1)
    labels = batch["valid"].to(offsets.dtype)
    invalid = labels < 0.5
    unsafe = (
        F.relu(minimum_slack[invalid] + weights.unsafe_margin).mean()
        if invalid.any()
        else offsets.sum() * 0.0
    )
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    sample_weights = torch.where(
        labels > 0.5,
        labels.numel() / (2.0 * positive),
        labels.numel() / (2.0 * negative),
    )
    validity = F.binary_cross_entropy_with_logits(
        minimum_slack / weights.validity_temperature,
        labels,
        weight=sample_weights,
    )
    auxiliary_loss = F.binary_cross_entropy_with_logits(
        auxiliary, labels, weight=sample_weights
    )
    quantization = F.mse_loss(offsets, torch_quantize_offsets(offsets.detach()))
    return (
        weights.unsafe * unsafe
        + weights.validity * validity
        + weights.quantization * quantization
        + weights.compactness * offsets.mean()
        + weights.auxiliary * auxiliary_loss
    )
