"""Flexible validity classifier using only inference-available BCES features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .surfacenet import SurfaceNetLite


class ValidityMLPLite(SurfaceNetLite):
    """Unconstrained decision-validity reference with the SurfaceNet backbone."""

    def __init__(self) -> None:
        super().__init__()
        del self.surface_head
        del self.auxiliary_head
        self.validity_head = nn.Sequential(
            nn.Linear(128 + 7, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        latent = self.encode(batch)
        value = torch.cat((latent, batch["normalized_drift"]), dim=1)
        return self.validity_head(value).squeeze(1)


@dataclass(frozen=True)
class ValidityLossWeights:
    positive: float = 1.0
    negative: float = 2.0


def validity_mlp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: ValidityLossWeights,
) -> torch.Tensor:
    labels = labels.to(logits.dtype)
    sample_weights = torch.where(
        labels > 0.5,
        torch.full_like(labels, weights.positive),
        torch.full_like(labels, weights.negative),
    )
    return F.binary_cross_entropy_with_logits(logits, labels, weight=sample_weights)
