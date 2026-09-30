"""Loss and metrics for the strongest learned scalar-TTL baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ScalarLossWeights:
    unsafe: float = 10.0
    validity: float = 1.0
    compactness: float = 0.01
    auxiliary: float = 0.25
    unsafe_margin: float = 0.02
    validity_temperature: float = 0.10


def scalar_slack(ttl: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
    return ttl - drift[:, 6]


def scalar_ttl_loss(
    ttl: torch.Tensor, auxiliary: torch.Tensor,
    batch: dict[str, torch.Tensor], weights: ScalarLossWeights,
) -> torch.Tensor:
    slack = scalar_slack(ttl, batch["normalized_drift"])
    labels = batch["valid"].to(ttl.dtype)
    invalid = labels < 0.5
    unsafe = F.relu(slack[invalid] + weights.unsafe_margin).mean() if invalid.any() else ttl.sum() * 0
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    sample_weights = torch.where(
        labels > 0.5, labels.numel() / (2 * positive), labels.numel() / (2 * negative)
    )
    validity = F.binary_cross_entropy_with_logits(
        slack / weights.validity_temperature, labels, weight=sample_weights
    )
    auxiliary_loss = F.binary_cross_entropy_with_logits(auxiliary, labels, weight=sample_weights)
    return (
        weights.unsafe * unsafe + weights.validity * validity
        + weights.compactness * ttl.mean() + weights.auxiliary * auxiliary_loss
    )
