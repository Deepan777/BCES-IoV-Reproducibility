"""SurfaceNet-Lite with the registered fixed-normal 16-offset head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MAX_PROTOCOL_OFFSET = 65535.0 / 32768.0


class ResidualBlock(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.second(F.relu(self.first(value)))
        return self.norm(value + residual)


class SurfaceNetLite(nn.Module):
    """Inference consumes only the exact frozen payload/query batch fields."""

    query_feature_count = 116

    def __init__(self) -> None:
        super().__init__()
        self.object_mlp = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        self.behavior_embedding = nn.Embedding(5, 8)
        self.risk_embedding = nn.Embedding(256, 4)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.query_feature_count, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(112, 128), nn.ReLU())
        self.residual_blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.surface_head = nn.Linear(128, 16)
        self.auxiliary_head = nn.Linear(128, 1)

    @staticmethod
    def _objects(features: torch.Tensor) -> torch.Tensor:
        value = features.clone()
        value[..., 0:2] /= 50.0
        value[..., 2:4] /= 20.0
        value[..., 4:6] /= 10.0
        value[..., 6] /= 3.0
        value[..., 8] /= 2.0
        return value

    def _query(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = batch["reference_state"]
        state_features = torch.stack(
            (
                state[:, 2] / 20.0,
                torch.sin(state[:, 3]),
                torch.cos(state[:, 3]),
                state[:, 4] / 8.0,
                state[:, 5] / 0.3,
            ),
            dim=1,
        )
        scale_features = torch.log1p(batch["drift_scales"])
        path = batch["reference_path"]
        dx = (path[:, :, 0] - state[:, None, 0]) / 50.0
        dy = (path[:, :, 1] - state[:, None, 1]) / 50.0
        heading_delta = path[:, :, 3] - state[:, None, 3]
        path_features = torch.stack(
            (
                dx,
                dy,
                path[:, :, 2] / 20.0,
                torch.sin(heading_delta),
                torch.cos(heading_delta),
                path[:, :, 4] / 0.3,
                path[:, :, 5] / 3.0,
            ),
            dim=2,
        ).flatten(1)
        identifiers = batch["identifiers"]
        behavior = self.behavior_embedding(identifiers[:, 0].long().clamp(0, 4))
        risk = self.risk_embedding(identifiers[:, 1].long().clamp(0, 255))
        bindings = torch.stack(
            (
                identifiers[:, 2] / float(2**32 - 1),
                identifiers[:, 3] / 8.0,
                identifiers[:, 4] / 8.0,
                identifiers[:, 5] / 255.0,
            ),
            dim=1,
        )
        context = torch.stack(
            (
                batch["object_count"] / 32.0,
                batch["occlusion_proxy"],
                batch["estimated_delay_s"] / 4.0,
                batch["map_context_flags"] / 255.0,
            ),
            dim=1,
        )
        value = torch.cat(
            (state_features, scale_features, path_features, behavior, risk, bindings, context),
            dim=1,
        )
        if value.shape[1] != self.query_feature_count:
            raise RuntimeError("query feature contract changed")
        return value

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["object_mask"].bool()
        encoded = self.object_mlp(self._objects(batch["object_features"]))
        masked = encoded * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        mean = masked.sum(dim=1) / denominator
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), negative).amax(dim=1)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        query = self.query_mlp(self._query(batch))
        return self.residual_blocks(self.fusion(torch.cat((mean, maximum, query), dim=1)))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(batch)
        offsets = F.softplus(self.surface_head(latent)).clamp_max(MAX_PROTOCOL_OFFSET)
        return offsets, self.auxiliary_head(latent).squeeze(1)


class ScalarTTLLite(SurfaceNetLite):
    """Strong behavior-conditioned scalar TTL using the identical backbone."""

    def __init__(self) -> None:
        super().__init__()
        del self.surface_head
        self.ttl_head = nn.Linear(128, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(batch)
        ttl = F.softplus(self.ttl_head(latent)).squeeze(1).clamp_max(MAX_PROTOCOL_OFFSET)
        return ttl, self.auxiliary_head(latent).squeeze(1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normal_tensor(*, device: torch.device | None = None) -> torch.Tensor:
    rows = []
    for axis in range(7):
        positive, negative = [0.0] * 7, [0.0] * 7
        positive[axis], negative[axis] = 1.0, -1.0
        rows.extend((positive, negative))
    diagonal = [1.0 / math.sqrt(7.0)] * 7
    rows.extend((diagonal, [-value for value in diagonal]))
    return torch.tensor(rows, dtype=torch.float32, device=device)
