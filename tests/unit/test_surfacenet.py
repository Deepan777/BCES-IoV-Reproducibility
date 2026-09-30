from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bces.models.dataset import SurfaceDataset
from bces.models.losses import LossWeights, surfacenet_loss, torch_quantize_offsets
from bces.models.scalar_ttl import ScalarLossWeights, scalar_ttl_loss
from bces.models.surfacenet import (
    ScalarTTLLite,
    SurfaceNetLite,
    trainable_parameter_count,
)
from bces.models.validity_mlp import (
    ValidityLossWeights,
    ValidityMLPLite,
    validity_mlp_loss,
)
from bces.protocol.codec import quantize_offsets

pytestmark = pytest.mark.phase6


def _batch(size: int = 3) -> dict[str, torch.Tensor]:
    mask = torch.zeros(size, 32, dtype=torch.bool)
    mask[:, :2] = True
    return {
        "object_features": torch.randn(size, 32, 10),
        "object_mask": mask,
        "reference_state": torch.tensor([[1.0, 2.0, 8.0, 0.2, 0.1, 0.01, 1000.0]]).repeat(size, 1),
        "drift_scales": torch.ones(size, 7),
        "reference_path": torch.randn(size, 12, 6),
        "identifiers": torch.tensor([[0.0, 1.0, 80693948.0, 1.0, 1.0, 0.0]]).repeat(size, 1),
        "object_count": torch.full((size,), 2.0),
        "occlusion_proxy": torch.zeros(size),
        "estimated_delay_s": torch.zeros(size),
        "map_context_flags": torch.zeros(size),
        "normalized_drift": torch.randn(size, 7) * 0.1,
        "valid": torch.tensor(([1.0, 0.0, 1.0] * size)[:size]),
        "boundary_targets": torch.full((size, 16), 0.5),
        "boundary_mask": torch.ones(size, 16, dtype=torch.bool),
    }


def test_registered_architecture_is_below_parameter_limit() -> None:
    assert trainable_parameter_count(SurfaceNetLite()) < 250_000


def test_forward_produces_finite_positive_protocol_offsets() -> None:
    offsets, auxiliary = SurfaceNetLite()(_batch())
    assert offsets.shape == (3, 16)
    assert auxiliary.shape == (3,)
    assert torch.isfinite(offsets).all() and torch.all(offsets >= 0)
    assert torch.all(offsets <= 65535 / 32768)


def test_registered_loss_is_finite_and_backpropagates() -> None:
    model = SurfaceNetLite()
    offsets, auxiliary = model(_batch())
    loss, components = surfacenet_loss(offsets, auxiliary, _batch(), LossWeights())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in components.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_torch_quantization_matches_real_codec() -> None:
    offsets = torch.tensor(
        [[0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]],
        dtype=torch.float32,
    )
    actual = torch_quantize_offsets(offsets)[0].tolist()
    expected = quantize_offsets(offsets[0].tolist()).dequantized_offsets
    assert actual == pytest.approx(expected, abs=1e-7)


def test_test_partition_is_locked_before_any_file_access(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="locked"):
        SurfaceDataset(tmp_path, "test")


def test_scalar_ttl_baseline_uses_same_features_and_budget() -> None:
    model = ScalarTTLLite()
    ttl, auxiliary = model(_batch())
    loss = scalar_ttl_loss(ttl, auxiliary, _batch(), ScalarLossWeights())
    loss.backward()
    assert ttl.shape == (3,)
    assert torch.isfinite(loss)
    assert trainable_parameter_count(model) <= trainable_parameter_count(SurfaceNetLite())


def test_validity_mlp_uses_drift_and_stays_within_budget() -> None:
    model = ValidityMLPLite()
    batch = _batch()
    logits = model(batch)
    loss = validity_mlp_loss(logits, batch["valid"], ValidityLossWeights())
    loss.backward()
    assert logits.shape == (3,)
    assert torch.isfinite(loss)
    assert trainable_parameter_count(model) <= 250_000
