from __future__ import annotations

import pytest
import torch

from bces.models.ablations import (
    NORMAL_COUNTS,
    AblationLossWeights,
    AblationSurfaceNetLite,
    ablation_normal_tensor,
    ablation_surface_loss,
)
from bces.models.surfacenet import trainable_parameter_count


def _batch() -> dict[str, torch.Tensor]:
    size = 2
    return {
        "object_features": torch.randn(size, 32, 10),
        "object_mask": torch.ones(size, 32, dtype=torch.bool),
        "reference_state": torch.randn(size, 7),
        "drift_scales": torch.ones(size, 7),
        "reference_path": torch.randn(size, 12, 6),
        "identifiers": torch.zeros(size, 6),
        "object_count": torch.full((size,), 32.0),
        "occlusion_proxy": torch.ones(size),
        "estimated_delay_s": torch.zeros(size),
        "map_context_flags": torch.zeros(size),
        "normalized_drift": torch.zeros(size, 7),
        "valid": torch.tensor([1.0, 0.0]),
    }


@pytest.mark.parametrize("count", NORMAL_COUNTS)
def test_ablation_normals_are_unit_and_centrally_symmetric(count: int) -> None:
    normals = ablation_normal_tensor(count)
    assert normals.shape == (count, 7)
    assert torch.linalg.vector_norm(normals, dim=1).tolist() == pytest.approx([1.0] * count)
    antipode_error = (normals[:, None, :] + normals[None, :, :]).abs().amax(dim=2)
    assert torch.all(antipode_error.amin(dim=1) < 1e-6)


@pytest.mark.parametrize("count", NORMAL_COUNTS)
def test_ablation_model_shapes_and_budget(count: int) -> None:
    model = AblationSurfaceNetLite(count, remove_behavior=True, remove_objects=True)
    offsets, auxiliary = model(_batch())
    assert offsets.shape == (2, count)
    assert auxiliary.shape == (2,)
    assert trainable_parameter_count(model) <= 250_000
    loss = ablation_surface_loss(
        offsets,
        auxiliary,
        _batch(),
        ablation_normal_tensor(count),
        AblationLossWeights(),
    )
    loss.backward()
    assert torch.isfinite(loss)
