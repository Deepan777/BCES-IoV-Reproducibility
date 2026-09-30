"""Development-only wire inference for the raw 448-feature empty-view models.

The registered FrozenWirePolicy intentionally uses a different invariant
reference_features transform. These candidate checkpoints were trained on the
raw ModelBatchRow flattening and must not be silently loaded through it.
This adapter verifies bytes/model/query identity; it does not certify that a
source observed an empty physical region or make a neural regret guarantee.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bces.models.bound_policy import FrozenWirePolicy
from bces.models.decision_surface import DecisionSurfaceNet
from bces.models.reference_input import freeze_reference_input
from bces.protocol.bound_exchange import BoundQuery
from bces.protocol.query import Behavior
from bces.utils.hashing import sha256_file

OLD_LABEL_REPORT_SHA256 = "a61c0717767f74f888526cd8e5ac119ee6ecf040421368eef6ad4d27ee88dd2e"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher", "surface_no_teacher", "scalar_ttl_no_teacher")


def raw_reference_vector(batch) -> np.ndarray:
    """Match the frozen development extractor exactly, without future drift."""
    batch = asdict(batch) if not isinstance(batch, dict) else batch
    values = [float(value) for row in batch["object_features"] for value in row]
    values.extend(float(value) for value in batch["object_mask"])
    values.extend(float(value) for value in batch["reference_state"])
    values.extend(float(value) for value in batch["drift_scales"])
    values.extend(float(value) for row in batch["reference_path"] for value in row)
    values.extend(float(value) for value in batch["identifiers"])
    values.extend((float(batch["object_count"]),
                   float(batch["occlusion_proxy"]),
                   float(batch["estimated_delay_s"]),
                   float(batch["map_context_flags"])))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (448,) or not np.isfinite(result).all():
        raise ValueError("raw development reference must have 448 finite features")
    return result


def raw_wire_features(bound: BoundQuery, payload: bytes) -> np.ndarray:
    """Rebuild frozen sender features only from a decoded query and payload."""
    if bound.feature_view != "frozen_inputs" or bound.margins:
        raise ValueError("raw model requires the base frozen-input query")
    message = bound.validate_payload(payload)
    frozen = freeze_reference_input(
        message, bound.query, generated_at_ms=bound.generated_ms,
        query_available_at_ms=bound.generated_ms,
        context_available_at_ms=bound.context_available_ms,
        query_provenance="received_query",
        occlusion_proxy=bound.occlusion_proxy,
        estimated_delay_s=bound.estimated_delay_s,
        map_context_flags=bound.map_context_flags,
    )
    return raw_reference_vector(frozen.batch)


class DevelopmentEmptyWirePolicy(FrozenWirePolicy):
    """Frozen candidate model under existing wire framing, not view attestation."""

    def __init__(self, checkpoint: Path, *, expected_sha256: str, policy_hash: int):
        if sha256_file(checkpoint) != expected_sha256:
            raise ValueError("checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["family"] not in FAMILIES
                or saved["feature_count"] != 448
                or saved["policy_hash"] != policy_hash
                or saved["source_report_sha256"] != OLD_LABEL_REPORT_SHA256
                or saved["scalar"] != saved["family"].startswith("scalar_ttl")):
            raise ValueError("candidate checkpoint contract mismatch")
        self.model = DecisionSurfaceNet(448, scalar=saved["scalar"])
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.mean, self.std = saved["mean"].numpy(), saved["std"].numpy()
        if (self.mean.shape != (448,) or self.std.shape != (448,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError("invalid frozen normalization")
        self.view = "frozen_inputs"
        self.family = "scalar_ttl" if saved["scalar"] else "surface"
        self.training_family = saved["family"]
        self.sha256, self.policy_hash = expected_sha256, policy_hash

    def predict(self, query_wire: bytes, payload: bytes):
        bound = BoundQuery.decode(query_wire)
        if (bound.model_sha256 != self.sha256 or bound.family != self.family
                or bound.feature_view != self.view
                or bound.query.policy_hash != self.policy_hash):
            raise ValueError("candidate model/policy/query binding mismatch")
        if (bound.query.behavior != Behavior.KEEP or bound.query.risk_class != 1
                or bound.query.calibration_id != 0
                or bound.map_context_flags != 1
                or bound.occlusion_proxy != 0.0
                or bound.estimated_delay_s != 0.0):
            raise ValueError("query context outside trained development contract")
        features = raw_wire_features(bound, payload)
        normalized = np.clip((features - self.mean) / self.std, -10, 10).astype(np.float32)
        with torch.no_grad():
            offsets, logits = self.model(torch.from_numpy(normalized).unsqueeze(0))
        return np.maximum(0.0, offsets[0].numpy().astype(float) - bound.shrinkage), bool(logits[0] >= 0)
