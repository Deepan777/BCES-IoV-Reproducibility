"""Frozen checkpoint inference from exact received query and payload bytes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch

from bces.evaluation.study_b import reference_features
from bces.geometry.drift import KinematicState, DriftScales
from bces.models.reference_input import freeze_reference_input
from bces.models.decision_surface import DecisionSurfaceNet, encode_conservative_surface
from bces.protocol.bindings import digest32
from bces.protocol.bound_exchange import BoundQuery, encode_response, scalar_extension
from bces.protocol.query import ReceiverQuery, PathPoint
from bces.utils.hashing import sha256_file


def reference_query(reference, *, query_id, sender_hash):
    batch = reference['frozen_input']['batch']
    s, ids = batch['reference_state'], batch['identifiers']
    return ReceiverQuery(query_id, sender_hash, KinematicState(*s[:6], int(s[6])),
        tuple(PathPoint(*p) for p in batch['reference_path']), ids[0], ids[1], ids[2], ids[5],
        DriftScales(*batch['drift_scales']), drift_schema_id=ids[3], normal_codebook_id=ids[4])


def bind_reference(reference, query, payload, *, model_sha256, view, family, shrinkage):
    batch = reference['frozen_input']['batch']
    if hashlib.sha256(payload).hexdigest() != reference['frozen_input']['payload_sha256']:
        raise ValueError('reference payload reconstruction mismatch')
    return BoundQuery(query, hashlib.sha256(payload).hexdigest(), model_sha256, view, family, shrinkage,
        tuple(reference['observable_margin_features']) if view == 'reference_margins' else (),
        reference['frozen_input']['context_available_at_ms'], batch['occlusion_proxy'],
        batch['estimated_delay_s'], batch['map_context_flags'])


def wire_features(bound, payload):
    message = bound.validate_payload(payload)
    frozen = freeze_reference_input(message, bound.query, generated_at_ms=bound.generated_ms,
        query_available_at_ms=bound.generated_ms, context_available_at_ms=bound.context_available_ms,
        query_provenance='received_query', occlusion_proxy=bound.occlusion_proxy,
        estimated_delay_s=bound.estimated_delay_s, map_context_flags=bound.map_context_flags)
    base = reference_features({'frozen_input': frozen.to_dict()})
    return np.concatenate((base, np.asarray(bound.margins, np.float32))) if bound.margins else base


class FrozenWirePolicy:
    def __init__(self, checkpoint: Path, *, expected_sha256: str, policy_hash: int):
        if sha256_file(checkpoint) != expected_sha256:
            raise ValueError('checkpoint hash mismatch')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = DecisionSurfaceNet(saved['feature_count'], scalar=saved['scalar'])
        self.model.load_state_dict(saved['model_state'])
        self.model.eval()
        self.mean, self.std = saved['mean'].numpy(), saved['std'].numpy()
        self.view = saved['view']
        self.family = 'scalar_ttl' if saved['scalar'] else 'surface'
        self.sha256, self.policy_hash = expected_sha256, policy_hash

    def predict(self, query_wire, payload):
        bound = BoundQuery.decode(query_wire)
        if (bound.model_sha256 != self.sha256 or bound.family != self.family
                or bound.feature_view != self.view or bound.query.policy_hash != self.policy_hash):
            raise ValueError('frozen model/policy contract mismatch')
        features = wire_features(bound, payload)
        x = ((features-self.mean)/self.std).clip(-10, 10).astype(np.float32)
        with torch.no_grad():
            offsets, logits = self.model(torch.from_numpy(x).unsqueeze(0))
        return np.maximum(0., offsets[0].numpy().astype(float)-bound.shrinkage), bool(logits[0] >= 0)

    def issue(self, query_wire, payload):
        bound = BoundQuery.decode(query_wire)
        offsets, origin = self.predict(query_wire, payload)
        q = bound.query
        if self.family == 'surface':
            extension = encode_conservative_surface(offsets, origin_permitted=origin, bindings={
                'behavior_id': int(q.behavior), 'query_id': q.query_id,
                'sender_id_hash': q.expected_sender_id_hash, 'payload_digest': digest32(payload),
                'policy_hash': q.policy_hash, 't_ref_ms': q.wire_t_ref_ms,
                'risk_class': q.risk_class, 'calibration_id': q.calibration_id})
        else:
            extension = scalar_extension(float(offsets[0]), origin)
        return encode_response(query_wire, extension)
