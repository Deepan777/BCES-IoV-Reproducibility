import numpy as np
import torch
from bces.models.decision_surface import DecisionSurfaceNet,encode_conservative_surface,inward_quantize,membership_score,CAP
from bces.models.surfacenet import trainable_parameter_count
from bces.protocol.codec import decode_packet,dequantize_offsets
from bces.geometry.surfaces import ExpirySurface


def test_compact_reference_model_and_monotone_quantization():
    model = DecisionSurfaceNet(500)
    assert trainable_parameter_count(model)<250000
    offsets,origin = model(torch.zeros(3,500))
    assert offsets.shape==(3,16) and origin.shape==(3,)
    assert torch.all(inward_quantize(offsets)<=offsets)


def test_wire_inclusion_never_expands_float_region():
    bindings = dict(behavior_id=0,query_id=1,sender_id_hash=2,payload_digest=3,
                    policy_hash=4,t_ref_ms=1000,risk_class=1,calibration_id=1)
    values = np.random.default_rng(29).uniform(0,CAP,16)
    encoded = encode_conservative_surface(values,origin_permitted=True,bindings=bindings)
    assert len(encoded)==48
    decoded = np.array(dequantize_offsets(decode_packet(encoded)))
    assert np.all(decoded<=values)
    abstained = encode_conservative_surface(values,origin_permitted=False,bindings=bindings)
    surface = ExpirySurface(tuple(dequantize_offsets(decode_packet(abstained))))
    for drift in np.random.default_rng(19).normal(size=(100,7)):
        assert surface.minimum_slack(drift)<.05
    assert surface.minimum_slack([0]*7)<.05


def test_origin_abstention_overrides_geometric_membership():
    offsets = torch.ones(2,16)
    scores = membership_score(offsets,torch.tensor([-1.,1.]),torch.zeros(2,7))
    assert scores[0]<0 and scores[1]>0
