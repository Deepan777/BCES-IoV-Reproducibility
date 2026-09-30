"""Compact Study-B reference-conditioned surfaces and feature-matched scalar TTL.

These are empirical models. The conditional regret theorem is not a certificate
for their learned offsets. Inference never consumes a future drift to regenerate
the reference offsets; drift is used only in the membership check.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bces.models.surfacenet import normal_tensor
from bces.protocol.codec import encode_packet,packet_from_offsets

CAP = 65535/32768
STEP = CAP/255
GUARD = .05


class DecisionSurfaceNet(nn.Module):
    def __init__(self,input_features:int,scalar:bool=False):
        super().__init__()
        self.scalar = scalar
        self.encoder = nn.Sequential(nn.Linear(input_features,128),nn.SiLU(),nn.LayerNorm(128),
            nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.offset_head = nn.Linear(64,1 if scalar else 16)
        self.origin_head = nn.Linear(64,1)

    def forward(self,reference):
        latent = self.encoder(reference)
        offsets = F.softplus(self.offset_head(latent)).clamp_max(CAP)
        return offsets,self.origin_head(latent).squeeze(-1)


def inward_quantize(offsets):
    return torch.floor(offsets/STEP).clamp(0,255)*STEP


def membership_score(offsets,origin_logits,drift,*,scalar=False,quantize=True):
    if quantize:
        offsets = inward_quantize(offsets)
    geometric = offsets[:,0]-drift[:,6]-GUARD if scalar else (offsets-drift@normal_tensor(device=drift.device).T).amin(1)-GUARD
    return torch.where(origin_logits>=0,geometric,torch.full_like(geometric,-CAP-GUARD))


def decision_surface_loss(offsets,origin_logits,drift,valid,origin_valid,*,scalar=False,teacher=None):
    quantized = offsets+(inward_quantize(offsets)-offsets).detach()
    slack = quantized[:,0]-drift[:,6]-GUARD if scalar else (quantized-drift@normal_tensor(device=drift.device).T).amin(1)-GUARD
    score = torch.minimum(slack/.05,origin_logits)
    weights = torch.where(valid>.5,1.,4.)
    inclusion = F.binary_cross_entropy_with_logits(score,valid,weight=weights)
    origin = F.binary_cross_entropy_with_logits(origin_logits,origin_valid)
    distillation = offsets.sum()*0 if teacher is None else F.smooth_l1_loss(offsets,teacher)
    return inclusion+origin+.2*distillation


def encode_conservative_surface(offsets,*,origin_permitted:bool,bindings:dict):
    """48-byte existing wire packet; inward quantization never enlarges a region.

    A zero packet plus strictly positive receiver guard band implements abstention.
    Bindings must be supplied from the exact payload and extended query context.
    """
    values = np.asarray(offsets,float)
    if values.shape!=(16,) or not np.isfinite(values).all() or (values<0).any() or (values>CAP).any():
        raise ValueError('16 finite offsets within protocol range required')
    codes = np.floor(values/STEP).clip(0,255).astype(int) if origin_permitted else np.zeros(16,int)
    packet = packet_from_offsets(offsets=[0.]*16,**bindings)
    packet = replace(packet,offset_scale_q15=65535,quantized_offsets=tuple(int(x) for x in codes))
    return encode_packet(packet)
