#!/usr/bin/env python3
"""Development-only labelled-slot risk audit of margin and lead-only gates.

This evaluates the unchanged frozen surface and TTL checkpoints on all labelled
points from the same 40 validation-partition development scenarios used by the
closed-loop screen. It does not inspect calibration or confirmation labels,
produce a new calibrated rule, or constitute independent evidence.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.study_b import feature_arrays
from bces.models.bound_policy import FrozenWirePolicy
from bces.models.decision_surface import membership_score
from bces.network.adaptive_margin_policy import select_reference_method
from bces.utils.hashing import sha256_file

BASE = ROOT / "outputs/study_b/adaptive_margin_gate_development_v1"
OUTPUT = ROOT / "outputs/study_b/margin_slot_risk_development_v2"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"


def summarize(valid: np.ndarray, accepted: np.ndarray, scenarios: np.ndarray) -> dict:
    count = int(accepted.sum())
    return {"points": len(valid), "accepted": count,
            "invalid_accepted": int((accepted & ~valid).sum()),
            "valid_accepted": int((accepted & valid).sum()),
            "accepted_scenarios": int(len(np.unique(scenarios[accepted]))),
            "invalid_accept_rate": float((accepted & ~valid).sum()/count) if count else None}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if len(base["selected"]) != 40:
        raise RuntimeError("wrong fixed development scenario count")
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    protocol = {
        "status": "FROZEN_DEVELOPMENT_SLOT_AUDIT_ONLY",
        "base_protocol_sha256": sha256_file(BASE / "protocol.json"),
        "v2_registration_sha256": sha256_file(REGISTRATION),
        "adaptive_source_sha256": sha256_file(ROOT / "bces/network/adaptive_margin_policy.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "selected_source_sha256": {row["development_source"]: row["development_source_sha256"] for row in base["selected"]},
        "policies": ["surface", "scalar_ttl", "margin_gate", "lead_only"],
        "estimand": "sampled controlled development-slot acceptance and invalid accepts; not online traffic or independent safety",
        "no_calibration_or_confirmation_labels": True,
        "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    references, points = {}, []
    scenario_family = {row["seed"]: row["family"] for row in base["selected"]}
    for row in base["selected"]:
        path = ROOT / row["development_source"]
        if sha256_file(path) != row["development_source_sha256"]:
            raise RuntimeError("development source changed: " + str(path))
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scene = json.load(handle)
        if scene["split"] != "validation" or int(scene["scenario_id"]) != row["seed"] or scene["family"] != row["family"]:
            raise RuntimeError("development scene identity mismatch")
        for ref in scene["references"]:
            if ref["reference_id"] in references:
                raise RuntimeError("duplicate reference")
            references[ref["reference_id"]] = ref
        points.extend(scene["points"])
    if not points:
        raise RuntimeError("no development points")
    arrays = feature_arrays(references, points)
    ids = arrays["reference_ids"]
    methods, evidence = [], []
    for key in ids:
        selected, observed = select_reference_method(references[key])
        methods.append(selected)
        evidence.append(observed)
    methods = np.asarray(methods)
    lead = np.asarray([item["lead_like"] for item in evidence], dtype=bool)
    gap = np.asarray([item["first_competitor_total_cost_gap"] for item in evidence], dtype=float)
    if not np.isfinite(gap).all() or not np.array_equal(methods == "scalar_ttl", lead & (gap >= 0.20)):
        raise RuntimeError("gate-selection audit mismatch")
    ref_idx = arrays["reference_index"]
    drift = torch.from_numpy(arrays["drift"])
    torch.set_num_threads(2)
    accepts = {}
    for name, rule in registration["model_rules"].items():
        model = FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                 policy_hash=registration["policy_hash"])
        raw = arrays["margin_reference"]
        x = ((raw-model.mean)/model.std).clip(-10, 10).astype(np.float32)
        with torch.no_grad():
            offsets, logits = model.model(torch.from_numpy(x))
            offsets = torch.clamp(offsets-float(rule["shrinkage"]), min=0)
            score = membership_score(offsets[ref_idx], logits[ref_idx], drift,
                                     scalar=(name == "scalar_ttl"))
        accepts[name] = score.numpy() >= 0
    accepts["margin_gate"] = np.where(methods[ref_idx] == "scalar_ttl", accepts["scalar_ttl"], accepts["surface"])
    accepts["lead_only"] = np.where(lead[ref_idx], accepts["scalar_ttl"], accepts["surface"])
    valid = np.asarray([bool(row["valid"]) for row in points])
    scenarios = np.asarray([int(row["scenario_id"]) for row in points])
    family = np.asarray([scenario_family[int(row["scenario_id"])] for row in points])
    low_margin_lead = lead[ref_idx] & (gap[ref_idx] < 0.20)
    margin, lead_only = accepts["margin_gate"], accepts["lead_only"]
    result = {
        "status": "DESCRIPTIVE_DEVELOPMENT_SLOT_AUDIT_ONLY",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "reference_count": len(ids), "scenario_count": len(np.unique(scenarios)),
        "point_count": len(points), "invalid_points": int((~valid).sum()),
        "lead_like_references": int(lead.sum()),
        "lead_like_below_margin_references": int((lead & (gap < 0.20)).sum()),
        "policies": {name: summarize(valid, accepted, scenarios) for name, accepted in accepts.items()},
        "low_margin_lead_stratum": {
            "points": int(low_margin_lead.sum()),
            "invalid_points": int((low_margin_lead & ~valid).sum()),
            "margin_gate": summarize(valid[low_margin_lead], margin[low_margin_lead], scenarios[low_margin_lead]),
            "lead_only": summarize(valid[low_margin_lead], lead_only[low_margin_lead], scenarios[low_margin_lead]),
            "lead_only_only_accepts": int((lead_only & ~margin & low_margin_lead).sum()),
            "lead_only_only_invalid": int((lead_only & ~margin & ~valid & low_margin_lead).sum()),
            "margin_only_accepts": int((margin & ~lead_only & low_margin_lead).sum()),
            "margin_only_invalid": int((margin & ~lead_only & ~valid & low_margin_lead).sum()),
        },
        "by_family": {name: {method: summarize(valid[family == name], accepted[family == name], scenarios[family == name])
                              for method, accepted in accepts.items()}
                      for name in sorted(set(family))},
        "caveats": ["development labels reused in prior model selection", "sampled slots not network or trajectory episodes",
                    "origin/check approximated by exact model shrinkage plus inward quantization and 0.05 guard",
                    "not independent confirmation or a proof of a regret bound"],
        "not_confirmatory": True,
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"references": result["reference_count"], "points": result["point_count"],
                      "policies": result["policies"], "low_margin_lead_stratum": result["low_margin_lead_stratum"]}, indent=2))


if __name__ == "__main__":
    main()
