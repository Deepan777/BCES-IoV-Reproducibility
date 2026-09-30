#!/usr/bin/env python3
"""Development-only audit of frozen BCES/TTL regions by scenario family."""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.study_b import feature_arrays, load_development
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.models.decision_surface import GUARD, STEP, DecisionSurfaceNet
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic

REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
DEVELOPMENT = ROOT / "outputs/study_b/controlled_development_v2"
OUTPUT = ROOT / "outputs/study_b/straight_geometry_development_audit_v1/result.json"


def load_rule(registration, method, features):
    rule = registration["model_rules"][method]
    checkpoint = ROOT / rule["path"]
    if sha256_file(checkpoint) != rule["sha256"]:
        raise RuntimeError(f"frozen checkpoint mismatch: {method}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["view"] != "reference_margins" or saved["scalar"] != (method == "scalar_ttl"):
        raise RuntimeError("unexpected frozen feature view or geometry")
    model = DecisionSurfaceNet(saved["feature_count"], scalar=saved["scalar"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    x = ((features - saved["mean"].numpy()) / saved["std"].numpy()).clip(-10, 10)
    with torch.no_grad():
        offsets, logits = model(torch.from_numpy(x.astype(np.float32)))
    offsets = np.maximum(0, offsets.numpy() - rule["shrinkage"])
    codes = np.floor(offsets / STEP).clip(0, 255)
    return {"offsets": codes * STEP, "origin": logits.numpy() >= 0,
            "checkpoint_sha256": rule["sha256"], "shrinkage": rule["shrinkage"]}


def main():
    if OUTPUT.exists():
        raise FileExistsError("development geometry audit is immutable")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    references, points, source = load_development(DEVELOPMENT)
    family_by_scenario = {}
    for name in source["artifact_sha256"]:
        with gzip.open(DEVELOPMENT / name, "rt", encoding="utf-8") as handle:
            scene = json.load(handle)
        family_by_scenario[scene["scenario_id"]] = scene["family"]
    arrays = feature_arrays(references, points)
    ids = arrays["reference_ids"]
    index = arrays["reference_index"]
    drift = arrays["drift"]
    valid = np.asarray([p["valid"] for p in points], bool)
    split = np.asarray([p["split"] for p in points])
    family = np.asarray([family_by_scenario[p["scenario_id"]] for p in points])
    normals = np.asarray(NORMAL_CODEBOOK_V1)
    surface = load_rule(registration, "surface", arrays["margin_reference"])
    ttl = load_rule(registration, "scalar_ttl", arrays["margin_reference"])
    surface_slacks = surface["offsets"][index] - drift @ normals.T
    surface_minimum = surface_slacks.min(axis=1)
    surface_limit = surface_slacks.argmin(axis=1)
    accepts_surface = surface["origin"][index] & (surface_minimum >= GUARD)
    ttl_slack = ttl["offsets"][index, 0] - drift[:, 6]
    accepts_ttl = ttl["origin"][index] & (ttl_slack >= GUARD)
    partitions = {}
    for name in ("validation", "calibration"):
        groups = {}
        for key in sorted(set(family[split == name])):
            mask = (split == name) & (family == key)
            s, t, y = accepts_surface[mask], accepts_ttl[mask], valid[mask]
            ttl_only = t & ~s
            limiting = Counter(int(i) for i in surface_limit[mask][ttl_only])
            groups[str(key)] = {
                "points": int(mask.sum()),
                "surface_accepted": int(s.sum()),
                "surface_unsafe_accepted": int((s & ~y).sum()),
                "ttl_accepted": int(t.sum()),
                "ttl_unsafe_accepted": int((t & ~y).sum()),
                "ttl_only_points": int(ttl_only.sum()),
                "ttl_only_valid": int((ttl_only & y).sum()),
                "ttl_only_invalid": int((ttl_only & ~y).sum()),
                "ttl_only_surface_origin_denied": int((ttl_only & ~surface["origin"][index][mask]).sum()),
                "ttl_only_limiting_normal_index": dict(sorted(limiting.items())),
                "surface_reference_origin_fraction": float(np.mean(surface["origin"][np.unique(index[mask])])),
                "ttl_reference_origin_fraction": float(np.mean(ttl["origin"][np.unique(index[mask])])),
            }
        partitions[name] = groups
    report = {"status": "DEVELOPMENT_DIAGNOSTIC_ONLY", "registration_sha256": sha256_file(REGISTRATION),
        "development_manifest_sha256": sha256_file(DEVELOPMENT / "run_manifest.json"),
        "normal_index_note": "indices 0..13 are +/- drift axes; 12 is positive elapsed time; 14..15 are diagonal",
        "surface_rule_sha256": surface["checkpoint_sha256"],
        "ttl_rule_sha256": ttl["checkpoint_sha256"],
        "partitions": partitions,
        "restriction": "No revised method may be chosen using v2 confirmation outcomes; future confirmation requires fresh seeds."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT, report)
    print(json.dumps({part: rows.get("straight_lead_braking") for part, rows in partitions.items()}, indent=2))


if __name__ == "__main__":
    main()
