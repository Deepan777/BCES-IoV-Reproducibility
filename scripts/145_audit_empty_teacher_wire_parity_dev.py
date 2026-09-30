#!/usr/bin/env python3
"""Reconstruct new-model features from exact query/payload bytes on 576 refs."""

from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.study_b import reference_features
from bces.models.bound_policy import reference_query
from bces.models.development_empty_wire_policy import (
    DevelopmentEmptyWirePolicy, raw_reference_vector, raw_wire_features,
)
from bces.oracle.world import WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundQuery, BoundReceiver, decode_response
from bces.protocol.query import Behavior
from bces.simulation.controlled_contract import ego_from_sample, make_message
from bces.utils.hashing import sha256_file

PLAN = ROOT / "docs/STUDY_B_EMPTY_TEACHER_WIRE_PARITY_DEV_V1_PLAN.md"
SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v2"
MODELS = ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_teacher_wire_parity_dev_v1"
SOURCE_SHA256 = "1ddfc3480b01f29df6152feac4f815ed135d711ca9005e2e479e04aaf5126983"
MODELS_SHA256 = "4575c74428dfe115495a2b1f8c0a6058c1096c6c1eb22294487be0d55d2076f6"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher")


def _negative_checks(bound, payload, policy):
    failed_closed = []
    cases = {
        "payload_digest": (bound.encode(), payload + b" "),
        "model_digest": (replace(bound, model_sha256="0" * 64).encode(), payload),
        "family": (replace(bound, family=("scalar_ttl" if bound.family == "surface" else "surface")).encode(), payload),
        "map_context": (replace(bound, map_context_flags=0).encode(), payload),
        "behavior": (replace(bound, query=replace(bound.query, behavior=Behavior.BRAKE)).encode(), payload),
    }
    for name, (query_wire, candidate_payload) in cases.items():
        try:
            policy.predict(query_wire, candidate_payload)
        except ValueError:
            failed_closed.append(name)
        else:
            raise RuntimeError("candidate wire policy accepted tampered " + name)
    return failed_closed


def main():
    if OUTPUT.exists():
        raise RuntimeError("parity audit output exists; preserve first run")
    if (sha256_file(SOURCE / "report.json") != SOURCE_SHA256
            or sha256_file(MODELS / "report.json") != MODELS_SHA256):
        raise RuntimeError("source cohort or model freeze changed")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    model_report = json.loads((MODELS / "report.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json")
                        .read_text(encoding="utf-8"))["config"]
    policies = {}
    for family in FAMILIES:
        checkpoint = MODELS / f"{family}.pt"
        digest = model_report["models"][family]["checkpoint_sha256"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        policies[family] = DevelopmentEmptyWirePolicy(
            checkpoint, expected_sha256=digest, policy_hash=saved["policy_hash"])
    OUTPUT.mkdir(parents=True)
    protocol = {
        "status": "DEVELOPMENT_WIRE_FEATURE_PARITY_AUDIT_V1",
        "plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": sha256_file(
            ROOT / "bces/models/development_empty_wire_policy.py"),
        "source_report_sha256": SOURCE_SHA256,
        "model_report_sha256": MODELS_SHA256,
        "synthetic_view_premise_transmitted": False,
        "independent_confirmation": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    records = []
    negative = None
    for name, digest in sorted(source_report["artifact_sha256"].items()):
        path = SOURCE / name
        if sha256_file(path) != digest:
            raise RuntimeError("raw trace changed")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if len(artifact["references"]) != 1:
            raise RuntimeError("trace reference identity changed")
        reference = artifact["references"][0]
        frame = artifact["frames"]["3000"]
        ego = ego_from_sample(frame["receiver"])
        world = tuple(WorldObject(**item) for item in frame["objects"])
        observed = tuple(item for item in world
                         if math.hypot(item.x_m - ego.x_m, item.y_m - ego.y_m)
                         <= config["cooperative_range_m"])
        message = make_message(observed, 3000, message_id=artifact["seed"])
        payload = message.encode()
        if hashlib.sha256(payload).hexdigest() != reference["frozen_input"]["payload_sha256"]:
            raise RuntimeError("reconstructed sender payload differs")
        query = reference_query(reference, query_id=1,
                                sender_hash=sender_id_hash(message.sender_id))
        batch = reference["frozen_input"]["batch"]
        expected = raw_reference_vector(batch)
        invariant = reference_features(reference)
        results = {}
        for family in FAMILIES:
            policy = policies[family]
            bound = BoundQuery(
                query=query, payload_sha256=hashlib.sha256(payload).hexdigest(),
                model_sha256=policy.sha256, feature_view="frozen_inputs",
                family=policy.family, shrinkage=0.0, margins=(),
                context_available_ms=reference["frozen_input"]["context_available_at_ms"],
                occlusion_proxy=batch["occlusion_proxy"],
                estimated_delay_s=batch["estimated_delay_s"],
                map_context_flags=batch["map_context_flags"],
                generated_ms=3000, payload_received_ms=3000,
            )
            query_wire = bound.encode()
            decoded = BoundQuery.decode(query_wire)
            received = raw_wire_features(decoded, payload)
            if not np.array_equal(expected, received):
                raise RuntimeError("wire-rebuilt raw features differ from saved development reference")
            offsets, origin = policy.predict(query_wire, payload)
            normalized = np.clip((expected - policy.mean) / policy.std, -10, 10).astype(np.float32)
            with torch.no_grad():
                direct_offsets, direct_logits = policy.model(torch.from_numpy(normalized).unsqueeze(0))
            if (not np.array_equal(offsets, direct_offsets[0].numpy().astype(float))
                    or origin != bool(direct_logits[0] >= 0)):
                raise RuntimeError("serialized wire inference differs from direct frozen model")
            response = policy.issue(query_wire, payload)
            extension = decode_response(response, query_wire)
            receiver = BoundReceiver()
            receiver.register(query_wire)
            receiver.install(response, payload)
            if len(extension) != (48 if policy.family == "surface" else 2):
                raise RuntimeError("extension length changed")
            results[family] = {
                "query_bytes": len(query_wire), "response_bytes": len(response),
                "extension_bytes": len(extension), "origin_permitted": origin,
            }
            if negative is None:
                negative = {family: _negative_checks(bound, payload, policy)}
            elif family not in negative:
                negative[family] = _negative_checks(bound, payload, policy)
        records.append({
            "artifact": name, "seed": artifact["seed"],
            "payload_bytes": len(payload),
            "trained_raw_features_equal_wire": True,
            "old_invariant_extractor_equal_raw": bool(np.array_equal(expected, invariant)),
            "methods": results,
        })
    if len(records) != 576 or set(negative) != set(FAMILIES):
        raise RuntimeError("incomplete 576-reference wire audit")
    summary = {
        "references": len(records),
        "wire_raw_feature_equal": sum(row["trained_raw_features_equal_wire"] for row in records),
        "old_invariant_feature_equal_raw": sum(row["old_invariant_extractor_equal_raw"] for row in records),
        "payload_bytes_min_max": [min(row["payload_bytes"] for row in records),
                                  max(row["payload_bytes"] for row in records)],
        "wire_lengths": {
            family: {
                key: sorted({row["methods"][family][key] for row in records})
                for key in ("query_bytes", "response_bytes", "extension_bytes")
            } for family in FAMILIES
        },
        "negative_checks": negative,
        "synthetic_view_premise_wire_bytes": 0,
        "full_communication_accounting_complete": False,
        "source_view_certified": False,
    }
    (OUTPUT / "records.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in records).encode("utf-8")))
    report = {
        "status": protocol["status"], "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "records_sha256": sha256_file(OUTPUT / "records.jsonl.gz"),
        "summary": summary, "independent_confirmation": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
