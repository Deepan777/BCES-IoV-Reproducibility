#!/usr/bin/env python3
"""Independent post-completion integrity and descriptive cluster audit."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic

MANIFEST = ROOT / "data/manifests/urbaning_labels_v1.json"
AUDIT = ROOT / "outputs/study_b/urbaning_schema_audit_v1.json"
RESULT = ROOT / "outputs/study_b/urbaning_track_proxy_v2/result.json"
OUT = ROOT / "outputs/study_b/urbaning_track_proxy_v2/independent_verification.json"
REFS = (20, 60, 100, 140)
OFFSETS = (5, 10, 15, 20)


def percentile(values, fraction):
    ordered = sorted(values)
    x = (len(ordered) - 1) * fraction
    lower = int(x)
    return ordered[lower] * (1 - (x - lower)) + ordered[min(lower + 1, len(ordered) - 1)] * (x - lower)


def differences(rows):
    n = len(rows)
    if n == 0:
        return None
    acceptance = sum(int(r["accepted"]["surface"]) - int(r["accepted"]["scalar_ttl"]) for r in rows) / n
    adverse = sum(int(r["accepted"]["surface"] and not r["proxy_valid"]) - int(r["accepted"]["scalar_ttl"] and not r["proxy_valid"]) for r in rows) / n
    return acceptance, adverse


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if len(manifest["records"]) != 34 or audit["manifest_sha256"] != sha256_file(MANIFEST):
        raise RuntimeError("label receipt/audit mismatch")
    if result["manifest_sha256"] != sha256_file(MANIFEST) or result["schema_audit_sha256"] != sha256_file(AUDIT):
        raise RuntimeError("result input hashes mismatch")
    if result["adapter_source_sha256"] != sha256_file(ROOT / "scripts/45_run_urbaning_track_proxy_v1.py"):
        raise RuntimeError("frozen runner hash mismatch")
    if result["v2_registration_sha256"] != sha256_file(ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"):
        raise RuntimeError("frozen model registration mismatch")
    for record in manifest["records"]:
        path = ROOT / record["relative_path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"label file changed: {path}")
    expected = {(s["sequence"], receiver["role"], ref, ref + offset)
                for s in audit["sequences"] for receiver in s["receivers"]
                for ref in REFS for offset in OFFSETS}
    actual = [(r["sequence"], r["receiver"], r["reference_index"], r["decision_index"]) for r in result["rows"]]
    if len(actual) != 1088 or len(set(actual)) != 1088 or set(actual) != expected:
        raise RuntimeError("slot denominator/identity mismatch")
    if dict(Counter(r["status"] for r in result["rows"])) != result["status_counts"]:
        raise RuntimeError("reported status counts mismatch")
    available = [r for r in result["rows"] if r["status"] == "proxy_label_available"]
    for method in ("surface", "scalar_ttl"):
        accepted = [r for r in available if r["accepted"][method]]
        expected_summary = {"labelled_slots": len(available), "accepted_slots": len(accepted),
                            "proxy_invalid_accepted": sum(not r["proxy_valid"] for r in accepted),
                            "accepted_sequences": len({r["sequence"] for r in accepted})}
        if result["methods"][method] != expected_summary:
            raise RuntimeError(f"method summary mismatch: {method}")
    groups = defaultdict(list)
    for row in available:
        groups[row["sequence"]].append(row)
    sequences = sorted(groups)
    point = differences(available)
    rng = random.Random(20260923)
    samples = []
    for _ in range(20000):
        draw = [rng.choice(sequences) for _ in sequences]
        rows = [row for seq in draw for row in groups[seq]]
        samples.append(differences(rows))
    interval = {"acceptance_difference": [percentile([x[0] for x in samples], 0.025), percentile([x[0] for x in samples], 0.975)],
                "adverse_acceptance_difference": [percentile([x[1] for x in samples], 0.025), percentile([x[1] for x in samples], 0.975)]}
    report = {"status": "INTEGRITY_VERIFIED_DESCRIPTIVE_ONLY", "source_result_sha256": sha256_file(RESULT),
              "source_manifest_sha256": sha256_file(MANIFEST), "source_audit_sha256": sha256_file(AUDIT),
              "verified_label_files": 34, "verified_candidate_slots": 1088,
              "evaluable_slots": len(available), "evaluable_sequences": len(sequences),
              "invalid_proxy_labels": sum(not r["proxy_valid"] for r in available),
              "both_accept": sum(r["accepted"]["surface"] and r["accepted"]["scalar_ttl"] for r in available),
              "surface_only_accept": sum(r["accepted"]["surface"] and not r["accepted"]["scalar_ttl"] for r in available),
              "ttl_only_accept": sum(not r["accepted"]["surface"] and r["accepted"]["scalar_ttl"] for r in available),
              "acceptance_difference": point[0], "adverse_acceptance_difference": point[1],
              "descriptive_sequence_cluster_bootstrap_95pct": interval,
              "bootstrap_seed": 20260923, "bootstrap_replicates": 20000,
              "no_matched_coverage_or_safety_inference": True}
    write_json_atomic(OUT, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
