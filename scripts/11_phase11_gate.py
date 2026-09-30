#!/usr/bin/env python3
"""Verify the immutable Phase-11 paper artifact package end to end."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.artifacts import artifact_hashes, prohibited_claims
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

PACKAGE = ROOT / "outputs/phase11/paper_artifacts_v1"
MANIFEST_PATH = PACKAGE / "artifact_manifest.json"
CONFIG = ROOT / "configs/evaluation/paper_artifacts_v1.yaml"
OUTPUT = ROOT / "outputs/phase11/phase11_gate.json"

REQUIRED = {
    "CLAIM_LEDGER.md",
    "LIMITATIONS.md",
    "RESULTS_SUMMARY.md",
    "figure_ablations.pdf",
    "figure_ablations.png",
    "figure_operating_points.pdf",
    "figure_operating_points.png",
    "figure_risk_coverage.pdf",
    "figure_risk_coverage.png",
    "figure_second_planner.pdf",
    "figure_second_planner.png",
    "figure_sumo_communication.pdf",
    "figure_sumo_communication.png",
    "provenance.json",
    "selected_cases.json",
    "storage_report.json",
    "table_ablations.csv",
    "table_closed_loop_sumo.csv",
    "table_compute_resources.csv",
    "table_failure_taxonomy.csv",
    "table_matched_coverage.csv",
    "table_offline_operating_points.csv",
    "table_primary_statistics.csv",
}


def _self_hash(document: dict[str, object], field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    return canonical_json_hash(payload)


def _csv_has_data(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return len(list(csv.DictReader(handle))) > 0


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("Phase-11 gate output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Phase-11 gate requires a clean committed revision")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    provenance_path = PACKAGE / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    storage = json.loads((PACKAGE / "storage_report.json").read_text(encoding="utf-8"))
    cases = json.loads((PACKAGE / "selected_cases.json").read_text(encoding="utf-8"))
    phase10 = json.loads((ROOT / "outputs/phase10/phase10_gate.json").read_text(encoding="utf-8"))

    actual_hashes = artifact_hashes(PACKAGE, exclude=("artifact_manifest.json",))
    source_hashes = {
        name: sha256_file(ROOT / record["path"])
        for name, record in provenance["sources"].items()
    }
    markdown_text = "\n".join(
        (PACKAGE / name).read_text(encoding="utf-8")
        for name in ("RESULTS_SUMMARY.md", "LIMITATIONS.md", "CLAIM_LEDGER.md")
    )
    png_valid = all((PACKAGE / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for name in REQUIRED if name.endswith(".png"))
    pdf_valid = all((PACKAGE / name).read_bytes()[:5] == b"%PDF-" for name in REQUIRED if name.endswith(".pdf"))
    csv_nonempty = all(_csv_has_data(PACKAGE / name) for name in REQUIRED if name.endswith(".csv"))
    budget = build_budget_report(Budget(workspace_root=ROOT, data_root=ROOT / "data"))

    checks = {
        "artifact_manifest_pass": manifest["status"] == "PASS",
        "registered_scientific_gate_rejected": manifest["scientific_success"] is False and phase10["scientific_success"] is False,
        "exact_required_artifact_set": set(actual_hashes) == REQUIRED and manifest["artifact_count"] == len(REQUIRED),
        "all_artifact_hashes_match": actual_hashes == manifest["artifact_sha256"],
        "manifest_self_hash_matches": _self_hash(manifest, "manifest_sha256") == manifest["manifest_sha256"],
        "provenance_self_hash_matches": _self_hash(provenance, "provenance_sha256") == provenance["provenance_sha256"],
        "source_hashes_match": source_hashes == manifest["source_sha256"] == {name: row["sha256"] for name, row in provenance["sources"].items()},
        "config_and_lock_hashes_match": manifest["config_sha256"] == provenance["config_sha256"] == sha256_file(CONFIG) and provenance["uv_lock_sha256"] == sha256_file(ROOT / "uv.lock"),
        "publication_files_parse": png_valid and pdf_valid and csv_nonempty,
        "case_categories_populated": set(cases) == {"surface_unsafe_accept", "surface_safe_accept", "surface_safe_reject"} and all(cases.values()),
        "claim_language_clean": not prohibited_claims(markdown_text) and not manifest["prohibited_claim_language_found"],
        "programmatic_generation_recorded": manifest["all_values_programmatically_generated"] is True,
        "storage_caps_preserved": storage["status"] == "pass" and budget["measurements"]["output_reserve_preserved"] and budget["measurements"]["workspace_bytes"] < 10_000_000_000,
        "phase10_integrity_gate_pass": phase10["status"] == "PASS" and phase10["engineering_complete"] is True,
    }
    report = {
        "schema_version": 1,
        "phase": 11,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "engineering_complete": all(checks.values()),
        "scientific_success": False,
        "checks": checks,
        "artifact_count": len(actual_hashes),
        "artifact_manifest_sha256": sha256_file(MANIFEST_PATH),
        "artifact_manifest_self_hash": manifest["manifest_sha256"],
        "config_sha256": sha256_file(CONFIG),
        "budget": budget["measurements"],
        "git": state,
        "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
