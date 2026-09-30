#!/usr/bin/env python3
"""Independent rectangle-geometry replay for the saved BCES/TTL branches."""

from __future__ import annotations

from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file


SOURCE = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_development_v1"
SOURCE_REPORT_SHA256 = "97efa491660881d7c712c1a3fb480cb64e34fd2f88d9d0ad43b5f78197ee1197"
GEOMETRY = ROOT / "scripts/93_audit_crossing_trajectory_geometry_dev.py"
GEOMETRY_SHA256 = "612dece0b8ebd4a6199086c19196cb4dc277fb9b01b58be287c229047dc6e416"
OUTPUT = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_geometry_audit_dev_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_REPORT_SHA256:
        raise RuntimeError("policy screen report changed")
    if sha256_file(GEOMETRY) != GEOMETRY_SHA256:
        raise RuntimeError("independent geometry auditor changed")
    spec = importlib.util.spec_from_file_location("independent_crossing_geometry", GEOMETRY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import geometry auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if source["branches"] != 64 or len(source["artifact_sha256"]) != 64:
        raise RuntimeError("wrong source branch cohort")
    rows = []
    for name, expected in sorted(source["artifact_sha256"].items()):
        if sha256_file(SOURCE / name) != expected:
            raise RuntimeError("policy branch changed: " + name)
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        row = module.branch_audit(branch)
        rows.append({"source_artifact": name, "method": branch["method"], **row})
    mismatches = [row["source_artifact"] for row in rows
                  if row["source_sampled_overlap_ticks"] != row["independent_sampled_overlap_ticks"]]
    counts = {}
    for method in ("surface", "scalar_ttl"):
        subset = [row for row in rows if row["method"] == method]
        counts[method] = {"branches": len(subset),
                          "sampled_positive": sum(bool(row["source_sampled_overlap_ticks"]) for row in subset),
                          "independent_sampled_positive": sum(bool(row["independent_sampled_overlap_ticks"])
                                                              for row in subset),
                          "interpolated_positive": sum(bool(row["interpolated_overlap_sample_count"])
                                                       for row in subset),
                          "interpolated_only": [row["source_artifact"] for row in subset
                                                if row["interpolated_overlap_sample_count"]
                                                and not row["source_sampled_overlap_ticks"]]}
    result = {"status": "POLICY_CROSSING_GEOMETRY_REPLAY_DEVELOPMENT_ONLY",
              "source_report_sha256": SOURCE_REPORT_SHA256,
              "geometry_script_sha256": GEOMETRY_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "interpolation_step_ms": 20,
              "sampled_tick_mismatches": mismatches,
              "by_method": counts, "rows": rows,
              "not_real_world_collision_validation": True,
              "not_independent_confirmation": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sampled_tick_mismatches": mismatches,
                      "by_method": counts}, indent=2))


if __name__ == "__main__":
    main()
