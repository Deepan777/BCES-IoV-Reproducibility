#!/usr/bin/env python3
"""Independent saved-trajectory geometry replay for gap-shadow pilot."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file


SOURCE = ROOT / "outputs/study_b/gap_shadow_crossing_development_v1"
SOURCE_SHA256 = "82d69726ac32b64a138d40c8d7cba292a091afb0e4eb075378ea8668270b5554"
AUDITOR = ROOT / "scripts/93_audit_crossing_trajectory_geometry_dev.py"
AUDITOR_SHA256 = "612dece0b8ebd4a6199086c19196cb4dc277fb9b01b58be287c229047dc6e416"
OUTPUT = ROOT / "outputs/study_b/gap_shadow_geometry_audit_dev_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("shadow pilot report changed")
    if sha256_file(AUDITOR) != AUDITOR_SHA256:
        raise RuntimeError("geometry auditor changed")
    spec = importlib.util.spec_from_file_location("independent_shadow_geometry", AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import geometry auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if source["branches"] != 32 or len(source["artifact_sha256"]) != 32:
        raise RuntimeError("incomplete shadow cohort")
    rows = []
    for name, expected in sorted(source["artifact_sha256"].items()):
        if sha256_file(SOURCE / name) != expected:
            raise RuntimeError("shadow artifact changed: " + name)
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        rows.append({"source_artifact": name, **module.branch_audit(branch)})
    mismatches = [row["source_artifact"] for row in rows
                  if row["source_sampled_overlap_ticks"] != row["independent_sampled_overlap_ticks"]]
    result = {"status": "GAP_SHADOW_GEOMETRY_REPLAY_DEVELOPMENT_ONLY",
              "source_report_sha256": SOURCE_SHA256,
              "auditor_sha256": AUDITOR_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "sampled_tick_mismatches": mismatches,
              "sampled_positive_branches": sum(bool(row["source_sampled_overlap_ticks"]) for row in rows),
              "interpolated_positive_branches": sum(bool(row["interpolated_overlap_sample_count"])
                                                    for row in rows),
              "interpolated_only": [row["source_artifact"] for row in rows
                                    if row["interpolated_overlap_sample_count"]
                                    and not row["source_sampled_overlap_ticks"]],
              "rows": rows,
              "not_real_world_collision_validation": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("sampled_tick_mismatches",
                      "sampled_positive_branches", "interpolated_positive_branches",
                      "interpolated_only")}, indent=2))


if __name__ == "__main__":
    main()
