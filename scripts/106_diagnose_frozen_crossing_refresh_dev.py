#!/usr/bin/env python3
"""Post hoc, descriptive refresh/visibility audit of the opened crossing run."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file


SOURCE = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_development_v1"
SOURCE_SHA256 = "97efa491660881d7c712c1a3fb480cb64e34fd2f88d9d0ad43b5f78197ee1197"
OUTPUT = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_refresh_diagnostic_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("opened development report changed")
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    observations = []
    for row in source["rows"]:
        if row["timing"] != "near" or row["periodic_severe"] or not row["local_severe"]:
            continue
        name = f'{row["seed"]}_near_{row["condition"]}_{row["method"]}.json.gz'
        if sha256_file(SOURCE / name) != source["artifact_sha256"][name]:
            raise RuntimeError("branch changed: " + name)
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        visibility = {point["timestamp_ms"]: point for point in branch["visibility_audit"]}
        overlaps = [point for point in branch["rows"]
                    if point.get("events", {}).get("geometric_overlap_ids")]
        first = overlaps[0] if overlaps else None
        session = branch["traffic"]["session_counts"]
        observations.append({"seed": row["seed"], "condition": row["condition"],
                             "method": row["method"], "severe": row["policy_severe"],
                             "reuse_decisions": branch["counts"].get("reuse_decisions", 0),
                             "refresh_decisions": session.get("decision:REFRESH", 0),
                             "installed_responses": session.get("installed_responses", 0),
                             "generated_packets": branch["traffic"]["generated_packets"],
                             "generated_bytes": branch["traffic"]["generated_bytes"],
                             "first_sampled_overlap_ms": first["timestamp_ms"] if first else None,
                             "reused_at_first_overlap": first["reused_payload"] if first else None,
                             "locally_visible_at_first_overlap":
                                 visibility[first["timestamp_ms"]]["target_locally_visible"] if first else None})
    by_method = {}
    for method in ("surface", "scalar_ttl"):
        subset = [row for row in observations if row["method"] == method]
        by_method[method] = {"branches": len(subset),
                             "sampled_severe": sum(row["severe"] for row in subset),
                             "mean_reuse_decisions": sum(row["reuse_decisions"] for row in subset) / len(subset),
                             "mean_refresh_decisions": sum(row["refresh_decisions"] for row in subset) / len(subset),
                             "mean_generated_packets": sum(row["generated_packets"] for row in subset) / len(subset),
                             "mean_generated_bytes": sum(row["generated_bytes"] for row in subset) / len(subset),
                             "sampled_overlaps_while_reusing":
                                 sum(row["reused_at_first_overlap"] is True for row in subset),
                             "sampled_overlaps_while_locally_visible":
                                 sum(row["locally_visible_at_first_overlap"] is True for row in subset)}
    result = {"status": "POST_HOC_CROSSING_REFRESH_DIAGNOSTIC_ONLY",
              "source_report_sha256": SOURCE_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "by_method": by_method, "rows": observations,
              "limitation": "Associations after divergent policy trajectories; no isolated causal mechanism test."}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_method, indent=2))


if __name__ == "__main__":
    main()
