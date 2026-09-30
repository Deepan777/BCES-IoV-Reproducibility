#!/usr/bin/env python3
"""Post hoc first-command divergence in paired crossing trajectories."""

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
OUTPUT = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_first_divergence_dev_v1"


def branch(report, seed, condition, method):
    name = f"{seed}_near_{condition}_{method}.json.gz"
    if sha256_file(SOURCE / name) != report["artifact_sha256"][name]:
        raise RuntimeError("branch changed: " + name)
    with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("source report changed")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    sensitive = sorted({(row["seed"], row["condition"])
                        for row in report["rows"] if row["timing"] == "near"
                        and not row["periodic_severe"] and row["local_severe"]})
    rows = []
    for seed, condition in sensitive:
        surface = branch(report, seed, condition, "surface")
        ttl = branch(report, seed, condition, "scalar_ttl")
        surface_rows = {row["timestamp_ms"]: row for row in surface["rows"]
                        if "command" in row and "requested_speed_mps" in row["command"]}
        ttl_rows = {row["timestamp_ms"]: row for row in ttl["rows"]
                    if "command" in row and "requested_speed_mps" in row["command"]}
        common = sorted(set(surface_rows) & set(ttl_rows))
        first = None
        for timestamp in common:
            a, b = surface_rows[timestamp], ttl_rows[timestamp]
            if (a["command"]["maneuver"] != b["command"]["maneuver"]
                    or abs(a["command"]["requested_speed_mps"]
                           - b["command"]["requested_speed_mps"]) > 1e-8):
                first = timestamp
                break
        if first is None:
            rows.append({"seed": seed, "condition": condition,
                         "first_divergent_command_result_ms": None,
                         "same_recorded_commands": True})
            continue
        a, b = surface_rows[first], ttl_rows[first]
        local_a = {item["timestamp_ms"]: item for item in surface["visibility_audit"]}
        local_b = {item["timestamp_ms"]: item for item in ttl["visibility_audit"]}
        rows.append({"seed": seed, "condition": condition,
                     "first_divergent_command_result_ms": first,
                     "surface_requested_speed_mps": a["command"]["requested_speed_mps"],
                     "ttl_requested_speed_mps": b["command"]["requested_speed_mps"],
                     "surface_maneuver": a["command"]["maneuver"],
                     "ttl_maneuver": b["command"]["maneuver"],
                     "surface_reused_at_divergent_step": a["reused_payload"],
                     "ttl_reused_at_divergent_step": b["reused_payload"],
                     "surface_actor_locally_visible_at_result":
                         local_a[first]["target_locally_visible"],
                     "ttl_actor_locally_visible_at_result":
                         local_b[first]["target_locally_visible"],
                     "surface_actor_range_m_at_result": local_a[first]["target_range_m"],
                     "ttl_actor_range_m_at_result": local_b[first]["target_range_m"]})
    summary = {"pairs": len(rows),
               "pairs_without_recorded_command_divergence":
                   sum(row.get("same_recorded_commands", False) for row in rows),
               "surface_local_when_ttl_reused_at_first_divergence":
                   sum(not row.get("surface_reused_at_divergent_step", False)
                       and row.get("ttl_reused_at_divergent_step", False) for row in rows),
               "both_reused_at_first_divergence":
                   sum(row.get("surface_reused_at_divergent_step", False)
                       and row.get("ttl_reused_at_divergent_step", False) for row in rows),
               "neither_reused_at_first_divergence":
                   sum(row.get("same_recorded_commands") is not True
                       and not row["surface_reused_at_divergent_step"]
                       and not row["ttl_reused_at_divergent_step"] for row in rows)}
    result = {"status": "POST_HOC_FIRST_DIVERGENCE_DIAGNOSTIC_ONLY",
              "source_report_sha256": SOURCE_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "summary": summary, "rows": rows,
              "limitation": "Aligned timestamps follow diverging endogenous trajectories; descriptive association, not a randomized isolation of policy mechanisms."}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
