#!/usr/bin/env python3
"""Scenario-clustered descriptive analysis of the exact-wire development audit.

This is an explicitly post hoc analysis of ten already-open low-margin straight
scenarios. Slots within a scenario are never treated as independent trials.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/margin_exact_wire_development_v1/report.json"
PROTOCOL = ROOT / "outputs/study_b/margin_exact_wire_development_v1/protocol.json"
OUTPUT = ROOT / "outputs/study_b/margin_exact_wire_development_v1/cluster_analysis.json"
SEED = 240926
REPLICATES = 10000


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha256_file(PROTOCOL) != source["protocol_sha256"] or len(source["low_margin_by_scenario"]) != 40:
        raise RuntimeError("source/protocol mismatch")
    if sha256_file(ROOT / "scripts/71_audit_margin_exact_wire_development.py") != protocol["script_sha256"]:
        raise RuntimeError("exact-wire source changed")
    rows = [row for row in source["low_margin_by_scenario"] if row["low_margin_points"] > 0]
    if len(rows) != 10 or any(row["family"] != "straight_lead_braking" for row in rows):
        raise RuntimeError("wrong fixed low-margin scenario stratum")
    accepted = np.asarray([row["margin_accepted"]-row["lead_only_accepted"] for row in rows], dtype=float)
    invalid = np.asarray([row["lead_only_invalid"]-row["margin_invalid"] for row in rows], dtype=float)
    draws = np.random.default_rng(SEED).integers(0, len(rows), size=(REPLICATES, len(rows)))
    accepted_draws = accepted[draws].mean(axis=1)
    invalid_draws = invalid[draws].mean(axis=1)
    nonzero = invalid[invalid != 0]
    result = {
        "status": "POST_HOC_DESCRIPTIVE_DEVELOPMENT_ONLY",
        "source_report_sha256": sha256_file(SOURCE),
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "script_sha256": sha256_file(Path(__file__)),
        "unit": "scenario; sampled points are nested within scenario",
        "selected_stratum": "lead-like reference with first-competitor total-cost gap < 0.20",
        "scenarios": len(rows),
        "points": int(sum(row["low_margin_points"] for row in rows)),
        "lead_only_minus_margin_invalid_accepts_per_scenario_mean": float(invalid.mean()),
        "invalid_count_difference_bootstrap_95pct_descriptive": np.quantile(invalid_draws, [0.025, 0.975]).tolist(),
        "invalid_count_scenarios": {"positive": int((invalid > 0).sum()),
                                    "negative": int((invalid < 0).sum()),
                                    "tied": int((invalid == 0).sum())},
        "conditional_one_sided_sign_p_invalid_count": float(binomtest(int((nonzero > 0).sum()),
                                                                         len(nonzero), 0.5,
                                                                         alternative="greater").pvalue) if len(nonzero) else 1.0,
        "margin_minus_lead_only_accepts_per_scenario_mean": float(accepted.mean()),
        "accepted_count_difference_bootstrap_95pct_descriptive": np.quantile(accepted_draws, [0.025, 0.975]).tolist(),
        "accepted_count_scenarios": {"positive": int((accepted > 0).sum()),
                                     "negative": int((accepted < 0).sum()),
                                     "tied": int((accepted == 0).sum())},
        "caveat": "post hoc, selected development stratum, ten scenarios; neither bootstrap nor sign test is a new confirmatory error guarantee",
        "not_confirmatory": True,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
