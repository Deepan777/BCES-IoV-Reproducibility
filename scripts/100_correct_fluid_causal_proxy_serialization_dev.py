#!/usr/bin/env python3
"""Rerun frozen causal proxy with only underpowered p95 serialized as null."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / "scripts/99_audit_fluid_causal_turn_proxy_dev.py"
ORIGINAL_SHA256 = "1e8ec04416268c822c9bb7d8a36721ac1fca52c292477478b186fba7450869d2"
FAILED_REPORT = ROOT / "outputs/study_b/fluid_causal_turn_proxy_development_v1/report.json"
FAILED_REPORT_SHA256 = "3c7e111739faf1f690759515fc08b1fef497047e72e9b1f6bc1ce593f3f26fc4"
OUTPUT = ROOT / "outputs/study_b/fluid_causal_turn_proxy_development_v1_corrected"


def reject_constant(value):
    raise ValueError("nonstandard JSON constant: " + value)


def normalize_nonfinite(value):
    if isinstance(value, dict):
        return {key: normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("original causal analysis changed")
    if hashlib.sha256(FAILED_REPORT.read_bytes()).hexdigest() != FAILED_REPORT_SHA256:
        raise RuntimeError("original nonstandard report changed")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    spec = importlib.util.spec_from_file_location("fluid_causal_proxy_original", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import original causal analysis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_quantile = module.upper_quantile

    def standard_quantile(values):
        value = original_quantile(values)
        return None if value is not None and not math.isfinite(value) else value

    module.upper_quantile = standard_quantile
    module.OUTPUT = OUTPUT
    module.main()
    report_path = OUTPUT / "report.json"
    corrected = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    original = json.loads(FAILED_REPORT.read_text(encoding="utf-8"))
    if corrected != normalize_nonfinite(original):
        raise RuntimeError("result changed beyond non-finite-to-null serialization correction")
    receipt = {"status": "SERIALIZATION_ONLY_CORRECTION",
               "original_analysis_sha256": ORIGINAL_SHA256,
               "original_nonstandard_report_sha256": FAILED_REPORT_SHA256,
               "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "corrected_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
               "correction": "underpowered diagnostic nearest-rank p95 changed from nonstandard Infinity to JSON null",
               "all_scientific_radii_and_coverage_metrics_unchanged": True}
    (OUTPUT / "correction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
