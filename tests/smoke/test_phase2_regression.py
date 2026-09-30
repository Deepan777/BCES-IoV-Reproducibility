from __future__ import annotations

import pytest

from bces.smoke.pipeline import EXPECTED_COVERAGE, EXPECTED_UAR, run_smoke_pipeline

pytestmark = pytest.mark.phase2


def test_fixed_smoke_uar_and_coverage() -> None:
    result = run_smoke_pipeline()
    assert result["metrics"]["coverage"] == EXPECTED_COVERAGE == 0.5
    assert result["metrics"]["unsafe_accept_rate"] == EXPECTED_UAR == 0.1
    assert result["evidence_use"] == "software_verification_only"
