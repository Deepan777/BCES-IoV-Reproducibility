from __future__ import annotations

from bces.evaluation.artifacts import artifact_hashes, prohibited_claims


def test_prohibited_claim_scan_is_case_insensitive() -> None:
    assert prohibited_claims("This is GUARANTEED SAFE.") == ("guaranteed safe",)
    assert prohibited_claims("Empirically decision-valid under this policy.") == ()


def test_artifact_hashes_are_relative_and_excludable(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    assert set(artifact_hashes(tmp_path, exclude=("b.txt",))) == {"a.txt"}

