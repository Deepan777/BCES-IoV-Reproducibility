from __future__ import annotations

import pytest

from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import RUN_MANIFEST_KEYS, new_run_manifest


pytestmark = pytest.mark.phase0


def test_canonical_hash_is_order_invariant() -> None:
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})


def test_run_manifest_contains_all_registered_fields() -> None:
    manifest = new_run_manifest(status="PASS", exit_code=0)
    assert all(key in manifest for key in RUN_MANIFEST_KEYS)
    assert manifest["status"] == "PASS"


def test_run_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        new_run_manifest(unregistered="value")

