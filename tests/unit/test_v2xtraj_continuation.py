from __future__ import annotations

import binascii
import importlib.util
from pathlib import Path

import pytest

from bces.data.remote_zip import RemoteZipMember

pytestmark = pytest.mark.phase3
ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_sizing_is_conservative_and_reproducible() -> None:
    planner = _load_script("03_plan_v2xtraj_continuation.py")
    lower = planner._wilson_lower(190, 300)
    assert lower == pytest.approx(0.5774356684994387)
    assert lower < 190 / 300
    assert planner.math.ceil(310 / lower) == 537


def test_largest_remainder_respects_capacity_and_total() -> None:
    planner = _load_script("03_plan_v2xtraj_continuation.py")
    first = planner._largest_remainder(17, {"a": 20, "b": 10, "c": 5}, salt="fixed")
    second = planner._largest_remainder(17, {"a": 20, "b": 10, "c": 5}, salt="fixed")
    assert first == second
    assert sum(first.values()) == 17
    assert all(first[key] <= capacity for key, capacity in {"a": 20, "b": 10, "c": 5}.items())


def test_resume_replaces_crc_invalid_file_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acquisition = _load_script("03_acquire_v2xtraj_tranche.py")
    payload = b"verified official payload"
    member = RemoteZipMember(
        name="v2x-traj/ego-trajectories/train/data/1.csv",
        compressed_bytes=len(payload),
        uncompressed_bytes=len(payload),
        crc32=binascii.crc32(payload) & 0xFFFFFFFF,
        compression_method=0,
        local_header_offset=0,
    )
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    monkeypatch.setattr(acquisition, "RAW_ROOT", tmp_path / "data" / "raw" / "v2xtraj_primary")
    monkeypatch.setattr(acquisition, "download_member", lambda session, url, item: payload)
    destination = acquisition._destination(member.name)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupted interrupted data")

    record = acquisition._download("https://official.example/archive.zip", member)

    assert destination.read_bytes() == payload
    assert record["crc32"] == f"{member.crc32:08x}"
    assert not destination.with_suffix(".csv.part").exists()
