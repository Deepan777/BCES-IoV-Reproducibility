from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from bces.data.download import (
    DownloadError,
    PreflightInfo,
    fetch_transactionally,
)
from bces.data.manifest import CatalogEntry, DataRole
from bces.data.remote_zip import RemoteZipMember, download_contiguous_members, download_member
from bces.utils.budget import Budget, BudgetExceeded

pytestmark = pytest.mark.phase3


class FakeTransport:
    def __init__(self, payload: bytes, *, declared_bytes: int | None = None) -> None:
        self.payload = payload
        self.declared_bytes = declared_bytes if declared_bytes is not None else len(payload)

    def preflight(self, url: str) -> PreflightInfo:
        return PreflightInfo(self.declared_bytes, url, "text/csv")

    def stream(self, url: str, chunk_bytes: int) -> Iterator[bytes]:
        yield self.payload[:7]
        yield self.payload[7:]


def _csv_payload() -> bytes:
    columns = "city,timestamp,id,type,sub_type,tag,x,y,z,length,width,height,theta,v_x,v_y,intersect_id,vic_tag,from_side,car_side_id,road_side_id\n"
    row = "B,0.0,v1,Vehicle,Car,AV,0,0,0,4,2,1,0,1,0,1,1,i,c,r\n"
    return (columns + row).encode()


def _entry(payload: bytes, **overrides: object) -> CatalogEntry:
    values = {
        "entry_id": "e1",
        "url": "https://raw.githubusercontent.com/AIR-THU/DAIR-V2X-Seq/main/scene.csv",
        "destination": "cooperative/scene.csv",
        "role": DataRole.COOPERATIVE,
        "scene_id": "scene",
        "intersection_id": "1",
        "selection_reason": "transaction integration test",
        "release": "official-example-2024",
        "license_url": "https://github.com/AIR-THU/DAIR-V2X-Seq",
        "expected_bytes": len(payload),
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
    }
    values.update(overrides)
    return CatalogEntry(**values)  # type: ignore[arg-type]


def _budget(tmp_path: Path, *, tranche: int = 100_000) -> Budget:
    return Budget(
        workspace_root=tmp_path,
        data_root=tmp_path / "data",
        initial_data_review_bytes=50_000,
        max_download_tranche_bytes=tranche,
        required_output_reserve_bytes=10_000,
        warn_workspace_bytes=900_000,
        max_workspace_bytes=1_000_000,
        max_cuda_allocated_bytes=1,
    )


def test_atomic_allowlisted_transfer_and_hash(tmp_path: Path) -> None:
    payload = _csv_payload()
    downloaded = fetch_transactionally(
        [_entry(payload)], budget=_budget(tmp_path), transport=FakeTransport(payload)
    )
    assert downloaded[0].path.read_bytes() == payload
    assert downloaded[0].sha256 == hashlib.sha256(payload).hexdigest()
    assert not list((tmp_path / "data" / "tmp").glob("*.part"))


def test_partial_stream_is_removed_on_length_failure(tmp_path: Path) -> None:
    payload = _csv_payload()
    with pytest.raises(DownloadError, match="exceeds"):
        fetch_transactionally(
            [_entry(payload, expected_bytes=None, expected_sha256=None)],
            budget=_budget(tmp_path),
            transport=FakeTransport(payload, declared_bytes=len(payload) - 1),
        )
    assert not list((tmp_path / "data" / "tmp").glob("*.part"))


def test_preflight_tranche_cap_fails_before_stream(tmp_path: Path) -> None:
    payload = _csv_payload()
    with pytest.raises(BudgetExceeded, match="tranche"):
        fetch_transactionally(
            [_entry(payload, expected_bytes=None)],
            budget=_budget(tmp_path, tranche=len(payload) - 1),
            transport=FakeTransport(payload),
        )


def test_hash_mismatch_removes_partial(tmp_path: Path) -> None:
    payload = _csv_payload()
    with pytest.raises(DownloadError, match="SHA-256"):
        fetch_transactionally(
            [_entry(payload, expected_sha256="0" * 64)],
            budget=_budget(tmp_path),
            transport=FakeTransport(payload),
        )
    assert not list((tmp_path / "data" / "tmp").glob("*.part"))


def test_existing_raw_destination_is_never_overwritten(tmp_path: Path) -> None:
    payload = _csv_payload()
    destination = tmp_path / "data" / "raw" / "cooperative" / "scene.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"preserve-me")
    with pytest.raises(DownloadError, match="already exists"):
        fetch_transactionally(
            [_entry(payload)], budget=_budget(tmp_path), transport=FakeTransport(payload)
        )
    assert destination.read_bytes() == b"preserve-me"


def test_remote_zip_member_ranges_verify_crc_and_content() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("v2x-traj/ego/train/1.csv", b"header\nrow\n")
        archive.writestr("v2x-traj/ego/train/2.csv", b"header\nother\n")
    payload = stream.getvalue()

    class Response:
        status_code = 206

        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, url: str, *, headers: dict[str, str], **kwargs: object) -> Response:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
            assert match
            start, end = map(int, match.groups())
            return Response(payload[start : end + 1])

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [
            RemoteZipMember(
                name=info.filename,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                crc32=info.CRC,
                compression_method=info.compress_type,
                local_header_offset=info.header_offset,
            )
            for info in archive.infolist()
        ]
    assert download_member(Session(), "https://official.example/archive.zip", members[0]) == b"header\nrow\n"  # type: ignore[arg-type]
    extracted, consumed = download_contiguous_members(
        Session(),  # type: ignore[arg-type]
        "https://official.example/archive.zip",
        members,
        archive_bytes=len(payload),
        maximum_range_bytes=len(payload),
    )
    assert extracted[members[1].name] == b"header\nother\n"
    assert consumed <= len(payload)
