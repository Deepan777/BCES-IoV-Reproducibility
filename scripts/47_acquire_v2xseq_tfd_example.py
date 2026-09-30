#!/usr/bin/env python3
"""Acquire and integrity-audit the publisher-linked V2X-Seq TFD example only."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import gdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.reproducibility import write_json_atomic

FILE_ID = "1-Ri92z6rkH14vAOFOx5xhfzvFxBptgAA"
SOURCE = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=drive_link"
ARCHIVE = ROOT / "data/raw/v2xseq_tfd_example_v1/V2X-Seq-TFD-Example.zip"
RECEIPT = ROOT / "data/manifests/v2xseq_tfd_example_archive_v1.json"
CAP = 350_000_000


def safe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and not any(part in ("", ".", "..") or ":" in part for part in path.parts)


def main() -> None:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        result = gdown.download(id=FILE_ID, output=str(ARCHIVE), quiet=False)
        if result != str(ARCHIVE):
            raise RuntimeError("official Google Drive example download did not complete")
    size = ARCHIVE.stat().st_size
    if not 0 < size <= CAP:
        raise RuntimeError(f"archive size outside allowed tranche: {size}")
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError("download is not a ZIP archive")
    members = []
    seen = set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            name = info.filename
            if not safe_member(name) or name in seen:
                raise RuntimeError(f"unsafe or duplicate ZIP member: {name}")
            seen.add(name)
            members.append({"name": name, "bytes": info.file_size,
                            "compressed_bytes": info.compress_size, "crc32": info.CRC})
        if sum(m["bytes"] for m in members) > 2_500_000_000:
            raise RuntimeError("unexpected ZIP expansion; review before extraction")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP member CRC failure: {bad}")
    receipt = {"status": "ARCHIVE_ACQUIRED_NO_BCES_OUTCOME", "official_repository": "https://github.com/AIR-THU/DAIR-V2X-Seq",
               "official_source_url": SOURCE, "google_drive_file_id": FILE_ID,
               "license_assessment": "Archive-specific data terms must be inspected; GitHub code license is not presumed to govern data",
               "relative_archive_path": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
               "archive_bytes": size, "archive_sha256": digest.hexdigest(),
               "member_count": len(members), "members": members}
    write_json_atomic(RECEIPT, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "archive_bytes", "archive_sha256", "member_count")}, indent=2))


if __name__ == "__main__":
    main()
