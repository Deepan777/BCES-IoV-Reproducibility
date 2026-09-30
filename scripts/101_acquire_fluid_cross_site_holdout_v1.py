#!/usr/bin/env python3
"""Acquire only one preselected public FLUID derived-data archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/fluid_cross_site_holdout_v1"
ARTICLE_ID = 29974954
ARTICLE_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
EXPECTED_DOI = "10.6084/m9.figshare.29974954.v2"
FILE_ID = 64179082  # smallest unopened derived-data archive, selected by size only
EXPECTED_SIZE = 96_533_946
MAX_DOWNLOAD_BYTES = 100_000_000
LOCAL_NAME = "derived_data_64179082.zip"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with urlopen(ARTICLE_URL, timeout=30) as response:
        article = json.loads(response.read(500_000))
    if (article.get("id") != ARTICLE_ID or article.get("version") != 2
            or article.get("doi") != EXPECTED_DOI
            or article.get("license", {}).get("name") != "CC BY 4.0"):
        raise RuntimeError("official FLUID article identity/version/license changed")
    matches = [item for item in article["files"] if item["id"] == FILE_ID]
    if len(matches) != 1:
        raise RuntimeError("selected file ID missing or duplicate")
    item = matches[0]
    md5 = item.get("computed_md5", "")
    if (item.get("name") != "derived_data.zip"
            or item.get("size") != EXPECTED_SIZE
            or EXPECTED_SIZE > MAX_DOWNLOAD_BYTES
            or len(md5) != 32 or any(ch not in "0123456789abcdef" for ch in md5.lower())):
        raise RuntimeError("official selected-file contract changed")
    if not item.get("download_url", "").startswith("https://"):
        raise RuntimeError("selected download URL is not HTTPS")

    OUTPUT.mkdir(parents=True)
    target = OUTPUT / LOCAL_NAME
    partial = OUTPUT / (LOCAL_NAME + ".part")
    sha256, local_md5, count = hashlib.sha256(), hashlib.md5(), 0
    with urlopen(item["download_url"], timeout=60) as response, partial.open("xb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > EXPECTED_SIZE:
                raise RuntimeError("download exceeded official size")
            sha256.update(chunk)
            local_md5.update(chunk)
            handle.write(chunk)
    if count != EXPECTED_SIZE or local_md5.hexdigest() != md5.lower():
        raise RuntimeError("download size or official MD5 mismatch")
    partial.rename(target)

    members = []
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if name.startswith("/") or ".." in path.parts or ":" in name:
                raise RuntimeError("unsafe ZIP member name")
            members.append({"name": name, "bytes": info.file_size,
                            "compressed_bytes": info.compress_size})
    manifest = {"status": "VERIFIED_UNOPENED_CROSS_SITE_HOLDOUT_ARCHIVE",
                "article_id": ARTICLE_ID, "article_version": 2,
                "doi": EXPECTED_DOI, "license": "CC BY 4.0",
                "file_id": FILE_ID, "official_name": item["name"],
                "official_bytes": EXPECTED_SIZE, "official_md5": md5.lower(),
                "download_url": item["download_url"],
                "local_name": LOCAL_NAME, "sha256": sha256.hexdigest(),
                "acquisition_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "video_archives_downloaded": False,
                "members": members}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_id": FILE_ID, "bytes": count,
                      "sha256": sha256.hexdigest(), "zip_members": len(members)}, indent=2))


if __name__ == "__main__":
    main()
