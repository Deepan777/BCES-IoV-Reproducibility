#!/usr/bin/env python3
"""Bounded, verified download of FLUID's smallest public development tranche."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/fluid_small_development_v1"
ARTICLE_ID = 29974954
ARTICLE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
EXPECTED_DOI = "10.6084/m9.figshare.29974954.v2"
FILES = (
    (61924402, "README.md", 22447, "e0ef4265daa22f774c1e3c8be642bc2e"),
    (61923805, "Conflict.zip", 548688, "014d6d2731ff211b9b6f36afda66423d"),
    (64179079, "derived_data_64179079.zip", 30932517, "fc2870464b3509130ce606c4f43a492b"),
)
MAX_TOTAL_BYTES = 32_000_000


def read_article():
    with urlopen(ARTICLE_API, timeout=30) as response:
        payload = response.read(500_000)
    article = json.loads(payload)
    if (article.get("id") != ARTICLE_ID or article.get("version") != 2
            or article.get("doi") != EXPECTED_DOI
            or article.get("license", {}).get("name") != "CC BY 4.0"):
        raise RuntimeError("official figshare article contract changed")
    by_id = {item["id"]: item for item in article["files"]}
    for file_id, source_name, size, md5 in FILES:
        item = by_id.get(file_id)
        if item is None or item["name"] != source_name or item["size"] != size:
            raise RuntimeError(f"official file inventory changed for {file_id}")
        if item.get("computed_md5") != md5:
            raise RuntimeError(f"official file digest changed for {file_id}")
    return article, by_id


def acquire(item, target, expected_size, expected_md5):
    if target.exists():
        raise FileExistsError(target)
    part = target.with_name(target.name + ".part")
    if part.exists():
        raise FileExistsError(part)
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    count = 0
    with urlopen(item["download_url"], timeout=45) as response, part.open("xb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > expected_size:
                raise RuntimeError("download exceeded official file size")
            sha256.update(chunk)
            md5.update(chunk)
            handle.write(chunk)
    if count != expected_size or md5.hexdigest() != expected_md5:
        raise RuntimeError(f"download size or MD5 mismatch for {target.name}")
    part.rename(target)
    return {"bytes": count, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def zip_index(path):
    result = []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or ":" in name:
                raise RuntimeError("unsafe zip member path")
            result.append({"name": name, "compressed_bytes": item.compress_size,
                           "uncompressed_bytes": item.file_size,
                           "is_directory": item.is_dir()})
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sum(item[2] for item in FILES) > MAX_TOTAL_BYTES:
        raise RuntimeError("planned download exceeds 32-MB cap")
    article, by_id = read_article()
    OUTPUT.mkdir(parents=True)
    records = []
    for file_id, source_name, size, md5 in FILES:
        target_name = source_name if source_name == "README.md" else (
            f"{source_name}" if file_id != 64179079 else "derived_data_64179079.zip")
        target = OUTPUT / target_name
        acquired = acquire(by_id[file_id], target, size, md5)
        record = {"figshare_file_id": file_id, "source_name": source_name,
                  "local_name": target_name, "download_url": by_id[file_id]["download_url"],
                  **acquired}
        if target.suffix.lower() == ".zip":
            record["zip_members"] = zip_index(target)
        records.append(record)
        print(json.dumps({"completed_files": len(records), "total_files": len(FILES),
                          "file_id": file_id, "bytes": acquired["bytes"]}), flush=True)
    manifest = {"status": "VERIFIED_SMALL_PUBLIC_FLUID_DEVELOPMENT_TRANCHE",
                "article_id": ARTICLE_ID, "article_version": article["version"],
                "doi": article["doi"], "license": article["license"],
                "article_url": article["url"],
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selection": "README, small conflict archive, smallest derived-data archive selected by official file size before contents were opened",
                "unopened_derived_data_file_ids": [64179082, 64179085],
                "video_archives_not_downloaded": True,
                "development_only": True, "files": records}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"downloaded_bytes": sum(row["bytes"] for row in records),
                      "file_ids": [row["figshare_file_id"] for row in records],
                      "derived_zip_members": len(records[-1]["zip_members"])}, indent=2))


if __name__ == "__main__":
    main()
