"""Fetch a default HTR recognition model into the local HTR storage.

The models are the multilingual PP-OCRv6 line recognizers published together
with kraken by Benjamin Kiessling (Apache-2.0). They are trained on handwritten
and machine-printed line images in 44 languages / 10 scripts and include
Russian/Cyrillic, which is what the diary pages of this project need.

Usage (from the ``backend`` directory)::

    python scripts/download_htr_model.py                  # medium (default)
    python scripts/download_htr_model.py --model small
    python scripts/download_htr_model.py --force

The file is written to ``<htr_storage_dir>/models/default/ppocrv6_<size>.safetensors``
which is exactly what ``htr_default_model_path`` points at. ``htr_storage`` is
git-ignored: models are provisioned, not committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

# Zenodo record -> file name per PP-OCRv6 variant (kraken >= 7.1.0).
ZENODO_RECORDS = {
    "tiny": (21788403, "tiny.safetensors"),
    "small": (21788405, "small.safetensors"),
    "medium": (21788410, "medium.safetensors"),
}
DEFAULT_VARIANT = "medium"
USER_AGENT = "life-logs-htr-model-fetcher/1.0"


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _file_entry(record_id: int, file_name: str) -> dict:
    record = _fetch_json(f"https://zenodo.org/api/records/{record_id}")
    for entry in record.get("files", []):
        if entry["key"] == file_name:
            return entry
    raise SystemExit(f"{file_name} not found in Zenodo record {record_id}")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(variant: str, force: bool = False) -> Path:
    record_id, file_name = ZENODO_RECORDS[variant]
    target_dir = Path(settings.htr_storage_dir) / "models" / "default"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"ppocrv6_{variant}.safetensors"

    entry = _file_entry(record_id, file_name)
    expected_md5 = (entry.get("checksum") or "").removeprefix("md5:")
    size_mb = entry["size"] / (1024 * 1024)

    if target.is_file() and not force:
        if expected_md5 and _md5(target) == expected_md5:
            print(f"already present: {target} ({size_mb:.1f} MiB)")
            return target
        print(f"checksum mismatch for {target}; re-downloading")

    url = entry["links"]["self"]
    print(f"downloading {file_name} ({size_mb:.1f} MiB) -> {target}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
    target.write_bytes(payload)

    if expected_md5:
        actual = _md5(target)
        if actual != expected_md5:
            target.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch: expected {expected_md5}, got {actual}")
    print(f"saved {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(ZENODO_RECORDS),
        default=DEFAULT_VARIANT,
        help=f"PP-OCRv6 variant to fetch (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    target = download(args.model, force=args.force)
    configured = Path(settings.htr_default_model_path)
    if target != configured:
        print(
            f"note: htr_default_model_path is {configured}; set "
            f"HTR_DEFAULT_MODEL_PATH to {target} to use this variant"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
