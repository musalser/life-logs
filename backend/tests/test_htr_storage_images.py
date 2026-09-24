"""Image handling of the HTR module: EXIF orientation and line crops.

Diary pages are photographed with a phone, so the stored JPEG is frequently
rotated by an EXIF tag. Both the stored page geometry and every crop must use
the visually oriented image, otherwise recognition runs on a sideways page.
"""
from __future__ import annotations

import io

import pytest

from app.htr.domain.entities import BoundingBox, LineGeometry
from app.htr.domain.errors import CorruptImageError
from app.htr.infrastructure.storage import (
    HTRStorage,
    PilLineCropper,
    open_oriented_image,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def jpeg_with_orientation(size=(40, 20), orientation=6) -> bytes:
    """Landscape JPEG whose EXIF tag says 'rotate 90° CW' (phone portrait)."""
    image = Image.new("RGB", size, "white")
    exif = Image.Exif()
    exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_probe_image_applies_exif_orientation(tmp_path):
    storage = HTRStorage(tmp_path)
    # 40x20 raw pixels are displayed (and processed) as 20x40
    assert storage.probe_image(jpeg_with_orientation((40, 20), 6)) == (20, 40)


def test_probe_image_keeps_dimensions_without_exif(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="PNG")
    assert HTRStorage(tmp_path).probe_image(buffer.getvalue()) == (40, 20)


def test_probe_image_rejects_undecodable_content(tmp_path):
    with pytest.raises(CorruptImageError):
        HTRStorage(tmp_path).probe_image(b"this is not an image")


def test_open_oriented_image_normalizes_rotation(tmp_path):
    path = tmp_path / "page.jpg"
    path.write_bytes(jpeg_with_orientation((40, 20), 6))

    with open_oriented_image(path) as image:
        assert image.size == (20, 40)


def test_open_oriented_image_reports_corrupt_file(tmp_path):
    path = tmp_path / "page.jpg"
    path.write_bytes(b"nope")

    with pytest.raises(CorruptImageError):
        open_oriented_image(path)


def test_line_crop_uses_oriented_geometry(tmp_path):
    page = tmp_path / "page.jpg"
    page.write_bytes(jpeg_with_orientation((40, 20), 6))
    output = tmp_path / "crops" / "line_1.png"

    PilLineCropper().crop_line(
        str(page), LineGeometry(BoundingBox(0, 0, 10, 20)), str(output)
    )

    with Image.open(output) as crop:
        assert crop.size == (10, 20)


def test_line_crop_clamps_boxes_to_the_image(tmp_path):
    page = tmp_path / "page.jpg"
    page.write_bytes(jpeg_with_orientation((40, 20), 6))
    output = tmp_path / "line.png"

    PilLineCropper().crop_line(
        str(page), LineGeometry(BoundingBox(-50, -50, 5000, 5000)), str(output)
    )

    with Image.open(output) as crop:
        assert crop.size == (20, 40)


def test_line_crop_reports_corrupt_page(tmp_path):
    page = tmp_path / "page.jpg"
    page.write_bytes(b"nope")

    with pytest.raises(CorruptImageError):
        PilLineCropper().crop_line(
            str(page), LineGeometry(BoundingBox(0, 0, 10, 10)), str(tmp_path / "x.png")
        )
