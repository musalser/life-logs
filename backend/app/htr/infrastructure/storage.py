"""Filesystem layout and image operations for the HTR module.

Layout:
    <root>/pages/author_<id>/<uuid>.<ext>
    <root>/crops/page_<id>/line_<id>.png
    <root>/models/author_<author_id>/v<version>/model.safetensors
"""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from pathlib import Path

from ..domain.entities import BoundingBox, LineGeometry
from ..domain.errors import CorruptImageError

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None


def _require_pil():
    if Image is None:
        raise CorruptImageError("Pillow is required for HTR image handling but is not installed")


def open_oriented_image(path: str | Path):
    """Open an image and normalize it according to its EXIF orientation.

    Phone photos of diary pages are frequently stored rotated with an EXIF
    orientation tag. Segmenting/recognizing the raw pixels would operate on a
    sideways page, so every consumer of a stored page image must go through
    this helper.
    """
    _require_pil()
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise CorruptImageError(f"Cannot open image {path}: {exc}") from exc
    fixed = ImageOps.exif_transpose(img) if ImageOps is not None else None
    if fixed is None or fixed is img:
        return img
    img.close()  # exif_transpose returned a rotated copy
    return fixed


class HTRStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- pages ----------------------------------------------------------

    def save_page_image(self, user_id: int, author_id: int, filename: str, content: bytes) -> str:
        ext = Path(filename).suffix.lower() or ".png"
        page_dir = self.root / "pages" / f"author_{author_id}"
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / f"{uuid.uuid4().hex}{ext}"
        path.write_bytes(content)
        return str(path)

    def probe_image(self, content: bytes) -> tuple[int, int]:
        _require_pil()
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.load()
                # EXIF orientation changes the visual (and therefore the
                # processed) dimensions of the stored image.
                oriented = ImageOps.exif_transpose(img) if ImageOps is not None else img
                return oriented.width, oriented.height
        except Exception as exc:
            raise CorruptImageError(f"Cannot decode image: {exc}") from exc

    def delete_page_assets(self, page_id: int, file_path: str) -> None:
        """Remove the stored page image and its line crops.

        Missing files are fine (legacy rows point at paths from the Windows
        deployment); the database row is the source of truth.
        """
        try:
            path = Path(file_path)
            if path.is_file():
                path.unlink()
        except OSError as exc:
            logger.warning("Could not remove page image %s: %s", file_path, exc)
        shutil.rmtree(self.root / "crops" / f"page_{page_id}", ignore_errors=True)

    # -- line crops ------------------------------------------------------

    def line_crop_path(self, page_id: int, line_id: int) -> str:
        crop_dir = self.root / "crops" / f"page_{page_id}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        return str(crop_dir / f"line_{line_id}.png")

    # -- models ----------------------------------------------------------

    def model_output_path(self, author_id: int, version: int) -> str:
        # kraken 7 writes safetensors; .mlmodel would need the CoreML loader,
        # which is not what the recognizer uses.
        model_dir = self.root / "models" / f"author_{author_id}" / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir / "model.safetensors")

    def training_work_dir(self) -> str:
        work_dir = self.root / "training_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)
        return str(work_dir)


class PilLineCropper:
    """LineCropper fallback: the axis-aligned envelope of the line.

    Used for lines without a polygon and as the safety net of
    :class:`~app.htr.infrastructure.kraken.lines.KrakenLineCropper`.
    """

    def crop_line(
        self, page_image_path: str, geometry: LineGeometry, output_path: str
    ) -> str:
        bbox = geometry.bbox
        try:
            with open_oriented_image(page_image_path) as img:
                x1 = max(0, min(bbox.x1, img.width))
                y1 = max(0, min(bbox.y1, img.height))
                x2 = max(x1 + 1, min(bbox.x2, img.width))
                y2 = max(y1 + 1, min(bbox.y2, img.height))
                crop = img.crop((x1, y1, x2, y2))
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                crop.save(output_path)
            return output_path
        except CorruptImageError:
            raise
        except Exception as exc:
            raise CorruptImageError(
                f"Cannot crop line from {page_image_path}: {exc}"
            ) from exc
