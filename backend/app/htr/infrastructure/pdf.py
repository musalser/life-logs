"""PDF → page image: the adapter between one scanned PDF and the page pipeline.

A manuscript is handed over as one PDF with a page per sheet, while everything
downstream speaks "one image = one page". This module renders every sheet to a
PNG that :class:`~app.htr.application.page_service.HandwritingPageService` can
store like any other upload.

Rendering goes through pypdfium2 (bundled PDFium: wheels for Linux/macOS/Windows,
no poppler or ghostscript to install). Pages are stored as PNG on purpose: the
stored image is the input of the recognizer, and JPEG ringing around thin pen
strokes costs recognition quality. A page whose render carries no colour is kept
as grayscale — a typescript or a black-and-white diary sheet is three times
smaller that way and the viewer shows the same picture.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from typing import Any

from ..domain.errors import CorruptPdfError

logger = logging.getLogger(__name__)

try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - exercised by the error path only
    pdfium = None

try:
    from PIL import Image, ImageChops
except ImportError:  # pragma: no cover
    Image = None
    ImageChops = None

#: how far the colour channels of one pixel may differ before a page is "colour"
COLOUR_TOLERANCE = 8


def _require_pdfium() -> None:
    if pdfium is None:
        raise CorruptPdfError(
            "Для загрузки PDF нужен pypdfium2: "
            "uv pip install --python backend/.venv/bin/python pypdfium2"
        )


def _require_pil() -> None:
    if Image is None:
        raise CorruptPdfError("Для загрузки PDF нужен Pillow (кодирование страниц)")


def is_greyscale(rgb: Any) -> bool:
    """True when the rendered page has no colour worth keeping.

    A bilevel or grayscale scan (the usual manuscript) renders to three equal
    channels; the difference between them is what tells it from a real colour
    page (a photo, a coloured stamp), which keeps all three.
    """
    red, green, blue = rgb.split()
    worst = 0
    for left, right in ((red, green), (green, blue)):
        _, high = ImageChops.difference(left, right).getextrema()
        worst = max(worst, high or 0)
    return worst <= COLOUR_TOLERANCE


class PdfRasterizer:
    """Renders the pages of a PDF to PNG bytes, one page at a time."""

    def __init__(self, dpi: int = 300) -> None:
        # below ~36 dpi a page is a thumbnail and HTR has nothing to read
        self.dpi = max(36, int(dpi))

    def page_count(self, content: bytes) -> int:
        _require_pdfium()
        document = self._open(content)
        try:
            return len(document)
        except Exception as exc:  # pragma: no cover - a malformed page tree
            raise CorruptPdfError(f"Не удалось прочитать PDF: {exc}") from exc
        finally:
            document.close()

    def render_pages(self, content: bytes) -> Iterator[bytes]:
        """PNG bytes of every page, in document order.

        A generator: a several-hundred-page scan is rendered and stored sheet by
        sheet instead of holding the whole rasterised document in memory.
        """
        _require_pdfium()
        _require_pil()
        document = self._open(content)
        try:
            for index in range(len(document)):
                yield self._render_page(document, index)
        finally:
            document.close()

    @staticmethod
    def _open(content: bytes):
        try:
            return pdfium.PdfDocument(content)
        except Exception as exc:
            raise CorruptPdfError(f"Не удалось открыть PDF: {exc}") from exc

    def _render_page(self, document, index: int) -> bytes:
        try:
            page = document[index]
            try:
                bitmap = page.render(scale=self.dpi / 72)
                try:
                    image = bitmap.to_pil()
                finally:
                    bitmap.close()
            finally:
                page.close()
        except CorruptPdfError:
            raise
        except Exception as exc:
            raise CorruptPdfError(
                f"Не удалось отрисовать страницу {index + 1} PDF: {exc}"
            ) from exc
        image = image.convert("RGB")
        if is_greyscale(image):
            image = image.convert("L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
