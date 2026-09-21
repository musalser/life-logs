"""Builds a framework-independent training dataset from confirmed pages.

Rules:
- only CONFIRMED pages of the given author are used, all of them;
- the line transcription (user correction, else accepted prediction) is the
  canonical training target and is never auto-corrected;
- explicitly-empty lines (corrected_text == "") are valid but excluded from
  training samples;
- pages with corrupt images are excluded;
- the dataset hash is deterministic for identical content.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Callable

from ..domain.entities import PageStatus, TrainingDataset, TrainingSample
from ..domain.errors import CorruptImageError, DatasetBuildError
from ..domain.interfaces import LineCropper, PageRepository

logger = logging.getLogger(__name__)


class TrainingDatasetBuilder:
    def __init__(
        self,
        page_repository: PageRepository,
        line_cropper: LineCropper,
        crop_path_provider: Callable[[int, int], str],
    ):
        self.page_repository = page_repository
        self.line_cropper = line_cropper
        self.crop_path_provider = crop_path_provider

    def build_for_author(self, author_id: int) -> TrainingDataset:
        pages = self.page_repository.get_confirmed_pages(author_id)
        samples: list[TrainingSample] = []
        for page in sorted(pages, key=lambda p: p.id):
            if page.status != PageStatus.CONFIRMED:
                raise DatasetBuildError(
                    f"Page {page.id} is {page.status}, only CONFIRMED pages may be used"
                )
            try:
                samples.extend(self._build_page_samples(page))
            except CorruptImageError:
                logger.warning(
                    "HTR dataset: skipping page_id=%s of author_id=%s, corrupt image %s",
                    page.id, author_id, page.file_path,
                )
        dataset_hash = self._compute_hash(samples)
        return TrainingDataset(author_id=author_id, samples=samples, dataset_hash=dataset_hash)

    def _build_page_samples(self, page) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        for line in sorted(page.lines, key=lambda l: l.order):
            if not line.has_valid_transcription:
                raise DatasetBuildError(
                    f"Line {line.id} of page {page.id} has no valid transcription"
                )
            text = line.effective_text or ""
            if not text.strip():
                # user explicitly confirmed the line is empty -> nothing to train on
                continue
            crop_path = self.line_cropper.crop_line(
                page.file_path,
                line.bbox,
                self.crop_path_provider(page.id, line.id),
            )
            samples.append(
                TrainingSample(
                    page_id=page.id,
                    line_id=line.id,
                    image_path=crop_path,
                    transcription=text,
                )
            )
        return samples

    @staticmethod
    def _compute_hash(samples: list[TrainingSample]) -> str:
        canonical = "\n".join(
            f"{s.page_id}|{s.line_id}|{s.transcription}"
            for s in sorted(samples, key=lambda s: (s.page_id, s.line_id))
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
