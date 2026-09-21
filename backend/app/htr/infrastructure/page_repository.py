"""SQLAlchemy implementation of PageRepository (maps ORM <-> domain views)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session, joinedload

from ...models import HTRLine, HTRPage, HTRWord
from ..domain.entities import (
    BoundingBox,
    LineView,
    PageStatus,
    PageView,
    RecognitionResult,
    WordView,
)
from ..domain.errors import NotFoundError


def _to_word_view(word: HTRWord) -> WordView:
    return WordView(
        id=word.id,
        order=word.order_index,
        bbox=BoundingBox(word.x1, word.y1, word.x2, word.y2),
        predicted_text=word.predicted_text,
        confidence=word.confidence,
        corrected_text=word.corrected_text,
    )


def _to_line_view(line: HTRLine) -> LineView:
    return LineView(
        id=line.id,
        order=line.order_index,
        bbox=BoundingBox(line.x1, line.y1, line.x2, line.y2),
        predicted_text=line.predicted_text,
        corrected_text=line.corrected_text,
        words=[_to_word_view(w) for w in line.words],
        words_stale=bool(line.words_stale),
    )


def _to_page_view(page: HTRPage) -> PageView:
    return PageView(
        id=page.id,
        user_id=page.user_id,
        author_id=page.author_id,
        file_path=page.file_path,
        status=PageStatus(page.status),
        width=page.width,
        height=page.height,
        created_at=page.created_at,
        confirmed_at=page.confirmed_at,
        recognition_model_version_id=page.recognition_model_version_id,
        prediction_cer=page.prediction_cer,
        prediction_wer=page.prediction_wer,
        lines=[_to_line_view(l) for l in page.lines],
    )


class SqlAlchemyPageRepository:
    def __init__(self, db: Session):
        self.db = db

    def _get_orm_page(self, page_id: int) -> HTRPage:
        page = (
            self.db.query(HTRPage)
            .options(joinedload(HTRPage.lines).joinedload(HTRLine.words))
            .filter(HTRPage.id == page_id)
            .first()
        )
        if page is None:
            raise NotFoundError(f"Page {page_id} not found")
        return page

    # ------------------------------------------------------------------

    def create_page(
        self, user_id: int, author_id: int, file_path: str, width: int, height: int
    ) -> PageView:
        page = HTRPage(
            user_id=user_id,
            author_id=author_id,
            file_path=file_path,
            status=PageStatus.UPLOADED.value,
            width=width,
            height=height,
        )
        self.db.add(page)
        self.db.commit()
        self.db.refresh(page)
        return _to_page_view(page)

    def get_page(self, page_id: int) -> PageView | None:
        try:
            return _to_page_view(self._get_orm_page(page_id))
        except NotFoundError:
            return None

    def get_confirmed_pages(self, author_id: int) -> list[PageView]:
        pages = (
            self.db.query(HTRPage)
            .options(joinedload(HTRPage.lines).joinedload(HTRLine.words))
            .filter(
                HTRPage.author_id == author_id,
                HTRPage.status == PageStatus.CONFIRMED.value,
            )
            .order_by(HTRPage.id)
            .all()
        )
        return [_to_page_view(p) for p in pages]

    # ------------------------------------------------------------------

    def save_recognition(
        self, page_id: int, result: RecognitionResult, model_version_id: int | None
    ) -> PageView:
        page = self._get_orm_page(page_id)
        # re-recognition replaces previous prediction entirely
        for line in list(page.lines):
            self.db.delete(line)
        page.lines = []
        for order, rec_line in enumerate(result.lines):
            line = HTRLine(
                order_index=order,
                x1=rec_line.bbox.x1,
                y1=rec_line.bbox.y1,
                x2=rec_line.bbox.x2,
                y2=rec_line.bbox.y2,
                predicted_text=rec_line.text,
                words_stale=False,
            )
            for word_order, rec_word in enumerate(rec_line.words):
                line.words.append(
                    HTRWord(
                        order_index=word_order,
                        x1=rec_word.bbox.x1,
                        y1=rec_word.bbox.y1,
                        x2=rec_word.bbox.x2,
                        y2=rec_word.bbox.y2,
                        predicted_text=rec_word.text,
                        confidence=rec_word.confidence,
                    )
                )
            page.lines.append(line)
        if result.page_width:
            page.width = result.page_width
        if result.page_height:
            page.height = result.page_height
        page.status = PageStatus.RECOGNIZED.value
        page.recognition_model_version_id = model_version_id
        self.db.commit()
        return _to_page_view(self._get_orm_page(page_id))

    # ------------------------------------------------------------------

    def apply_word_update(
        self,
        page_id: int,
        line_id: int,
        word_id: int,
        corrected_text: str,
        line_corrected_text: str,
        words_stale: bool,
    ) -> PageView:
        page = self._get_orm_page(page_id)
        line = next((l for l in page.lines if l.id == line_id), None)
        if line is None:
            raise NotFoundError(f"Line {line_id} not found on page {page_id}")
        word = next((w for w in line.words if w.id == word_id), None)
        if word is None:
            raise NotFoundError(f"Word {word_id} not found in line {line_id}")
        word.corrected_text = corrected_text
        line.corrected_text = line_corrected_text
        line.words_stale = words_stale
        page.status = PageStatus.EDITING.value
        self.db.commit()
        return _to_page_view(self._get_orm_page(page_id))

    def apply_line_update(self, page_id: int, line_id: int, corrected_text: str) -> PageView:
        page = self._get_orm_page(page_id)
        line = next((l for l in page.lines if l.id == line_id), None)
        if line is None:
            raise NotFoundError(f"Line {line_id} not found on page {page_id}")
        line.corrected_text = corrected_text
        # tokenization may have changed; word alignment is no longer trusted
        line.words_stale = True
        page.status = PageStatus.EDITING.value
        self.db.commit()
        return _to_page_view(self._get_orm_page(page_id))

    def confirm_page(
        self,
        page_id: int,
        confirmed_at: datetime,
        prediction_cer: float | None,
        prediction_wer: float | None,
    ) -> PageView:
        page = self._get_orm_page(page_id)
        page.status = PageStatus.CONFIRMED.value
        page.confirmed_at = confirmed_at
        page.prediction_cer = prediction_cer
        page.prediction_wer = prediction_wer
        self.db.commit()
        return _to_page_view(self._get_orm_page(page_id))
