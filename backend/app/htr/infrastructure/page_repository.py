"""SQLAlchemy implementation of PageRepository (maps ORM <-> domain views)."""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ...models import HTRLine, HTRPage, HTRSuggestionEvent, HTRWord
from ..domain.entities import (
    BoundingBox,
    LineView,
    PageStatus,
    PageSummary,
    PageView,
    RecognitionResult,
    WordView,
)
from ..domain.errors import NotFoundError
from .lexicon import AuthorCorpusCache, author_corpus_cache


def _load_polygon(raw: str | None) -> list[tuple[int, int]] | None:
    if not raw:
        return None
    try:
        points = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(points, list) or len(points) < 3:
        return None
    try:
        return [(int(point[0]), int(point[1])) for point in points]
    except (TypeError, IndexError, ValueError):
        return None


def _dump_polygon(points) -> str | None:
    if not points:
        return None
    return json.dumps([[int(x), int(y)] for x, y in points])


def _to_word_view(word: HTRWord) -> WordView:
    return WordView(
        id=word.id,
        order=word.order_index,
        bbox=BoundingBox(word.x1, word.y1, word.x2, word.y2),
        predicted_text=word.predicted_text,
        confidence=word.confidence,
        corrected_text=word.corrected_text,
        polygon=_load_polygon(word.polygon),
    )


def _to_line_view(line: HTRLine) -> LineView:
    return LineView(
        id=line.id,
        order=line.order_index,
        bbox=BoundingBox(line.x1, line.y1, line.x2, line.y2),
        predicted_text=line.predicted_text,
        corrected_text=line.corrected_text,
        corrected_by=getattr(line, "corrected_by", None),
        words=[_to_word_view(w) for w in line.words],
        words_stale=bool(line.words_stale),
        polygon=_load_polygon(line.polygon),
        suggested_text=getattr(line, "suggested_text", None),
        suggested_by=getattr(line, "suggested_by", None),
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
    def __init__(self, db: Session, corpus_cache: AuthorCorpusCache | None = None):
        self.db = db
        # any write to pages/lines changes the author's vocabulary snapshot
        self.corpus_cache = corpus_cache if corpus_cache is not None else author_corpus_cache

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

    @staticmethod
    def _line(page: HTRPage, line_id: int):
        line = next((l for l in page.lines if l.id == line_id), None)
        if line is None:
            raise NotFoundError(f"Line {line_id} not found on page {page.id}")
        return line

    def _invalidate_corpus(self, author_id: int | None) -> None:
        if author_id is not None:
            self.corpus_cache.invalidate(int(author_id))

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
        self._invalidate_corpus(page.author_id)
        return _to_page_view(page)

    def get_page(self, page_id: int) -> PageView | None:
        try:
            return _to_page_view(self._get_orm_page(page_id))
        except NotFoundError:
            return None

    def delete_page(self, page_id: int) -> None:
        page = self._get_orm_page(page_id)
        author_id = page.author_id
        # HTRPage.lines / HTRLine.words cascade, so lines and words go with it
        self.db.delete(page)
        self.db.commit()
        self._invalidate_corpus(author_id)

    def list_page_summaries(self, author_id: int) -> list[PageSummary]:
        rows = (
            self.db.query(
                HTRPage,
                func.count(HTRLine.id).label("line_count"),
            )
            .outerjoin(HTRLine, HTRLine.page_id == HTRPage.id)
            .filter(HTRPage.author_id == author_id)
            .group_by(HTRPage.id)
            .order_by(HTRPage.id.desc())
            .all()
        )
        return [
            PageSummary(
                id=page.id,
                author_id=page.author_id,
                status=PageStatus(page.status),
                file_path=page.file_path,
                created_at=page.created_at,
                confirmed_at=page.confirmed_at,
                line_count=int(line_count or 0),
                prediction_cer=page.prediction_cer,
                prediction_wer=page.prediction_wer,
            )
            for page, line_count in rows
        ]

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
                polygon=_dump_polygon(rec_line.polygon),
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
                        polygon=_dump_polygon(rec_word.polygon),
                    )
                )
            page.lines.append(line)
        if result.page_width:
            page.width = result.page_width
        if result.page_height:
            page.height = result.page_height
        page.status = PageStatus.RECOGNIZED.value
        page.recognition_model_version_id = model_version_id
        # a replaced prediction invalidates any previous confirmation
        page.confirmed_at = None
        page.prediction_cer = None
        page.prediction_wer = None
        self.db.commit()
        self._invalidate_corpus(page.author_id)
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
        line.corrected_by = "user"
        line.words_stale = words_stale
        # the user typed their own text: the pending proposal is obsolete
        line.suggested_text = None
        line.suggested_by = None
        page.status = PageStatus.EDITING.value
        self.db.commit()
        self._invalidate_corpus(page.author_id)
        return _to_page_view(self._get_orm_page(page_id))

    def apply_line_update(self, page_id: int, line_id: int, corrected_text: str) -> PageView:
        page = self._get_orm_page(page_id)
        line = next((l for l in page.lines if l.id == line_id), None)
        if line is None:
            raise NotFoundError(f"Line {line_id} not found on page {page_id}")
        line.corrected_text = corrected_text
        line.corrected_by = "user"
        # tokenization may have changed; word alignment is no longer trusted
        line.words_stale = True
        line.suggested_text = None
        line.suggested_by = None
        page.status = PageStatus.EDITING.value
        self.db.commit()
        self._invalidate_corpus(page.author_id)
        return _to_page_view(self._get_orm_page(page_id))

    def save_line_suggestion(
        self,
        page_id: int,
        line_id: int,
        suggested_text: str,
        suggested_by: str,
    ) -> PageView:
        """Store a model proposal without touching the transcription."""
        page = self._get_orm_page(page_id)
        line = self._line(page, line_id)
        line.suggested_text = suggested_text
        line.suggested_by = suggested_by
        self.db.commit()
        return _to_page_view(self._get_orm_page(page_id))

    def accept_line_suggestions(self, page_id: int, line_ids: list[int]) -> PageView:
        """Apply proposals: the text becomes the user's, the proposal is gone."""
        page = self._get_orm_page(page_id)
        wanted = set(line_ids)
        for line in page.lines:
            if line.id not in wanted or not line.suggested_text:
                continue
            line.corrected_text = line.suggested_text
            line.corrected_by = "user"
            # accepting changes the tokenization as often as typing does
            line.words_stale = True
            line.suggested_text = None
            line.suggested_by = None
            page.status = PageStatus.EDITING.value
        self.db.commit()
        self._invalidate_corpus(page.author_id)
        return _to_page_view(self._get_orm_page(page_id))

    def apply_line_change(self, page_id: int, line_id: int, corrected_text: str) -> PageView:
        """Store a partially accepted proposal and keep the proposal itself."""
        page = self._get_orm_page(page_id)
        line = self._line(page, line_id)
        line.corrected_text = corrected_text
        line.corrected_by = "user"
        # a single accepted word changes the tokenization as much as typing
        line.words_stale = True
        page.status = PageStatus.EDITING.value
        self.db.commit()
        self._invalidate_corpus(page.author_id)
        return _to_page_view(self._get_orm_page(page_id))

    def clear_line_suggestions(self, page_id: int, line_ids: list[int]) -> PageView:
        """Drop proposals the user dismissed."""
        page = self._get_orm_page(page_id)
        wanted = set(line_ids)
        for line in page.lines:
            if line.id in wanted:
                line.suggested_text = None
                line.suggested_by = None
        self.db.commit()
        return _to_page_view(self._get_orm_page(page_id))

    def record_suggestion_event(
        self,
        *,
        page_id: int,
        line_id: int,
        user_id: int,
        action: str,
        suggested_text: str | None = None,
        original_text: str | None = None,
        change_before: str | None = None,
        change_after: str | None = None,
        in_lexicon: bool | None = None,
        suggested_by: str | None = None,
    ) -> None:
        """Append to the feedback log of the LLM correction loop."""
        self.db.add(
            HTRSuggestionEvent(
                page_id=page_id,
                line_id=line_id,
                user_id=user_id,
                action=action,
                suggested_text=suggested_text,
                original_text=original_text,
                change_before=change_before,
                change_after=change_after,
                in_lexicon=in_lexicon,
                suggested_by=suggested_by,
            )
        )
        self.db.commit()

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
        # the transcription is frozen: pending proposals are no longer offered
        for line in page.lines:
            line.suggested_text = None
            line.suggested_by = None
        page.prediction_cer = prediction_cer
        page.prediction_wer = prediction_wer
        self.db.commit()
        self._invalidate_corpus(page.author_id)
        return _to_page_view(self._get_orm_page(page_id))
