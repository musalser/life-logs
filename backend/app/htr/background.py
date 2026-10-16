"""Background recognition: a page is read before anyone opens it.

Recognition used to start when the user opened an uploaded page, so every page
cost a minute of waiting in front of a spinner. This queue recognizes uploaded
pages on its own — right after an upload or a PDF import, and the pages still
waiting when the backend starts — so by the time the user has corrected one
page, the next ones are ready.

Rules:

* only ``UPLOADED`` pages are touched; a page that was recognized meanwhile (or
  edited, confirmed, deleted) is skipped, so the queue never overwrites work;
* one lock per page is shared with the HTTP route: a user who presses
  «Распознать» while the queue is reading that page waits for it and gets its
  result instead of a second run;
* a failure is logged and the page stays ``UPLOADED`` (the button still works);
  the same page is not retried by the queue until the backend restarts;
* the queue waits while a model is being trained — both want the GPU, and the
  training run must not be slowed down by pages nobody is looking at yet.

The queue lives in the backend process (threads, not Celery): recognition needs
the prepared models that the process already holds. Several pages run at once
(``htr_background_workers``) because a page spends most of its time on the CPU
(segmentation, beam search) and only a few seconds on the GPU.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

#: how long a worker sleeps while a model is being trained before it looks again
TRAINING_POLL_SECONDS = 15.0

QUEUED = "queued"
RUNNING = "running"
IDLE = "idle"


class RecognitionQueue:
    """Recognizes uploaded pages in worker threads.

    ``recognize_if_uploaded(page_id)`` does the actual work in its own database
    session and returns True when the page was recognized; ``pending_pages()``
    lists the pages waiting when the queue starts. Both are injected, so the
    queue itself knows nothing about SQLAlchemy or kraken.
    """

    def __init__(
        self,
        recognize_if_uploaded: Callable[[int], bool],
        pending_pages: Callable[[], Iterable[int]] = lambda: (),
        workers: int = 2,
        training_in_progress: Callable[[], bool] = lambda: False,
    ):
        self._recognize = recognize_if_uploaded
        self._pending_pages = pending_pages
        self._workers = max(1, int(workers))
        self._training_in_progress = training_in_progress
        self._queue: "queue.Queue[int]" = queue.Queue()
        self._queued: set[int] = set()
        self._running: set[int] = set()
        self._failed: set[int] = set()
        self._page_locks: dict[int, threading.Lock] = {}
        self._state = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()

    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._state:
            if self._threads:
                return
            for index in range(self._workers):
                thread = threading.Thread(
                    target=self._work, name=f"htr-recognition-{index}", daemon=True
                )
                self._threads.append(thread)
                thread.start()
        logger.info("HTR background recognition: %s worker thread(s)", self._workers)

    def stop(self) -> None:
        self._stopping.set()

    def enqueue(self, page_ids: Iterable[int]) -> int:
        """Queue pages for recognition; returns how many were actually added."""
        added = 0
        with self._state:
            for page_id in page_ids:
                page_id = int(page_id)
                if page_id in self._queued or page_id in self._running or page_id in self._failed:
                    continue
                self._queued.add(page_id)
                self._queue.put(page_id)
                added += 1
        if added:
            self.start()
        return added

    def enqueue_pending(self) -> int:
        """Queue every page that is still waiting for recognition."""
        try:
            pending = list(self._pending_pages())
        except Exception:  # pragma: no cover - a startup query must not kill the app
            logger.exception("HTR background recognition: cannot list pending pages")
            return 0
        added = self.enqueue(pending)
        if added:
            logger.info("HTR background recognition: %s waiting page(s) queued", added)
        return added

    def state(self, page_id: int) -> str:
        with self._state:
            if page_id in self._running:
                return RUNNING
            if page_id in self._queued:
                return QUEUED
        return IDLE

    @contextmanager
    def page_lock(self, page_id: int):
        """Hold the page for a recognition; yields True if the queue had it.

        The HTTP route recognizes inside this block. When the queue was reading
        the page at that moment, the route waited for it, and the page may well
        be recognized by now — the route then returns that result.
        """
        was_running = self.state(page_id) == RUNNING
        lock = self._lock_for(page_id)
        with lock:
            yield was_running

    # ------------------------------------------------------------------

    def _lock_for(self, page_id: int) -> threading.Lock:
        with self._state:
            return self._page_locks.setdefault(page_id, threading.Lock())

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                page_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            while self._training_in_progress() and not self._stopping.is_set():
                time.sleep(TRAINING_POLL_SECONDS)
            lock = self._lock_for(page_id)
            with lock:
                with self._state:
                    self._queued.discard(page_id)
                    self._running.add(page_id)
                started = time.monotonic()
                try:
                    if self._recognize(page_id):
                        logger.info(
                            "HTR background recognition: page %s done in %.1f s",
                            page_id, time.monotonic() - started,
                        )
                except Exception:
                    with self._state:
                        self._failed.add(page_id)
                    logger.exception("HTR background recognition: page %s failed", page_id)
                finally:
                    with self._state:
                        self._running.discard(page_id)
            self._queue.task_done()

    def join(self, timeout: float | None = None) -> bool:
        """Wait until the queue is empty (tests and scripts)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._state:
                if not self._queued and not self._running:
                    return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# The process-wide queue, wired to the database and the page service
# ---------------------------------------------------------------------------

_QUEUE: RecognitionQueue | None = None
_QUEUE_LOCK = threading.Lock()


def _recognize_if_uploaded(page_id: int) -> bool:
    from ..db import SessionLocal
    from .domain.entities import PageStatus
    from .domain.errors import NotFoundError
    from .factory import build_page_service

    db = SessionLocal()
    try:
        service = build_page_service(db)
        try:
            page = service.get_page(page_id)
        except NotFoundError:
            return False
        if page.status != PageStatus.UPLOADED:
            return False
        service.recognize_page(page_id)
        return True
    finally:
        db.close()


def _pending_pages() -> list[int]:
    from ..db import SessionLocal
    from ..models import HTRPage

    db = SessionLocal()
    try:
        rows = (
            db.query(HTRPage.id)
            .filter(HTRPage.status == "UPLOADED")
            .order_by(HTRPage.author_id, HTRPage.order_index, HTRPage.id)
            .all()
        )
        return [row.id for row in rows]
    finally:
        db.close()


def _training_in_progress() -> bool:
    from .application.training_service import any_training_in_progress

    return any_training_in_progress()


def get_recognition_queue() -> RecognitionQueue:
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE is None:
            from ..config import settings

            _QUEUE = RecognitionQueue(
                recognize_if_uploaded=_recognize_if_uploaded,
                pending_pages=_pending_pages,
                workers=settings.htr_background_workers,
                training_in_progress=_training_in_progress,
            )
        return _QUEUE


def background_enabled() -> bool:
    from ..config import settings

    return bool(settings.htr_background_recognition)


def enqueue_pages(page_ids: Iterable[int]) -> None:
    """Queue freshly uploaded pages, when background recognition is on."""
    if not background_enabled():
        return
    get_recognition_queue().enqueue(page_ids)
