"""The background recognition queue."""
from __future__ import annotations

import threading
import time

from app.htr import background
from app.htr.background import QUEUED, RUNNING, RecognitionQueue


def test_queued_pages_are_recognized_once():
    seen = []
    queue = RecognitionQueue(lambda page_id: seen.append(page_id) or True, workers=2)

    assert queue.enqueue([1, 2, 2, 3]) == 3
    assert queue.join(timeout=5)
    assert sorted(seen) == [1, 2, 3]


def test_a_failed_page_is_not_retried():
    calls = []

    def recognize(page_id):
        calls.append(page_id)
        raise RuntimeError("no model")

    queue = RecognitionQueue(recognize, workers=1)
    queue.enqueue([5])
    assert queue.join(timeout=5)
    assert queue.enqueue([5]) == 0
    assert calls == [5]


def test_pending_pages_are_queued_at_start():
    seen = []
    queue = RecognitionQueue(lambda page_id: seen.append(page_id) or True, pending_pages=lambda: [7, 8])

    assert queue.enqueue_pending() == 2
    assert queue.join(timeout=5)
    assert sorted(seen) == [7, 8]


def test_the_route_waits_for_a_page_the_queue_is_reading():
    started, release = threading.Event(), threading.Event()

    def recognize(page_id):
        started.set()
        release.wait(5)
        return True

    queue = RecognitionQueue(recognize, workers=1)
    queue.enqueue([11])
    assert started.wait(5)
    assert queue.state(11) == RUNNING

    waited = []

    def route():
        with queue.page_lock(11) as queue_had_it:
            waited.append(queue_had_it)

    thread = threading.Thread(target=route)
    thread.start()
    time.sleep(0.1)
    assert waited == []          # still blocked behind the queue
    release.set()
    thread.join(5)
    assert waited == [True]


def test_the_queue_waits_while_a_model_is_trained(monkeypatch):
    monkeypatch.setattr(background, "TRAINING_POLL_SECONDS", 0.05)
    training = {"on": True}
    seen = []
    queue = RecognitionQueue(
        lambda page_id: seen.append(page_id) or True,
        workers=1,
        training_in_progress=lambda: training["on"],
    )
    queue.enqueue([3])
    time.sleep(0.3)
    assert seen == [] and queue.state(3) == QUEUED
    training["on"] = False
    assert queue.join(timeout=5)
    assert seen == [3]


def test_uploads_are_not_queued_when_background_recognition_is_off(monkeypatch):
    monkeypatch.setattr(background, "get_recognition_queue", lambda: (_ for _ in ()).throw(AssertionError))
    background.enqueue_pages([1])  # settings.htr_background_recognition is False in tests
