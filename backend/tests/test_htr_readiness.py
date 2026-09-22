import pytest

from app.htr.application.readiness import TrainingReadinessPolicy
from app.htr.domain.entities import TrainingDataset, TrainingSample


def dataset(texts: list[str]) -> TrainingDataset:
    samples = [
        TrainingSample(page_id=1, line_id=i, image_path=f"/crops/{i}.png", transcription=t)
        for i, t in enumerate(texts)
    ]
    return TrainingDataset(author_id=1, samples=samples, dataset_hash="h")


def test_below_line_threshold_is_not_ready():
    policy = TrainingReadinessPolicy(min_lines=50)
    readiness = policy.evaluate(dataset(["два слова"] * 31))

    assert readiness.ready is False
    assert readiness.lines == 31
    assert readiness.min_lines == 50
    assert "31/50" in readiness.reason


def test_line_threshold_reached_is_ready():
    policy = TrainingReadinessPolicy(min_lines=3)
    readiness = policy.evaluate(dataset(["раз", "два", "три"]))

    assert readiness.ready is True
    assert readiness.reason is None


def test_word_threshold_is_checked_when_enabled():
    policy = TrainingReadinessPolicy(min_lines=1, min_words=5)
    readiness = policy.evaluate(dataset(["раз два", "три"]))

    assert readiness.ready is False
    assert readiness.words == 3
    assert readiness.min_words == 5
    assert "3/5" in readiness.reason


def test_word_threshold_disabled_by_default():
    policy = TrainingReadinessPolicy(min_lines=1)
    assert policy.evaluate(dataset(["раз"])).ready is True


def test_at_least_one_threshold_must_be_positive():
    with pytest.raises(ValueError):
        TrainingReadinessPolicy(min_lines=0, min_words=0)


def test_empty_dataset_is_not_ready():
    readiness = TrainingReadinessPolicy(min_lines=50).evaluate(dataset([]))
    assert readiness.ready is False
    assert readiness.lines == 0
