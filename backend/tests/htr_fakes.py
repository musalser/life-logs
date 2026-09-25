"""In-memory fakes for HTR unit tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.htr.domain.entities import (
    BoundingBox,
    LineView,
    ModelRef,
    ModelVersionInfo,
    ModelVersionStatus,
    PageStatus,
    PageView,
    RecognitionResult,
    TrainingConfig,
    TrainingDataset,
    TrainingRunResult,
    WordView,
)
from app.htr.domain.errors import CorruptImageError, TrainingError
from app.htr.domain.interfaces import HTRRecognizer, HTRTrainer

DEFAULT_MODEL = ModelRef(id="default", path="/models/default.safetensors")


def make_word(word_id, order=0, text="слово", confidence=0.95, corrected=None):
    return WordView(
        id=word_id,
        order=order,
        bbox=BoundingBox(order * 10, 0, order * 10 + 9, 10),
        predicted_text=text,
        confidence=confidence,
        corrected_text=corrected,
    )


def make_line(line_id, order=0, predicted="Уж очень дед", corrected=None, words=None):
    return LineView(
        id=line_id,
        order=order,
        bbox=BoundingBox(0, order * 20, 100, order * 20 + 19),
        predicted_text=predicted,
        corrected_text=corrected,
        words=words or [],
    )


def make_page(page_id, author_id=1, user_id=1, status=PageStatus.CONFIRMED, lines=None,
              file_path="/pages/p.png"):
    return PageView(
        id=page_id,
        user_id=user_id,
        author_id=author_id,
        file_path=file_path,
        status=status,
        width=100,
        height=100,
        created_at=None,
        lines=lines if lines is not None else [make_line(page_id * 100)],
    )


class FakePageRepository:
    def __init__(self, pages: list[PageView]):
        self.pages = pages

    def get_confirmed_pages(self, author_id: int) -> list[PageView]:
        return [
            p for p in self.pages
            if p.author_id == author_id and p.status == PageStatus.CONFIRMED
        ]


class FakeLineCropper:
    def __init__(self, corrupt_paths: set[str] | None = None):
        self.corrupt_paths = corrupt_paths or set()
        self.calls: list[tuple[str, BoundingBox, str]] = []

    def crop_line(self, page_image_path, bbox, output_path):
        if page_image_path in self.corrupt_paths:
            raise CorruptImageError(f"corrupt: {page_image_path}")
        self.calls.append((page_image_path, bbox, output_path))
        return output_path


class FakeModelRepository:
    def __init__(self, default: ModelRef = DEFAULT_MODEL):
        self._default = default
        self.versions: dict[int, ModelVersionInfo] = {}
        self._next_id = 1

    def get_default_model(self) -> ModelRef:
        return self._default

    def get_active_model(self, author_id: int) -> ModelVersionInfo | None:
        for v in self.versions.values():
            if v.author_id == author_id and v.status == ModelVersionStatus.ACTIVE:
                return v
        return None

    def get_active_model_ref(self, author_id: int) -> ModelRef:
        active = self.get_active_model(author_id)
        if active is None:
            return self._default
        return ModelRef(id=f"author_{author_id}_v{active.version}", path=active.file_path)

    def create_version(self, author_id, base_model_id, training_config, dataset_hash,
                       environment=None) -> ModelVersionInfo:
        number = 1 + max(
            (v.version for v in self.versions.values() if v.author_id == author_id),
            default=0,
        )
        info = ModelVersionInfo(
            id=self._next_id,
            author_id=author_id,
            version=number,
            base_model_id=base_model_id,
            file_path=f"/models/author_{author_id}/v{number}/model.safetensors",
            status=ModelVersionStatus.TRAINING,
            dataset_hash=dataset_hash,
            training_config=training_config,
        )
        self.versions[self._next_id] = info
        self._next_id += 1
        return info

    def mark_ready(self, model_version_id, model_path, metrics):
        v = self.versions[model_version_id]
        v.status = ModelVersionStatus.READY
        v.file_path = model_path
        v.metrics = metrics

    def mark_failed(self, model_version_id, error):
        self.versions[model_version_id].status = ModelVersionStatus.FAILED

    def fail_stale_training(self, author_id, reason):
        stale = [
            v for v in self.versions.values()
            if v.author_id == author_id and v.status == ModelVersionStatus.TRAINING
        ]
        for version in stale:
            version.status = ModelVersionStatus.FAILED
        return len(stale)

    def get_version(self, model_version_id):
        return self.versions.get(model_version_id)

    def activate_model(self, author_id, model_version_id):
        v = self.versions[model_version_id]
        assert v.status == ModelVersionStatus.READY
        for other in self.versions.values():
            if other.author_id == author_id and other.status == ModelVersionStatus.ACTIVE:
                other.status = ModelVersionStatus.READY
        v.status = ModelVersionStatus.ACTIVE

    def clear_active_model(self, author_id):
        for v in self.versions.values():
            if v.author_id == author_id and v.status == ModelVersionStatus.ACTIVE:
                v.status = ModelVersionStatus.READY

    def list_versions(self, author_id):
        return sorted(
            (v for v in self.versions.values() if v.author_id == author_id),
            key=lambda v: v.version,
        )


class FakeTrainingRunRepository:
    def __init__(self):
        self.runs: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    def start_run(self, author_id, dataset_hash):
        run_id = self._next_id
        self._next_id += 1
        self.runs[run_id] = {
            "author_id": author_id,
            "dataset_hash": dataset_hash,
            "status": "RUNNING",
        }
        return run_id

    def finish_run(self, run_id, status, metrics=None, error=None, model_version_id=None):
        self.runs[run_id].update(
            status=status, metrics=metrics, error=error, model_version_id=model_version_id
        )


@dataclass
class TrainerCall:
    base_model: ModelRef
    dataset: TrainingDataset
    output_model_path: str
    config: TrainingConfig


class RecordingTrainer(HTRTrainer):
    def __init__(
        self,
        fail: bool = False,
        write_artifact: bool = False,
        validation_metrics: dict | None = None,
        baseline_metrics: dict | None = None,
    ):
        self.fail = fail
        self.write_artifact = write_artifact
        self.calls: list[TrainerCall] = []
        # default: the fine-tune improves on the base model on the same split
        self.validation_metrics = (
            validation_metrics if validation_metrics is not None
            else {"cer": 0.02, "wer": 0.05}
        )
        self.baseline_metrics = (
            baseline_metrics if baseline_metrics is not None
            else {"cer": 0.10, "wer": 0.20}
        )

    def train(self, base_model, dataset, output_model_path, config):
        self.calls.append(TrainerCall(base_model, dataset, output_model_path, config))
        if self.fail:
            raise TrainingError("training exploded")
        if self.write_artifact:
            from pathlib import Path

            Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_model_path).write_bytes(b"model")
        return TrainingRunResult(
            model_path=output_model_path,
            training_metrics={"loss": 0.1},
            validation_metrics=self.validation_metrics,
            baseline_metrics=self.baseline_metrics,
            holdout_used=False,
        )


@dataclass
class FakeDatasetBuilder:
    dataset: TrainingDataset | None = None
    error: Exception | None = None
    calls: list[int] = field(default_factory=list)

    def build_for_author(self, author_id):
        self.calls.append(author_id)
        if self.error is not None:
            raise self.error
        assert self.dataset is not None
        return self.dataset


class FakeCorrector:
    """Stands in for the LLM corrector: maps predictions to corrections."""

    model = "fake-llm"
    host = "http://fake-ollama:11434"

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        fail: bool = False,
        available: bool = True,
    ):
        self.mapping = dict(mapping or {})
        self.fail = fail
        self.available = available
        self.calls: list[tuple[str, list[str], Any]] = []

    def is_available(self) -> bool:
        return self.available

    def correct_line(self, text: str, context_lines: list[str], context) -> str | None:
        self.calls.append((text, list(context_lines), context))
        if self.fail:
            return None
        return self.mapping.get(text, text)


class FakeWordList:
    """In-memory stand-in for the dictionary artifact."""

    def __init__(self, words: Iterable[str] = (), available: bool = True):
        self.words = frozenset(words)
        self._available = available

    @property
    def is_available(self) -> bool:
        return self._available

    def __contains__(self, word: object) -> bool:
        return word in self.words


class FakeLexiconProvider:
    """Stands in for the dictionary: only ``known`` words are in the lexicon."""

    def __init__(
        self,
        known: Iterable[str] = (),
        texts: dict[int, list[str]] | None = None,
        available: bool = True,
    ):
        from app.htr.infrastructure.lexicon import LayeredLexiconChecker

        self._checker = LayeredLexiconChecker(
            base=None,
            extra={word.casefold() for word in known},
            available=available,
        )
        self.texts = texts or {}

    def checker(self, author_id: int):
        return self._checker

    def page_transcriptions(self, author_id: int):
        return self.texts


class FakeRecognizer(HTRRecognizer):
    """Records the (image, model) pairs it is asked to recognize."""

    def __init__(
        self,
        result: RecognitionResult | None = None,
        error: Exception | None = None,
    ):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def recognize(
        self,
        image_path: str,
        model_path: str,
        author_id: int | None = None,
        word_checker=None,
    ) -> RecognitionResult:
        self.calls.append((image_path, model_path))
        self.last_word_checker = word_checker
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _quad(box, skew=0):
    """Polygon of a bbox with an optional vertical skew (curved baseline)."""
    return [
        (box.x1, box.y1),
        (box.x2, box.y1 + skew),
        (box.x2, box.y2),
        (box.x1, box.y2 - skew),
    ]


def recognition_result(width=200, height=120) -> RecognitionResult:
    """Two lines with word geometry, as a real recognizer would return."""
    from app.htr.domain.entities import RecognizedLine, RecognizedWord

    line1 = BoundingBox(0, 0, width, 40)
    line2 = BoundingBox(0, 50, width, 90)
    words1 = [
        RecognizedWord("w1", BoundingBox(0, 0, 30, 40), "Уж", 0.98,
                       polygon=_quad(BoundingBox(0, 0, 30, 40))),
        RecognizedWord("w2", BoundingBox(35, 0, 90, 40), "очень", 0.55,
                       polygon=_quad(BoundingBox(35, 0, 90, 40), skew=6)),
        RecognizedWord("w3", BoundingBox(95, 0, 130, 40), "дед", 0.95,
                       polygon=_quad(BoundingBox(95, 0, 130, 40))),
    ]
    words2 = [
        RecognizedWord("w4", BoundingBox(0, 50, 40, 90), "на", 0.97,
                       polygon=_quad(BoundingBox(0, 50, 40, 90))),
        RecognizedWord("w5", BoundingBox(45, 50, 120, 90), "еврея", 0.75,
                       polygon=_quad(BoundingBox(45, 50, 120, 90), skew=4)),
    ]
    return RecognitionResult(
        page_width=width,
        page_height=height,
        lines=[
            RecognizedLine(id="l1", bbox=line1, text="Уж очень дед",
                           words=words1, polygon=_quad(line1, skew=8)),
            RecognizedLine(id="l2", bbox=line2, text="на еврея",
                           words=words2, polygon=_quad(line2, skew=6)),
        ],
    )
