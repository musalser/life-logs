"""In-memory fakes for HTR unit tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.htr.domain.entities import (
    BoundingBox,
    LineView,
    ModelRef,
    ModelVersionInfo,
    ModelVersionStatus,
    PageStatus,
    PageView,
    TrainingConfig,
    TrainingDataset,
    TrainingRunResult,
    WordView,
)
from app.htr.domain.errors import CorruptImageError, TrainingError
from app.htr.domain.interfaces import HTRTrainer

DEFAULT_MODEL = ModelRef(id="default", path="/models/default.mlmodel")


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
            file_path=f"/models/author_{author_id}/v{number}/model.mlmodel",
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

    def activate_model(self, author_id, model_version_id):
        v = self.versions[model_version_id]
        assert v.status == ModelVersionStatus.READY
        for other in self.versions.values():
            if other.author_id == author_id and other.status == ModelVersionStatus.ACTIVE:
                other.status = ModelVersionStatus.READY
        v.status = ModelVersionStatus.ACTIVE

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
    def __init__(self, fail: bool = False, write_artifact: bool = False):
        self.fail = fail
        self.write_artifact = write_artifact
        self.calls: list[TrainerCall] = []

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
            validation_metrics={"cer": 0.02, "wer": 0.05},
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
