"""SQLAlchemy implementations of ModelRepository and TrainingRunRepository."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ...models import HTRModelVersion, HTRTrainingRun
from ..domain.entities import ModelRef, ModelVersionInfo, ModelVersionStatus
from ..domain.errors import HTRError, NotFoundError
from .storage import HTRStorage


def _to_version_info(row: HTRModelVersion) -> ModelVersionInfo:
    return ModelVersionInfo(
        id=row.id,
        author_id=row.author_id,
        version=row.version,
        base_model_id=row.base_model_id,
        file_path=row.file_path,
        status=ModelVersionStatus(row.status),
        dataset_hash=row.dataset_hash,
        metrics=json.loads(row.metrics) if row.metrics else None,
        training_config=json.loads(row.training_config) if row.training_config else None,
        created_at=row.created_at,
    )


class SqlAlchemyModelRepository:
    def __init__(self, db: Session, storage: HTRStorage, default_model: ModelRef):
        self.db = db
        self.storage = storage
        self.default_model = default_model

    def get_default_model(self) -> ModelRef:
        return self.default_model

    def get_active_model(self, author_id: int) -> ModelVersionInfo | None:
        row = (
            self.db.query(HTRModelVersion)
            .filter(
                HTRModelVersion.author_id == author_id,
                HTRModelVersion.status == ModelVersionStatus.ACTIVE.value,
            )
            .first()
        )
        return _to_version_info(row) if row else None

    def get_active_model_ref(self, author_id: int) -> ModelRef:
        active = self.get_active_model(author_id)
        if active is None:
            return self.default_model
        return ModelRef(id=f"author_{author_id}_v{active.version}", path=active.file_path)

    def create_version(
        self,
        author_id: int,
        base_model_id: str,
        training_config: dict[str, Any],
        dataset_hash: str,
        environment: dict[str, Any] | None = None,
    ) -> ModelVersionInfo:
        max_version = (
            self.db.query(HTRModelVersion.version)
            .filter(HTRModelVersion.author_id == author_id)
            .order_by(HTRModelVersion.version.desc())
            .limit(1)
            .scalar()
        )
        version = (max_version or 0) + 1
        row = HTRModelVersion(
            author_id=author_id,
            version=version,
            base_model_id=base_model_id,
            file_path=self.storage.model_output_path(author_id, version),
            status=ModelVersionStatus.TRAINING.value,
            dataset_hash=dataset_hash,
            training_config=json.dumps(training_config, ensure_ascii=False),
            environment=json.dumps(environment, ensure_ascii=False) if environment else None,
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return _to_version_info(row)

    def _get_row(self, model_version_id: int) -> HTRModelVersion:
        row = self.db.query(HTRModelVersion).filter(HTRModelVersion.id == model_version_id).first()
        if row is None:
            raise NotFoundError(f"Model version {model_version_id} not found")
        return row

    def mark_ready(self, model_version_id: int, model_path: str, metrics: dict[str, Any]) -> None:
        row = self._get_row(model_version_id)
        row.status = ModelVersionStatus.READY.value
        row.file_path = model_path
        row.metrics = json.dumps(metrics, ensure_ascii=False)
        self.db.commit()

    def mark_failed(self, model_version_id: int, error: str) -> None:
        row = self._get_row(model_version_id)
        row.status = ModelVersionStatus.FAILED.value
        row.error = error
        self.db.commit()

    def fail_stale_training(self, author_id: int, reason: str) -> int:
        rows = (
            self.db.query(HTRModelVersion)
            .filter(
                HTRModelVersion.author_id == author_id,
                HTRModelVersion.status == ModelVersionStatus.TRAINING.value,
            )
            .all()
        )
        for row in rows:
            row.status = ModelVersionStatus.FAILED.value
            row.error = reason
        if rows:
            self.db.commit()
        return len(rows)

    def get_version(self, model_version_id: int) -> ModelVersionInfo | None:
        row = (
            self.db.query(HTRModelVersion)
            .filter(HTRModelVersion.id == model_version_id)
            .first()
        )
        return _to_version_info(row) if row else None

    def activate_model(self, author_id: int, model_version_id: int) -> None:
        row = self._get_row(model_version_id)
        if row.author_id != author_id:
            raise HTRError(
                f"Model version {model_version_id} does not belong to author {author_id}"
            )
        if row.status != ModelVersionStatus.READY.value:
            raise HTRError(
                f"Only READY model versions can be activated, got {row.status}"
            )
        # single active model per author
        (
            self.db.query(HTRModelVersion)
            .filter(
                HTRModelVersion.author_id == author_id,
                HTRModelVersion.status == ModelVersionStatus.ACTIVE.value,
            )
            .update({HTRModelVersion.status: ModelVersionStatus.READY.value})
        )
        row.status = ModelVersionStatus.ACTIVE.value
        row.activated_at = datetime.now(timezone.utc)
        self.db.commit()

    def clear_active_model(self, author_id: int) -> None:
        """Roll back to the default model: no version of the author is active."""
        (
            self.db.query(HTRModelVersion)
            .filter(
                HTRModelVersion.author_id == author_id,
                HTRModelVersion.status == ModelVersionStatus.ACTIVE.value,
            )
            .update({HTRModelVersion.status: ModelVersionStatus.READY.value})
        )
        self.db.commit()

    def list_versions(self, author_id: int) -> list[ModelVersionInfo]:
        rows = (
            self.db.query(HTRModelVersion)
            .filter(HTRModelVersion.author_id == author_id)
            .order_by(HTRModelVersion.version)
            .all()
        )
        return [_to_version_info(r) for r in rows]


class SqlAlchemyTrainingRunRepository:
    def __init__(self, db: Session):
        self.db = db

    def start_run(self, author_id: int, dataset_hash: str) -> int:
        run = HTRTrainingRun(
            author_id=author_id,
            dataset_hash=dataset_hash or None,
            status="RUNNING",
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run.id

    def finish_run(
        self,
        run_id: int,
        status: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        model_version_id: int | None = None,
    ) -> None:
        run = self.db.query(HTRTrainingRun).filter(HTRTrainingRun.id == run_id).first()
        if run is None:
            raise NotFoundError(f"Training run {run_id} not found")
        run.status = status
        run.finished_at = datetime.now(timezone.utc)
        run.error = error
        run.metrics = json.dumps(metrics, ensure_ascii=False) if metrics else None
        run.model_version_id = model_version_id
        self.db.commit()
