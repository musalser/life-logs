"""Pydantic schemas of the HTR HTTP API (API layer only)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AuthorCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class AuthorResponse(BaseModel):
    id: int
    name: str
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


# -- recognition result import -------------------------------------------------
# Only used to import results produced by an external HTR tool. The engine that
# ships with this project is used through POST /pages/{page_id}/recognize; the
# geometry of imported lines is validated strictly so that placeholder payloads
# cannot be stored as if they were real predictions.


def _require_positive_area(bbox: BoundingBoxSchema) -> None:
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        raise ValueError("bbox must have a positive area (x2 > x1 and y2 > y1)")


class RecognizedWordIn(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_bbox(self) -> "RecognizedWordIn":
        _require_positive_area(self.bbox)
        return self


class RecognizedLineIn(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    words: list[RecognizedWordIn] = []

    @model_validator(mode="after")
    def _validate_bbox(self) -> "RecognizedLineIn":
        _require_positive_area(self.bbox)
        return self


class RecognitionResultIn(BaseModel):
    page_width: int | None = Field(default=None, gt=0)
    page_height: int | None = Field(default=None, gt=0)
    lines: list[RecognizedLineIn] = Field(min_length=1)


# -- page views ---------------------------------------------------------------

class WordResponse(BaseModel):
    id: int
    order: int
    bbox: BoundingBoxSchema
    predicted_text: str | None
    corrected_text: str | None
    effective_text: str | None
    confidence: float | None
    confidence_level: str


class LineResponse(BaseModel):
    id: int
    order: int
    bbox: BoundingBoxSchema
    predicted_text: str | None
    corrected_text: str | None
    effective_text: str | None
    words_stale: bool
    words: list[WordResponse]


class ConfidenceThresholdsResponse(BaseModel):
    warning: float
    critical: float


class PageResponse(BaseModel):
    page_id: int
    author_id: int
    status: str
    file_path: str
    width: int | None
    height: int | None
    created_at: datetime | None
    confirmed_at: datetime | None
    recognition_model_version_id: int | None
    prediction_cer: float | None
    prediction_wer: float | None
    confidence_thresholds: ConfidenceThresholdsResponse
    lines: list[LineResponse]


class PageUploadResponse(BaseModel):
    page_id: int
    status: str


class WordUpdateRequest(BaseModel):
    corrected_text: str


class LineUpdateRequest(BaseModel):
    corrected_text: str


# -- training -----------------------------------------------------------------

class ModelVersionResponse(BaseModel):
    id: int
    author_id: int
    version: int
    base_model_id: str
    file_path: str
    status: str
    dataset_hash: str | None
    metrics: dict[str, Any] | None
    training_config: dict[str, Any] | None
    created_at: datetime | None


class TrainingResultResponse(BaseModel):
    outcome: str
    author_id: int
    message: str | None = None
    dataset_hash: str | None = None
    training_run_id: int | None = None
    metrics: dict[str, Any] | None = None
    model_version: ModelVersionResponse | None = None
    # progress towards the fine-tuning threshold (INSUFFICIENT_DATA results)
    lines_collected: int | None = None
    lines_required: int | None = None
    words_collected: int | None = None
    words_required: int | None = None


class PageConfirmResponse(BaseModel):
    page: PageResponse
    training: TrainingResultResponse
