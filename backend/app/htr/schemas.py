"""Pydantic schemas of the HTR HTTP API (API layer only)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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


# -- recognition result import (until a real HTRRecognizer is wired in) ------

class RecognizedWordIn(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class RecognizedLineIn(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    words: list[RecognizedWordIn] = []


class RecognitionResultIn(BaseModel):
    page_width: int | None = None
    page_height: int | None = None
    lines: list[RecognizedLineIn]


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


class PageConfirmResponse(BaseModel):
    page: PageResponse
    training: TrainingResultResponse
