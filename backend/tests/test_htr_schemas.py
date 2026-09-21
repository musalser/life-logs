"""Validation of the external recognition-result import payload.

The reported bug was that an unfilled Swagger example body (bbox 0,0,0,0 and
text "string") was accepted and stored as a prediction. Degenerate geometry must
be rejected loudly instead.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.htr.schemas import RecognitionResultIn


def payload(lines=None, **overrides):
    body = {
        "page_width": 4640,
        "page_height": 3472,
        "lines": lines
        if lines is not None
        else [{"bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 40}, "text": "Уж очень дед"}],
    }
    body.update(overrides)
    return body


def test_accepts_well_formed_result():
    result = RecognitionResultIn(**payload())
    assert result.lines[0].text == "Уж очень дед"
    assert result.lines[0].words == []


def test_accepts_words_with_geometry():
    result = RecognitionResultIn(
        **payload(
            lines=[
                {
                    "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 40},
                    "text": "Уж очень",
                    "words": [
                        {"bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 40}, "text": "Уж", "confidence": 0.9},
                    ],
                }
            ]
        )
    )
    assert result.lines[0].words[0].confidence == pytest.approx(0.9)


def test_rejects_the_swagger_placeholder_body():
    with pytest.raises(ValidationError):
        RecognitionResultIn(
            **payload(
                lines=[{"bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}, "text": "string"}]
            )
        )


def test_rejects_negative_area_or_inverted_boxes():
    with pytest.raises(ValidationError):
        RecognitionResultIn(
            **payload(lines=[{"bbox": {"x1": 10, "y1": 0, "x2": 5, "y2": 40}, "text": "x"}])
        )
    with pytest.raises(ValidationError):
        RecognitionResultIn(
            **payload(lines=[{"bbox": {"x1": 0, "y1": 40, "x2": 100, "y2": 40}, "text": "x"}])
        )


def test_rejects_degenerate_word_boxes():
    with pytest.raises(ValidationError):
        RecognitionResultIn(
            **payload(
                lines=[
                    {
                        "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 40},
                        "text": "Уж",
                        "words": [
                            {"bbox": {"x1": 5, "y1": 5, "x2": 5, "y2": 5}, "text": "Уж", "confidence": 0.5},
                        ],
                    }
                ]
            )
        )


def test_rejects_result_without_lines():
    with pytest.raises(ValidationError):
        RecognitionResultIn(**payload(lines=[]))


def test_rejects_non_positive_page_size():
    with pytest.raises(ValidationError):
        RecognitionResultIn(**payload(page_width=0))
    with pytest.raises(ValidationError):
        RecognitionResultIn(**payload(page_height=-1))


def test_page_size_is_optional():
    result = RecognitionResultIn(**payload(page_width=None, page_height=None))
    assert result.page_width is None
