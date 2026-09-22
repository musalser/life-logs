"""Opt-in smoke test of the REAL kraken training backend.

Runs a genuine (tiny) fine-tuning of the configured default model and asserts
the exported safetensors model plus CER/WER metrics. It is skipped unless
``HTR_RUN_REAL_TRAINING=1`` because it needs the optional kraken backend, the
downloaded default model and a few minutes of CPU/GPU time::

    HTR_RUN_REAL_TRAINING=1 .venv/bin/python -m pytest tests/test_htr_kraken_training_smoke.py -s

Device can be overridden with ``HTR_TEST_DEVICE`` (default ``cpu``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import settings
from app.htr.domain.entities import ModelRef, TrainingConfig, TrainingDataset, TrainingSample
from app.htr.infrastructure.kraken.trainer import KrakenTrainer

LINES = [
    "Уж очень дед был похож на еврея",
    "и мать моя тоже была похожа",
    "на еврейку только красивее",
    "я помню этот день хорошо",
    "мы шли по длинной улице",
    "и солнце светило в глаза",
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

pytestmark = pytest.mark.skipif(
    os.environ.get("HTR_RUN_REAL_TRAINING") != "1",
    reason="set HTR_RUN_REAL_TRAINING=1 to run real kraken fine-tuning",
)


def _render_lines(directory: Path) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont

    if not Path(FONT).is_file() or not Path(settings.htr_default_model_path).is_file():
        pytest.skip("DejaVu font or default HTR model is not available")

    font = ImageFont.truetype(FONT, 48)
    paths = []
    for index, text in enumerate(LINES):
        probe = ImageDraw.Draw(Image.new("L", (10, 10), 255)).textbbox((0, 0), text, font=font)
        image = Image.new("L", (probe[2] - probe[0] + 40, probe[3] - probe[1] + 40), 255)
        ImageDraw.Draw(image).text((20 - probe[0], 20 - probe[1]), text, font=font, fill=0)
        path = directory / f"line_{index}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def test_real_fine_tuning_workflow(tmp_path):
    pytest.importorskip("kraken")
    # KrakenTrainer disables torch.compile by default; this runs the real path
    image_paths = _render_lines(tmp_path)
    dataset = TrainingDataset(
        author_id=1,
        samples=[
            TrainingSample(page_id=1, line_id=i, image_path=path, transcription=text)
            for i, (path, text) in enumerate(zip(image_paths, LINES))
        ],
        dataset_hash="smoke",
    )
    config = TrainingConfig(
        device=os.environ.get("HTR_TEST_DEVICE", "cpu"),
        epochs=1,
        batch_size=2,
        validation_split=0.34,
        backend_options={"num_workers": 0, "augment": False},
    )
    output = tmp_path / "author_model.safetensors"

    result = KrakenTrainer().train(
        ModelRef(id="default", path=settings.htr_default_model_path),
        dataset,
        str(output),
        config,
    )

    assert Path(result.model_path).is_file()
    assert result.model_path.endswith(".safetensors")
    assert result.holdout_used is False
    assert result.note
    # the exported model must be loadable exactly like the recognizer loads it
    from kraken.models import load_safetensors

    loaded = load_safetensors(result.model_path, tasks=["recognition"])
    assert loaded
    assert result.validation_metrics is not None
    assert "cer" in result.validation_metrics and "wer" in result.validation_metrics
    # the base model is scored on the very same validation split
    assert result.baseline_metrics is not None
    assert "cer" in result.baseline_metrics and "wer" in result.baseline_metrics
