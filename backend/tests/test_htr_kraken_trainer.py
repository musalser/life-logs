"""Unit tests of the kraken adapter's framework-independent plumbing.

The real training loop is covered by the integration test (fake trainer) and by
``test_htr_kraken_training_smoke.py``, which runs only when explicitly enabled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app.htr.domain.entities import (
    ModelRef,
    TrainingConfig,
    TrainingDataset,
    TrainingSample,
)
from app.htr.domain.errors import TrainingError
from app.htr.infrastructure.kraken.trainer import KrakenTrainer


# -- fakes standing in for kraken config/data classes -------------------------


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakePPOCRModule:
    _arch = "ppocrv6"
    _config_class = FakeConfig
    _data_config_class = FakeConfig
    _data_module_class = staticmethod(lambda config: config)


class FakeVGSLModule:
    _arch = "vgsl"
    _config_class = FakeConfig
    _data_config_class = FakeConfig
    _data_module_class = staticmethod(lambda config: config)


def make_gt_dir(tmp_path, count=4):
    paths = []
    for i in range(count):
        image = tmp_path / f"line_{i}.png"
        image.write_bytes(b"png")
        paths.append(str(image))
    return paths


def make_dataset(paths):
    samples = [
        TrainingSample(page_id=1, line_id=i, image_path=p, transcription=f"строка {i}")
        for i, p in enumerate(paths)
    ]
    return TrainingDataset(author_id=1, samples=samples, dataset_hash="h")


# -- device mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    "device,expected",
    [
        ("cpu", ("cpu", "auto")),
        ("auto", ("auto", "auto")),
        ("cuda:0", ("gpu", [0])),
        ("cuda:1", ("gpu", [1])),
        ("cuda", ("gpu", "auto")),
        ("gpu", ("gpu", "auto")),
        ("mps", ("mps", "auto")),
    ],
)
def test_device_settings(device, expected):
    assert KrakenTrainer._device_settings(device) == expected


# -- ground truth materialization ---------------------------------------------


def test_ground_truth_pairs_are_written(tmp_path):
    paths = make_gt_dir(tmp_path)
    dataset = make_dataset(paths)
    config = TrainingConfig(validation_split=0.25, random_seed=1)

    train_files, eval_files = KrakenTrainer._prepare_ground_truth(dataset, config, tmp_path / "work")

    assert len(train_files) + len(eval_files) == 4
    assert len(eval_files) == 1
    for image in train_files + eval_files:
        assert image.endswith(".png")
        gt = image[: -len(".png")] + ".gt.txt"
        assert gt.endswith(".gt.txt")
        assert Path(gt).is_file()


def test_ground_truth_split_is_deterministic(tmp_path):
    paths = make_gt_dir(tmp_path, count=6)
    dataset = make_dataset(paths)
    config = TrainingConfig(validation_split=0.33, random_seed=7)

    first = KrakenTrainer._prepare_ground_truth(dataset, config, tmp_path / "a")
    second = KrakenTrainer._prepare_ground_truth(dataset, config, tmp_path / "b")

    names = lambda files: [Path(f).name for f in files]  # noqa: E731
    assert names(first[1]) == names(second[1])
    assert names(first[0]) == names(second[0])

def test_ground_truth_requires_two_samples(tmp_path):
    dataset = make_dataset(make_gt_dir(tmp_path, count=1))
    with pytest.raises(TrainingError):
        KrakenTrainer._prepare_ground_truth(dataset, TrainingConfig(), tmp_path / "work")


def test_missing_crop_fails(tmp_path):
    dataset = make_dataset([str(tmp_path / "nope.png")])
    with pytest.raises(TrainingError):
        KrakenTrainer._prepare_ground_truth(dataset, TrainingConfig(), tmp_path / "work")


# -- config construction ------------------------------------------------------


def test_ppocr_config_gets_architecture_specific_options(tmp_path):
    config = TrainingConfig(
        device="cpu",
        epochs=5,
        batch_size=2,
        learning_rate=1e-4,
        backend_options={
            "variant": "medium",
            "height": 128,
            "max_width": 1024,
            "resize": "union",
            "num_workers": 0,
            "augment": True,
        },
    )
    trainer = KrakenTrainer()

    model_config = trainer._build_model_config(FakePPOCRModule, "ppocrv6", config, tmp_path)
    data_config = trainer._data_config(FakePPOCRModule, config, ["a.png"], ["b.png"])

    assert model_config.variant == "medium"
    assert model_config.height == 128
    assert model_config.resize == "union"
    assert model_config.accelerator == "cpu"
    assert model_config.weights_format == "safetensors"
    assert model_config.quit == "fixed"
    assert data_config.max_width == 1024
    assert data_config.augment is True
    assert data_config.num_workers == 0


def test_data_config_uses_user_text_verbatim(tmp_path):
    config = TrainingConfig(validation_split=0.1)
    trainer = KrakenTrainer()

    data_config = trainer._data_config(FakePPOCRModule, config, ["a.png", "b.png"], ["c.png"])

    # no whitespace/spelling/case changes to the user's ground truth
    assert data_config.normalize_whitespace is False
    # NFD is canonical equivalence and must match the model's decomposed codec
    assert data_config.normalization == "NFD"
    assert data_config.format_type == "path"
    assert data_config.evaluation_data == ["c.png"]
    # explicit evaluation data disables kraken's random partition
    assert data_config.partition == 1.0


def test_normalization_can_be_disabled_explicitly(tmp_path):
    config = TrainingConfig(backend_options={"normalization": None})
    data_config = KrakenTrainer()._data_config(FakePPOCRModule, config, ["a.png"], ["b.png"])
    assert data_config.normalization is None


def test_validation_data_config_uses_test_data(tmp_path):
    config = TrainingConfig()
    data_config = KrakenTrainer()._data_config(
        FakePPOCRModule, config, ["a.png"], ["b.png"], test_files=["b.png"]
    )
    assert data_config.test_data == ["b.png"]
    assert data_config.evaluation_data is None
    # kraken only builds a test_set when no training data is configured
    assert data_config.training_data is None


def test_vgsl_config_has_no_ppocr_keys(tmp_path):
    config = TrainingConfig(device="cuda:0")
    model_config = KrakenTrainer()._build_model_config(FakeVGSLModule, "vgsl", config, tmp_path)
    assert not hasattr(model_config, "variant")
    assert model_config.accelerator == "gpu"


# -- guards -------------------------------------------------------------------


def test_train_rejects_missing_base_model():
    dataset = make_dataset(["/crops/1.png", "/crops/2.png"])
    with pytest.raises(TrainingError, match="does not exist"):
        KrakenTrainer().train(
            ModelRef(id="default", path="/nope/default.safetensors"),
            dataset,
            "/out/model.safetensors",
            TrainingConfig(),
        )


def test_train_rejects_empty_dataset():
    with pytest.raises(TrainingError, match="empty"):
        KrakenTrainer().train(
            ModelRef(id="default", path=__file__),
            TrainingDataset(author_id=1, samples=[], dataset_hash="h"),
            "/out/model.safetensors",
            TrainingConfig(),
        )


def test_runtime_disables_torch_compile_by_default(monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")  # restored on teardown
    KrakenTrainer._configure_runtime(TrainingConfig())
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    # torch.compile must become a no-op, not merely be configured
    import torch

    def fn(x):
        return x + 1

    assert torch.compile(fn) is fn


def test_runtime_keeps_torch_compile_when_enabled(monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")  # restored on teardown
    KrakenTrainer._configure_runtime(
        TrainingConfig(backend_options={"compile": True})
    )
    assert "TORCHDYNAMO_DISABLE" not in os.environ


def test_runtime_sets_matmul_precision_on_gpu():
    import torch

    previous = torch.get_float32_matmul_precision()
    try:
        KrakenTrainer._configure_runtime(
            TrainingConfig(device="cuda:0", backend_options={"matmul_precision": "high"})
        )
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_train_reports_missing_kraken_backend(monkeypatch, tmp_path):
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"x")
    dataset = make_dataset(["/crops/1.png", "/crops/2.png"])

    monkeypatch.setitem(sys.modules, "kraken.train", None)
    with pytest.raises(TrainingError, match="not installed"):
        KrakenTrainer().train(
            ModelRef(id="default", path=str(base)),
            dataset,
            str(tmp_path / "out.safetensors"),
            TrainingConfig(),
        )
