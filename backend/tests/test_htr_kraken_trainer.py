"""Unit tests of the kraken adapter's framework-independent plumbing.

The real training loop is covered by the integration test (fake trainer) and by
``test_htr_kraken_training_smoke.py``, which runs only when explicitly enabled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


class FakeNet(torch.nn.Module):
    """A PP-OCRv6-shaped net: a backbone plus an exposed output projection."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4, 4),
        )
        self.output = torch.nn.Linear(4, 2)

    @property
    def _output_proj(self):
        return self.output


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


def test_runtime_device_falls_back_to_cpu_without_cuda(monkeypatch):
    from app.htr.infrastructure.kraken import recognizer as module

    monkeypatch.setattr(module, "gpu_available", lambda: False)
    assert KrakenTrainer._runtime_device(TrainingConfig(device="cuda:0")) == ("cpu", "auto")

    monkeypatch.setattr(module, "gpu_available", lambda: True)
    assert KrakenTrainer._runtime_device(TrainingConfig(device="cuda:0")) == ("gpu", [0])


# -- freezing the backbone ----------------------------------------------------


def test_freeze_backbone_auto_is_one_epoch():
    trainer = KrakenTrainer()
    # 62 samples / batch 4 -> 16 steps per epoch
    assert trainer._freeze_backbone_steps(
        TrainingConfig(batch_size=4, backend_options={"freeze_backbone": -1}), 62
    ) == 16
    # never zero: one epoch of a tiny corpus must still freeze something
    assert trainer._freeze_backbone_steps(
        TrainingConfig(batch_size=64, backend_options={"freeze_backbone": -1}), 3
    ) == 1


def test_freeze_backbone_can_be_disabled_or_set_exactly():
    trainer = KrakenTrainer()
    assert trainer._freeze_backbone_steps(
        TrainingConfig(backend_options={"freeze_backbone": 0}), 100
    ) == 0
    assert trainer._freeze_backbone_steps(
        TrainingConfig(backend_options={"freeze_backbone": 500}), 100
    ) == 500
    # missing key means "no freezing", so an old config cannot change behaviour
    assert trainer._freeze_backbone_steps(TrainingConfig(), 100) == 0


def test_freeze_callback_keeps_only_the_output_projection_trainable():
    from app.htr.infrastructure.kraken.callbacks import FreezeBackboneForSteps

    net = FakeNet()
    module = SimpleNamespace(net=net)
    callback = FreezeBackboneForSteps(unfreeze_at_step=5)

    callback.on_train_start(trainer=None, pl_module=module)

    assert net.backbone[0].weight.requires_grad is False
    assert net.backbone[3].weight.requires_grad is False
    assert net.output.weight.requires_grad is True
    assert net.output.bias.requires_grad is True
    # a frozen backbone must not keep updating its BatchNorm statistics either
    assert net.backbone.training is False
    assert net.output.training is True


def test_freeze_callback_unfreezes_after_the_configured_step():
    from app.htr.infrastructure.kraken.callbacks import FreezeBackboneForSteps

    net = FakeNet()
    module = SimpleNamespace(net=net)
    callback = FreezeBackboneForSteps(unfreeze_at_step=3)
    callback.on_train_start(trainer=None, pl_module=module)

    callback.on_train_batch_start(SimpleNamespace(global_step=1), module, None, 0)
    assert net.backbone[0].weight.requires_grad is False

    # Lightning switches the model back to train mode after a validation run;
    # while frozen the callback re-asserts its state on every batch
    net.backbone.train()
    callback.on_train_batch_start(SimpleNamespace(global_step=1), module, None, 0)
    assert net.backbone.training is False

    # a step that jumped over the threshold still unfreezes (kraken uses `==`)
    callback.on_train_batch_start(SimpleNamespace(global_step=4), module, None, 0)
    assert net.backbone[0].weight.requires_grad is True
    assert net.backbone[3].weight.requires_grad is True
    assert net.output.weight.requires_grad is True
    assert net.backbone.training is True


def test_batch_norm_statistics_stay_frozen_after_unfreezing():
    """Batch of 4 crops cannot re-estimate 61 pretrained BatchNorm layers."""
    from app.htr.infrastructure.kraken.callbacks import FreezeBackboneForSteps

    net = FakeNet()
    module = SimpleNamespace(net=net)
    callback = FreezeBackboneForSteps(unfreeze_at_step=1)
    callback.on_train_start(trainer=None, pl_module=module)
    batch_norm = net.backbone[1]
    assert batch_norm.training is False

    callback.on_train_batch_start(SimpleNamespace(global_step=5), module, None, 0)

    # the backbone is trainable again, but its statistics are not re-estimated
    assert net.backbone[3].weight.requires_grad is True
    assert batch_norm.training is False
    assert net.output.training is True


def test_batch_norm_freezing_can_be_disabled():
    from app.htr.infrastructure.kraken.callbacks import FreezeBackboneForSteps

    net = FakeNet()
    module = SimpleNamespace(net=net)
    callback = FreezeBackboneForSteps(unfreeze_at_step=1, freeze_batch_norm=False)
    callback.on_train_start(trainer=None, pl_module=module)
    callback.on_train_batch_start(SimpleNamespace(global_step=5), module, None, 0)

    net.backbone.train()  # what Lightning does after every validation
    callback.on_train_batch_start(SimpleNamespace(global_step=6), module, None, 0)
    assert net.backbone[1].training is True


def test_ppocrv6_is_trained_without_the_auxiliary_nrtr_head():
    """The base checkpoint has no NRTR head; a random one must not steer training."""
    pytest.importorskip("kraken.train.ppocr")
    from app.htr.infrastructure.kraken.modules import CtcFineTunePPOCRv6

    without = TrainingConfig(backend_options={"aux_nrtr": False})
    with_head = TrainingConfig(backend_options={"aux_nrtr": True})

    assert KrakenTrainer._training_module_class(FakePPOCRModule, without) is CtcFineTunePPOCRv6
    assert KrakenTrainer._training_module_class(FakePPOCRModule, with_head) is FakePPOCRModule
    # other architectures are left to kraken
    assert KrakenTrainer._training_module_class(FakeVGSLModule, without) is FakeVGSLModule


def test_ctc_only_module_zeroes_the_auxiliary_loss():
    pytest.importorskip("kraken.train.ppocr")
    from app.htr.infrastructure.kraken.modules import CtcFineTunePPOCRv6

    loss = CtcFineTunePPOCRv6._gtc_loss(None, torch.zeros(2, 5), None, None)
    assert loss.shape == ()
    assert float(loss) == 0.0


def test_freeze_callback_falls_back_to_the_last_layer_of_a_sequential_net():
    """A VGSL net is an nn.Sequential: kraken's own `net[:-1]` case."""
    from app.htr.infrastructure.kraken.callbacks import trainable_output_parameters

    net = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
    )
    keep = trainable_output_parameters(net)
    assert keep == {id(p) for p in net[-1].parameters()}


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


def test_vgsl_config_has_no_ppocr_keys(tmp_path, monkeypatch):
    from app.htr.infrastructure.kraken import recognizer as module

    monkeypatch.setattr(module, "gpu_available", lambda: True)
    config = TrainingConfig(device="cuda:0")
    model_config = KrakenTrainer()._build_model_config(FakeVGSLModule, "vgsl", config, tmp_path)
    assert not hasattr(model_config, "variant")
    assert model_config.accelerator == "gpu"

    # without CUDA the same config is downgraded instead of failing
    monkeypatch.setattr(module, "gpu_available", lambda: False)
    cpu_config = KrakenTrainer()._build_model_config(FakeVGSLModule, "vgsl", config, tmp_path)
    assert cpu_config.accelerator == "cpu"


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
