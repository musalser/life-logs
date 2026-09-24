"""Lightning callbacks the kraken backend needs on top of kraken's own ones.

kraken's ``--freeze-backbone`` is implemented by ``KrakenFreezeBackbone``, which
freezes ``pl_module.net[:-1]``. Slicing only works for the VGSL family (an
``nn.Sequential``); this project fine-tunes a PP-OCRv6 model, which is a plain
``nn.Module``:

    >>> net[:-1]
    TypeError: 'PPOCRv6Model' object is not subscriptable

kraken does expose the layer that matters — ``_output_proj``, the codec-sized
CTC projection — so the freezing is reimplemented here against that property.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


def output_projection(net: Any) -> Any:
    """The one layer that keeps learning while the backbone is frozen.

    After a ``resize`` this is the freshly initialized part of the network, so
    it must train from the first step. VGSL nets (no ``_output_proj``) fall back
    to their last child, which is what kraken's ``net[:-1]`` means.
    """
    projection = getattr(net, "_output_proj", None)
    if projection is None:
        children = list(net.children())
        projection = children[-1] if children else net
    return projection


def trainable_output_parameters(net: Any) -> set[int]:
    """``id()`` of the parameters of that output layer."""
    return {id(parameter) for parameter in output_projection(net).parameters()}


def frozen_modules(net: Any) -> list[Any]:
    """Modules that make up the frozen backbone (everything but the head).

    Used to switch off more than the gradients: a frozen backbone must also keep
    its pretrained BatchNorm statistics and stay out of dropout, exactly as
    Lightning's ``BaseFinetuning.freeze`` does for kraken's own callback.
    """
    keep_params = trainable_output_parameters(net)
    keep_modules = {id(module) for module in output_projection(net).modules()}
    return [
        module
        for module in net.modules()
        if id(module) not in keep_modules
        and not any(id(parameter) in keep_params for parameter in module.parameters())
    ]


class FreezeBackboneForSteps(Callback):
    """Keep everything but the output layer frozen for the first N steps.

    On a small corpus the fresh output layer pushes large gradients into a
    pretrained backbone and the model forgets what it knew — the reason the
    first fine-tune of this project came out worse than the base model. Freezing
    the backbone while the new head settles, then fine-tuning everything at the
    configured learning rate, avoids that.

    Unlike kraken's callback this only toggles ``requires_grad`` and the module
    mode: the parameters stay in the optimizer's parameter group, so after
    unfreezing they continue with the same learning rate and schedule as the
    rest of the network instead of a new group at ``lr / 10``.
    """

    def __init__(self, unfreeze_at_step: int, freeze_batch_norm: bool = True):
        self.unfreeze_at_step = max(0, int(unfreeze_at_step))
        # a batch of 4 crops is far too small to re-estimate the statistics of
        # 61 pretrained BatchNorm layers; letting them drift wrecks a model that
        # was previously fine (this project's batches are 4 lines)
        self.freeze_batch_norm = freeze_batch_norm
        self._frozen_parameters: list[Any] = []
        self._frozen_modules: list[Any] = []
        self._batch_norm_modules: list[Any] = []

    # ------------------------------------------------------------------

    def _freeze(self, net: Any) -> None:
        self._frozen_parameters = [
            parameter
            for parameter in net.parameters()
            if parameter.requires_grad and id(parameter) not in trainable_output_parameters(net)
        ]
        self._frozen_modules = frozen_modules(net)
        self._apply()

    def _apply(self) -> None:
        """(Re-)assert the frozen state; Lightning resets module modes itself."""
        for module in self._frozen_modules:
            module.eval()
        for parameter in self._frozen_parameters:
            parameter.requires_grad_(False)

    def _unfreeze(self) -> None:
        for module in self._frozen_modules:
            module.train()
        for parameter in self._frozen_parameters:
            parameter.requires_grad_(True)

    # -- Lightning hooks -----------------------------------------------

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: D102
        net = getattr(pl_module, "net", None)
        if net is None:
            return
        if self.freeze_batch_norm:
            self._batch_norm_modules = [
                module for module in net.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            ]
        self._freeze(net)
        logger.info(
            "HTR training: backbone frozen for the first %s steps (%s tensors, %s modules)",
            self.unfreeze_at_step,
            len(self._frozen_parameters),
            len(self._frozen_modules),
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: D102
        if self._frozen_parameters:
            # ``>=`` rather than kraken's ``==``: a skipped step must not leave
            # the backbone frozen for the whole run
            if trainer.global_step >= self.unfreeze_at_step:
                self._unfreeze()
                logger.info(
                    "HTR training: backbone unfrozen at step %s (%s tensors)",
                    trainer.global_step,
                    len(self._frozen_parameters),
                )
                self._frozen_parameters = []
                self._frozen_modules = []
            else:
                self._apply()
        # BatchNorm statistics stay frozen for the whole run: Lightning switches
        # the model back to train mode after every validation
        for module in self._batch_norm_modules:
            module.eval()
