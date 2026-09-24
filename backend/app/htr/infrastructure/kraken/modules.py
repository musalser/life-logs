"""PP-OCRv6 training module tweaks that kraken does not offer as options.

kraken's ppocrv6 recipe trains a *joint* objective::

    loss = ctc_loss + nrtr_loss          # kraken/train/ppocr.py

where the NRTR head is a 4-layer transformer decoder (18.9 M parameters, more
than the 15.9 M of the recognizer itself) that is **not used at inference** —
recognition decodes the CTC output. That is fine when training from scratch or
from a checkpoint that already contains the head, but this project fine-tunes an
exported inference model, which has none: kraken then builds the head from
random weights and adds its loss with weight 1.

On a 79-line corpus the effect is visible in the logs — the backbone is fine
while frozen (val accuracy 0.79), and the epoch after it is unfrozen the CTC
accuracy collapses to 0.32 as the features are dragged towards a random
auxiliary objective. ``htr_training_aux_nrtr = false`` (the default) therefore
swaps in the module below.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from kraken.train.ppocr import PPOCRv6RecognitionModel

logger = logging.getLogger(__name__)


class CtcFineTunePPOCRv6(PPOCRv6RecognitionModel):
    """PP-OCRv6 fine-tuning of the CTC path only, without the auxiliary head."""

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        # frozen *before* configure_optimizers() runs, so kraken's parameter
        # grouping (which skips requires_grad=False) leaves the head out of the
        # optimizer entirely: no state, no updates, no wasted memory
        if self.nrtr_head is not None:
            self.nrtr_head.requires_grad_(False)
            logger.info(
                "HTR training: auxiliary NRTR head disabled (%s params stay frozen)",
                sum(parameter.numel() for parameter in self.nrtr_head.parameters()),
            )

    def _gtc_loss(self, feat: Any, tgt: Any, out_lens: Any) -> torch.Tensor:
        """Zero auxiliary loss: the CTC objective is what the model deploys."""
        return feat.new_zeros(())
