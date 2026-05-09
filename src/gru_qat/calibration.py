"""Calibration utilities — Phase 4.

Phase 4 work item: implement `calibrate(layer, loader, n_batches)` that
runs the layer in min_max observer mode for n_batches and then freezes.

The interface below is the target. Bodies are stubs.

Design note: we want calibration to be *separable* from the model so
calibration data is decoupled from training data. Typical workflow:

    layer = GRULayer(..., recipe=recipe_with_min_max_mode)
    train(layer, train_loader)  # QAT
    calibrate(layer, val_loader, n_batches=64)  # gather act stats
    layer.freeze()
    export(layer)  # to inference kernel

We do *not* re-use training stats for calibration because training-time
augmentation can shift activation distributions in ways the deployed
model never sees.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


@torch.no_grad()
def calibrate(
    module: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    n_batches: int = 64,
) -> dict[str, dict[str, float]]:
    """Run the module on n_batches in observer mode; return a stats summary.

    TODO(phase=4):
      1. Set every FakeQuantize in module to mode="min_max"
      2. Run forward passes for n_batches
      3. Collect (running_min, running_max) per quantizer keyed by qualname
      4. Return summary; do NOT freeze automatically — caller decides
    """
    raise NotImplementedError("phase=4")


def freeze_all(module: nn.Module) -> None:
    """Freeze every FakeQuantize in module. After this, scales are read-only."""
    from gru_qat.quantizers import FakeQuantize

    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.freeze()
