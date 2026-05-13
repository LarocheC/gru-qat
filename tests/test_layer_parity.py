"""Layer-parity tests — Phase 1 audit.

Validates that GRULayer with all quantizers set to Identity (use_triton=False,
dense, no structure_hidden) matches torch.nn.GRU(num_layers=1, bidirectional=
False, batch_first=False) on (out, h_T) forward, on the six weight gradients
plus dx and dh_0 backward, and on h_0 != 0 initial state. Tolerance < 1e-4
across a T x B x H = 5 x 3 x 5 = 75-combo grid (see Plan 02).

If this fails, the unroll math (or its time-loop orchestration) is wrong and
every later phase's reference is contaminated.

This module sets ``torch.set_float32_matmul_precision('highest')`` at import
time because Phase 1 audits the math, not TF32. Diverges from
``tests/test_triton_*.py`` (which use 'high' so the kernel runs under realistic
conditions) — that's intentional. Cell-level parity at < 1e-5 is pinned by
``tests/test_parity.py``; this file is the *layer*-level counterpart at < 1e-4.
"""

from __future__ import annotations

import pytest  # noqa: F401  # used by tests added in Task 2 of this plan
import torch
import torch.nn as nn

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS

# Module-level: we audit math, not TF32. "highest" forces IEEE-754 fp32 matmul,
# so the only drift is algorithm, not arithmetic mode. Diverges from the
# Triton kernel tests (which use "high" to test the kernel under realistic
# conditions). See .planning/phases/01-reference-path-parity-vs-nn-gru/01-CONTEXT.md
# D-07 for the locked decision.
torch.set_float32_matmul_precision("highest")


def _make_dense_fp32_layer(input_size: int, hidden_size: int) -> GRULayer:
    """fp32 dense reference layer: Identity quantizers, no structure, no Triton.

    This is the path Phase 1 audits. ``recipe=PRESETS['fp32']`` selects the
    three Identity quantizers (weight / input_act / hidden, all bits=32, all
    axis=None — see ``src/gru_qat/quantizers.py:284-289``). No
    ``structure_input`` / ``structure_hidden`` arg → dense W_i*/W_h* parameters.
    ``use_triton`` is left at the default "auto" which resolves to ``False``
    here because the layer is not fast-dispatch eligible (no structured hidden
    + ``gate_layout='split'``).
    """
    return GRULayer(
        input_size,
        hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
    )


def _translate_cell_to_nn_gru(layer: GRULayer) -> nn.GRU:
    """Build a ``torch.nn.GRU(num_layers=1, bidirectional=False, batch_first=False)``
    whose weights and biases reproduce ``layer`` exactly.

    Per the PyTorch GRU docs
    (https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) gate
    order is ``(r, z, n)`` for both sides, matching ``gru_cell.py``'s
    ``W_ir / W_iz / W_in`` family. Translation is just ``torch.cat`` along
    axis 0::

        weight_ih_l0 = cat([W_ir, W_iz, W_in], dim=0)   # [3H, IN]
        weight_hh_l0 = cat([W_hr, W_hz, W_hn], dim=0)   # [3H, H]
        bias_ih_l0   = cat([b_ir, b_iz, b_in])           # [3H]
        bias_hh_l0   = cat([b_hr, b_hz, b_hn])           # [3H]

    The n-gate asymmetry (``r_t * (W_hn h + b_hn)`` *inside* the tanh,
    multiplying only the hidden contribution) is preserved by this layout
    because both sides apply ``r_t`` identically. See
    ``src/gru_qat/gru_cell.py:1-15`` (module docstring) for why this asymmetric
    placement matters — many home-grown GRU implementations get it wrong.

    Primary direction for the Phase 1 grid (Plans 02-04): build a ``GRULayer``
    first with random weights as source-of-truth, then build an ``nn.GRU``
    from this helper and compare. Inverse direction (
    ``_translate_nn_gru_to_cell``) is exercised by a single round-trip smoke
    test only.
    """
    cell = layer.cell
    gru = nn.GRU(
        input_size=cell.input_size,
        hidden_size=cell.hidden_size,
        num_layers=1,
        bidirectional=False,
        batch_first=False,
    )
    with torch.no_grad():
        gru.weight_ih_l0.copy_(torch.cat([cell.W_ir, cell.W_iz, cell.W_in], dim=0))
        gru.weight_hh_l0.copy_(torch.cat([cell.W_hr, cell.W_hz, cell.W_hn], dim=0))
        gru.bias_ih_l0.copy_(torch.cat([cell.b_ir, cell.b_iz, cell.b_in]))
        gru.bias_hh_l0.copy_(torch.cat([cell.b_hr, cell.b_hz, cell.b_hn]))
    return gru


def _translate_nn_gru_to_cell(gru: nn.GRU) -> GRULayer:
    """Inverse of ``_translate_cell_to_nn_gru`` — used by the round-trip smoke
    test only.

    Direct mirror of ``tests/test_parity.py:18-44`` ``_copy_weights`` but at
    the layer level: ``chunk(3, dim=0)`` splits ``nn.GRU``'s concatenated
    weight / bias parameters back into the per-gate ``W_ir / W_iz / W_in``
    family. ``bin_`` (trailing underscore) is mandatory because ``bin`` is a
    Python built-in — same convention as ``tests/test_parity.py:29``.
    """
    layer = GRULayer(
        gru.input_size,
        gru.hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
    )
    cell = layer.cell
    Wir, Wiz, Win = gru.weight_ih_l0.chunk(3, dim=0)
    Whr, Whz, Whn = gru.weight_hh_l0.chunk(3, dim=0)
    bir, biz, bin_ = gru.bias_ih_l0.chunk(3)
    bhr, bhz, bhn = gru.bias_hh_l0.chunk(3)
    with torch.no_grad():
        cell.W_ir.copy_(Wir)
        cell.W_iz.copy_(Wiz)
        cell.W_in.copy_(Win)
        cell.W_hr.copy_(Whr)
        cell.W_hz.copy_(Whz)
        cell.W_hn.copy_(Whn)
        cell.b_ir.copy_(bir)
        cell.b_iz.copy_(biz)
        cell.b_in.copy_(bin_)
        cell.b_hr.copy_(bhr)
        cell.b_hz.copy_(bhz)
        cell.b_hn.copy_(bhn)
    return layer
