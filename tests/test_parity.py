"""Parity tests — Phase 2 exit test.

Validates that GRUCellQuant with all quantizers set to Identity matches
torch.nn.GRUCell exactly. If this fails, the unroll math is wrong and
nothing else matters.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gru_qat.gru_cell import GRUCellQuant
from gru_qat.quantizers import PRESETS


def _copy_weights(src: nn.GRUCell, dst: GRUCellQuant) -> None:
    """Copy weights from torch.nn.GRUCell to GRUCellQuant.

    nn.GRUCell uses concatenated weights:
        weight_ih: [3*hidden, input]    (r, z, n stacked)
        weight_hh: [3*hidden, hidden]
    We unstack them.
    """
    h = dst.hidden_size
    Wir, Wiz, Win = src.weight_ih.chunk(3, dim=0)
    Whr, Whz, Whn = src.weight_hh.chunk(3, dim=0)
    bir, biz, bin_ = src.bias_ih.chunk(3)
    bhr, bhz, bhn = src.bias_hh.chunk(3)

    with torch.no_grad():
        dst.W_ir.copy_(Wir)
        dst.W_iz.copy_(Wiz)
        dst.W_in.copy_(Win)
        dst.W_hr.copy_(Whr)
        dst.W_hz.copy_(Whz)
        dst.W_hn.copy_(Whn)
        dst.b_ir.copy_(bir)
        dst.b_iz.copy_(biz)
        dst.b_in.copy_(bin_)
        dst.b_hr.copy_(bhr)
        dst.b_hz.copy_(bhz)
        dst.b_hn.copy_(bhn)


@pytest.mark.parametrize(
    "input_size,hidden_size,batch",
    [
        (16, 32, 4),
        (1, 8, 1),
        (128, 256, 16),
    ],
)
def test_cell_matches_torch_gru_cell(
    input_size: int, hidden_size: int, batch: int
) -> None:
    torch.manual_seed(0)
    ref = nn.GRUCell(input_size, hidden_size)
    ours = GRUCellQuant(input_size, hidden_size, recipe=PRESETS["fp32"])
    _copy_weights(ref, ours)

    x = torch.randn(batch, input_size)
    h = torch.randn(batch, hidden_size)

    h_ref = ref(x, h)
    h_ours = ours(x, h)

    max_diff = (h_ref - h_ours).abs().max().item()
    assert max_diff < 1e-5, f"max diff {max_diff} exceeds 1e-5"


def test_cell_with_zero_hidden() -> None:
    ref = nn.GRUCell(8, 16)
    ours = GRUCellQuant(8, 16, recipe=PRESETS["fp32"])
    _copy_weights(ref, ours)

    x = torch.randn(4, 8)
    h = torch.zeros(4, 16)

    assert torch.allclose(ref(x, h), ours(x, h), atol=1e-5)


def test_cell_with_zero_input() -> None:
    ref = nn.GRUCell(8, 16)
    ours = GRUCellQuant(8, 16, recipe=PRESETS["fp32"])
    _copy_weights(ref, ours)

    x = torch.zeros(4, 8)
    h = torch.randn(4, 16)

    assert torch.allclose(ref(x, h), ours(x, h), atol=1e-5)


def test_cell_with_large_magnitude() -> None:
    ref = nn.GRUCell(8, 16)
    ours = GRUCellQuant(8, 16, recipe=PRESETS["fp32"])
    _copy_weights(ref, ours)

    x = torch.randn(4, 8) * 100
    h = torch.randn(4, 16) * 100

    assert torch.allclose(ref(x, h), ours(x, h), atol=1e-4, rtol=1e-4)
