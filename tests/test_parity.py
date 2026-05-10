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


@pytest.mark.parametrize(
    "input_size,hidden_size,batch",
    [(16, 32, 4), (128, 256, 16)],
)
def test_fused_gate_matches_split(
    input_size: int, hidden_size: int, batch: int
) -> None:
    """Fused gate layout must match split layout to fp32 noise.

    Both compute x @ cat([W_ir, W_iz, W_in]).T + cat(b...) versus three
    separate F.linear calls — algebraically identical, but cuBLAS may
    pick different tile sizes. Tolerance is loose enough to absorb that.
    """
    torch.manual_seed(0)
    split = GRUCellQuant(input_size, hidden_size, recipe=PRESETS["int8_per_channel"])
    fused = GRUCellQuant(
        input_size,
        hidden_size,
        recipe=PRESETS["int8_per_channel"],
        gate_layout="fused",
    )
    fused.load_state_dict(split.state_dict())

    x = torch.randn(batch, input_size)
    h = torch.randn(batch, hidden_size)

    out_split = split(x, h)
    out_fused = fused(x, h)
    max_diff = (out_split - out_fused).abs().max().item()
    assert max_diff < 1e-5, f"fused vs split differ by {max_diff}"


def test_layer_prebatched_input_matches_per_step() -> None:
    """Pre-batched input projection must match the per-step path when
    activation quant is a no-op (fp32 Identity).

    With per-tensor dynamic activation quant the two paths differ by
    construction (one sequence-wide scale vs. one scale per timestep);
    that case is covered by a separate "finite & sensible" smoke test.
    """
    from gru_qat.gru_layer import GRULayer
    from gru_qat.quantizers import QuantizerConfig, QuantRecipe

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    torch.manual_seed(0)
    a = GRULayer(8, 16, recipe=rec, gate_layout="fused", pre_batch_input=False)
    b = GRULayer(8, 16, recipe=rec, gate_layout="fused", pre_batch_input=True)
    b.load_state_dict(a.state_dict())

    x = torch.randn(7, 4, 8)  # T=7, B=4
    out_a, h_a = a(x)
    out_b, h_b = b(x)
    assert (out_a - out_b).abs().max().item() < 1e-5
    assert (h_a - h_b).abs().max().item() < 1e-5


def test_layer_prebatched_input_int8_finite() -> None:
    """Pre-batched + int8 fused: output must be finite. Numerical drift vs.
    the per-step path is expected (one sequence-wide x scale)."""
    from gru_qat.gru_layer import GRULayer

    layer = GRULayer(
        8,
        16,
        recipe=PRESETS["int8_per_channel"],
        gate_layout="fused",
        pre_batch_input=True,
    )
    x = torch.randn(7, 4, 8)
    out, h = layer(x)
    assert torch.isfinite(out).all() and torch.isfinite(h).all()


def test_prebatch_rejects_split_layout() -> None:
    from gru_qat.gru_layer import GRULayer

    with pytest.raises(ValueError, match="gate_layout='fused'"):
        GRULayer(
            8,
            16,
            recipe=PRESETS["int8_per_channel"],
            gate_layout="split",
            pre_batch_input=True,
        )


def test_fused_gate_rejects_per_tensor_weights() -> None:
    """Per-tensor weight quant + fused gates is unsafe (one scale across
    three differently-distributed gate matrices) — must raise."""
    from gru_qat.quantizers import QuantizerConfig, QuantRecipe

    bad_recipe = QuantRecipe(
        weight=QuantizerConfig(bits=8, axis=None),  # per-tensor
    )
    with pytest.raises(ValueError, match="axis=0"):
        GRUCellQuant(8, 16, recipe=bad_recipe, gate_layout="fused")
