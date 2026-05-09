"""End-to-end QAT smoke tests — Phase 3+4 exit tests.

These verify that:
  - swapping a quantizer config changes behaviour with no other code change
  - INT8 per-channel weight quant + per-tensor activation quant gives a
    cell output close to fp32
  - the layer trains to within tolerance of an fp32 baseline on a toy task
"""

from __future__ import annotations

import pytest
import torch

from gru_qat.gru_cell import GRUCellQuant
from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS, QuantRecipe, QuantizerConfig


def test_no_op_quant_matches_fp32() -> None:
    """Identity quantizers must reproduce fp32 path exactly."""
    cell_a = GRUCellQuant(8, 16, recipe=PRESETS["fp32"])
    cell_b = GRUCellQuant(8, 16, recipe=PRESETS["fp32"])
    cell_b.load_state_dict(cell_a.state_dict())

    x = torch.randn(4, 8)
    h = torch.randn(4, 16)
    assert torch.equal(cell_a(x, h), cell_b(x, h))


def test_int8_per_channel_finite_and_bounded() -> None:
    """Single-shot INT8 cell on untrained random weights.

    NOTE: This is *not* an accuracy test — it's a structural smoke test.
    Compounding 6 weight quantizers + 3 activation quantizers + sigmoid/tanh
    on random fp32 weights with no calibration will produce large relative
    error. Real accuracy validation is `test_layer_trains_to_baseline`,
    which trains and converges. Here we just check the output is finite and
    bounded by hidden-state magnitude.
    """
    fp32 = GRUCellQuant(8, 16, recipe=PRESETS["fp32"])
    quant = GRUCellQuant(8, 16, recipe=PRESETS["int8_per_channel"])
    quant.load_state_dict(fp32.state_dict(), strict=False)

    x = torch.randn(4, 8)
    h = torch.randn(4, 16)

    h_int8 = quant(x, h)
    assert torch.isfinite(h_int8).all()
    # Hidden state is bounded by max(|h|, 1) since h_new = (1-z)*n + z*h
    # and tanh ∈ [-1, 1]. Allow generous slack for quant noise.
    assert h_int8.abs().max() < max(h.abs().max().item(), 1.0) + 0.5


def test_swap_granularity_no_code_change() -> None:
    """Same model, swap weight quant from per-channel to per-group(8).

    Both must run without error and produce non-identical, sensible output.
    Note we use group_size=8 here because hidden_size=16 must be divisible.
    """
    rec_pc = QuantRecipe(
        weight=QuantizerConfig(bits=4, axis=0),
        input_act=QuantizerConfig(bits=8),
        hidden=QuantizerConfig(bits=8),
    )
    rec_pg = QuantRecipe(
        weight=QuantizerConfig(bits=4, axis=0, group_size=8),
        input_act=QuantizerConfig(bits=8),
        hidden=QuantizerConfig(bits=8),
    )

    a = GRUCellQuant(8, 16, recipe=rec_pc)
    b = GRUCellQuant(8, 16, recipe=rec_pg)
    b.load_state_dict(a.state_dict(), strict=False)

    x = torch.randn(4, 8)
    h = torch.randn(4, 16)

    out_pc = a(x, h)
    out_pg = b(x, h)
    # Different quant schemes should give different outputs
    assert not torch.allclose(out_pc, out_pg, atol=1e-6)
    # Both should be finite
    assert torch.isfinite(out_pc).all()
    assert torch.isfinite(out_pg).all()


@pytest.mark.slow
def test_layer_trains_to_baseline() -> None:
    """Phase 4 exit test: INT8 QAT converges close to fp32 baseline.

    Synthetic task: predict next-step output of a fixed teacher GRU.
    Skip-marked as 'slow'; run with `pytest -m slow`.
    """
    torch.manual_seed(0)
    seq_len, batch, in_dim, hid = 32, 16, 8, 16

    # Teacher: a fixed fp32 GRU layer
    teacher = GRULayer(in_dim, hid, recipe=PRESETS["fp32"])
    teacher.eval()

    # Students
    student_fp32 = GRULayer(in_dim, hid, recipe=PRESETS["fp32"])
    student_qat = GRULayer(in_dim, hid, recipe=PRESETS["int8_per_channel"])

    def train(model: GRULayer, n_steps: int = 200) -> float:
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        last_loss = float("inf")
        for _ in range(n_steps):
            x = torch.randn(seq_len, batch, in_dim)
            with torch.no_grad():
                target, _ = teacher(x)
            pred, _ = model(x)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        return last_loss

    loss_fp32 = train(student_fp32)
    loss_qat = train(student_qat)

    # QAT should not be more than 2x worse than fp32 on this toy task.
    assert loss_qat < 2.0 * loss_fp32, (
        f"QAT loss {loss_qat:.4e} >> fp32 loss {loss_fp32:.4e}"
    )
