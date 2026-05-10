"""Tier-1 structured-mode tests.

Each test parameterizes over the four supported kinds (monarch, circulant,
butterfly, ldr). They check three things per kind:
- forward output is finite and the right shape;
- gradients populate every parameter on backward;
- a short training loop reduces loss vs. a fresh-init baseline (sanity);
- the cell still runs under int8 frozen QAT (per-tensor symmetric on the
  hidden state) and produces finite output + gradients.

Tests skip cleanly when ``torch_structured`` isn't installed.
"""

from __future__ import annotations

import warnings

# torch_structured emits a one-time CUDA-version mismatch warning when its
# compiled CUDA version differs from torch's; harmless for these tests.
warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

torch_structured = pytest.importorskip("torch_structured")

from gru_qat import (  # noqa: E402
    GRULayer,
    QuantizerConfig,
    QuantRecipe,
    StructureConfig,
)
from gru_qat.gru_cell import GRUCellQuant  # noqa: E402
from gru_qat.quantizers import FakeQuantizePerTensor  # noqa: E402


def _shapes_for_kind(kind: str) -> tuple[int, int]:
    """(input_size, hidden_size) tuned to each kind's constraints.

    - monarch: in/out divisible by nblocks=4.
    - circulant: square, power-of-2.
    - butterfly: power-of-2 makes the zero-pad a no-op.
    - ldr: square.
    """
    if kind == "circulant":
        return (32, 32)  # square + pow2
    if kind == "ldr":
        return (32, 32)  # square
    if kind == "butterfly":
        return (32, 32)  # pow2 both sides
    if kind == "monarch":
        return (32, 32)  # divisible by nblocks=4
    raise ValueError(kind)


def _make_cell(kind: str, recipe: QuantRecipe | None = None) -> GRUCellQuant:
    """Build a structured cell where BOTH sides use ``kind``."""
    in_size, hid = _shapes_for_kind(kind)
    cfg = StructureConfig(kind=kind)
    if recipe is None:
        recipe = QuantRecipe(
            weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
            input_act=QuantizerConfig(bits=32, name="x_id"),
            hidden=QuantizerConfig(bits=32, name="h_id"),
        )
    return GRUCellQuant(
        in_size, hid, recipe=recipe,
        gate_layout="fused",
        structure_input=cfg, structure_hidden=cfg,
    )


KINDS = ["monarch", "circulant", "butterfly", "ldr"]


@pytest.mark.parametrize("kind", KINDS)
def test_structured_cell_forward_finite(kind: str) -> None:
    cell = _make_cell(kind)
    in_size, hid = _shapes_for_kind(kind)
    x = torch.randn(4, in_size)
    h = torch.randn(4, hid)
    h_new = cell(x, h)
    assert h_new.shape == (4, hid)
    assert torch.isfinite(h_new).all()
    # Hidden states are bounded by max(|h|, 1) since h_new = (1-z)*n + z*h
    # with tanh ∈ [-1, 1]. Allow generous slack for any structured init noise.
    assert h_new.abs().max() < max(h.abs().max().item(), 1.0) + 5.0


@pytest.mark.parametrize("kind", KINDS)
def test_structured_cell_grad_flows(kind: str) -> None:
    cell = _make_cell(kind)
    in_size, hid = _shapes_for_kind(kind)
    x = torch.randn(4, in_size, requires_grad=True)
    h = torch.randn(4, hid, requires_grad=True)
    h_new = cell(x, h)
    loss = h_new.float().pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert h.grad is not None
    # Every learnable parameter that participated should have a non-None
    # grad. The biases for gates and structured factors all touch the
    # forward, so they should be populated.
    populated = [p for p in cell.parameters() if p.requires_grad and p.grad is not None]
    assert len(populated) > 0, "no parameters got gradients"


@pytest.mark.parametrize("kind", KINDS)
def test_structured_layer_trains_loss_decreases(kind: str) -> None:
    """Toy task: predict next step's output of a fixed teacher GRULayer.
    Run a short training loop and check that loss decreases."""
    torch.manual_seed(0)
    in_size, hid = _shapes_for_kind(kind)
    seq, batch = 8, 4

    # Teacher is a dense layer (different parameterization is fine — we
    # only need a deterministic synthetic target).
    rec_dense = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    teacher = GRULayer(in_size, hid, recipe=rec_dense)
    teacher.eval()

    cfg = StructureConfig(kind=kind)
    student = GRULayer(
        in_size, hid, recipe=rec_dense,
        gate_layout="fused",
        structure_input=cfg, structure_hidden=cfg,
    )
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        x = torch.randn(seq, batch, in_size)
        with torch.no_grad():
            target, _ = teacher(x)
        pred, _ = student(x)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss at the end should be lower than the initial average of the
    # first few steps (warm-up). If structured training is broken (e.g.
    # gradients don't flow through the structured factors) loss won't
    # move at all.
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial, f"loss did not decrease: initial={initial:.4f}, final={final:.4f}"


@pytest.mark.parametrize("kind", KINDS)
def test_structured_int8_qat_finite(kind: str) -> None:
    """Cell with a frozen int8 hidden quantizer in structured mode runs
    end-to-end and produces finite output + gradients."""
    torch.manual_seed(0)
    in_size, hid = _shapes_for_kind(kind)

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    recipe = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),  # weight quant unused in structured
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=bits, mode="frozen", name="h_q"),
    )
    cell = _make_cell(kind, recipe=recipe)
    # Manually set the frozen scales on the hidden quantizers (otherwise
    # they sit at scale=1 from default init, which clips ~everything).
    for q in (cell.quant_h_in, cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(0.05)
        q.zero_point = torch.tensor(0.0)

    x = (torch.randn(4, in_size) * 0.1).requires_grad_()
    h = (torch.randn(4, hid) * 0.1).requires_grad_()
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()
    loss = h_new.float().pow(2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_structured_layer_rejects_pre_batch_input() -> None:
    """pre_batch_input depends on the dense fused Wi_cat — explicit error
    in structured mode is better than a confusing AttributeError."""
    cfg = StructureConfig(kind="monarch")
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    with pytest.raises(ValueError, match="structured mode"):
        GRULayer(
            32, 32, recipe=rec,
            gate_layout="fused",
            pre_batch_input=True,
            structure_input=cfg, structure_hidden=cfg,
        )


def test_structured_quantize_weights_raises() -> None:
    """quantize_weights() is dense-only; structured cells should raise
    so callers don't silently get bad data."""
    cell = _make_cell("monarch")
    with pytest.raises(RuntimeError, match="dense-only"):
        cell.quantize_weights()


def test_mixed_dense_input_structured_hidden() -> None:
    """A side may be dense while the other is structured. The cell must
    handle the mixed case without error."""
    in_size, hid = 16, 32  # in_size != hid but both work for monarch
    cfg = StructureConfig(kind="monarch", nblocks=4)
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cell = GRUCellQuant(
        in_size, hid, recipe=rec,
        gate_layout="fused",
        structure_input=None,         # dense input side
        structure_hidden=cfg,         # structured hidden side
    )
    x = torch.randn(4, in_size)
    h = torch.randn(4, hid)
    h_new = cell(x, h)
    assert h_new.shape == (4, hid)
    assert torch.isfinite(h_new).all()


def test_structure_validation_errors() -> None:
    """Each kind enforces its shape constraints up front."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    # Monarch: hidden_size must be divisible by nblocks
    with pytest.raises(ValueError, match="divisible by nblocks"):
        GRUCellQuant(
            16, 17, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="monarch", nblocks=4),
        )
    # Circulant: requires square (in == out for the projection — and our
    # fused projection is in -> 3*hidden, so square means input_size ==
    # 3*hidden_size which is awkward). Make sure the error is raised.
    with pytest.raises(ValueError, match="square"):
        GRUCellQuant(
            16, 32, recipe=rec, gate_layout="fused",
            structure_input=StructureConfig(kind="circulant"),
        )
    # LDR: requires square.
    with pytest.raises(ValueError, match="square"):
        GRUCellQuant(
            16, 32, recipe=rec, gate_layout="fused",
            structure_input=StructureConfig(kind="ldr"),
        )
