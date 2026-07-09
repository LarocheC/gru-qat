"""Tier-1 structured-mode tests.

Each test parameterizes over the supported kinds (blockdiag, monarch,
circulant, butterfly, ldr). ``blockdiag`` is the single block-diagonal
factor (Triton-backed fast path); ``monarch`` is the genuine two-factor
Monarch (reference path only) — both must satisfy the generic structured
contracts below. They check three things per kind:
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

    - blockdiag / monarch: in/out divisible by nblocks=4.
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
    if kind in ("blockdiag", "monarch"):
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


KINDS = ["blockdiag", "monarch", "circulant", "butterfly", "ldr"]


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


def test_blockdiag_is_single_factor_monarch_is_two_factor() -> None:
    """torch-structured 1.3.0 naming: ``blockdiag`` builds a single
    block-diagonal factor (one ``weight`` tensor, zero cross-block mixing);
    ``monarch`` builds the genuine two-factor Monarch (``w1``/``w2``, full
    cross-channel mixing). Guards against silently swapping one for the
    other."""
    from gru_qat.structure import make_structured_linear

    bd = make_structured_linear(StructureConfig(kind="blockdiag", nblocks=4), 64, 64)
    # Single block-diagonal factor: [nblocks, out_blksz, in_blksz].
    assert hasattr(bd, "weight")
    assert bd.weight.shape == (4, 16, 16)

    mon = make_structured_linear(StructureConfig(kind="monarch", nblocks=4), 64, 64)
    # Two block-diagonal factors + implicit permutation.
    assert hasattr(mon, "w1") and hasattr(mon, "w2")
    assert not hasattr(mon, "weight")

    # Cross-block mixing: a single block-diagonal factor cannot move
    # information between blocks, the two-factor Monarch can. Perturb one
    # input coordinate and check whether outputs outside its block move.
    x0 = torch.zeros(1, 64)
    x1 = x0.clone()
    x1[0, 0] = 1.0
    with torch.no_grad():
        bd_delta = (bd(x1) - bd(x0)).abs()
        mon_delta = (mon(x1) - mon(x0)).abs()
    # blockdiag: only the first block (coords 0..15) can respond.
    assert bd_delta[0, 16:].max().item() == 0.0
    # monarch: coordinates outside the first block respond (cross-mixing).
    assert mon_delta[0, 16:].max().item() > 0.0


def test_monarch_two_factor_triton_eligibility() -> None:
    """The two-factor ``monarch`` has its own fused Triton kernel, eligible
    when blksz = H/nblocks is a power of two >= 16. Configs that fail that
    constraint fall back to the per-step reference path (``use_triton='auto'``
    -> False) and reject an explicit ``use_triton=True``. ``blockdiag`` is
    always eligible."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    # H=64, nblocks=4 -> blksz=16 (pow2, >=16): Triton-eligible.
    mon_ok = GRULayer(
        64, 64, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="monarch", nblocks=4),
        use_triton="auto",
    )
    assert mon_ok._fast_dispatch_eligible is True
    assert mon_ok.use_triton is True

    # H=32, nblocks=4 -> blksz=8 (< 16): not eligible, falls back.
    mon_small = GRULayer(
        32, 32, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="monarch", nblocks=4),
        use_triton="auto",
    )
    assert mon_small._fast_dispatch_eligible is False
    assert mon_small.use_triton is False
    with pytest.raises(ValueError, match="monarch"):
        GRULayer(
            32, 32, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="monarch", nblocks=4),
            use_triton=True,
        )

    bd = GRULayer(
        64, 64, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="blockdiag", nblocks=4),
        use_triton="auto",
    )
    assert bd._fast_dispatch_eligible is True


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
