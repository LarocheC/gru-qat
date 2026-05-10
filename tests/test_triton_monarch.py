"""Tier-2 Monarch persistent kernel tests.

Stage A: factor extraction + PyTorch reference. The reference must match
the tier-1 cell's structured forward/backward so it can serve as ground
truth for the Triton kernels in stages B and C.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

torch_structured = pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402
from gru_qat.gru_cell import GRUCellQuant  # noqa: E402
from gru_qat.triton_kernels.scan_monarch import (  # noqa: E402
    extract_monarch_factors,
    gru_scan_monarch_backward_pytorch,
    gru_scan_monarch_forward_pytorch,
)


def _make_monarch_layer(
    in_size: int, hid: int, nblocks: int = 4
) -> GRULayer:
    """Build a structured-Monarch layer with no quant on weights or
    activations (fp32 path) — keeps reference math clean."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cfg = StructureConfig(kind="monarch", nblocks=nblocks)
    return GRULayer(
        in_size, hid, recipe=rec,
        gate_layout="fused",
        structure_input=None,        # input side stays dense for tier 2
        structure_hidden=cfg,
    )


def _build_gi_from_cell(layer: GRULayer, x: torch.Tensor) -> torch.Tensor:
    """Reproduce what the cell's structured input projection produces:
    dense Wi (per-gate, with quant_W_*) for each gate, concatenated along
    the last dim, then sliced by time. The dense input side already
    matches what the persistent kernel expects.

    For the structured-monarch test we want to feed the SAME ``gi`` to
    both the cell-based reference (which goes through quant_h_in / etc.)
    and our PyTorch monarch reference. Constructing it explicitly here
    keeps both paths in sync.
    """
    cell = layer.cell
    T, B, _ = x.shape
    Wi_cat = torch.cat(
        [
            cell.quant_W_ir(cell.W_ir),
            cell.quant_W_iz(cell.W_iz),
            cell.quant_W_in(cell.W_in),
        ],
        dim=0,
    )
    bi_cat = torch.cat([cell.b_ir, cell.b_iz, cell.b_in])
    xq = cell.quant_x(x)
    return torch.nn.functional.linear(xq, Wi_cat, bi_cat)


@pytest.mark.parametrize("T,B,H,nblocks", [(8, 4, 32, 4), (16, 8, 64, 4)])
def test_monarch_pytorch_forward_matches_cell(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """The PyTorch monarch reference must match the tier-1 layer's
    structured forward — same weights, same bias, fp32-Identity quant."""
    torch.manual_seed(0)
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks)
    layer.eval()

    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        mon_out = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref_out - mon_out).abs().max().item()
    rel = max_diff / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-5, f"forward rel diff {rel:.4e}"


@pytest.mark.parametrize("T,B,H,nblocks", [(8, 4, 32, 4), (16, 8, 64, 4)])
def test_monarch_pytorch_backward_matches_cell(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Gradients from the PyTorch monarch reference must match autograd
    through the tier-1 cell. We compare gradients of the inputs (gi, h0)
    and of the bias (bh) — both representations share these. The Wh
    parameter gradients live in different layouts (cell has three [nblocks,
    blksz, blksz]; reference has one [3, nblocks, blksz, blksz]) so we
    stack the cell's grads to compare."""
    torch.manual_seed(0)
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks)

    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)

    # ---- Reference path: autograd through the tier-1 cell ----
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()
    ref_dWh = torch.stack(
        [
            layer.cell.struct_Wh_r.weight.grad,
            layer.cell.struct_Wh_z.weight.grad,
            layer.cell.struct_Wh_n.weight.grad,
        ],
        dim=0,
    )
    ref_dbh = torch.cat(
        [layer.cell.b_hr.grad, layer.cell.b_hz.grad, layer.cell.b_hn.grad]
    )

    # ---- Monarch reference path ----
    Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
    with torch.no_grad():
        gi = _build_gi_from_cell(layer, x)
        out_fwd = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        # dout matches the gradient of (.pow(2).sum()) wrt the layer's output:
        #   d(sum(out^2))/d(out) = 2 * out
        dout = 2.0 * out_fwd

    dgi, dh0, dWh_struct, dbh = gru_scan_monarch_backward_pytorch(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )

    # Compare h0 gradients
    diff_h0 = (dh0 - ref_h0.grad).abs().max().item()
    rel_h0 = diff_h0 / max(ref_h0.grad.abs().max().item(), 1e-6)
    assert rel_h0 < 1e-4, f"dh0 rel diff {rel_h0:.4e}"

    # Compare Wh gradients
    diff_Wh = (dWh_struct - ref_dWh).abs().max().item()
    rel_Wh = diff_Wh / max(ref_dWh.abs().max().item(), 1e-6)
    assert rel_Wh < 1e-4, f"dWh rel diff {rel_Wh:.4e}"

    # Compare bh gradients
    diff_bh = (dbh - ref_dbh).abs().max().item()
    rel_bh = diff_bh / max(ref_dbh.abs().max().item(), 1e-6)
    assert rel_bh < 1e-4, f"dbh rel diff {rel_bh:.4e}"
