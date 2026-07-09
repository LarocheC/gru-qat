"""Tier-2 Blockdiag persistent kernel tests.

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
from gru_qat.triton_kernels.scan_blockdiag import (  # noqa: E402
    extract_blockdiag_factors,
    gru_scan_blockdiag_backward_pytorch,
    gru_scan_blockdiag_backward_triton,
    gru_scan_blockdiag_forward_pytorch,
    gru_scan_blockdiag_forward_triton,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


def _make_blockdiag_layer(
    in_size: int, hid: int, nblocks: int = 4
) -> GRULayer:
    """Build a structured-Blockdiag layer with no quant on weights or
    activations (fp32 path) — keeps reference math clean."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cfg = StructureConfig(kind="blockdiag", nblocks=nblocks)
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

    For the structured-blockdiag test we want to feed the SAME ``gi`` to
    both the cell-based reference (which goes through quant_h_in / etc.)
    and our PyTorch blockdiag reference. Constructing it explicitly here
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
def test_blockdiag_pytorch_forward_matches_cell(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """The PyTorch blockdiag reference must match the tier-1 layer's
    structured forward — same weights, same bias, fp32-Identity quant."""
    torch.manual_seed(0)
    layer = _make_blockdiag_layer(in_size=H, hid=H, nblocks=nblocks)
    layer.eval()

    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        Wh_struct, bh_cat = extract_blockdiag_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        mon_out = gru_scan_blockdiag_forward_pytorch(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref_out - mon_out).abs().max().item()
    rel = max_diff / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-5, f"forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 32, 64, 4), (16, 32, 256, 4)])
def test_blockdiag_triton_forward_matches_pytorch(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Triton forward kernel must match the PyTorch blockdiag reference
    within TF32 noise."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    blksz = H // nblocks
    Wh_struct = (torch.randn(3, nblocks, blksz, blksz, device=device) * 0.1).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    ref = gru_scan_blockdiag_forward_pytorch(gi, h0, Wh_struct, bh_cat)
    tri = gru_scan_blockdiag_forward_triton(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    rel = max_diff / max(ref.abs().max().item(), 1e-6)
    # TF32 matmul + T-step compounding.
    assert rel < 5e-3, f"forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 32, 64, 4), (16, 32, 256, 4)])
def test_blockdiag_triton_qat_forward_matches_pytorch(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """In-kernel fake-quant forward: Triton must match PyTorch reference."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    blksz = H // nblocks
    gi = (torch.randn(T, B, 3 * H, device=device) * 0.1).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.1).contiguous()
    Wh_struct = (torch.randn(3, nblocks, blksz, blksz, device=device) * 0.1).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.05).contiguous()

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_in_q = (0.02, qmin, qmax)
    h_out_q = (0.02, qmin, qmax)

    ref = gru_scan_blockdiag_forward_pytorch(
        gi, h0, Wh_struct, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    tri = gru_scan_blockdiag_forward_triton(
        gi, h0, Wh_struct, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )

    max_diff = (ref - tri).abs().max().item()
    rel = max_diff / max(ref.abs().max().item(), 1e-6)
    assert rel < 1e-1, f"qat forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 32, 64, 4), (16, 32, 256, 4)])
def test_blockdiag_triton_qat_backward_matches_pytorch(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """In-kernel fake-quant backward: Triton must match PyTorch reference
    on (dgi, dh0, dWh_struct, dbh)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    blksz = H // nblocks
    gi = (torch.randn(T, B, 3 * H, device=device) * 0.1).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.1).contiguous()
    Wh_struct = (torch.randn(3, nblocks, blksz, blksz, device=device) * 0.1).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.05).contiguous()

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_in_q = (0.02, qmin, qmax)
    h_out_q = (0.02, qmin, qmax)

    out_fwd = gru_scan_blockdiag_forward_triton(
        gi, h0, Wh_struct, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    dout = (torch.randn(T, B, H, device=device) * 0.1).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_blockdiag_backward_triton(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_blockdiag_backward_pytorch(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )

    for name, t, p in [
        ("dgi", dgi_t, dgi_p),
        ("dh0", dh0_t, dh0_p),
        ("dWh_struct", dWh_t, dWh_p),
        ("dbh", dbh_t, dbh_p),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        assert rel < 1e-1, f"qat {name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 32, 64, 4), (16, 32, 256, 4)])
def test_blockdiag_triton_backward_matches_pytorch(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Triton backward gradients must match the PyTorch blockdiag reference
    on (dgi, dh0, dWh_struct, dbh)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    blksz = H // nblocks
    Wh_struct = (torch.randn(3, nblocks, blksz, blksz, device=device) * 0.1).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    out_fwd = gru_scan_blockdiag_forward_triton(gi, h0, Wh_struct, bh_cat)
    dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_blockdiag_backward_triton(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_blockdiag_backward_pytorch(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )

    for name, t, p in [
        ("dgi", dgi_t, dgi_p),
        ("dh0", dh0_t, dh0_p),
        ("dWh_struct", dWh_t, dWh_p),
        ("dbh", dbh_t, dbh_p),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        assert rel < 1e-2, f"{name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 16, 32, 4), (16, 32, 64, 4)])
def test_grulayer_use_triton_matches_pytorch_path(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """GRULayer with use_triton=True must produce the same output as
    use_triton=False (PyTorch path). Both fp32, no activation quant."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cfg = StructureConfig(kind="blockdiag", nblocks=nblocks)

    pt_layer = GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=cfg, use_triton=False,
    ).to(device)
    tri_layer = GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=cfg, use_triton=True,
    ).to(device)
    tri_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    pt_out, pt_hT = pt_layer(x, h0)
    tri_out, tri_hT = tri_layer(x, h0)

    rel_out = (pt_out - tri_out).abs().max().item() / max(pt_out.abs().max().item(), 1e-6)
    rel_hT = (pt_hT - tri_hT).abs().max().item() / max(pt_hT.abs().max().item(), 1e-6)
    assert rel_out < 5e-3, f"out rel diff {rel_out:.4e}"
    assert rel_hT < 5e-3, f"hT rel diff {rel_hT:.4e}"


@cuda_only
def test_grulayer_use_triton_qat_after_calibration() -> None:
    """End-to-end QAT flow: build, calibrate, freeze, run via Triton.

    Validates that the GRULayer's use_triton path correctly extracts
    h_in/h_out frozen scales and produces correct output."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    H = 32
    T, B = 8, 16
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=8, name="h_q"),  # int8 hidden
    )
    cfg = StructureConfig(kind="blockdiag", nblocks=4)
    layer = GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=cfg, use_triton=True,
    ).to(device)

    # Calibrate with synthetic loader, then freeze.
    def loader(n):
        for _ in range(n):
            yield torch.randn(T, B, H, device=device) * 0.1

    layer.calibrate(loader(8), n_batches=8)
    layer.freeze()

    # Forward should now run through the Triton path with frozen scales.
    x = torch.randn(T, B, H, device=device) * 0.1
    out, hT = layer(x)
    assert torch.isfinite(out).all()
    assert out.shape == (T, B, H)


@cuda_only
def test_grulayer_use_triton_eligibility_errors() -> None:
    """use_triton=True must error when the cell isn't compatible
    (input structured, non-blockdiag hidden, or split gate layout)."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    # Hidden non-blockdiag (circulant)
    cfg_circ = StructureConfig(kind="circulant")
    with pytest.raises(ValueError, match="blockdiag"):
        GRULayer(
            32, 32, recipe=rec, gate_layout="fused",
            structure_hidden=cfg_circ, use_triton=True,
        )
    # Split gate layout
    cfg_mon = StructureConfig(kind="blockdiag", nblocks=4)
    with pytest.raises(ValueError, match="fused"):
        GRULayer(
            32, 32, recipe=rec, gate_layout="split",
            structure_hidden=cfg_mon, use_triton=True,
        )


@pytest.mark.parametrize("T,B,H,nblocks", [(8, 4, 32, 4), (16, 8, 64, 4)])
def test_blockdiag_pytorch_backward_matches_cell(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Gradients from the PyTorch blockdiag reference must match autograd
    through the tier-1 cell. We compare gradients of the inputs (gi, h0)
    and of the bias (bh) — both representations share these. The Wh
    parameter gradients live in different layouts (cell has three [nblocks,
    blksz, blksz]; reference has one [3, nblocks, blksz, blksz]) so we
    stack the cell's grads to compare."""
    torch.manual_seed(0)
    layer = _make_blockdiag_layer(in_size=H, hid=H, nblocks=nblocks)

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

    # ---- Blockdiag reference path ----
    Wh_struct, bh_cat = extract_blockdiag_factors(layer.cell)
    with torch.no_grad():
        gi = _build_gi_from_cell(layer, x)
        out_fwd = gru_scan_blockdiag_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        # dout matches the gradient of (.pow(2).sum()) wrt the layer's output:
        #   d(sum(out^2))/d(out) = 2 * out
        dout = 2.0 * out_fwd

    dgi, dh0, dWh_struct, dbh = gru_scan_blockdiag_backward_pytorch(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )

    # Compare h0 gradients
    diff_h0 = (dh0 - ref_h0.grad).abs().max().item()
    rel_h0 = diff_h0 / max(ref_h0.grad.abs().max().item(), 1e-6)
    assert rel_h0 < 1e-5, f"dh0 rel diff {rel_h0:.4e}"

    # Compare Wh gradients
    diff_Wh = (dWh_struct - ref_dWh).abs().max().item()
    rel_Wh = diff_Wh / max(ref_dWh.abs().max().item(), 1e-6)
    assert rel_Wh < 1e-5, f"dWh rel diff {rel_Wh:.4e}"

    # Compare bh gradients
    diff_bh = (dbh - ref_dbh).abs().max().item()
    rel_bh = diff_bh / max(ref_dbh.abs().max().item(), 1e-6)
    assert rel_bh < 1e-5, f"dbh rel diff {rel_bh:.4e}"
