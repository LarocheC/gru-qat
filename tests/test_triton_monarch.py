"""Two-factor Monarch fast-path tests.

Covers the genuine two-factor Monarch (``kind="monarch"``) hidden scan:
- the PyTorch reference matches the tier-1 cell's structured forward;
- the fused Triton forward matches the PyTorch reference (TF32 tolerance);
- the autograd wrapper's gradients match the reference;
- ``GRULayer(use_triton=True)`` matches ``use_triton=False`` end to end,
  including an int8-hidden calibrate -> freeze -> deploy flow.

Distinct from ``tests/test_triton_blockdiag.py`` (single block-diagonal
factor). The two-factor kernel does two block-diagonal matmuls with a
transpose-permute between them, so it mixes across all blocks.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

torch_structured = pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402
from gru_qat.triton_kernels.scan_monarch import (  # noqa: E402
    extract_monarch_factors,
    gru_scan_monarch,
    gru_scan_monarch_backward_pytorch,
    gru_scan_monarch_backward_triton,
    gru_scan_monarch_forward_pytorch,
    gru_scan_monarch_forward_triton,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


def _fp32_recipe() -> QuantRecipe:
    return QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )


def _make_monarch_layer(H: int, nblocks: int, use_triton: bool | str) -> GRULayer:
    return GRULayer(
        H, H, recipe=_fp32_recipe(), gate_layout="fused",
        structure_hidden=StructureConfig(kind="monarch", nblocks=nblocks),
        use_triton=use_triton,
    )


def _build_gi(cell: object, x: torch.Tensor) -> torch.Tensor:
    """Reproduce the cell's dense input projection for a matched gi."""
    Wi = torch.cat(
        [cell.quant_W_ir(cell.W_ir), cell.quant_W_iz(cell.W_iz), cell.quant_W_in(cell.W_in)],
        dim=0,
    )
    bi = torch.cat([cell.b_ir, cell.b_iz, cell.b_in])
    return torch.nn.functional.linear(cell.quant_x(x), Wi, bi)


@pytest.mark.parametrize("T,B,H,nb", [(8, 4, 64, 4), (16, 8, 128, 8)])
def test_monarch_pytorch_matches_cell(T: int, B: int, H: int, nb: int) -> None:
    """The PyTorch two-factor reference must match the tier-1 cell's
    structured forward (same weights, fp32-Identity quant)."""
    torch.manual_seed(0)
    layer = _make_monarch_layer(H, nb, use_triton=False).eval()
    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)
    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        W1, W2, bh = extract_monarch_factors(layer.cell)
        gi = _build_gi(layer.cell, x)
        py = gru_scan_monarch_forward_pytorch(gi, h0, W1, W2, bh)
    rel = (ref_out - py).abs().max().item() / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-5, f"pyref vs cell rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nb", [(8, 32, 64, 4), (16, 32, 256, 4), (8, 32, 128, 8)])
def test_monarch_triton_forward_matches_pytorch(T: int, B: int, H: int, nb: int) -> None:
    """Fused Triton forward must match the PyTorch reference within TF32 noise
    (two chained tl.dot matmuls compounded over T)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    blksz = H // nb
    gi = (torch.randn(T, B, 3 * H, device=dev) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=dev) * 0.5).contiguous()
    W1 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    bh = (torch.randn(3 * H, device=dev) * 0.1).contiguous()
    ref = gru_scan_monarch_forward_pytorch(gi, h0, W1, W2, bh)
    tri = gru_scan_monarch_forward_triton(gi, h0, W1, W2, bh)
    rel = (ref - tri).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert rel < 5e-3, f"forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nb", [(8, 32, 64, 4), (8, 32, 256, 4)])
def test_monarch_triton_qat_forward_matches_pytorch(T: int, B: int, H: int, nb: int) -> None:
    """In-kernel fake-quant forward: Triton must match the PyTorch reference."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    blksz = H // nb
    gi = (torch.randn(T, B, 3 * H, device=dev) * 0.1).contiguous()
    h0 = (torch.randn(B, H, device=dev) * 0.1).contiguous()
    W1 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    bh = (torch.randn(3 * H, device=dev) * 0.05).contiguous()
    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    q = (0.02, qmin, qmax)
    ref = gru_scan_monarch_forward_pytorch(gi, h0, W1, W2, bh, h_in_quant=q, h_out_quant=q)
    tri = gru_scan_monarch_forward_triton(gi, h0, W1, W2, bh, h_in_quant=q, h_out_quant=q)
    rel = (ref - tri).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert rel < 1e-1, f"qat forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize(
    "T,B,H,nb", [(8, 32, 64, 4), (16, 32, 256, 4), (8, 96, 128, 8)]
)
def test_monarch_triton_backward_matches_reference(T: int, B: int, H: int, nb: int) -> None:
    """Hand-derived Triton backward must match the PyTorch reference backward
    on (dgi, dh0, dW1, dW2, dbh), fp32. B=96 exercises cross-program atomic
    weight-grad accumulation (n_pid_b=3)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    blksz = H // nb
    gi = (torch.randn(T, B, 3 * H, device=dev) * 0.3).contiguous()
    h0 = (torch.randn(B, H, device=dev) * 0.2).contiguous()
    W1 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    bh = (torch.randn(3 * H, device=dev) * 0.1).contiguous()
    out = gru_scan_monarch_forward_triton(gi, h0, W1, W2, bh)
    dout = (torch.randn(T, B, H, device=dev) * 0.1).contiguous()

    tg = gru_scan_monarch_backward_triton(gi, h0, W1, W2, bh, out, dout)
    rg = gru_scan_monarch_backward_pytorch(gi, h0, W1, W2, bh, dout)
    for name, a, b in zip(["dgi", "dh0", "dW1", "dW2", "dbh"], tg, rg):
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)
        assert rel < 5e-3, f"{name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nb", [(8, 32, 64, 4), (12, 16, 128, 8)])
def test_monarch_triton_qat_backward_matches_reference(T: int, B: int, H: int, nb: int) -> None:
    """Hand-derived Triton backward under in-kernel fake-quant must match the
    reference backward (looser tolerance: STE rounding-boundary flips)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    blksz = H // nb
    gi = (torch.randn(T, B, 3 * H, device=dev) * 0.3).contiguous()
    h0 = (torch.randn(B, H, device=dev) * 0.2).contiguous()
    W1 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    W2 = (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1).contiguous()
    bh = (torch.randn(3 * H, device=dev) * 0.1).contiguous()
    q = (0.02, -127, 127)
    out = gru_scan_monarch_forward_triton(gi, h0, W1, W2, bh, h_in_quant=q, h_out_quant=q)
    dout = (torch.randn(T, B, H, device=dev) * 0.1).contiguous()

    tg = gru_scan_monarch_backward_triton(gi, h0, W1, W2, bh, out, dout, h_in_quant=q, h_out_quant=q)
    rg = gru_scan_monarch_backward_pytorch(gi, h0, W1, W2, bh, dout, h_in_quant=q, h_out_quant=q)
    for name, a, b in zip(["dgi", "dh0", "dW1", "dW2", "dbh"], tg, rg):
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)
        assert rel < 5e-2, f"{name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nb", [(8, 8, 64, 4), (12, 16, 128, 8)])
def test_monarch_autograd_grads_match_reference(T: int, B: int, H: int, nb: int) -> None:
    """The autograd wrapper's gradients (Triton fwd, reference bwd) must match
    autograd through the pure PyTorch reference."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    blksz = H // nb

    def leaves() -> list[torch.Tensor]:
        g = [
            (torch.randn(T, B, 3 * H, device=dev) * 0.1),
            (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1),
            (torch.randn(3, nb, blksz, blksz, device=dev) * 0.1),
            (torch.randn(3 * H, device=dev) * 0.1),
        ]
        return g

    torch.manual_seed(1)
    gi0, W1_0, W2_0, bh0 = leaves()
    h0 = torch.zeros(B, H, device=dev)

    gi = gi0.clone().requires_grad_()
    W1 = W1_0.clone().requires_grad_()
    W2 = W2_0.clone().requires_grad_()
    bh = bh0.clone().requires_grad_()
    gru_scan_monarch(gi, h0, W1, W2, bh).float().pow(2).sum().backward()
    got = [gi.grad, W1.grad, W2.grad, bh.grad]

    gi2 = gi0.clone().requires_grad_()
    W1b = W1_0.clone().requires_grad_()
    W2b = W2_0.clone().requires_grad_()
    bhb = bh0.clone().requires_grad_()
    gru_scan_monarch_forward_pytorch(gi2, h0, W1b, W2b, bhb).float().pow(2).sum().backward()
    ref = [gi2.grad, W1b.grad, W2b.grad, bhb.grad]

    for name, a, b in zip(["dgi", "dW1", "dW2", "dbh"], got, ref):
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)
        # gradients differ only by the TF32 forward noise flowing into dout.
        assert rel < 2e-2, f"{name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nb", [(8, 16, 64, 4), (16, 16, 128, 8)])
def test_grulayer_monarch_triton_matches_pytorch_path(T: int, B: int, H: int, nb: int) -> None:
    """GRULayer use_triton=True must match use_triton=False for two-factor
    Monarch, forward and final hidden state, fp32."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    pt = _make_monarch_layer(H, nb, use_triton=False).to(dev)
    tri = _make_monarch_layer(H, nb, use_triton=True).to(dev)
    tri.load_state_dict(pt.state_dict())
    assert tri.use_triton is True and pt.use_triton is False

    x = torch.randn(T, B, H, device=dev) * 0.1
    h0 = torch.randn(B, H, device=dev) * 0.1
    pt_out, pt_hT = pt(x, h0)
    tri_out, tri_hT = tri(x, h0)
    rel_out = (pt_out - tri_out).abs().max().item() / max(pt_out.abs().max().item(), 1e-6)
    rel_hT = (pt_hT - tri_hT).abs().max().item() / max(pt_hT.abs().max().item(), 1e-6)
    assert rel_out < 5e-3, f"out rel diff {rel_out:.4e}"
    assert rel_hT < 5e-3, f"hT rel diff {rel_hT:.4e}"


@cuda_only
def test_grulayer_monarch_triton_backward_matches_pytorch_path() -> None:
    """Backward through the GRULayer monarch fast path must match the per-step
    reference path (fp32)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    T, B, H, nb = 8, 16, 64, 4
    pt = _make_monarch_layer(H, nb, use_triton=False).to(dev)
    tri = _make_monarch_layer(H, nb, use_triton=True).to(dev)
    tri.load_state_dict(pt.state_dict())

    x_pt = (torch.randn(T, B, H, device=dev) * 0.1).requires_grad_()
    x_tri = x_pt.detach().clone().requires_grad_()
    h0 = torch.randn(B, H, device=dev) * 0.1
    pt(x_pt, h0)[0].float().pow(2).sum().backward()
    tri(x_tri, h0)[0].float().pow(2).sum().backward()
    rel_dx = (x_pt.grad - x_tri.grad).abs().max().item() / max(x_pt.grad.abs().max().item(), 1e-6)
    assert rel_dx < 2e-2, f"dx rel diff {rel_dx:.4e}"

    for (n_pt, p_pt), (n_tri, p_tri) in zip(pt.named_parameters(), tri.named_parameters()):
        assert n_pt == n_tri
        if p_pt.grad is None:
            continue
        rel = (p_pt.grad - p_tri.grad).abs().max().item() / max(p_pt.grad.abs().max().item(), 1e-6)
        assert rel < 3e-2, f"{n_pt} grad rel diff {rel:.4e}"


@cuda_only
def test_grulayer_monarch_qat_after_calibration() -> None:
    """End-to-end QAT: build monarch fast path, calibrate, freeze, run int8."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    H, T, B, nb = 64, 8, 16, 4
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=8, name="h_q"),
    )
    layer = GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="monarch", nblocks=nb),
        use_triton=True,
    ).to(dev)

    def loader(n: int):
        for _ in range(n):
            yield torch.randn(T, B, H, device=dev) * 0.1

    layer.calibrate(loader(8), n_batches=8)
    layer.freeze()
    out, hT = layer(torch.randn(T, B, H, device=dev) * 0.1)
    assert torch.isfinite(out).all()
    assert out.shape == (T, B, H)
    assert hT.shape == (B, H)
