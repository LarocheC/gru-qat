"""Butterfly fast-path dispatch tests.

Validates ``GRULayer(use_triton=True, structure_hidden=ButterflyCfg)``:
- Forward parity with the per-step PyTorch path.
- End-to-end QAT (calibrate -> freeze -> forward) runs and produces
  finite output.
- gru_scan_butterfly directly: backward gradients exist on all params.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

torch_structured = pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantizerConfig, QuantRecipe, StructureConfig  # noqa: E402
from gru_qat.triton_kernels.scan_butterfly import (  # noqa: E402
    extract_butterfly_factors,
    gru_scan_butterfly,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="butterfly dispatch path is CUDA-only"
)


def _make_layer(
    H: int, *, use_triton: bool, hidden_bits: int = 32
) -> GRULayer:
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=hidden_bits, name="h" if hidden_bits < 32 else "h_id"),
    )
    return GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="butterfly"),
        use_triton=use_triton,
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 4, 32), (16, 8, 64)])
def test_butterfly_dispatch_matches_per_step(T: int, B: int, H: int) -> None:
    """use_triton=True for butterfly must produce the same forward output
    as the PyTorch per-step path (use_triton=False)."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    pt_out, pt_hT = pt_layer(x, h0)
    fast_out, fast_hT = fast_layer(x, h0)

    rel_out = (pt_out - fast_out).abs().max().item() / max(pt_out.abs().max().item(), 1e-6)
    rel_hT = (pt_hT - fast_hT).abs().max().item() / max(pt_hT.abs().max().item(), 1e-6)
    assert rel_out < 1e-4, f"out rel diff {rel_out:.4e}"
    assert rel_hT < 1e-4, f"hT rel diff {rel_hT:.4e}"


@cuda_only
def test_butterfly_grulayer_qat_after_calibration() -> None:
    """End-to-end: train (synthetic), calibrate, freeze, forward via
    butterfly fast path. Output finite, no errors."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    H = 32
    T, B = 8, 16
    layer = _make_layer(H, use_triton=True, hidden_bits=8).to(device)

    def loader(n):
        for _ in range(n):
            yield torch.randn(T, B, H, device=device) * 0.1

    layer.calibrate(loader(8), n_batches=8)
    layer.freeze()

    x = torch.randn(T, B, H, device=device) * 0.1
    out, hT = layer(x)
    assert torch.isfinite(out).all()
    assert out.shape == (T, B, H)
    assert hT.shape == (B, H)


@cuda_only
def test_butterfly_grulayer_dispatch_grad_flows() -> None:
    """Backward must populate gradients on all learnable params when the
    fast dispatch is used (parameters live inside the Butterfly modules
    + dense input weights + biases)."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    H = 32
    T, B = 6, 8
    layer = _make_layer(H, use_triton=True).to(device)

    x = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_()
    h0 = torch.randn(B, H, device=device) * 0.1
    out, _ = layer(x, h0)
    loss = out.float().pow(2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Every learnable parameter that participated in the forward should
    # have a grad tensor populated.
    for name, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


@cuda_only
def test_butterfly_extract_and_gru_scan_directly() -> None:
    """Calling gru_scan_butterfly with factors extracted from a layer
    must produce the same result as routing through GRULayer."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    H = 32
    T, B = 8, 4
    layer = _make_layer(H, use_triton=True).to(device).eval()

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        layer_out, _ = layer(x, h0)

    # Same flow but stitched together by hand.
    with torch.no_grad():
        xq = layer.cell.quant_x(x)
        Wi_cat, bi_cat = layer.cell.quantize_input_weights()
        gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)
        modules, bh_cat = extract_butterfly_factors(layer.cell)
        manual_out = gru_scan_butterfly(gi, h0, modules, bh_cat)

    rel = (layer_out - manual_out).abs().max().item() / max(layer_out.abs().max().item(), 1e-6)
    assert rel < 1e-5, f"manual vs layer rel diff {rel:.4e}"
