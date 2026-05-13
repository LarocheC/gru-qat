"""Strict-tier parity tests for the Butterfly Triton kernel — Phase 2 audit.

Validates ``gru_scan_butterfly_forward_triton`` and
``gru_scan_butterfly_backward_triton`` against the CUDA-op per-step reference
path (``gru_scan_butterfly``, which routes through
``torch_structured.butterfly_multiply``) at the strict tier::

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 1e-5     # absolute, not relative

Butterfly has **no pure-PyTorch reference distinct from the kernel under
test** — the CUDA-op path goes through ``butterfly_multiply`` from
``torch_structured``, and that path serves as ground truth here. Strict-tier
divergence vs the realistic-tier sibling (``tests/test_butterfly_dispatch.py``,
TF32 / rel < 5e-2) is the precision regime: ``'highest'`` IEEE fp32 matmul
eliminates the ~10-bit TF32 mantissa drift and lets us assert absolute
< 1e-5 instead of relative > 1e-2.

Both files coexist; this file does NOT loosen the existing one (D-20). The
realistic-tier sibling exercises the kernel under deployment conditions
(TF32); this file audits the math.

Note: the per-program scratch-OOB regression for the butterfly fwd kernel
(commit ``d8218d4``, finding TRI-04) is covered at
``tests/test_butterfly_dispatch.py:164``
(``test_butterfly_triton_forward_scratch_oob_regression``). That test runs at
(T=16, B=32, H=512) under TF32 with ``rel < 5e-2``; this strict file does
NOT duplicate it per D-22. Phase-exit verification (Plan 02-06) confirms the
OOB regression still passes; if it regresses, the bug surfaces there, not
here.

Butterfly requires H to be a power of 2 (the kernel only supports H = 2^k);
per D-16 the strict grid is restricted to H ∈ {32, 128, 512}.

The cell-parity contract in ``tests/test_parity.py`` and the layer-parity
contract in ``tests/test_layer_parity.py`` are LOCKED by D-28 and are NOT
duplicated here.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest  # noqa: E402
import torch  # noqa: E402

triton = pytest.importorskip("triton")
torch_structured = pytest.importorskip("torch_structured")

from gru_qat import (  # noqa: E402
    GRULayer,
    QuantizerConfig,
    QuantRecipe,
    StructureConfig,
)
from gru_qat.triton_kernels.scan_butterfly import (  # noqa: E402
    extract_butterfly_factors,  # noqa: F401  (imported for symmetry with sibling)
    extract_butterfly_twiddles,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_butterfly,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_butterfly_backward_triton,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_butterfly_forward_triton,  # noqa: F401  (imported for symmetry with sibling)
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (tests/test_butterfly_dispatch.py) uses 'high' to exercise the kernel
# under deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="butterfly dispatch path is CUDA-only"
)


# duplicated per D-18 (< 30 LOC, inline beats shared module).
# Strict-tier callers always pass hidden_bits=32 (fp32-Identity per CONTEXT —
# Phase 2 is fp32-Identity only; quant-on is Phase 4).
def _make_layer(
    H: int, *, use_triton: bool, hidden_bits: int = 32
) -> GRULayer:
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(
            bits=hidden_bits, name="h" if hidden_bits < 32 else "h_id"
        ),
    )
    return GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="butterfly"),
        use_triton=use_triton,
    )


# Butterfly requires H to be a power of 2 per src/gru_qat/structure.py shape
# validators. Per D-16: H in {32, 128, 512}.
FAST_BFLY_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)  # powers of 2; butterfly requires 2^k
]  # 27 cases

SLOW_BFLY_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_BFLY_GRID)
def test_butterfly_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton butterfly forward must match the CUDA-op per-step reference
    (``gru_scan_butterfly``) to < 1e-5 absolute under ``'highest'`` precision.

    Both the reference and the Triton kernel run with IEEE fp32 matmul; the
    only sources of drift are reduction order across log_H stages and the
    per-step nonlinearities. The strict-tier bound asserts that drift stays
    below the algorithmic-noise floor (1e-5 abs).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        pt_out, _ = pt_layer(x, h0)
        fast_out, _ = fast_layer(x, h0)

    max_diff = (pt_out - fast_out).abs().max().item()
    # Strict tier: absolute error under IEEE fp32 matmul. Realistic-tier
    # sibling (tests/test_butterfly_dispatch.py:160) uses < 2e-2 rel under
    # TF32 — that's correct for its regime; not loosened by us.
    assert max_diff < 1e-5, (
        f"butterfly fwd max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_BFLY_GRID)
def test_butterfly_fwd_strict_matches_reference_slow(
    T: int, B: int, H: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T ∈ {512, 1024})."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        pt_out, _ = pt_layer(x, h0)
        fast_out, _ = fast_layer(x, h0)

    max_diff = (pt_out - fast_out).abs().max().item()
    assert max_diff < 1e-5, (
        f"butterfly fwd max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


def _assert_grad_close(
    name: str, ref_g: torch.Tensor | None, tri_g: torch.Tensor | None,
    T: int, B: int, H: int,
) -> None:
    """Strict-tier per-grad assertion. Raises on shape mismatch / missing
    grads so failures are diagnosable per-grad (named) rather than a
    bare tensor-equality blowup.

    Returns silently when both grads are None (e.g. a frozen parameter
    that didn't participate in the forward — skip rather than fail).
    """
    if ref_g is None and tri_g is None:
        return
    assert ref_g is not None, f"{name}: reference grad is None but triton grad is not"
    assert tri_g is not None, f"{name}: triton grad is None but reference grad is not"
    max_diff = (ref_g - tri_g).abs().max().item()
    assert max_diff < 1e-5, (
        f"{name} grad max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_BFLY_GRID)
def test_butterfly_bwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton butterfly backward must match autograd through the CUDA-op
    per-step reference path to < 1e-5 absolute under ``'highest'``.

    Pattern: dual-layer-with-shared-state. ``pt_layer`` runs the per-step
    PyTorch path (``use_triton=False`` — autograd flows through
    ``gru_scan_butterfly`` and its ``butterfly_multiply`` closure);
    ``fast_layer`` runs the Triton kernel. State is shared via
    ``load_state_dict``, so each parameter sees the same value on both
    sides — the only difference is the kernel doing the math.

    Compares gradients on (x, h0) AND on every learnable parameter in the
    layer's ``named_parameters()``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    # Inputs require_grad on both sides; allocate the base tensor first
    # (``* 0.1`` returns a non-leaf tensor and would not preserve
    # requires_grad on the result), then flip the flag in-place.
    x_pt = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_()
    h0_pt = (torch.randn(B, H, device=device) * 0.1).requires_grad_()
    x_tri = x_pt.detach().clone().requires_grad_()
    h0_tri = h0_pt.detach().clone().requires_grad_()

    pt_out, _ = pt_layer(x_pt, h0_pt)
    pt_out.float().pow(2).sum().backward()

    tri_out, _ = fast_layer(x_tri, h0_tri)
    tri_out.float().pow(2).sum().backward()

    # Per-parameter gradient parity. Strict tier: every learnable parameter
    # that participated in both forwards must have matching gradients to
    # < 1e-5 abs.
    fast_params = dict(fast_layer.named_parameters())
    for name, p_pt in pt_layer.named_parameters():
        p_tri = fast_params[name]
        _assert_grad_close(name, p_pt.grad, p_tri.grad, T, B, H)

    # Input gradients.
    for name, ref_g, tri_g in [
        ("x", x_pt.grad, x_tri.grad),
        ("h0", h0_pt.grad, h0_tri.grad),
    ]:
        _assert_grad_close(name, ref_g, tri_g, T, B, H)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_BFLY_GRID)
def test_butterfly_bwd_strict_matches_reference_slow(
    T: int, B: int, H: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T ∈ {512, 1024})."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x_pt = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_()
    h0_pt = (torch.randn(B, H, device=device) * 0.1).requires_grad_()
    x_tri = x_pt.detach().clone().requires_grad_()
    h0_tri = h0_pt.detach().clone().requires_grad_()

    pt_out, _ = pt_layer(x_pt, h0_pt)
    pt_out.float().pow(2).sum().backward()

    tri_out, _ = fast_layer(x_tri, h0_tri)
    tri_out.float().pow(2).sum().backward()

    fast_params = dict(fast_layer.named_parameters())
    for name, p_pt in pt_layer.named_parameters():
        p_tri = fast_params[name]
        _assert_grad_close(name, p_pt.grad, p_tri.grad, T, B, H)

    for name, ref_g, tri_g in [
        ("x", x_pt.grad, x_tri.grad),
        ("h0", h0_pt.grad, h0_tri.grad),
    ]:
        _assert_grad_close(name, ref_g, tri_g, T, B, H)
