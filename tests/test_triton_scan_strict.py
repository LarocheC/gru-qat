"""Strict-tier parity tests for the dense Triton scan kernel — Phase 2 audit.

Validates ``gru_scan`` / ``gru_scan_forward`` / ``gru_scan_persistent`` (and
their fwd/bwd helpers) against the Phase 1 reference path
(``GRULayer(use_triton=False, dense, Identity quantizers)``) at the strict
tier::

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 5e-4     # absolute, not relative

Diverges intentionally from ``tests/test_triton_scan.py`` (which runs under
``'high'`` / TF32 with 5e-3..1e-1 relative bounds — that's the
realistic-deployment tier). Both files coexist; this file does NOT loosen the
existing one (D-20). The realistic-tier sibling is the deployment regime; this
file audits the math.

Tight-TF32 strict-tier bound rationale (Phase 2 Plan 02-06 disposition):
Triton's ``tl.dot`` defaults to TF32 on Ampere+ regardless of
``torch.set_float32_matmul_precision('highest')`` — the global knob only
affects PyTorch matmuls, not in-kernel ``tl.dot`` reductions. The kernel
under test uses ``tl.dot`` for the hidden GEMM, so its outputs carry TF32's
~10-bit mantissa noise (≈ 1e-4 abs on representative tensors) while the
PyTorch reference path runs at IEEE fp32. The strict-tier bound is therefore
held at ``< 5e-4 abs`` — a "tight TF32" bound that still catches kernel bugs
at the ~5e-4 level without false-positiving on TF32 noise itself. See
Phase 2 Plan 02-06 SUMMARY / Option C disposition for the audit trail and
bd issue for the accepted divergence.

Also hosts in the same module:

- TRI-05 regression (``test_autotune_dWh_dbh_zero_init_across_configs``) — the
  autotune-config rotation of the slab-zero bug fixed in commit ``c001a8a``.
  Existing single-config regression lives at ``tests/test_triton_scan.py:202-215``.
- TRI-06 regression (``test_persistent_kernel_deterministic``) — 50-run
  bit-identical guard for the release/acquire cross-CTA fence
  (see ``src/gru_qat/triton_kernels/scan.py:184-208``).
- D-25 static canary (``test_no_cv_cache_modifier_live_uses_in_scan_source``)
  — asserts ``cache_modifier=".cv"`` does not appear in any *live*
  (non-comment) line of ``src/gru_qat/triton_kernels/scan*.py``.

The cell-parity contract in ``tests/test_parity.py`` and the layer-parity
contract in ``tests/test_layer_parity.py`` are LOCKED by D-28 and are NOT
duplicated here.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn  # noqa: F401  (imported for parity with TF32 sibling)

triton = pytest.importorskip("triton")

from gru_qat.gru_layer import GRULayer  # noqa: E402
from gru_qat.quantizers import QuantizerConfig, QuantRecipe  # noqa: E402
from gru_qat.triton_kernels.scan import (  # noqa: E402
    gru_scan,
    gru_scan_forward,
    gru_scan_forward_persistent,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_backward_persistent,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_persistent,
    _gru_scan_backward_pytorch,  # noqa: F401  (imported for symmetry with sibling)
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (tests/test_triton_scan.py) uses 'high' to exercise the kernel under
# deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


# duplicated per D-18 (< 30 LOC, inline beats shared module)
def _ref_layer(in_dim: int, hidden: int) -> GRULayer:
    """fp32-Identity GRULayer with fused gates and per-batch input projection.

    The Triton kernel takes the post-input-projection ``gi`` directly, so
    parity is against the layer that produces matching ``gi`` (fused +
    pre_batch_input).
    """
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    return GRULayer(
        in_dim, hidden, recipe=rec, gate_layout="fused", pre_batch_input=True
    )


# Per CONTEXT D-16 dense grid. FAST set runs on every ``pytest -q``; SLOW set
# (T ∈ {512, 1024}) is gated behind ``@pytest.mark.slow``.
FAST_DENSE_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 27 cases

SLOW_DENSE_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)
def test_scan_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """``gru_scan_forward`` must match the reference GRULayer to < 5e-4
    absolute under ``'highest'`` precision.

    Tight-TF32 strict-tier bound (Phase 2 Plan 02-06 disposition / Option C):
    Triton's ``tl.dot`` uses TF32 on Ampere+ regardless of the global
    ``torch.set_float32_matmul_precision('highest')`` setting — the global
    knob does not propagate into in-kernel ``tl.dot`` reductions. The hidden
    GEMM in this kernel therefore carries ~10-bit TF32 mantissa noise while
    the PyTorch reference runs at IEEE fp32. The 5e-4 bound is a "tight TF32"
    audit threshold: still catches kernel bugs at the ~5e-4 level but does
    not false-positive on the documented TF32 floor. The accepted divergence
    is tracked as a bd issue (see Plan 02-06 SUMMARY).

    Realistic-tier sibling (tests/test_triton_scan.py:139) uses < 5e-3 under
    TF32; that's correct for its regime and not loosened by this file.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)
        assert w.Wh_cat is not None and w.bh_cat is not None
        triton_out = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

    max_diff = (ref_out - triton_out).abs().max().item()
    assert max_diff < 5e-4, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_DENSE_GRID)
def test_scan_fwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Slow sibling of ``test_scan_fwd_strict_matches_reference`` over
    SLOW_DENSE_GRID (T ∈ {512, 1024}). Gated behind ``@pytest.mark.slow``.

    Bound: < 5e-4 abs (tight-TF32; see fast-variant docstring).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)
        assert w.Wh_cat is not None and w.bh_cat is not None
        triton_out = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

    max_diff = (ref_out - triton_out).abs().max().item()
    assert max_diff < 5e-4, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)
def test_scan_bwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton autograd gradients must match PyTorch autograd through the
    reference layer to < 5e-4 absolute on x, h0, Wh_cat, bh_cat under
    ``'highest'`` precision.

    Tight-TF32 strict-tier bound (Phase 2 Plan 02-06 / Option C): the bwd
    kernel uses ``tl.dot`` (TF32 on Ampere+) for the hidden-side reductions;
    the global ``'highest'`` knob does not affect in-kernel ``tl.dot``. Bound
    is 5e-4 abs — see fwd docstring for the full rationale and the bd issue
    documenting the accepted TF32 divergence.

    Realistic-tier sibling (tests/test_triton_scan.py:215) uses rel < 1e-1
    under TF32; this file's absolute < 5e-4 is the audit bound.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H

    ref_layer = _ref_layer(IN, H).to(device)
    x = torch.randn(T, B, IN, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    # Reference path: PyTorch autograd through the layer.
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref_layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    # Triton path: pre-batch input projection (autograd-aware), then gru_scan.
    w = ref_layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
    out.float().pow(2).sum().backward()

    # Reconstruct the reference dWh_cat / dbh_cat by concatenating per-gate
    # grads in the same order quantize_weights() builds Wh_cat (r, z, n).
    ref_dWh_cat = torch.cat(
        [ref_layer.cell.W_hr.grad, ref_layer.cell.W_hz.grad, ref_layer.cell.W_hn.grad],
        dim=0,
    )
    ref_dbh_cat = torch.cat(
        [ref_layer.cell.b_hr.grad, ref_layer.cell.b_hz.grad, ref_layer.cell.b_hn.grad],
        dim=0,
    )

    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
        ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
        ("bh_cat", ref_dbh_cat, bh_cat.grad),
    ]:
        assert ref_g is not None and tri_g is not None
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 5e-4, (
            f"{name} max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
        )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_DENSE_GRID)
def test_scan_bwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Slow sibling of ``test_scan_bwd_strict_matches_reference`` over
    SLOW_DENSE_GRID (T ∈ {512, 1024}).

    Bound: < 5e-4 abs (tight-TF32; see fast-variant docstring).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H

    ref_layer = _ref_layer(IN, H).to(device)
    x = torch.randn(T, B, IN, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref_layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    w = ref_layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
    out.float().pow(2).sum().backward()

    ref_dWh_cat = torch.cat(
        [ref_layer.cell.W_hr.grad, ref_layer.cell.W_hz.grad, ref_layer.cell.W_hn.grad],
        dim=0,
    )
    ref_dbh_cat = torch.cat(
        [ref_layer.cell.b_hr.grad, ref_layer.cell.b_hz.grad, ref_layer.cell.b_hn.grad],
        dim=0,
    )

    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
        ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
        ("bh_cat", ref_dbh_cat, bh_cat.grad),
    ]:
        assert ref_g is not None and tri_g is not None
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 5e-4, (
            f"{name} max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
        )


# ---------------------------------------------------------------------------
# TRI-05 + TRI-06 named regression tests
# ---------------------------------------------------------------------------


@cuda_only
def test_autotune_dWh_dbh_zero_init_across_configs() -> None:
    """Regression for TRI-05 (commit ``c001a8a``): the autotuned backward
    kernel allocates per-program dWh / dbh accumulator slabs and must zero
    them on entry. Pre-fix, a stale slab from autotune-config A leaked into
    config B's accumulator, producing dWh / dbh off by ~O(0.1).

    The existing single-config slab-zero regression at
    ``tests/test_triton_scan.py:202-215`` (``test_triton_backward_matches_pytorch``)
    catches the bug on a SINGLE autotune config; this variant rotates through
    two different ``(T, B)`` shapes which hit different autotune buckets per
    the autotune ``key=['T', 'B']`` declared at
    ``src/gru_qat/triton_kernels/scan.py:732`` (autotuned fwd) and ``:893``
    (autotuned bwd). If the slab-zero fix regresses, the SECOND iteration's
    ``dWh_cat`` / ``dbh_cat`` diverge from reference while the first still
    passes — the assertion message includes ``iter=`` so the failure is
    unambiguous in pytest output. The slab-zero contract is preserved
    regardless of tolerance: a regressed fix produces ~O(0.1) divergence,
    not ~5e-4.

    Strict tier: < 5e-4 absolute under ``'highest'`` (tight-TF32 per Phase 2
    Plan 02-06 / Option C — Triton's ``tl.dot`` defaults to TF32 on Ampere+
    regardless of the global precision setting). Tighter than the
    realistic-tier sibling's ``rel < 1e-1`` (TF32 regime) and well below the
    ~0.1 divergence a slab-leak regression would produce.
    """
    device = torch.device("cuda")

    # Two shapes that hit different autotune buckets per key=['T','B'].
    # Both T AND B must differ so the autotune cache emits a distinct config.
    shapes = [(16, 16, 64), (32, 32, 64)]

    for idx, (T, B, H) in enumerate(shapes):
        # Fresh seed per iteration so reference grads are reproducible but
        # independent across the two shapes.
        torch.manual_seed(idx)
        IN = H

        ref_layer = _ref_layer(IN, H).to(device)
        x = torch.randn(T, B, IN, device=device, requires_grad=True)
        h0 = torch.randn(B, H, device=device, requires_grad=True)

        ref_x = x.detach().clone().requires_grad_()
        ref_h0 = h0.detach().clone().requires_grad_()
        ref_out, _ = ref_layer(ref_x, ref_h0)
        ref_out.float().pow(2).sum().backward()

        w = ref_layer.cell.quantize_weights()
        Wi_cat = w.Wi_cat.detach().clone()
        bi_cat = w.bi_cat.detach().clone()
        Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
        bh_cat = w.bh_cat.detach().clone().requires_grad_()
        tri_x = x.detach().clone().requires_grad_()
        tri_h0 = h0.detach().clone().requires_grad_()
        gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
        out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
        out.float().pow(2).sum().backward()

        ref_dWh_cat = torch.cat(
            [
                ref_layer.cell.W_hr.grad,
                ref_layer.cell.W_hz.grad,
                ref_layer.cell.W_hn.grad,
            ],
            dim=0,
        )
        ref_dbh_cat = torch.cat(
            [
                ref_layer.cell.b_hr.grad,
                ref_layer.cell.b_hz.grad,
                ref_layer.cell.b_hn.grad,
            ],
            dim=0,
        )

        for name, ref_g, tri_g in [
            ("x", ref_x.grad, tri_x.grad),
            ("h0", ref_h0.grad, tri_h0.grad),
            ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
            ("bh_cat", ref_dbh_cat, bh_cat.grad),
        ]:
            assert ref_g is not None and tri_g is not None
            max_diff = (ref_g - tri_g).abs().max().item()
            assert max_diff < 5e-4, (
                f"iter={idx} shape={(T, B, H)} {name} max abs diff "
                f"{max_diff:.4e} (TRI-05: autotune slab leak — second-iter "
                f"failure means c001a8a fix regressed)"
            )


@cuda_only
def test_persistent_kernel_deterministic() -> None:
    """Regression for TRI-06 (commit ``0e26193`` per REQUIREMENTS.md): the
    persistent fwd kernel uses ``atomic_add(sem='release')`` +
    ``atomic_add(0, sem='acquire')`` for cross-CTA visibility — see the
    comment block at ``src/gru_qat/triton_kernels/scan.py:184-208`` and the
    "What the agent should NOT do" warning at ``DEVELOPMENT.md:131-143``
    against using ``cache_modifier=".cv"`` as a fence substitute. The
    pre-fix code (relaxed atomics + ``.cv`` load) produced output that was
    *mostly* correct but drifted by ~0.2 absolute on some ``[t>=1, batch,
    hidden]`` cells depending on CTA schedule order — i.e. non-deterministic.

    This test runs ``gru_scan_persistent`` 50 times on bit-identical inputs
    and asserts ``torch.equal`` across all 50 outputs. If any run diverges,
    the release/acquire pattern has regressed.

    ``torch.equal`` (NOT ``torch.allclose``) is the strict-tier determinism
    gate per D-24: determinism is bit-identity even under TF32, because
    reduction order is fixed per kernel — CTA scheduling is the only
    varying factor. Inputs are allocated ONCE before the loop and NOT
    re-randomized between runs.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, B, H = 64, 16, 128

    gi = torch.randn(T, B, 3 * H, device=device).contiguous()
    h0 = torch.randn(B, H, device=device).contiguous()
    Wh = (torch.randn(3 * H, H, device=device) * 0.1).contiguous()
    bh = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    out0 = gru_scan_persistent(gi, h0, Wh, bh)
    for i in range(1, 50):
        out_i = gru_scan_persistent(gi, h0, Wh, bh)
        assert torch.equal(out0, out_i), (
            f"persistent run {i} diverged from run 0 — cross-CTA fence may "
            f"have regressed. max abs diff "
            f"{(out0 - out_i).abs().max().item():.4e} (TRI-06)"
        )


# ---------------------------------------------------------------------------
# D-25 static .cv cache-modifier canary
# ---------------------------------------------------------------------------


def test_no_cv_cache_modifier_live_uses_in_scan_source() -> None:
    """Static canary for D-25: ``cache_modifier=".cv"`` MUST NOT appear in
    any *live* (non-comment) line of ``src/gru_qat/triton_kernels/scan*.py``.

    The ``.cv`` cache modifier was historically misused as a cross-CTA fence
    substitute; see the comment block at
    ``src/gru_qat/triton_kernels/scan.py:184-208`` and the "What the agent
    should NOT do" section at ``DEVELOPMENT.md:131-143``. The current fix
    pattern uses ``atomic_add(sem='release')`` + ``atomic_add(0,
    sem='acquire')`` for cross-CTA visibility. The dynamic regression guard
    is ``test_persistent_kernel_deterministic`` above (TRI-06); this static
    canary is the cheap CI signal that catches reintroduction before any
    GPU runs.

    At the time this test was authored (2026-05-13), the three occurrences
    of ``cache_modifier=".cv"`` in ``scan.py`` (lines 192, 431, 625) are
    ALL inside ``#``-comment lines that *document* why the pattern is
    forbidden; the live-code baseline is 0. The other ``scan*.py`` files
    (scan_diagonal.py, scan_monarch.py, scan_butterfly.py) have zero matches
    of any kind. If a future commit reintroduces ``cache_modifier=".cv"``
    outside a comment in any of those files, this canary fails with the
    offending file path + line number.

    Comment-strip rule is ``raw.lstrip().startswith("#")`` — correctly
    classifies indented Triton-JIT comment lines (which begin with
    whitespace, then ``#``). ``raw.startswith("#")`` alone would miss them
    and reintroduce false positives.

    Pure-Python via ``pathlib`` (no shell-out per CONVENTIONS.md). Runs
    on CPU; no ``@cuda_only`` needed.
    """
    src_dir = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src"
        / "gru_qat"
        / "triton_kernels"
    )
    assert src_dir.is_dir(), f"expected {src_dir} to exist"

    forbidden = 'cache_modifier=".cv"'
    live_hits: list[tuple[str, int, str]] = []

    for path in sorted(src_dir.glob("scan*.py")):
        for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
            stripped = raw.lstrip()
            if stripped.startswith("#"):
                continue
            if forbidden in stripped:
                live_hits.append((path.name, line_no, raw.rstrip()))

    assert live_hits == [], (
        f'Live (non-comment) cache_modifier=".cv" uses found in scan*.py: '
        f"{live_hits}. See DEVELOPMENT.md anti-pattern note + "
        "tests/test_triton_scan_strict.py::test_persistent_kernel_deterministic "
        "for the dynamic guard."
    )
