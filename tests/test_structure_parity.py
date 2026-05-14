"""Hand-rolled parity tests for the Circulant and LDR PyTorch fallback paths — Phase 3 audit.

Pins ``_CirculantLinear`` and ``_LDRLinear`` in ``src/gru_qat/structure.py``
against independent hand-rolled mathematical references at < 1e-5 abs (fwd +
bwd via autograd-grad comparison). Plus STR-03 graceful-degradation: when
``torch-structured`` is missing, optional-dep kinds (monarch, butterfly, ldr)
must raise ImportError with a clear install hint, while local-impl kinds
(circulant, diagonal, dense) continue to work.

Pure PyTorch — no Triton, no CUDA. Pairs with ``tests/test_structure.py``
(smoke/integration tier: finite-output + gradient-flow + training-loop +
int8-QAT). Two clear tiers, one file each.

This module sets ``torch.set_float32_matmul_precision('highest')`` at import
time because Phase 3 audits the math (per Phase 3 CONTEXT D-40 — TF32 'high'
is for Triton kernel files only). The < 1e-5 abs bound is achievable on
fp32-vs-fp32 without TF32 in play.

This module does NOT call ``pytest.importorskip("torch_structured")`` at
module top — the circulant family is a local impl (see
``src/gru_qat/structure.py:207-225``) and must run on machines without
``torch-structured`` installed. LDR-specific imports (plan 03-02) will be
guarded per-section.
"""

from __future__ import annotations

import pytest  # noqa: F401 — used by parametrize decorators added in subsequent task
import torch

# Per Phase 3 CONTEXT D-40: pure PyTorch (no tl.dot), so 'highest' is
# achievable and < 1e-5 abs is the strict bound. Diverges from the Triton
# kernel test files (which use 'high' to test under realistic TF32
# conditions). Module-level because set_float32_matmul_precision is global
# state — set once at import is the cleanest signal.
torch.set_float32_matmul_precision("highest")

from gru_qat.structure import _CirculantLinear  # noqa: E402, F401 — used by tests added in subsequent task


def _build_toeplitz_from_kernel(c: torch.Tensor) -> torch.Tensor:
    """Build the H x H circulant matrix C from the length-H kernel vector c.

    Per Phase 3 PATTERNS.md lines 286-302 (convention reconciliation), the
    production ``_CirculantLinear.forward`` computes::

        y[b, k] = sum_j col[(k - j) mod n] * x[b, j]

    i.e., circular convolution of ``col`` with ``x``. The matrix C that
    represents this operation in row-vector form (``y = x @ C.T``) has::

        C[i, j] = c[(i - j) mod H]

    so C's first column equals c. Each subsequent column is c cyclically
    shifted down by one.

    Returns a tensor with the same dtype/device as ``c``. Used as one of the
    two independent references in the self-consistency check (FFT-form vs
    Toeplitz-form) BEFORE either is compared to ``_CirculantLinear``.
    """
    H = c.shape[0]
    idx = torch.arange(H, device=c.device)
    # Vectorized outer arithmetic: i_minus_j[i, j] = (i - j) mod H.
    i_minus_j_mod_H = (idx[:, None] - idx[None, :]) % H
    return c[i_minus_j_mod_H]


def _circulant_via_fft(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply the circulant matrix defined by kernel ``c`` to ``x`` via full
    complex FFT.

    Deliberately uses ``torch.fft.fft`` / ``torch.fft.ifft`` (NOT
    ``rfft``/``irfft``) — the production path uses ``rfft``/``irfft``, so
    this is the genuinely independent FFT reference for the self-consistency
    check.

    For ``x`` of shape ``(B, H)``::

        y = real(ifft(fft(c, n=H) * fft(x, n=H, dim=-1), n=H, dim=-1))

    The ``.real`` cast is safe because ``c`` and ``x`` are real-valued; any
    imaginary component is floating-point noise (fp64-relative).
    """
    H = c.shape[0]
    c_f = torch.fft.fft(c, n=H)
    x_f = torch.fft.fft(x, n=H, dim=-1)
    y = torch.fft.ifft(c_f * x_f, n=H, dim=-1)
    return y.real


# Shape grids per Phase 3 CONTEXT D-36.
# Circulant: square; power-of-2 (per src/gru_qat/structure.py:95-98 validator).
FAST_CIRC_GRID: list[tuple[int, int]] = [
    (B, H)
    for B in (1, 4, 32)
    for H in (8, 32, 128)
]  # 9 cases
SLOW_CIRC_GRID: list[tuple[int, int]] = [
    (B, H)
    for B in (1, 4, 32)
    for H in (512,)
]  # 3 cases
