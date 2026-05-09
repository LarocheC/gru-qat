"""Triton GRU cell — Phase 5.

The kernel-level contract this module must implement is documented here
so the Phase 5 agent has a clear target before writing a single line of
Triton.

NOTHING IN THIS FILE RUNS YET. It is an interface design.

Kernel variants (in priority order)
-----------------------------------
1. fp16 weights, fp16 acts. Validates loop structure against cuDNN.
2. int8 per-channel weights, fp16 acts.
3. int4 per-group weights, fp16 acts. The actual prize.
4. int8 weights, int8 acts. For embedded targets only.

Each variant is a single `triton.jit` function. They share most code via
`tl.constexpr` parameters:

    @triton.jit
    def gru_cell_kernel(
        x_ptr, h_ptr, out_ptr,
        Wir_ptr, Wiz_ptr, Win_ptr,
        Whr_ptr, Whz_ptr, Whn_ptr,
        b_ir_ptr, b_iz_ptr, b_in_ptr,
        b_hr_ptr, b_hz_ptr, b_hn_ptr,
        scale_W_ptr, zp_W_ptr,        # weight scales (per-channel/group)
        scale_x, scale_h,             # activation scales (frozen scalars)
        BATCH, INPUT_SIZE, HIDDEN_SIZE,
        BITS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SYMMETRIC: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        ...

Tile structure
--------------
- Grid: (cdiv(BATCH, BLOCK_B), cdiv(HIDDEN_SIZE, BLOCK_H))
- Each program computes a [BLOCK_B, BLOCK_H] tile of h_t.
- Inside the kernel, each gate is a separate matmul over INPUT_SIZE +
  HIDDEN_SIZE reduction (we may fuse along the K axis since x and h
  contributions add). Order: r, z, n; then sigmoid/tanh in float; then
  the (1-z)*n + z*h combination.

Dequant pattern
---------------
For per-group int4 weights, on each K-tile load:
  1. Load packed int4 weights as int8 with bit unpack
  2. Subtract zero point (per-group)
  3. Multiply by scale (per-group, broadcast across the group)
  4. Cast to fp16 and feed into the GEMM accumulator

This matches the bitsandbytes / AWQ-style dequant-in-shared-memory
pattern. Reference: bitsandbytes/csrc/ops.cu and the AWQ Triton kernel.

Dispatch
--------
The Python wrapper `triton_gru_cell(x, h, weights, scales, ...)` picks
the kernel variant from the recipe metadata stored on the (frozen)
quantizers. It does NOT introspect tensor dtypes alone — recipe is
authoritative.

Parity test
-----------
For each variant: random inputs, run the PyTorch fake-quant cell with
matching frozen scales, run the Triton kernel, compare. Tolerance:

    fp16/fp16:   1e-3 relative
    int8/fp16:   1e-2 relative (rounding compounds)
    int4/fp16:   2e-2 relative

Anything looser than these is a bug in the kernel, not a tolerance issue.

Benchmarking
------------
`bench/bench_triton_vs_cudnn.py` (Phase 5 deliverable):
    - Sweep (batch, hidden) ∈ {(1,128), (16,256), (64,512), (128,1024)}
    - Report ms/iter for cuDNN GRU, fp16 Triton, int8 Triton, int4 Triton
    - Target: int8 Triton ≥ 0.7× cuDNN at batch ≥ 16

The Python entrypoints below are placeholders to import-test; they raise
on call.
"""

from __future__ import annotations

import torch


def triton_gru_cell(
    x: torch.Tensor,
    h: torch.Tensor,
    *,
    weights: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    zero_points: dict[str, torch.Tensor],
    biases: dict[str, torch.Tensor],
    bits: int,
    group_size: int | None,
    symmetric: bool,
) -> torch.Tensor:
    """Single-step Triton GRU cell. Phase 5."""
    raise NotImplementedError("phase=5")


def is_available() -> bool:
    """Return True if Triton is importable and a CUDA device is present."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()
