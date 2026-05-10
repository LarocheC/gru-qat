"""Butterfly hidden weights for the multi-step GRU scan.

Two paths live here:

1. ``gru_scan_butterfly`` — the API-parity path. Python time loop
   calling ``torch_structured.butterfly_multiply`` per step. Backward
   via standard PyTorch autograd. ~as fast as the tier-1 structured
   step path; mostly exists so ``GRULayer(use_triton="auto")`` works
   uniformly across structured kinds.

2. ``gru_scan_butterfly_triton`` — multi-step persistent Triton kernel.
   Implements the butterfly multiply directly in Triton (log_N stages
   of strided 2×2 mixing) and fuses the recurrence across timesteps.
   No tensor-core utilization (butterfly's 2×2 mixing isn't a GEMM
   shape), so the win comes purely from launch-count reduction:
   T×ops_per_step launches → one launch per train-step half.

Forward kernel layout (Triton path):
- Grid: (cdiv(B, BLOCK_B),). Each program owns [BLOCK_B, H] state and
  runs ALL T timesteps independently. Butterfly's recurrence is
  per-batch-row independent so no cross-CTA sync is needed.
- Per timestep: 3 butterfly multiplies (one per gate) into per-gate
  scratch buffers in global memory, then gate compose, then store h_t.
- Per butterfly stage: load self + (XOR stride) partner, apply the 2×2
  twiddle for this stage's pair index, scatter back. Triton's register
  tensors don't allow dynamic gather/scatter so the running state
  passes through global memory between stages — L2 absorbs the cost.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from gru_qat.ste import fake_quant_ste


def extract_butterfly_factors(
    cell: nn.Module,
) -> tuple[list[nn.Module], torch.Tensor]:
    """Pull the three hidden-side Butterfly modules out of a tier-1 cell.

    Returns the underlying ``torch_structured.Butterfly`` instances
    rather than raw twiddles — Butterfly's forward pre/post-processes
    its input (reshape into ``[batch, nstacks, in_size]`` etc.) so it's
    cleaner to call the module than to reproduce the wrapping ourselves.

    Args:
        cell: a ``GRUCellQuant`` whose ``structure_hidden`` was a
              ``StructureConfig(kind="butterfly", ...)``.

    Returns:
        modules: list of three ``Butterfly`` modules, one per gate
                 (r, z, n).
        bh_cat:  [3*H] — concat of (b_hr, b_hz, b_hn).
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # struct_Wh_* are _ButterflyLinear wrappers; .b is the underlying
    # torch_structured.Butterfly nn.Module.
    modules = [
        cell.struct_Wh_r.b,
        cell.struct_Wh_z.b,
        cell.struct_Wh_n.b,
    ]
    sample = next(modules[0].parameters())
    if cell.b_hr is None:
        bh_cat = torch.zeros(
            3 * cell.hidden_size, device=sample.device, dtype=sample.dtype,
        )
    else:
        bh_cat = torch.cat([cell.b_hr, cell.b_hz, cell.b_hn])
    return modules, bh_cat


def _maybe_fake_quant(
    x: torch.Tensor, params: tuple[float, int, int] | None
) -> torch.Tensor:
    """Apply per-tensor symmetric fake-quant when params is provided."""
    if params is None:
        return x
    scale, qmin, qmax = params
    s = torch.tensor(scale, device=x.device, dtype=x.dtype)
    zp = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return fake_quant_ste(x, s, zp, qmin, qmax)


def gru_scan_butterfly(
    gi: torch.Tensor,
    h0: torch.Tensor,
    butterfly_modules: list[nn.Module],
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Differentiable Butterfly-hidden-side GRU scan.

    Mirror of ``gru_scan_monarch`` but the matmul per step goes through
    ``torch_structured.Butterfly``'s CUDA op. No multi-step Triton fusion.

    Args:
        gi:       [T, B, 3H] — pre-batched input projection (with bi).
        h0:       [B, H]
        butterfly_modules: list of three ``torch_structured.Butterfly``
            modules, one per gate. Get from ``extract_butterfly_factors``.
        bh_cat:   [3*H]
        h_in_quant / h_out_quant: optional ``(scale, qmin, qmax)`` —
            same semantics as ``gru_scan_monarch``.

    Returns:
        out: [T, B, H]
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    Wr_m, Wz_m, Wn_m = butterfly_modules
    bh = bh_cat.view(3, H)

    out = []
    h = h0
    for t in range(T):
        hq = _maybe_fake_quant(h, h_in_quant)
        gh_r = Wr_m(hq) + bh[0]
        gh_z = Wz_m(hq) + bh[1]
        gh_n = Wn_m(hq) + bh[2]

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_new = (1.0 - z) * n + z * h
        h_new = _maybe_fake_quant(h_new, h_out_quant)
        out.append(h_new)
        h = h_new

    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Multi-step persistent Triton butterfly kernel
# ---------------------------------------------------------------------------


@triton.jit
def gru_scan_butterfly_fwd_kernel(
    gi_ptr,             # [T, B, 3H], fp32
    h0_ptr,             # [B, H], fp32
    twiddle_ptr,        # [3, log_n, n//2, 2, 2], fp32 — one per gate
    bh_ptr,             # [3H], fp32
    out_ptr,            # [T, B, H], fp32
    # Per-program scratch: 3 gate buffers + 1 hq buffer, each [BLOCK_B, H].
    # Accessed only by this program (no cross-CTA), so disjoint across pid_b.
    scratch_ptr,        # [num_pid_b, 4, BLOCK_B, H], fp32
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    st_g, st_s, st_p, st_m_new, st_m_old,
    so_t, so_b,
    sscr_pid, sscr_buf, sscr_b,
    H: tl.constexpr,
    LOG_H: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Persistent forward over the butterfly recurrence.

    Each program holds [BLOCK_B, H] state across T timesteps. Within a
    timestep, three gate-specific butterfly multiplies run in sequence
    on a per-program scratch buffer, then the gate compose / recurrence
    update produces h_t. No cross-CTA sync — butterfly is per-row
    independent so each batch tile can run in isolation.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_h = tl.arange(0, H)

    # Pre-load bias per gate.
    bhr = tl.load(bh_ptr + 0 * H + offs_h)
    bhz = tl.load(bh_ptr + 1 * H + offs_h)
    bhn = tl.load(bh_ptr + 2 * H + offs_h)

    # This program's scratch slab.
    scr_base = scratch_ptr + pid_b * sscr_pid

    # Pointer to the current "h_in" — starts at h0, then walks out[t-1].
    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(T):
        # Stage 0 of butterfly: copy h_in into all three gate scratch
        # buffers (we'll mutate them in place across stages).
        h_self = tl.load(
            h_in_ptr + offs_b[:, None] * sh_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        )
        for g in range(3):
            scr_g = scr_base + g * sscr_buf
            tl.store(
                scr_g + offs_b[:, None] * sscr_b + offs_h[None, :],
                h_self,
                mask=mask_b[:, None],
            )

        # Run log_H butterfly stages on each gate's scratch buffer.
        for s in range(LOG_H):
            stride_s = 1 << s
            partner = offs_h ^ stride_s
            # member ∈ {0, 1}: which side of the pair this position is.
            member = (offs_h >> s) & 1
            # pair index in the [n//2] flat layout — matches torch_structured's
            # twiddle.view(n//(2*stride), stride, 2, 2) layout (block * stride + k).
            pair_idx = (offs_h >> (s + 1)) * stride_s + (offs_h & (stride_s - 1))

            for g in range(3):
                scr_g = scr_base + g * sscr_buf
                # Load self and partner at the current state.
                a = tl.load(
                    scr_g + offs_b[:, None] * sscr_b + offs_h[None, :],
                    mask=mask_b[:, None], other=0.0,
                )
                b = tl.load(
                    scr_g + offs_b[:, None] * sscr_b + partner[None, :],
                    mask=mask_b[:, None], other=0.0,
                )
                # Twiddle entries for this position:
                #   t_self_self = t[s, pair_idx, member, member]
                #   t_self_partner = t[s, pair_idx, member, 1 - member]
                t_offset = (
                    g * st_g
                    + s * st_s
                    + pair_idx * st_p
                    + member * st_m_new
                )
                t_ss = tl.load(twiddle_ptr + t_offset + member * st_m_old)
                t_sp = tl.load(twiddle_ptr + t_offset + (1 - member) * st_m_old)
                new_val = t_ss[None, :] * a + t_sp[None, :] * b
                # Scatter back. We're overwriting `a` (the same offset we
                # just read from), but `b` (partner) is also being written
                # by another half of the threads in parallel; the read-then-
                # -write ordering is safe because we read `b` before any
                # write happens (Triton sequentializes within the program).
                tl.store(
                    scr_g + offs_b[:, None] * sscr_b + offs_h[None, :],
                    new_val,
                    mask=mask_b[:, None],
                )

        # After log_H stages, scratch[g] = butterfly_g(h). Now run the
        # gate compose + recurrence to produce h_new.
        scr_r = scr_base + 0 * sscr_buf
        scr_z = scr_base + 1 * sscr_buf
        scr_n = scr_base + 2 * sscr_buf
        gh_r = tl.load(
            scr_r + offs_b[:, None] * sscr_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhr[None, :]
        gh_z = tl.load(
            scr_z + offs_b[:, None] * sscr_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhz[None, :]
        gh_n = tl.load(
            scr_n + offs_b[:, None] * sscr_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhn[None, :]

        gi_base = (
            gi_ptr + t * sg_t + offs_b[:, None] * sg_b + offs_h[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_b[:, None], other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_b[:, None], other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_b[:, None], other=0.0)

        r = tl.sigmoid(gir + gh_r)
        z = tl.sigmoid(giz + gh_z)
        n = tl.extra.libdevice.tanh(gin + r * gh_n)
        h_new = (1.0 - z) * n + z * h_self

        out_ptrs = (
            out_ptr + t * so_t + offs_b[:, None] * so_b + offs_h[None, :]
        )
        tl.store(out_ptrs, h_new, mask=mask_b[:, None])

        # Next step reads from out[t] for h_in.
        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def gru_scan_butterfly_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    twiddles: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 8,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Multi-step persistent Triton butterfly forward.

    Args:
        gi:       [T, B, 3H] — pre-batched input projection (with bias).
        h0:       [B, H]
        twiddles: [3, log_H, H//2, 2, 2] — three gates' butterfly twiddles
            stacked along dim 0. Each shaped like a single Butterfly's
            twiddle (with nstacks=1, nblocks=1 squeezed out).
        bh_cat:   [3H]
    Returns:
        out: [T, B, H]
    """
    assert gi.is_cuda
    T, B, three_H = gi.shape
    H = three_H // 3
    assert h0.shape == (B, H)
    assert (H & (H - 1)) == 0, "butterfly requires H to be a power of 2"
    log_H = int(math.log2(H))
    n_gates, log_n_t, n_div_2_t, two1, two2 = twiddles.shape
    assert n_gates == 3 and log_n_t == log_H and n_div_2_t == H // 2
    assert two1 == 2 and two2 == 2

    gi = gi.contiguous()
    h0 = h0.contiguous()
    twiddles = twiddles.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)

    n_pid_b = triton.cdiv(B, block_b)
    # Scratch: 4 buffers per program (3 gates + 1 unused/aligned), each
    # [BLOCK_B, H]. We allocate 4 to keep stride math simple; the 4th is
    # unused at the moment but reserved for the backward kernel.
    scratch = torch.empty(
        (n_pid_b, 4, block_b, H), device=gi.device, dtype=gi.dtype,
    )

    grid = (n_pid_b,)
    gru_scan_butterfly_fwd_kernel[grid](
        gi, h0, twiddles, bh_cat, out, scratch,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        twiddles.stride(0), twiddles.stride(1), twiddles.stride(2),
        twiddles.stride(3), twiddles.stride(4),
        out.stride(0), out.stride(1),
        scratch.stride(0), scratch.stride(1), scratch.stride(2),
        H=H,
        LOG_H=log_H,
        BLOCK_B=block_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def extract_butterfly_twiddles(
    cell: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull the three hidden-side Butterfly twiddles into a single tensor.

    Returns:
        twiddles: [3, log_H, H//2, 2, 2] — gates stacked. Each gate's
            twiddle has nstacks=1, nblocks=1 squeezed out. The triton
            kernel works in this flat layout.
        bh_cat:   [3H]
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # cell.struct_Wh_*.b.twiddle: [nstacks=1, nblocks=1, log_n, n//2, 2, 2]
    Wr = cell.struct_Wh_r.b.twiddle.squeeze(0).squeeze(0)
    Wz = cell.struct_Wh_z.b.twiddle.squeeze(0).squeeze(0)
    Wn = cell.struct_Wh_n.b.twiddle.squeeze(0).squeeze(0)
    twiddles = torch.stack([Wr, Wz, Wn], dim=0)  # [3, log_n, n//2, 2, 2]
    if cell.b_hr is None:
        bh_cat = torch.zeros(
            3 * cell.hidden_size, device=twiddles.device, dtype=twiddles.dtype,
        )
    else:
        bh_cat = torch.cat([cell.b_hr, cell.b_hz, cell.b_hn])
    return twiddles, bh_cat
