"""Genuine two-factor Monarch hidden weights for the multi-step GRU scan.

Distinct from ``scan_blockdiag`` (a single block-diagonal factor). Here each
gate's hidden weight is the two-factor Monarch construction
(``torch_structured.monarch.MonarchLinear``): a block-diagonal factor ``w1``,
a transpose-permutation, then a second block-diagonal factor ``w2``. Unlike
the single block-diagonal factor, this mixes information across all blocks by
construction (full dense-equivalent rank), which is why it needs both a
first-factor and a second-factor matmul with a reshuffle between them.

Math (square hidden, ``H`` channels, ``nblocks`` blocks, ``blksz = H/nblocks``):

    X[k, p]      = h[k * blksz + p]                       # reshape to blocks
    out1[k, q]   = sum_p w1[k, q, p] * X[k, p]            # factor 1 (block-diag)
    mid[l, r]    = out1_flat[r * nblocks + l]             # transpose-permute
    gh[s, l]     = sum_r w2[l, s, r] * mid[l, r]          # factor 2 (block-diag)
    y[s * nblocks + l] = gh[s, l]                         # natural H layout

The permutation ``mid[l, r] = out1_flat[r*nblocks + l]`` is the crux: block
``l`` of the second factor reads one element from each first-factor block, so
the two factors cannot both be block-local. The kernel handles it with a
per-program scratch round-trip (mirrors ``scan_butterfly``'s inter-stage
scratch): factor 1 writes ``out1`` contiguously, an intra-CTA barrier, then
factor 2 loads its inputs with a strided (permuted) gather.

Grid is ``(n_pid_b,)`` — one program per batch tile owns ALL H channels for
its rows, so the recurrence needs no cross-CTA barrier (each program feeds its
own ``h_t`` back into ``h_{t+1}``); only an intra-CTA ``tl.debug_barrier()``
separates the two factors, whose scratch traffic crosses warp boundaries.

Backward is a hand-derived reverse-time Triton kernel (same grid /
scratch-round-trip structure as the forward): per step it recomputes the
forward, runs the gate backward, factor-2 backward (``dmid`` / ``dW2``, with
the inverse-permute), then factor-1 backward (``dX`` / ``dW1``), folding the
matmul term into the carry. Weight gradients accumulate via ``atomic_add``
(uncontended when ``B <= BLOCK_B``). ``gru_scan_monarch_backward_pytorch``
(autograd through the forward reference) is the parity ground truth.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import triton
import triton.language as tl

from gru_qat.ste import fake_quant_ste


def extract_monarch_factors(
    cell: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull the three hidden-side two-factor Monarch weights out of a cell.

    Args:
        cell: a ``GRUCellQuant`` whose ``structure_hidden`` was a Monarch
              ``StructureConfig``. Must have ``struct_Wh_r/z/n`` MonarchLinear
              modules, each exposing ``w1``/``w2`` factor tensors.

    Returns:
        W1: [3, nblocks, blksz, blksz] — first factors, gates in (r, z, n).
        W2: [3, nblocks, blksz, blksz] — second factors, gates in (r, z, n).
        bh_cat: [3*H] — concat of (b_hr, b_hz, b_hn).
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")

    def _w(name: str, attr: str) -> torch.Tensor:
        mod = cast(nn.Module, getattr(cell, name))
        return cast(torch.Tensor, getattr(mod, attr))

    W1 = torch.stack(
        [_w("struct_Wh_r", "w1"), _w("struct_Wh_z", "w1"), _w("struct_Wh_n", "w1")],
        dim=0,
    )
    W2 = torch.stack(
        [_w("struct_Wh_r", "w2"), _w("struct_Wh_z", "w2"), _w("struct_Wh_n", "w2")],
        dim=0,
    )
    if cell.b_hr is None:
        bh_cat = torch.zeros(3 * cell.hidden_size, device=W1.device, dtype=W1.dtype)
    else:
        bh_cat = torch.cat(
            [
                cast(torch.Tensor, cell.b_hr),
                cast(torch.Tensor, cell.b_hz),
                cast(torch.Tensor, cell.b_hn),
            ]
        )
    return W1, W2, bh_cat


def _permute_index(nblocks: int, blksz: int, device: torch.device) -> torch.Tensor:
    """Gather index realizing mid[l, r] = out1_flat[r*nblocks + l].

    Returned as a flat [nblocks*blksz] index into out1_flat, laid out
    row-major over (l, r) so ``out1_flat[idx].view(nblocks, blksz) == mid``.
    """
    lb = torch.arange(nblocks, device=device).view(nblocks, 1)
    rb = torch.arange(blksz, device=device).view(1, blksz)
    return (rb * nblocks + lb).reshape(-1)


def _ste(x: torch.Tensor, params: tuple[float, int, int] | None) -> torch.Tensor:
    """Differentiable per-tensor symmetric fake-quant (STE). None passes through."""
    if params is None:
        return x
    scale, qmin, qmax = params
    return fake_quant_ste(
        x,
        torch.tensor(scale, device=x.device, dtype=x.dtype),
        torch.zeros((), device=x.device, dtype=x.dtype),
        float(qmin),
        float(qmax),
    )


def gru_scan_monarch_forward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Reference forward for the two-factor Monarch scan, in PyTorch.

    Differentiable (STE fake-quant), so it doubles as the backward reference.

    Args:
        gi: [T, B, 3H] — pre-batched input projection (already with bi).
        h0: [B, H]
        W1, W2: [3, nblocks, blksz, blksz] — per-gate Monarch factors.
        bh_cat: [3H]
        h_in_quant: optional (scale, qmin, qmax) for the matmul-side h.
        h_out_quant: optional (scale, qmin, qmax) for h_new before store.
    Returns:
        out: [T, B, H] — hidden state at every timestep.
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, blksz, in_blksz = W1.shape
    assert n_gates == 3
    assert blksz == in_blksz, "square Monarch only"
    assert nblocks * blksz == H, f"nblocks*blksz={nblocks * blksz} != H={H}"

    perm = _permute_index(nblocks, blksz, gi.device)
    bh = bh_cat.view(3, H)

    def monarch2(h_mm: torch.Tensor, g: int) -> torch.Tensor:
        X = h_mm.view(B, nblocks, blksz)
        out1 = torch.einsum("kqp,bkp->bkq", W1[g], X).reshape(B, nblocks * blksz)
        mid = out1[:, perm].view(B, nblocks, blksz)
        out2 = torch.einsum("lsr,blr->bsl", W2[g], mid)  # [B, s, l]
        return out2.reshape(B, blksz * nblocks)  # flat s*nblocks + l

    h = h0
    outs: list[torch.Tensor] = []
    for t in range(T):
        h_mm = _ste(h, h_in_quant)
        gh_r = monarch2(h_mm, 0) + bh[0]
        gh_z = monarch2(h_mm, 1) + bh[1]
        gh_n = monarch2(h_mm, 2) + bh[2]

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H : 2 * H]
        gi_n = gi[t, :, 2 * H : 3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_new = (1.0 - z) * n + z * h
        h_new = _ste(h_new, h_out_quant)
        outs.append(h_new)
        h = h_new

    return torch.stack(outs, dim=0)


@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_monarch_fwd_kernel(  # type: ignore[no-untyped-def]
    gi_ptr,            # [T, B, 3H], fp32
    h0_ptr,            # [B, H], fp32
    W1_ptr,            # [3, nblocks, blksz, blksz], fp32
    W2_ptr,            # [3, nblocks, blksz, blksz], fp32
    bh_ptr,            # [3H], fp32
    scratch_ptr,       # [3, B, H], fp32 (per-program rows; factor-1 output)
    out_ptr,           # [T, B, H], fp32
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    sW_g, sW_n, sW_o,   # W1/W2 strides (identical layout): gate, block, out-row
    ss_g, ss_b,         # scratch strides: gate, batch
    so_t, so_b,
    h_in_scale, h_in_qmin, h_in_qmax,
    h_out_scale, h_out_qmin, h_out_qmax,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,   # power of 2, >= 16
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Forward over the two-factor Monarch recurrence.

    Grid ``(pid_b,)``: each program owns ``BLOCK_B`` batch rows across ALL H
    channels. Per timestep it runs factor 1 for all three gates into a scratch
    buffer, a CTA barrier, then factor 2 + the gate recurrence per output
    block. No cross-CTA barrier is needed because a program's rows never touch
    another program's rows.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    ar = tl.arange(0, BLKSZ)  # within-block index (p / q / r / s), BLKSZ is pow2

    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(0, T):
        # ---- factor 1: out1[g, :, k, q] -> scratch[g] at flat k*BLKSZ + q ----
        for k in range(0, NBLOCKS):
            cols = k * BLKSZ + ar  # p positions in h (contiguous block k)
            h_ptrs = h_in_ptr + offs_b[:, None] * sh_b + cols[None, :]
            hk = tl.load(h_ptrs, mask=mask_b[:, None], other=0.0)
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(hk / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                hk = q * h_in_scale
            for g in range(0, 3):
                w1 = tl.load(
                    W1_ptr + g * sW_g + k * sW_n + ar[:, None] * sW_o + ar[None, :]
                )  # [q, p]
                o1 = tl.dot(hk, tl.trans(w1), input_precision="tf32")  # [BB, q]
                s_ptrs = (
                    scratch_ptr + g * ss_g + offs_b[:, None] * ss_b + cols[None, :]
                )
                tl.store(s_ptrs, o1, mask=mask_b[:, None])

        tl.debug_barrier()

        # ---- factor 2 + gate recurrence, per output block lb ----
        for lb in range(0, NBLOCKS):
            chan = lb + NBLOCKS * ar  # output channels o = s*NBLOCKS + lb; mid r positions

            # gate r/z/n second factor: gh[:, s] = mid[:, r] @ w2[lb].T
            mid_r = tl.load(
                scratch_ptr + 0 * ss_g + offs_b[:, None] * ss_b + chan[None, :],
                mask=mask_b[:, None], other=0.0,
            )
            mid_z = tl.load(
                scratch_ptr + 1 * ss_g + offs_b[:, None] * ss_b + chan[None, :],
                mask=mask_b[:, None], other=0.0,
            )
            mid_n = tl.load(
                scratch_ptr + 2 * ss_g + offs_b[:, None] * ss_b + chan[None, :],
                mask=mask_b[:, None], other=0.0,
            )
            w2r = tl.load(W2_ptr + 0 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :])
            w2z = tl.load(W2_ptr + 1 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :])
            w2n = tl.load(W2_ptr + 2 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :])
            ghr = tl.dot(mid_r, tl.trans(w2r), input_precision="tf32")
            ghz = tl.dot(mid_z, tl.trans(w2z), input_precision="tf32")
            ghn = tl.dot(mid_n, tl.trans(w2n), input_precision="tf32")

            bhr = tl.load(bh_ptr + 0 * H + chan)
            bhz = tl.load(bh_ptr + 1 * H + chan)
            bhn = tl.load(bh_ptr + 2 * H + chan)
            ghr += bhr[None, :]
            ghz += bhz[None, :]
            ghn += bhn[None, :]

            gi_base = gi_ptr + t * sg_t + offs_b[:, None] * sg_b + chan[None, :]
            gir = tl.load(gi_base + 0 * H, mask=mask_b[:, None], other=0.0)
            giz = tl.load(gi_base + 1 * H, mask=mask_b[:, None], other=0.0)
            gin = tl.load(gi_base + 2 * H, mask=mask_b[:, None], other=0.0)

            r = tl.sigmoid(gir + ghr)
            z = tl.sigmoid(giz + ghz)
            n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

            h_old = tl.load(
                h_in_ptr + offs_b[:, None] * sh_b + chan[None, :],
                mask=mask_b[:, None], other=0.0,
            )
            h_new = (1.0 - z) * n + z * h_old
            if QUANT_H_OUT:
                q = tl.extra.cuda.libdevice.rint(h_new / h_out_scale)
                q = tl.minimum(tl.maximum(q, h_out_qmin), h_out_qmax)
                h_new = q * h_out_scale

            out_ptrs = out_ptr + t * so_t + offs_b[:, None] * so_b + chan[None, :]
            tl.store(out_ptrs, h_new, mask=mask_b[:, None])

        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def gru_scan_monarch_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Triton forward for the two-factor Monarch hidden-side scan."""
    if not (gi.is_cuda and W1.is_cuda and W2.is_cuda):
        raise ValueError("gi, W1, W2 must be CUDA tensors")
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, blksz, in_blksz = W1.shape
    if n_gates != 3 or W2.shape[0] != 3:
        raise ValueError(f"W1/W2 dim 0 (n_gates) must be 3; got {n_gates}, {W2.shape[0]}")
    if not (blksz == in_blksz and nblocks * blksz == H):
        raise ValueError(
            f"Monarch factors must be square and tile H: got nblocks={nblocks}, "
            f"blksz={blksz}x{in_blksz}, H={H}"
        )
    if tuple(W2.shape) != (3, nblocks, blksz, blksz):
        raise ValueError(f"W2 shape {tuple(W2.shape)} != W1 shape {tuple(W1.shape)}")
    if blksz < 16 or (blksz & (blksz - 1)) != 0:
        raise ValueError(
            f"Triton Monarch kernel requires blksz (=H/nblocks) a power of 2 >= 16; "
            f"got blksz={blksz} (H={H}, nblocks={nblocks}). Use the reference path."
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    W1 = W1.contiguous()
    W2 = W2.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)
    scratch = torch.empty((3, B, H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -(2**31), 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -(2**31), 2**31 - 1)

    grid = (triton.cdiv(B, block_b),)
    gru_scan_monarch_fwd_kernel[grid](
        gi, h0, W1, W2, bh_cat, scratch, out,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        W1.stride(0), W1.stride(1), W1.stride(2),
        scratch.stride(0), scratch.stride(1),
        out.stride(0), out.stride(1),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLKSZ=blksz,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def gru_scan_monarch_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    bh_cat: torch.Tensor,
    dout: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference backward: autograd through the differentiable forward.

    Returns ``(dgi, dh0, dW1, dW2, dbh)``. Ground truth for the Triton
    backward kernel.
    """
    leaves = [t.detach().requires_grad_(True) for t in (gi, h0, W1, W2, bh_cat)]
    out = gru_scan_monarch_forward_pytorch(
        *leaves, h_in_quant=h_in_quant, h_out_quant=h_out_quant
    )
    grads = torch.autograd.grad(out, leaves, grad_outputs=dout, allow_unused=True)
    return tuple(  # type: ignore[return-value]
        g if g is not None else torch.zeros_like(t) for g, t in zip(grads, leaves)
    )


@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_monarch_bwd_kernel(  # type: ignore[no-untyped-def]
    gi_ptr,            # [T, B, 3H]
    hprev_ptr,         # [T, B, H]   h_prev[t] = h_{t-1}  (= cat([h0], out[:-1]))
    W1_ptr,            # [3, nblocks, blksz, blksz]
    W2_ptr,            # [3, nblocks, blksz, blksz]
    bh_ptr,            # [3H]
    dout_ptr,          # [T, B, H]   upstream grad
    dgi_ptr,           # [T, B, 3H]  (out)
    dh0_ptr,           # [B, H]      (out)
    dW1_ptr,           # [3, nblocks, blksz, blksz]  (out, zero-init, atomic)
    dW2_ptr,           # [3, nblocks, blksz, blksz]  (out, zero-init, atomic)
    dbh_ptr,           # [3H]        (out, zero-init, atomic)
    carry_ptr,         # [B, H]      scratch (zero-init)
    out1_ptr,          # [3, B, H]   scratch (factor-1 recompute)
    dout1_ptr,         # [3, B, H]   scratch (inverse-permuted dmid)
    T,
    B,
    sg_t, sg_b,
    shp_t, shp_b,
    sW_g, sW_n, sW_o,
    sd_t, sd_b,          # dout strides
    sdgi_t, sdgi_b,
    ss_g, ss_b,
    scarry_b,
    h_in_scale, h_in_qmin, h_in_qmax,
    h_out_scale, h_out_qmin, h_out_qmax,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Reverse-time backward for the two-factor Monarch recurrence.

    Grid ``(pid_b,)``: each program owns ``BLOCK_B`` batch rows across all H
    and walks t from T-1 down to 0, carrying ``dh`` (grad w.r.t. the current
    hidden state) in the ``carry`` buffer. Per step it recomputes the forward
    (factor 1 -> scratch, factor 2), runs the gate backward, factor-2 backward
    (dmid + dW2) with the inverse-permute into ``dout1``, then factor-1
    backward (dX + dW1) folding the matmul term into the carry. Weight grads
    accumulate via ``atomic_add`` (uncontended when B <= BLOCK_B).
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    mb = mask_b[:, None]
    ar = tl.arange(0, BLKSZ)

    for t in range(T - 1, -1, -1):
        hprev_row = hprev_ptr + t * shp_t + offs_b[:, None] * shp_b

        # ---- recompute factor 1 -> out1 scratch (all 3 gates) ----
        for k in range(0, NBLOCKS):
            cols = k * BLKSZ + ar
            hk = tl.load(hprev_row + cols[None, :], mask=mb, other=0.0)
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(hk / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                hk = q * h_in_scale
            for g in range(0, 3):
                w1 = tl.load(W1_ptr + g * sW_g + k * sW_n + ar[:, None] * sW_o + ar[None, :])
                o1 = tl.dot(hk, tl.trans(w1), input_precision="tf32")
                tl.store(out1_ptr + g * ss_g + offs_b[:, None] * ss_b + cols[None, :], o1, mask=mb)
        tl.debug_barrier()

        # ---- factor 2 recompute + gate backward + factor 2 backward ----
        for lb in range(0, NBLOCKS):
            chan = lb + NBLOCKS * ar
            cptr = offs_b[:, None] * ss_b + chan[None, :]
            mid_r = tl.load(out1_ptr + 0 * ss_g + cptr, mask=mb, other=0.0)
            mid_z = tl.load(out1_ptr + 1 * ss_g + cptr, mask=mb, other=0.0)
            mid_n = tl.load(out1_ptr + 2 * ss_g + cptr, mask=mb, other=0.0)
            w2r = tl.load(W2_ptr + 0 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :])
            w2z = tl.load(W2_ptr + 1 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :])
            w2n = tl.load(W2_ptr + 2 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :])
            ghr = tl.dot(mid_r, tl.trans(w2r), input_precision="tf32") + tl.load(bh_ptr + 0 * H + chan)[None, :]
            ghz = tl.dot(mid_z, tl.trans(w2z), input_precision="tf32") + tl.load(bh_ptr + 1 * H + chan)[None, :]
            ghn = tl.dot(mid_n, tl.trans(w2n), input_precision="tf32") + tl.load(bh_ptr + 2 * H + chan)[None, :]

            gi_base = gi_ptr + t * sg_t + offs_b[:, None] * sg_b + chan[None, :]
            gir = tl.load(gi_base + 0 * H, mask=mb, other=0.0)
            giz = tl.load(gi_base + 1 * H, mask=mb, other=0.0)
            gin = tl.load(gi_base + 2 * H, mask=mb, other=0.0)
            r = tl.sigmoid(gir + ghr)
            z = tl.sigmoid(giz + ghz)
            n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)
            h_old = tl.load(hprev_row + chan[None, :], mask=mb, other=0.0)
            h_new_pre = (1.0 - z) * n + z * h_old

            g_ht = (
                tl.load(dout_ptr + t * sd_t + offs_b[:, None] * sd_b + chan[None, :], mask=mb, other=0.0)
                + tl.load(carry_ptr + offs_b[:, None] * scarry_b + chan[None, :], mask=mb, other=0.0)
            )
            if QUANT_H_OUT:
                qo = tl.extra.cuda.libdevice.rint(h_new_pre / h_out_scale)
                mask_out = (qo >= h_out_qmin) & (qo <= h_out_qmax)
                g_hp = g_ht * mask_out
            else:
                g_hp = g_ht

            dz = g_hp * (h_old - n)
            dn = g_hp * (1.0 - z)
            dh_old_direct = g_hp * z
            dan = dn * (1.0 - n * n)
            dgh_n = dan * r
            dr = dan * ghn
            dgi_n = dan
            dar = dr * r * (1.0 - r)
            dgh_r = dar
            dgi_r = dar
            daz = dz * z * (1.0 - z)
            dgh_z = daz
            dgi_z = daz

            dgi_base = dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + chan[None, :]
            tl.store(dgi_base + 0 * H, dgi_r, mask=mb)
            tl.store(dgi_base + 1 * H, dgi_z, mask=mb)
            tl.store(dgi_base + 2 * H, dgi_n, mask=mb)
            # z-part of dh_{t-1} (overwrite carry at these output channels)
            tl.store(carry_ptr + offs_b[:, None] * scarry_b + chan[None, :], dh_old_direct, mask=mb)

            # dbh: sum over batch rows (invalid rows contribute 0 via g_ht=0)
            tl.atomic_add(dbh_ptr + 0 * H + chan, tl.sum(dgh_r, axis=0))
            tl.atomic_add(dbh_ptr + 1 * H + chan, tl.sum(dgh_z, axis=0))
            tl.atomic_add(dbh_ptr + 2 * H + chan, tl.sum(dgh_n, axis=0))

            # factor 2 backward per gate: dmid + dW2
            dmid_r = tl.dot(dgh_r, w2r, input_precision="tf32")
            dmid_z = tl.dot(dgh_z, w2z, input_precision="tf32")
            dmid_n = tl.dot(dgh_n, w2n, input_precision="tf32")
            tl.atomic_add(
                dW2_ptr + 0 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :],
                tl.dot(tl.trans(dgh_r), mid_r, input_precision="tf32"),
            )
            tl.atomic_add(
                dW2_ptr + 1 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :],
                tl.dot(tl.trans(dgh_z), mid_z, input_precision="tf32"),
            )
            tl.atomic_add(
                dW2_ptr + 2 * sW_g + lb * sW_n + ar[:, None] * sW_o + ar[None, :],
                tl.dot(tl.trans(dgh_n), mid_n, input_precision="tf32"),
            )
            # inverse-permute: dout1_flat[r*NBLOCKS + lb] = dmid[lb, r] -> chan positions
            tl.store(dout1_ptr + 0 * ss_g + cptr, dmid_r, mask=mb)
            tl.store(dout1_ptr + 1 * ss_g + cptr, dmid_z, mask=mb)
            tl.store(dout1_ptr + 2 * ss_g + cptr, dmid_n, mask=mb)
        tl.debug_barrier()

        # ---- factor 1 backward per input block: dX + dW1, fold into carry ----
        for k in range(0, NBLOCKS):
            cols = k * BLKSZ + ar
            kptr = offs_b[:, None] * ss_b + cols[None, :]
            hk = tl.load(hprev_row + cols[None, :], mask=mb, other=0.0)
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(hk / h_in_scale)
                mask_in = (q >= h_in_qmin) & (q <= h_in_qmax)
                xk = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax) * h_in_scale
            else:
                xk = hk
            dhmm = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
            for g in range(0, 3):
                dout1_g = tl.load(dout1_ptr + g * ss_g + kptr, mask=mb, other=0.0)
                w1 = tl.load(W1_ptr + g * sW_g + k * sW_n + ar[:, None] * sW_o + ar[None, :])
                dhmm += tl.dot(dout1_g, w1, input_precision="tf32")
                tl.atomic_add(
                    dW1_ptr + g * sW_g + k * sW_n + ar[:, None] * sW_o + ar[None, :],
                    tl.dot(tl.trans(dout1_g), xk, input_precision="tf32"),
                )
            if QUANT_H_IN:
                dhmm = dhmm * mask_in
            cur = tl.load(carry_ptr + offs_b[:, None] * scarry_b + cols[None, :], mask=mb, other=0.0)
            tl.store(carry_ptr + offs_b[:, None] * scarry_b + cols[None, :], cur + dhmm, mask=mb)

    # dh0 = final carry
    for k in range(0, NBLOCKS):
        cols = k * BLKSZ + ar
        c = tl.load(carry_ptr + offs_b[:, None] * scarry_b + cols[None, :], mask=mb, other=0.0)
        tl.store(dh0_ptr + offs_b[:, None] * H + cols[None, :], c, mask=mb)


def gru_scan_monarch_backward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    block_b: int = 32,
    num_warps: int = 4,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton backward for the two-factor Monarch scan.

    Returns ``(dgi, dh0, dW1, dW2, dbh)``. ``out`` is the forward output
    (used to build ``h_prev[t] = h_{t-1}``).
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    _, nblocks, blksz, _ = W1.shape

    gi = gi.contiguous()
    dout = dout.contiguous()
    W1 = W1.contiguous()
    W2 = W2.contiguous()
    bh_cat = bh_cat.contiguous()
    # h_prev[t] = h_{t-1}: h0 for t=0, out[t-1] otherwise.
    h_prev = torch.cat([h0.unsqueeze(0), out[:-1]], dim=0).contiguous()

    dgi = torch.empty_like(gi)
    dh0 = torch.empty_like(h0)
    dW1 = torch.zeros_like(W1)
    dW2 = torch.zeros_like(W2)
    dbh = torch.zeros_like(bh_cat)
    carry = torch.zeros((B, H), device=gi.device, dtype=gi.dtype)
    out1 = torch.empty((3, B, H), device=gi.device, dtype=gi.dtype)
    dout1 = torch.empty((3, B, H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -(2**31), 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -(2**31), 2**31 - 1)

    grid = (triton.cdiv(B, block_b),)
    gru_scan_monarch_bwd_kernel[grid](
        gi, h_prev, W1, W2, bh_cat, dout,
        dgi, dh0, dW1, dW2, dbh,
        carry, out1, dout1,
        T, B,
        gi.stride(0), gi.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        W1.stride(0), W1.stride(1), W1.stride(2),
        dout.stride(0), dout.stride(1),
        dgi.stride(0), dgi.stride(1),
        out1.stride(0), out1.stride(1),
        carry.stride(0),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLKSZ=blksz,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=1,
    )
    return dgi, dh0, dW1, dW2, dbh


class GRUScanMonarchFunction(torch.autograd.Function):
    """Autograd bridge: fused Triton forward and hand-derived Triton backward."""

    @staticmethod
    def forward(
        ctx: Any,
        gi: torch.Tensor,
        h0: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        bh_cat: torch.Tensor,
        h_in_quant: tuple[float, int, int] | None,
        h_out_quant: tuple[float, int, int] | None,
    ) -> torch.Tensor:
        out = gru_scan_monarch_forward_triton(
            gi, h0, W1, W2, bh_cat,
            h_in_quant=h_in_quant, h_out_quant=h_out_quant,
        )
        ctx.save_for_backward(gi, h0, W1, W2, bh_cat, out)
        ctx.h_in_quant = h_in_quant
        ctx.h_out_quant = h_out_quant
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> Any:
        gi, h0, W1, W2, bh_cat, out = ctx.saved_tensors
        dgi, dh0, dW1, dW2, dbh = gru_scan_monarch_backward_triton(
            gi, h0, W1, W2, bh_cat, out, dout.contiguous(),
            h_in_quant=ctx.h_in_quant, h_out_quant=ctx.h_out_quant,
        )
        return dgi, dh0, dW1, dW2, dbh, None, None


def gru_scan_monarch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Public autograd wrapper for the two-factor Monarch scan.

    - ``gi: [T, B, 3H]`` pre-batched input projection.
    - ``W1, W2: [3, nblocks, blksz, blksz]`` per-gate Monarch factors.
    - ``bh_cat: [3H]`` hidden bias.
    Returns ``out: [T, B, H]``.
    """
    return cast(
        torch.Tensor,
        GRUScanMonarchFunction.apply(  # type: ignore[no-untyped-call]
            gi, h0, W1, W2, bh_cat, h_in_quant, h_out_quant
        ),
    )
