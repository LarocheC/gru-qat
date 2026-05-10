"""Monarch (block-diagonal) hidden weights for the multi-step GRU scan.

Tier-2 work: structured-hidden-side variant of ``gru_scan_persistent``.
The hidden weight ``Wh`` is parameterized as three Monarch factors (one
per gate), each ``[nblocks, blksz, blksz]`` with ``blksz = H/nblocks``.
The per-step matmul becomes ``nblocks`` independent ``[B, blksz] x
[blksz, blksz]`` block matmuls — same total FLOPs in the input-bound
regime, but ``nblocks``× smaller K-reduction per output block, ``nblocks``×
smaller per-block working set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


def extract_monarch_factors(cell: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull the three hidden-side Monarch weights out of a tier-1 cell.

    Args:
        cell: a ``GRUCellQuant`` whose ``structure_hidden`` was a Monarch
              ``StructureConfig``. Must have ``struct_Wh_r``, ``struct_Wh_z``,
              and ``struct_Wh_n`` BlockdiagLinear modules.

    Returns:
        Wh_struct: [3, nblocks, blksz, blksz] — gates stacked in (r, z, n)
            order, each layer's underlying ``[nblocks, out_blksz, in_blksz]``
            weight tensor.
        bh_cat: [3*H] — concat of (b_hr, b_hz, b_hn).
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # All three layers share the same shape (square BlockdiagLinear with
    # in_features == out_features == H).
    # struct_Wh_* are BlockdiagLinear instances; their `.weight` is the
    # [nblocks, out_blksz, in_blksz] factor tensor.
    Wr = cell.struct_Wh_r.weight
    Wz = cell.struct_Wh_z.weight
    Wn = cell.struct_Wh_n.weight
    Wh_struct = torch.stack([Wr, Wz, Wn], dim=0)  # [3, nblocks, blksz, blksz]
    if cell.b_hr is None:
        bh_cat = torch.zeros(3 * cell.hidden_size, device=Wh_struct.device, dtype=Wh_struct.dtype)
    else:
        bh_cat = torch.cat([cell.b_hr, cell.b_hz, cell.b_hn])
    return Wh_struct, bh_cat


def gru_scan_monarch_forward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
) -> torch.Tensor:
    """Reference forward for the block-diagonal scan, in PyTorch.

    Args:
        gi: [T, B, 3H] — pre-batched input projection (already with bi).
        h0: [B, H]
        Wh_struct: [3, nblocks, blksz, blksz]
        bh_cat: [3H]
    Returns:
        out: [T, B, H] — hidden state at every timestep.
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    assert n_gates == 3
    assert out_blksz == in_blksz, "square Monarch only"
    assert nblocks * out_blksz == H, f"nblocks*blksz={nblocks*out_blksz} != H={H}"

    blksz = out_blksz
    out = torch.empty(T, B, H, device=gi.device, dtype=gi.dtype)
    h = h0
    bh = bh_cat.view(3, H)

    for t in range(T):
        # h: [B, H] -> [B, nblocks, blksz]
        h_chunks = h.view(B, nblocks, blksz)
        # Block-diagonal matmul per gate:
        #   gh[g, b, n, o] = sum_i h_chunks[b, n, i] * Wh_struct[g, n, o, i]
        gh = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)  # [B, 3, nblocks, blksz]
        gh = gh.reshape(B, 3, H) + bh  # add bias per gate
        gh_r, gh_z, gh_n = gh[:, 0, :], gh[:, 1, :], gh[:, 2, :]

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_new = (1.0 - z) * n + z * h
        out[t] = h_new
        h = h_new

    return out


@triton.jit
def gru_scan_monarch_fwd_kernel(
    gi_ptr,            # [T, B, 3H], fp32
    h0_ptr,            # [B, H], fp32
    Wh_ptr,            # [3, nblocks, blksz, blksz], fp32
    bh_ptr,            # [3H], fp32
    out_ptr,           # [T, B, H], fp32
    barrier_ptr,       # [T], int32
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    sW_g, sW_n, sW_o,
    so_t, so_b,
    NUM_PROGRAMS,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent forward over the block-diagonal recurrence.

    Grid (pid_b, pid_block): each program handles ALL 3 gates for ONE
    block, producing [BLOCK_B, blksz] of h_t for that block. Block
    boundaries don't mix in the matmul (block-diagonal), so the K
    reduction is only over blksz instead of full H — that's where the
    win comes from.
    """
    pid_b = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = tl.arange(0, BLKSZ)  # output rows within this block

    # Output position in flat 3H layout: gate * H + pid_block * BLKSZ + offs_oh.
    # h-input range: pid_block * BLKSZ + offs_k.

    # Pre-load the bias slice for this (gate, block) — 3 chunks of BLKSZ.
    bh_offset = pid_block * BLKSZ
    bhr_tile = tl.load(bh_ptr + 0 * H + bh_offset + offs_oh)
    bhz_tile = tl.load(bh_ptr + 1 * H + bh_offset + offs_oh)
    bhn_tile = tl.load(bh_ptr + 2 * H + bh_offset + offs_oh)

    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(0, T):
        ghr = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)

        for k in range(0, BLKSZ, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < BLKSZ

            # h_block tile: [BLOCK_B, BLOCK_K] read from the current h_in
            # at the input slice of pid_block.
            h_ptrs = (
                h_in_ptr
                + offs_b[:, None] * sh_b
                + (pid_block * BLKSZ + offs_k)[None, :]
            )
            h_tile = tl.load(
                h_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )

            # Three W tiles, one per gate. Each is [BLKSZ, BLOCK_K].
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )

            ghr += tl.dot(h_tile, tl.trans(Wr_tile), input_precision="tf32")
            ghz += tl.dot(h_tile, tl.trans(Wz_tile), input_precision="tf32")
            ghn += tl.dot(h_tile, tl.trans(Wn_tile), input_precision="tf32")

        ghr += bhr_tile[None, :]
        ghz += bhz_tile[None, :]
        ghn += bhn_tile[None, :]

        # gi[t] tile for this block, three gate slices.
        gi_base = (
            gi_ptr
            + t * sg_t
            + offs_b[:, None] * sg_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_b[:, None], other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_b[:, None], other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_b[:, None], other=0.0)

        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.libdevice.tanh(gin + r * ghn)

        # h_t = (1-z)*n + z*h_prev at THIS block's output positions.
        h_old_ptrs = (
            h_in_ptr
            + offs_b[:, None] * sh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        h_old = tl.load(h_old_ptrs, mask=mask_b[:, None], other=0.0)
        h_new = (1.0 - z) * n + z * h_old

        out_ptrs = (
            out_ptr
            + t * so_t
            + offs_b[:, None] * so_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        tl.store(out_ptrs, h_new, mask=mask_b[:, None])

        # Cross-CTA barrier: pair release/acquire same as dense persistent.
        tl.atomic_add(barrier_ptr + t, 1, sem="release")
        done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")
        while done < NUM_PROGRAMS:
            done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")

        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def gru_scan_monarch_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 16,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Triton forward for the Monarch hidden-side scan."""
    assert gi.is_cuda and Wh_struct.is_cuda
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    assert n_gates == 3
    assert out_blksz == in_blksz == H // nblocks

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_struct = Wh_struct.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)
    barrier = torch.zeros((T,), device=gi.device, dtype=torch.int32)

    n_pid_b = triton.cdiv(B, block_b)
    num_programs = n_pid_b * nblocks

    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    if num_programs > sm_count:
        raise RuntimeError(
            f"persistent grid {num_programs} > SM count {sm_count}; "
            f"would deadlock on the spin-wait barrier."
        )

    grid = (n_pid_b, nblocks)
    gru_scan_monarch_fwd_kernel[grid](
        gi, h0, Wh_struct, bh_cat, out,
        barrier,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_struct.stride(0), Wh_struct.stride(1), Wh_struct.stride(2),
        out.stride(0), out.stride(1),
        num_programs,
        H=H,
        BLKSZ=out_blksz,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def gru_scan_monarch_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference backward.

    Returns:
        dgi:        [T, B, 3H]
        dh0:        [B, H]
        dWh_struct: [3, nblocks, blksz, blksz]
        dbh_cat:    [3H]
    """
    T, B, _ = gi.shape
    H = h0.shape[-1]
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    blksz = out_blksz

    dgi = torch.zeros_like(gi)
    dWh_struct = torch.zeros_like(Wh_struct)
    dbh = torch.zeros_like(bh_cat)
    dh_acc = torch.zeros_like(h0)

    for t in reversed(range(T)):
        h_prev = h0 if t == 0 else out[t - 1]

        # Forward recompute
        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]
        h_chunks = h_prev.view(B, nblocks, blksz)
        gh = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)
        gh = gh.reshape(B, 3, H) + bh_cat.view(3, H)
        gh_r = gh[:, 0, :]
        gh_z = gh[:, 1, :]
        gh_n = gh[:, 2, :]
        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)

        dh_t = dout[t] + dh_acc

        # h_t = (1-z)*n + z*h_prev
        dn = dh_t * (1.0 - z)
        dz = dh_t * (h_prev - n)
        dh_prev_direct = dh_t * z

        # n = tanh(gn_pre); gn_pre = gi_n + r*gh_n
        dgn_pre = dn * (1.0 - n * n)
        dgi_n = dgn_pre
        dr = dgn_pre * gh_n
        dgh_n = dgn_pre * r

        # z = sigmoid(gi_z + gh_z)
        dgz_pre = dz * z * (1.0 - z)
        dgi_z = dgz_pre
        dgh_z = dgz_pre

        # r = sigmoid(gi_r + gh_r)
        dgr_pre = dr * r * (1.0 - r)
        dgi_r = dgr_pre
        dgh_r = dgr_pre

        dgi[t] = torch.cat([dgi_r, dgi_z, dgi_n], dim=-1)

        # Stack dgh per gate into [B, 3, H] -> reshape to [B, 3, nblocks, blksz]
        dgh = torch.stack([dgh_r, dgh_z, dgh_n], dim=1)  # [B, 3, H]
        dbh += dgh.sum(dim=0).reshape(-1)  # accumulate over batch and time
        dgh_chunks = dgh.view(B, 3, nblocks, blksz)

        # gh = einsum('bni,gnoi->bgno', h_chunks, Wh_struct)
        # Backward:
        #   dWh_struct[g, n, o, i] += sum_b dgh[b, g, n, o] * h_chunks[b, n, i]
        #   dh_chunks[b, n, i] += sum_{g,o} dgh[b, g, n, o] * Wh_struct[g, n, o, i]
        dWh_struct += torch.einsum("bgno,bni->gnoi", dgh_chunks, h_chunks)
        dh_via_W_chunks = torch.einsum(
            "bgno,gnoi->bni", dgh_chunks, Wh_struct
        )
        dh_via_W = dh_via_W_chunks.reshape(B, H)

        dh_acc = dh_prev_direct + dh_via_W

    return dgi, dh_acc, dWh_struct, dbh
