"""Monarch (block-diagonal) hidden weights for the multi-step GRU scan.

Tier-2 work: structured-hidden-side variant of ``gru_scan_persistent``.
The hidden weight ``Wh`` is parameterized as three Monarch factors (one
per gate), each ``[nblocks, blksz, blksz]`` with ``blksz = H/nblocks``.
The per-step matmul becomes ``nblocks`` independent ``[B, blksz] x
[blksz, blksz]`` block matmuls — same total FLOPs in the input-bound
regime, but ``nblocks``× smaller K-reduction per output block, ``nblocks``×
smaller per-block working set.

Stage A (this commit): factor extraction from a tier-1 cell, plus a
PyTorch reference forward/backward pair. The Triton kernels in
follow-up commits will validate against these references.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
