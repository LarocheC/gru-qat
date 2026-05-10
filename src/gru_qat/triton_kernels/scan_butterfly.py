"""Butterfly hidden weights for the multi-step GRU scan.

API-parity counterpart to ``scan_monarch``: same shape of public
function (``gru_scan_butterfly`` ↔ ``gru_scan_monarch``), but the
implementation is a Python time loop calling ``torch_structured``'s
``butterfly_multiply`` CUDA op per step. There's no multi-step Triton
kernel here — Butterfly's compute pattern (log_N stages of strided 2×2
mixing) doesn't decompose into clean GEMM tiles the way Monarch's
block-diagonal does, and the existing torch_structured CUDA kernel for
the matmul itself is already hand-tuned.

What you get:
- Unified API: ``GRULayer(use_triton="auto", structure_hidden=ButterflyCfg)``
  routes through here. Same calibration + freeze + forward flow as
  Monarch.
- Backward via standard PyTorch autograd through butterfly_multiply
  (which has its own backward registered as a custom op).
- In-kernel-style fake-quant on hidden state via fake_quant_ste — the
  STE backward is exactly the same primitive used elsewhere in this
  package, so gradients flow correctly under QAT.

What you do NOT get vs. Monarch:
- No multi-step kernel fusion (per-step Python overhead remains).
- No persistent grid; each butterfly_multiply launches as its own CUDA
  op on each timestep.

Net: at our shapes this is ~as fast as the per-step structured path
plus compile_step. Use it because it's the same end-to-end pipeline
as Monarch, not because the recurrence got faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
