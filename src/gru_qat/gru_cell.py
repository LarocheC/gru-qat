"""Single-timestep GRU cell, manually unrolled.

This is the heart of the library. Every line that touches a quantizable
quantity is annotated. The math follows PyTorch's GRUCell exactly:

    r_t = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z_t = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n_t = tanh   (W_in x + b_in + r_t * (W_hn h + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h

Note `r_t` is applied *inside* the tanh argument and only multiplies the
*hidden* contribution to `n_t`, not the input contribution. This matches
`torch.nn.GRUCell`. CuDNN matches it. Keep it that way; many home-grown
implementations get this wrong and silently lose 1-2% accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from gru_qat.quantizers import (
    FakeQuantize,
    QuantRecipe,
    QuantizerConfig,
    factory,
    make_quantizer,
)

GateLayout = Literal["split", "fused"]


@dataclass
class CellWeights:
    """Bag of fake-quantized weights for one cell.

    Returned by `GRUCellQuant.quantize_weights()` so a multi-step layer can
    quantize once per forward and reuse the result across all timesteps.
    Weights don't change inside a forward pass; running `quant_W_*` per
    step costs 6 × seq_len pointless module calls.

    `Wi_cat`/`Wh_cat`/`bi_cat`/`bh_cat` are populated when the cell uses
    `gate_layout="fused"`; the per-step path then runs two large GEMMs
    instead of six small ones. Concat is along axis 0 (the per-channel
    axis), so per-channel/per-group weight quant scales survive intact.
    """

    Wir: torch.Tensor
    Wiz: torch.Tensor
    Win: torch.Tensor
    Whr: torch.Tensor
    Whz: torch.Tensor
    Whn: torch.Tensor
    Wi_cat: torch.Tensor | None = None
    Wh_cat: torch.Tensor | None = None
    bi_cat: torch.Tensor | None = None
    bh_cat: torch.Tensor | None = None


class GRUCellQuant(nn.Module):
    """GRU cell with pluggable fake-quant at every insertion point.

    Insertion points (each one is a `FakeQuantize` module, swappable):

    1. `quant_x`       — input activation x_t
    2. `quant_h_in`    — hidden state h_{t-1} on the read side
    3. `quant_W_ir/iz/in` — input-to-gate weights (3 separate quantizers
                            so per-tensor schemes work; per-channel could
                            share but doesn't gain anything)
    4. `quant_W_hr/hz/hn` — hidden-to-gate weights
    5. `quant_h_out`   — hidden state h_t on the write side. Often shares
                         config with `quant_h_in`; pass the same recipe.

    Bias is fp32. Sigmoid/tanh are fp32.

    Args:
        input_size, hidden_size: as in nn.GRUCell.
        recipe: QuantRecipe (see quantizers.PRESETS).
        gate_layout: "split" (default; matches insertion point design) or
            "fused" (Phase 5+; concatenates W_i* and shares quant_W_i).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        *,
        gate_layout: GateLayout = "split",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if gate_layout == "fused":
            # Fusing concatenates W_ir/W_iz/W_in along axis 0 (output rows).
            # Per-channel and per-group weight quant along axis 0 survive
            # because each row keeps its own scale. Per-tensor weight quant
            # would silently share one scale across all three gate matrices,
            # which is exactly the regime SCOPE.md §4 says to avoid.
            if recipe.weight.axis != 0:
                raise ValueError(
                    "fused gate layout requires recipe.weight.axis=0; "
                    f"got axis={recipe.weight.axis}. Per-tensor weight quant "
                    "is not supported with fused gates."
                )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_layout = gate_layout

        # ---- weights (split-gate layout) ----
        # Each weight is [hidden_size, *_size]; bias is [hidden_size].
        def _w(out_dim: int, in_dim: int) -> nn.Parameter:
            return nn.Parameter(torch.empty(out_dim, in_dim))

        self.W_ir = _w(hidden_size, input_size)
        self.W_iz = _w(hidden_size, input_size)
        self.W_in = _w(hidden_size, input_size)
        self.W_hr = _w(hidden_size, hidden_size)
        self.W_hz = _w(hidden_size, hidden_size)
        self.W_hn = _w(hidden_size, hidden_size)

        if bias:
            self.b_ir = nn.Parameter(torch.zeros(hidden_size))
            self.b_iz = nn.Parameter(torch.zeros(hidden_size))
            self.b_in = nn.Parameter(torch.zeros(hidden_size))
            self.b_hr = nn.Parameter(torch.zeros(hidden_size))
            self.b_hz = nn.Parameter(torch.zeros(hidden_size))
            self.b_hn = nn.Parameter(torch.zeros(hidden_size))
        else:
            for name in ("b_ir", "b_iz", "b_in", "b_hr", "b_hz", "b_hn"):
                self.register_parameter(name, None)

        self.reset_parameters()

        # ---- quantizers (one module each so they hold independent state) ----
        # Activation quantizers
        self.quant_x = make_quantizer(recipe.input_act)
        self.quant_h_in = make_quantizer(recipe.hidden)
        self.quant_h_out = make_quantizer(recipe.hidden)

        # Weight quantizers — six independent modules so each one's
        # observer / learnable scale is independent. They share `recipe.weight`
        # as a *config* but each instance has its own buffers.
        self.quant_W_ir = make_quantizer(recipe.weight)
        self.quant_W_iz = make_quantizer(recipe.weight)
        self.quant_W_in = make_quantizer(recipe.weight)
        self.quant_W_hr = make_quantizer(recipe.weight)
        self.quant_W_hz = make_quantizer(recipe.weight)
        self.quant_W_hn = make_quantizer(recipe.weight)

        # Optional gate-preact quantizers — wired in but identity unless a
        # gate_act config is provided in the recipe.
        gate_cfg = recipe.gate_act or QuantizerConfig(bits=32, name="gate_id")
        self.quant_gate_r = make_quantizer(gate_cfg)
        self.quant_gate_z = make_quantizer(gate_cfg)
        self.quant_gate_n = make_quantizer(gate_cfg)

    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        # Match nn.GRUCell init: uniform(-k, k) where k = 1/sqrt(hidden_size)
        k = self.hidden_size**-0.5
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    # ------------------------------------------------------------------

    def quantize_weights(self) -> CellWeights:
        """Run all six weight quantizers once and return the result.

        Hoist this out of the per-step loop in multi-step layers — weights
        are constant across time, so per-step re-quantization is wasted.
        For fused gate layout, also concatenate the three input/hidden
        weights and biases once so the per-step path can issue two big
        GEMMs instead of six small ones.
        """
        Wir = self.quant_W_ir(self.W_ir)
        Wiz = self.quant_W_iz(self.W_iz)
        Win = self.quant_W_in(self.W_in)
        Whr = self.quant_W_hr(self.W_hr)
        Whz = self.quant_W_hz(self.W_hz)
        Whn = self.quant_W_hn(self.W_hn)
        cw = CellWeights(Wir, Wiz, Win, Whr, Whz, Whn)
        if self.gate_layout == "fused":
            cw.Wi_cat = torch.cat([Wir, Wiz, Win], dim=0)
            cw.Wh_cat = torch.cat([Whr, Whz, Whn], dim=0)
            if self.b_ir is not None:
                cw.bi_cat = torch.cat([self.b_ir, self.b_iz, self.b_in])
                cw.bh_cat = torch.cat([self.b_hr, self.b_hz, self.b_hn])
        return cw

    def step(
        self, x: torch.Tensor, h: torch.Tensor, w: CellWeights
    ) -> torch.Tensor:
        """One step with pre-quantized weights.

        Args:
            x: [batch, input_size]
            h: [batch, hidden_size]
            w: weights already passed through their FakeQuantize modules.
        Returns:
            h_new: [batch, hidden_size]
        """
        # ---- 1. Quantize activations on the read side ----
        xq = self.quant_x(x)
        hq = self.quant_h_in(h)

        # ---- 2. Gate pre-activations (in float; bias unquantized) ----
        # F.linear computes x @ W.T + b. Fused layout stacks the three
        # gate matrices along axis 0 and runs one GEMM per branch.
        if self.gate_layout == "fused":
            assert w.Wi_cat is not None and w.Wh_cat is not None
            gi = F.linear(xq, w.Wi_cat, w.bi_cat)
            gh = F.linear(hq, w.Wh_cat, w.bh_cat)
            gi_r, gi_z, gi_n = gi.chunk(3, dim=-1)
            gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
            gate_r = gi_r + gh_r
            gate_z = gi_z + gh_z
            n_input_branch = gi_n
            n_hidden_branch = gh_n
        else:
            gate_r = F.linear(xq, w.Wir, self.b_ir) + F.linear(hq, w.Whr, self.b_hr)
            gate_z = F.linear(xq, w.Wiz, self.b_iz) + F.linear(hq, w.Whz, self.b_hz)
            # n-gate: NOTE the asymmetry — r_t scales only the hidden branch.
            n_input_branch = F.linear(xq, w.Win, self.b_in)
            n_hidden_branch = F.linear(hq, w.Whn, self.b_hn)

        # Optional fake-quant on gate pre-activations (Phase 3 toggle).
        gate_r = self.quant_gate_r(gate_r)
        gate_z = self.quant_gate_z(gate_z)

        # ---- 3. Nonlinearities (fp32) ----
        r = torch.sigmoid(gate_r)
        z = torch.sigmoid(gate_z)

        # n-gate combination: r * (W_hn h + b_hn) is the asymmetric step.
        gate_n = n_input_branch + r * n_hidden_branch
        gate_n = self.quant_gate_n(gate_n)
        n = torch.tanh(gate_n)

        # ---- 4. Hidden update ----
        h_new = (1.0 - z) * n + z * h
        # Note: we use unquantized h on the carry side so the fp32 path is
        # bit-identical to nn.GRUCell when all quantizers are Identity.
        # The "stored" h_new is the quantized one — see GRULayer.

        # ---- 5. Quantize on the write side (so next step reads quant) ----
        return self.quant_h_out(h_new)

    def input_projection(
        self, x_seq: torch.Tensor, w: CellWeights
    ) -> torch.Tensor:
        """Pre-compute x @ W_i + b_i for the whole sequence in one GEMM.

        Args:
            x_seq: [T, B, input_size]
            w: must have `Wi_cat` populated (fused gate layout).
        Returns:
            gi: [T, B, 3 * hidden_size] with bias added; chunk along the
                last dim per step into (r, z, n) slices.

        Side effect for activation quant: `quant_x` runs once on the full
        `[T, B, in]` tensor instead of per-step on `[B, in]`. For per-tensor
        dynamic mode this means *one* scale across the whole sequence —
        closer to the eventual frozen-inference kernel's behaviour, and
        a meaningful behaviour change vs. the per-step path. Per-channel
        activation quant on a non-time axis is unaffected.
        """
        if self.gate_layout != "fused":
            raise RuntimeError(
                "input_projection requires gate_layout='fused'"
            )
        assert w.Wi_cat is not None
        xq = self.quant_x(x_seq)
        return F.linear(xq, w.Wi_cat, w.bi_cat)

    def step_with_gi(
        self,
        gi_t: torch.Tensor,
        h: torch.Tensor,
        w: CellWeights,
    ) -> torch.Tensor:
        """One step using a pre-computed input projection.

        Args:
            gi_t: [B, 3*hidden_size] — output of input_projection at time t.
            h:    [B, hidden_size]
            w:    pre-quantized weights (fused layout).
        """
        assert self.gate_layout == "fused"
        assert w.Wh_cat is not None
        hq = self.quant_h_in(h)
        gh = F.linear(hq, w.Wh_cat, w.bh_cat)
        gi_r, gi_z, gi_n = gi_t.chunk(3, dim=-1)
        gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
        gate_r = self.quant_gate_r(gi_r + gh_r)
        gate_z = self.quant_gate_z(gi_z + gh_z)
        r = torch.sigmoid(gate_r)
        z = torch.sigmoid(gate_z)
        gate_n = self.quant_gate_n(gi_n + r * gh_n)
        n = torch.tanh(gate_n)
        h_new = (1.0 - z) * n + z * h
        return self.quant_h_out(h_new)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """One step. Convenience wrapper that quantizes weights inline.

        Multi-step callers should use `quantize_weights()` + `step()` to
        avoid re-quantizing weights on every timestep.
        """
        return self.step(x, h, self.quantize_weights())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_recipe(
        cls,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        **kwargs: object,
    ) -> "GRUCellQuant":
        return cls(input_size, hidden_size, recipe, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # State transitions: training / calibration / inference
    # ------------------------------------------------------------------

    def freeze_quantizers(self) -> None:
        """Switch every quantizer in this cell to frozen mode.

        After calibration, call this once before exporting to the inference
        kernel. From this point, scales are read-only.
        """
        for module in self.modules():
            if isinstance(module, FakeQuantize) and module is not self:
                module.freeze()

    # TODO(phase=5): export_int_weights() returning a dict of int tensors,
    # scales, and zero points in the layout expected by the Triton kernel.
    # Defer until the kernel layout is fixed.
