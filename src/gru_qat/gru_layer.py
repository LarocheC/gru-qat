"""Multi-timestep GRU layer.

Iterates a `GRUCellQuant` over a sequence. Single layer, single direction;
that's deliberate (see SCOPE.md non-goals). Stack two of these for a
2-layer GRU; bidirectionality is a wrapper around two of them.

Hidden state carry is the subtle part:
  - At training time, h_{t-1} is the (fake-quantized) output of step t-1,
    which is exactly what the quant_h_out call inside GRUCellQuant
    produces. So `h_carry = cell(x_t, h_carry)` is correct.
  - At inference time with a frozen hidden quantizer, the same code path
    works because frozen-mode quantize-dequantize is the same op shape; we
    just stop updating the scale.
  - Streaming inference: the user calls `step(x_t, h)` themselves and
    feeds h_t back in next call. `forward` is for full-sequence training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gru_qat.gru_cell import GateLayout, GRUCellQuant
from gru_qat.quantizers import QuantRecipe
from gru_qat.structure import StructureConfig


class GRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        *,
        batch_first: bool = False,
        gate_layout: GateLayout = "split",
        compile_step: bool = False,
        pre_batch_input: bool = False,
        structure_input: StructureConfig | None = None,
        structure_hidden: StructureConfig | None = None,
    ) -> None:
        super().__init__()
        if pre_batch_input and gate_layout != "fused":
            raise ValueError(
                "pre_batch_input=True requires gate_layout='fused'"
            )
        self.cell = GRUCellQuant(
            input_size, hidden_size, recipe,
            gate_layout=gate_layout,
            structure_input=structure_input,
            structure_hidden=structure_hidden,
        )
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        # Structured cells go through a different per-step path that
        # doesn't pre-quantize a CellWeights bag. pre_batch_input also
        # depends on the dense fused layout so it's force-disabled there.
        if self.cell.is_structured and pre_batch_input:
            raise ValueError(
                "pre_batch_input is not supported in structured mode "
                "(no dense Wi_cat to pre-project)."
            )
        self.pre_batch_input = pre_batch_input
        # When compile_step is True, wrap the per-step body in torch.compile
        # so Inductor fuses the elementwise ops (sigmoid/tanh/mul/add) with
        # the matmul epilogue. Static shapes only — bind one specialization
        # per (batch, hidden) seen.
        #
        # We deliberately do NOT use mode="reduce-overhead": that enables
        # CUDA Graphs which captures input/output buffers statically, but
        # the GRU loop feeds the previous step's output back as the next
        # step's input — the graph then overwrites a tensor that the next
        # invocation is still holding a pointer to. Plain "default" gets
        # the kernel fusion win without the graph-capture footgun.
        if self.cell.is_structured:
            body = self.cell.step_structured
        elif pre_batch_input:
            body = self.cell.step_with_gi
        else:
            body = self.cell.step
        self._compiled_step = (
            torch.compile(body, mode="default", dynamic=False)
            if compile_step
            else body
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the cell over the time dimension.

        Args:
            x: [seq, batch, input_size] (or [batch, seq, input_size] if
               batch_first).
            h0: [batch, hidden_size]. Defaults to zeros.

        Returns:
            outputs: [seq, batch, hidden_size] (or [batch, seq, ...]).
            h_T:     [batch, hidden_size], the final hidden state.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if h0 is None:
            h0 = x.new_zeros(batch_size, self.hidden_size)

        h = h0
        outputs: list[torch.Tensor] = []
        step = self._compiled_step
        if self.cell.is_structured:
            # Structured per-step path takes (x, h) only — there's no
            # pre-quantized CellWeights bag to thread through.
            for t in range(seq_len):
                h = step(x[t], h)
                outputs.append(h)
        else:
            # Hoist weight quantization out of the time loop — weights are
            # invariant across timesteps, so calling the six FakeQuantize
            # modules per step is wasted work (it dominates int8 training cost).
            w = self.cell.quantize_weights()
            if self.pre_batch_input:
                # Run x @ W_i + b_i once over the whole sequence so the per-step
                # body only does the hidden-projection GEMM. Big win at large
                # T where the input GEMM is no longer launch-bound.
                gi = self.cell.input_projection(x, w)  # [T, B, 3*hidden]
                for t in range(seq_len):
                    h = step(gi[t], h, w)
                    outputs.append(h)
            else:
                for t in range(seq_len):
                    h = step(x[t], h, w)
                    outputs.append(h)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, h

    # ------------------------------------------------------------------
    # Calibration / freezing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(self, loader: object) -> None:
        """Run forward passes in min_max observer mode to gather stats.

        Caller must set the recipe's mode to "min_max" before constructing
        the layer (or set it on each quantizer manually). After calibrate(),
        call freeze() to fix the scales for inference.

        TODO(phase=4): take a real DataLoader, run N batches, return stats
        summary. For now this is a stub.
        """
        raise NotImplementedError("phase=4")

    def freeze(self) -> None:
        self.cell.freeze_quantizers()


# TODO(phase=5): TritonGRULayer — same interface, different cell.
# def __init__ should accept the same QuantRecipe and either dispatch to
# the matching kernel variant or raise if the recipe doesn't have a
# kernel.
