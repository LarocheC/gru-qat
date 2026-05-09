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

from gru_qat.gru_cell import GRUCellQuant
from gru_qat.quantizers import QuantRecipe


class GRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        *,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.cell = GRUCellQuant(input_size, hidden_size, recipe)
        self.hidden_size = hidden_size
        self.batch_first = batch_first

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
        for t in range(seq_len):
            h = self.cell(x[t], h)
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
