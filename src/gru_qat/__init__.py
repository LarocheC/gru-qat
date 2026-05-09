"""gru_qat — pluggable QAT and quantized inference for GRU.

See SCOPE.md for design and DEVELOPMENT.md for the implementation plan.
"""

from gru_qat.gru_cell import GRUCellQuant
from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import (
    PRESETS,
    FakeQuantize,
    FakeQuantizePerChannel,
    FakeQuantizePerGroup,
    FakeQuantizePerTensor,
    Identity,
    QuantizerConfig,
    QuantRecipe,
)
from gru_qat.ste import STEClamp, STERound

__all__ = [
    "GRUCellQuant",
    "GRULayer",
    "FakeQuantize",
    "FakeQuantizePerTensor",
    "FakeQuantizePerChannel",
    "FakeQuantizePerGroup",
    "Identity",
    "QuantizerConfig",
    "QuantRecipe",
    "PRESETS",
    "STERound",
    "STEClamp",
]
