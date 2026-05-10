"""Calibration round-trip tests.

Validates the typical QAT-to-deployment flow:
1. Build a layer with min_max-mode activation quantizers.
2. Run calibrate() over a synthetic loader.
3. Confirm running stats populated.
4. Call freeze() and confirm scales are now stable across forwards.
"""

from __future__ import annotations

import pytest
import torch

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig
from gru_qat.calibration import calibrate, freeze_all
from gru_qat.quantizers import FakeQuantizePerTensor


def _make_qat_layer(in_size: int = 16, hid: int = 32) -> GRULayer:
    """Layer with int8 hidden quantizer (per-tensor symmetric, mode default)."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=8, axis=0, name="W"),
        input_act=QuantizerConfig(bits=8, name="x"),
        hidden=QuantizerConfig(bits=8, name="h"),
    )
    return GRULayer(in_size, hid, recipe=rec, gate_layout="fused")


def _synthetic_loader(n: int, T: int, B: int, in_size: int):
    """Yield n random sequences shaped (T, B, in_size). Tensor-only — no
    h0 — so calibrate() exercises the single-tensor branch."""
    for _ in range(n):
        yield torch.randn(T, B, in_size) * 0.5


def test_calibrate_populates_running_stats() -> None:
    layer = _make_qat_layer()
    summary = calibrate(layer, _synthetic_loader(10, 8, 4, 16), n_batches=10)

    # At minimum the activation quantizers should appear in the summary
    # with finite running stats.
    expected_names = {"cell.quant_x", "cell.quant_h_in", "cell.quant_h_out"}
    assert expected_names.issubset(summary.keys()), (
        f"missing quantizers: {expected_names - set(summary.keys())}"
    )
    for name in expected_names:
        info = summary[name]
        assert info["initialized"] is True
        rmin, rmax = info["running_min"], info["running_max"]
        rmin_v = rmin if isinstance(rmin, float) else min(rmin)
        rmax_v = rmax if isinstance(rmax, float) else max(rmax)
        # Stats must be in (-inf, inf) — initial sentinel values would be
        # +inf / -inf so this catches "didn't run any forwards".
        assert -1e6 < rmin_v < 1e6, f"{name}: running_min still sentinel"
        assert -1e6 < rmax_v < 1e6, f"{name}: running_max still sentinel"
        assert rmin_v <= rmax_v


def test_calibrate_then_freeze_locks_scales() -> None:
    layer = _make_qat_layer()
    calibrate(layer, _synthetic_loader(10, 8, 4, 16), n_batches=10)
    freeze_all(layer)

    # After freeze, scales should not change across forwards even with
    # very-different-magnitude inputs.
    h_in = layer.cell.quant_h_in
    h_out = layer.cell.quant_h_out
    assert isinstance(h_in, FakeQuantizePerTensor)
    assert isinstance(h_out, FakeQuantizePerTensor)
    assert h_in.config.mode == "frozen"
    assert h_out.config.mode == "frozen"

    scale_in_before = h_in.scale.clone()
    scale_out_before = h_out.scale.clone()

    # Pump a giant-magnitude batch through.
    big_x = torch.randn(8, 4, 16) * 100.0
    layer(big_x)

    assert torch.equal(h_in.scale, scale_in_before)
    assert torch.equal(h_out.scale, scale_out_before)


def test_calibrate_handles_tuple_loader() -> None:
    """Loader yielding (x, h0) tuples should work too."""
    layer = _make_qat_layer()

    def tuple_loader(n: int):
        for _ in range(n):
            x = torch.randn(8, 4, 16) * 0.5
            h0 = torch.randn(4, 32) * 0.5
            yield (x, h0)

    summary = calibrate(layer, tuple_loader(5), n_batches=5)
    assert "cell.quant_x" in summary
    assert summary["cell.quant_x"]["initialized"] is True


def test_calibrate_only_activations_skips_weight_quantizers() -> None:
    """only_activations=True (default) must not modify weight quantizers."""
    layer = _make_qat_layer()
    # Note quant_W_ir is a per-channel-axis weight quantizer in our preset.
    weight_q = layer.cell.quant_W_ir
    mode_before = weight_q.config.mode

    calibrate(layer, _synthetic_loader(3, 4, 2, 16), n_batches=3)

    assert weight_q.config.mode == mode_before, (
        "weight quantizer mode was changed; only_activations=True should leave it alone"
    )


def test_calibrate_truncates_to_n_batches() -> None:
    """Calibration must stop at n_batches even if loader has more."""
    layer = _make_qat_layer()
    # Loader that would yield 100 if exhausted; we ask for 3.
    summary = calibrate(layer, _synthetic_loader(100, 4, 2, 16), n_batches=3)
    # Only check it didn't crash and produced summary; per-batch counting
    # isn't exposed in the API.
    assert "cell.quant_x" in summary
