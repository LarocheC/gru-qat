"""Tests for quantizers.py — Phase 1 exit tests."""

from __future__ import annotations

import pytest
import torch

from gru_qat.quantizers import (
    FakeQuantizePerChannel,
    FakeQuantizePerGroup,
    FakeQuantizePerTensor,
    Identity,
    QuantizerConfig,
    make_quantizer,
)


def test_identity_is_passthrough() -> None:
    q = Identity()
    x = torch.randn(8, 16)
    assert torch.equal(q(x), x)


def test_per_tensor_roundtrip_no_clip() -> None:
    cfg = QuantizerConfig(bits=8, symmetric=True)
    q = FakeQuantizePerTensor(cfg)
    x = torch.randn(64, 32) * 0.5  # well within range for 8-bit symmetric
    out = q(x)
    # Step size after one forward
    scale, _ = q._compute_scale_zp(x)
    assert (out - x).abs().max() <= scale.item()


def test_per_channel_independent_scales() -> None:
    cfg = QuantizerConfig(bits=8, axis=0, symmetric=True)
    q = FakeQuantizePerChannel(cfg)
    # Row 0: small magnitudes; row 1: large.
    x = torch.stack([torch.randn(16) * 0.01, torch.randn(16) * 10.0])
    scale, _ = q._compute_scale_zp(x)
    assert scale.shape == (2, 1)
    assert scale[1, 0] > 100 * scale[0, 0]


def test_per_group_scale_count() -> None:
    cfg = QuantizerConfig(bits=4, axis=0, group_size=8, symmetric=True)
    q = FakeQuantizePerGroup(cfg)
    x = torch.randn(32, 16)  # 32 / 8 = 4 groups along axis 0
    out = q(x)
    assert out.shape == x.shape


def test_per_group_requires_divisibility() -> None:
    cfg = QuantizerConfig(bits=4, axis=0, group_size=7, symmetric=True)
    q = FakeQuantizePerGroup(cfg)
    x = torch.randn(32, 16)
    with pytest.raises(ValueError, match="not divisible"):
        q(x)


def test_make_quantizer_dispatch() -> None:
    assert isinstance(
        make_quantizer(QuantizerConfig(bits=8)), FakeQuantizePerTensor
    )
    assert isinstance(
        make_quantizer(QuantizerConfig(bits=8, axis=0)), FakeQuantizePerChannel
    )
    assert isinstance(
        make_quantizer(QuantizerConfig(bits=4, axis=0, group_size=64)),
        FakeQuantizePerGroup,
    )
    assert isinstance(
        make_quantizer(QuantizerConfig(bits=32)), Identity
    )


def test_freeze_locks_scale() -> None:
    cfg = QuantizerConfig(bits=8, mode="min_max")
    q = FakeQuantizePerTensor(cfg)
    # warm up the observer
    for _ in range(10):
        q(torch.randn(64) * 2.0)
    q.freeze()
    scale_before = q.scale.clone()
    q(torch.randn(64) * 100.0)  # would shift dynamic scale dramatically
    assert torch.equal(q.scale, scale_before)


# TODO(phase=2): test_simulator_parity — requires the existing simulator
# in PYTHONPATH; mark xfail until then.
@pytest.mark.skip(reason="phase=2 — requires simulator import")
def test_matches_simulator_quantize_dequantize() -> None:
    pass


def test_per_channel_min_max_observer_per_channel_running_stats() -> None:
    """QNT-04 (D-44 / D-45): the per-channel ``min_max`` observer must produce
    PER-CHANNEL ``running_min`` / ``running_max`` tensors, not scalars.

    The current implementation at ``src/gru_qat/quantizers.py:135-146`` calls
    ``x.detach().min()`` / ``.max()`` — global scalar reductions, broken for
    per-channel axes. After the fix in Commit B (per-axis reduction via
    ``x.amin(dim=other_dims)``), ``running_min`` / ``running_max`` should be
    shape ``[num_channels]`` with channel-distinct values.

    Construct a tensor with channel 0 in ``[-1, 1]`` and channel 1 in
    ``[-10, 10]`` (per CONTEXT specifics). After one forward, assert:

    - ``running_min.shape == (2,)`` — NOT scalar.
    - ``running_min[0] != running_min[1]`` — channel-distinct values.

    Two-commit failing-test-before-fix per D-37 / D-45: this test is
    Commit A (failing-before-fix); Commit B fixes ``_update_observer`` at
    ``src/gru_qat/quantizers.py:135-146``; CI green => ``bd close`` for the
    QNT-04 / ACT-01 issue.

    Pattern mirrors ``test_per_channel_independent_scales`` at lines 34-41
    but exercises the ``min_max`` observer path (``mode='min_max'``) rather
    than the default ``dynamic`` ``_compute_scale_zp`` path.
    """
    cfg = QuantizerConfig(bits=8, axis=0, symmetric=True, mode="min_max")
    q = FakeQuantizePerChannel(cfg)
    # Channel 0 in [-1, 1]; channel 1 in [-10, 10]. Distinct per-channel.
    x = torch.stack([torch.randn(16) * 1.0, torch.randn(16) * 10.0])
    # Force values to span the intended range so min/max are unambiguous.
    x[0, 0] = -1.0
    x[0, -1] = 1.0
    x[1, 0] = -10.0
    x[1, -1] = 10.0
    q(x)  # one forward; min_max observer updates running stats
    # Assertions that FAIL pre-fix (scalar reduction produces 0-d running_min):
    assert q.running_min.ndim > 0, (
        f"running_min should be per-channel; got scalar "
        f"(ndim={q.running_min.ndim}, shape={tuple(q.running_min.shape)})"
    )
    assert q.running_min.shape == (2,), (
        f"running_min should be shape (2,); got {tuple(q.running_min.shape)}"
    )
    assert q.running_max.shape == (2,), (
        f"running_max should be shape (2,); got {tuple(q.running_max.shape)}"
    )
    # Channel 0 in [-1, 1]; channel 1 in [-10, 10] => running_min must differ.
    assert q.running_min[0] != q.running_min[1], (
        f"running_min should differ per channel; got values "
        f"{q.running_min.tolist()}"
    )
    assert q.running_max[1] > q.running_max[0], (
        f"running_max[1] should exceed running_max[0]; got values "
        f"{q.running_max.tolist()}"
    )
