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
