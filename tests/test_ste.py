"""Tests for ste.py — Phase 1 exit tests."""

from __future__ import annotations

import pytest
import torch

from gru_qat.ste import STEClamp, STERound, fake_quant_ste


class TestSTERound:
    def test_forward_rounds(self) -> None:
        x = torch.tensor([0.1, 0.6, -0.4, -0.7, 1.5, 2.5])
        out = STERound.apply(x)
        # round-half-to-even: 1.5 -> 2, 2.5 -> 2
        expected = torch.tensor([0.0, 1.0, -0.0, -1.0, 2.0, 2.0])
        assert torch.allclose(out, expected)

    def test_backward_is_identity(self) -> None:
        x = torch.tensor([0.3, 1.7, -0.9], requires_grad=True)
        out = STERound.apply(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))


class TestSTEClamp:
    def test_forward_clamps(self) -> None:
        x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        out = STEClamp.apply(x, -2.0, 2.0)
        assert torch.allclose(out, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))

    def test_backward_zero_outside_range(self) -> None:
        x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], requires_grad=True)
        out = STEClamp.apply(x, -2.0, 2.0)
        out.sum().backward()
        assert x.grad is not None
        # Inside range: 1.0; outside: 0.0
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
        assert torch.allclose(x.grad, expected)


class TestFakeQuant:
    def test_roundtrip_zero_error_at_grid(self) -> None:
        # Values that fall exactly on the quantization grid should round-trip
        # losslessly.
        scale = torch.tensor(0.5)
        zp = torch.tensor(0.0)
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  # multiples of scale
        out = fake_quant_ste(x, scale, zp, qmin=-127, qmax=127)
        assert torch.allclose(out, x)

    def test_clipping(self) -> None:
        # Values beyond the representable range should clip.
        scale = torch.tensor(1.0)
        zp = torch.tensor(0.0)
        x = torch.tensor([-200.0, 0.0, 200.0])
        out = fake_quant_ste(x, scale, zp, qmin=-127, qmax=127)
        assert out[0].item() == pytest.approx(-127.0)
        assert out[2].item() == pytest.approx(127.0)
        assert out[1].item() == pytest.approx(0.0)

    def test_gradient_flows(self) -> None:
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        scale = torch.tensor(0.1)
        zp = torch.tensor(0.0)
        out = fake_quant_ste(x, scale, zp, qmin=-127, qmax=127)
        out.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))
