"""Layer-parity tests — Phase 1 audit.

Validates that GRULayer with all quantizers set to Identity (use_triton=False,
dense, no structure_hidden) matches torch.nn.GRU(num_layers=1, bidirectional=
False, batch_first=False) on (out, h_T) forward, on the six weight gradients
plus dx and dh_0 backward, and on h_0 != 0 initial state. Tolerance < 1e-4
across a T x B x H = 5 x 3 x 5 = 75-combo grid (see Plan 02).

If this fails, the unroll math (or its time-loop orchestration) is wrong and
every later phase's reference is contaminated.

This module sets ``torch.set_float32_matmul_precision('highest')`` at import
time because Phase 1 audits the math, not TF32. Diverges from
``tests/test_triton_*.py`` (which use 'high' so the kernel runs under realistic
conditions) — that's intentional. Cell-level parity at < 1e-5 is pinned by
``tests/test_parity.py``; this file is the *layer*-level counterpart at < 1e-4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS

# Module-level: we audit math, not TF32. "highest" forces IEEE-754 fp32 matmul,
# so the only drift is algorithm, not arithmetic mode. Diverges from the
# Triton kernel tests (which use "high" to test the kernel under realistic
# conditions). See .planning/phases/01-reference-path-parity-vs-nn-gru/01-CONTEXT.md
# D-07 for the locked decision.
torch.set_float32_matmul_precision("highest")


def _make_dense_fp32_layer(input_size: int, hidden_size: int) -> GRULayer:
    """fp32 dense reference layer: Identity quantizers, no structure, no Triton.

    This is the path Phase 1 audits. ``recipe=PRESETS['fp32']`` selects the
    three Identity quantizers (weight / input_act / hidden, all bits=32, all
    axis=None — see ``src/gru_qat/quantizers.py:284-289``). No
    ``structure_input`` / ``structure_hidden`` arg → dense W_i*/W_h* parameters.
    ``use_triton`` is left at the default "auto" which resolves to ``False``
    here because the layer is not fast-dispatch eligible (no structured hidden
    + ``gate_layout='split'``).
    """
    return GRULayer(
        input_size,
        hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
    )


def _translate_cell_to_nn_gru(layer: GRULayer) -> nn.GRU:
    """Build a ``torch.nn.GRU(num_layers=1, bidirectional=False, batch_first=False)``
    whose weights and biases reproduce ``layer`` exactly.

    Per the PyTorch GRU docs
    (https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) gate
    order is ``(r, z, n)`` for both sides, matching ``gru_cell.py``'s
    ``W_ir / W_iz / W_in`` family. Translation is just ``torch.cat`` along
    axis 0::

        weight_ih_l0 = cat([W_ir, W_iz, W_in], dim=0)   # [3H, IN]
        weight_hh_l0 = cat([W_hr, W_hz, W_hn], dim=0)   # [3H, H]
        bias_ih_l0   = cat([b_ir, b_iz, b_in])           # [3H]
        bias_hh_l0   = cat([b_hr, b_hz, b_hn])           # [3H]

    The n-gate asymmetry (``r_t * (W_hn h + b_hn)`` *inside* the tanh,
    multiplying only the hidden contribution) is preserved by this layout
    because both sides apply ``r_t`` identically. See
    ``src/gru_qat/gru_cell.py:1-15`` (module docstring) for why this asymmetric
    placement matters — many home-grown GRU implementations get it wrong.

    Primary direction for the Phase 1 grid (Plans 02-04): build a ``GRULayer``
    first with random weights as source-of-truth, then build an ``nn.GRU``
    from this helper and compare. Inverse direction (
    ``_translate_nn_gru_to_cell``) is exercised by a single round-trip smoke
    test only.
    """
    cell = layer.cell
    gru = nn.GRU(
        input_size=cell.input_size,
        hidden_size=cell.hidden_size,
        num_layers=1,
        bidirectional=False,
        batch_first=False,
    )
    with torch.no_grad():
        gru.weight_ih_l0.copy_(torch.cat([cell.W_ir, cell.W_iz, cell.W_in], dim=0))
        gru.weight_hh_l0.copy_(torch.cat([cell.W_hr, cell.W_hz, cell.W_hn], dim=0))
        gru.bias_ih_l0.copy_(torch.cat([cell.b_ir, cell.b_iz, cell.b_in]))
        gru.bias_hh_l0.copy_(torch.cat([cell.b_hr, cell.b_hz, cell.b_hn]))
    return gru


def _translate_nn_gru_to_cell(gru: nn.GRU) -> GRULayer:
    """Inverse of ``_translate_cell_to_nn_gru`` — used by the round-trip smoke
    test only.

    Direct mirror of ``tests/test_parity.py:18-44`` ``_copy_weights`` but at
    the layer level: ``chunk(3, dim=0)`` splits ``nn.GRU``'s concatenated
    weight / bias parameters back into the per-gate ``W_ir / W_iz / W_in``
    family. ``bin_`` (trailing underscore) is mandatory because ``bin`` is a
    Python built-in — same convention as ``tests/test_parity.py:29``.
    """
    layer = GRULayer(
        gru.input_size,
        gru.hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
    )
    cell = layer.cell
    Wir, Wiz, Win = gru.weight_ih_l0.chunk(3, dim=0)
    Whr, Whz, Whn = gru.weight_hh_l0.chunk(3, dim=0)
    bir, biz, bin_ = gru.bias_ih_l0.chunk(3)
    bhr, bhz, bhn = gru.bias_hh_l0.chunk(3)
    with torch.no_grad():
        cell.W_ir.copy_(Wir)
        cell.W_iz.copy_(Wiz)
        cell.W_in.copy_(Win)
        cell.W_hr.copy_(Whr)
        cell.W_hz.copy_(Whz)
        cell.W_hn.copy_(Whn)
        cell.b_ir.copy_(bir)
        cell.b_iz.copy_(biz)
        cell.b_in.copy_(bin_)
        cell.b_hr.copy_(bhr)
        cell.b_hz.copy_(bhz)
        cell.b_hn.copy_(bhn)
    return layer


# ----------------------------------------------------------------------------
# Gate-ordering / n-gate-asymmetry micro-tests (Plan 01-01, Task 2; D-04)
# ----------------------------------------------------------------------------
#
# These three tests are NOT parametrized — they are one-shot smoke tests run
# BEFORE the 75-combo grid (which lives in Plan 02). If any of these fail, the
# helper is compensating for a real cell-math bug that the grid would mask;
# they isolate the assumption that gate order is (r, z, n) on both sides and
# that the n-gate's r_t multiplier is applied only to the hidden contribution.
#
# Tolerance: < 1e-4 using the relative-error idiom from
# tests/test_triton_diagonal.py:120-121. The 1e-6 floor on the denominator is
# non-negotiable — prevents division by near-zero on degenerate cases.


def test_gate_order_r_only() -> None:
    """Set W_ir=ones, W_iz=W_in=zeros (all hidden weights and biases zero);
    nn.GRU and ours must agree that only the r-gate's sigmoid fires.

    If the cell's gate order is wrong (e.g. (z, r, n) instead of (r, z, n)),
    the grid tests will still pass because the translation helper would
    compensate. This micro-test isolates the gate-order assumption by
    activating only the r-gate.
    """
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        cell.W_ir.fill_(1.0)
        cell.W_iz.zero_()
        cell.W_in.zero_()
        cell.W_hr.zero_()
        cell.W_hz.zero_()
        cell.W_hn.zero_()
        for b in (cell.b_ir, cell.b_iz, cell.b_in, cell.b_hr, cell.b_hz, cell.b_hn):
            b.zero_()
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(1, 2, 4)  # [T=1, B=2, IN=4]
    h0 = torch.zeros(2, 4)  # [B=2, H=4]
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, (
        f"r-only rel diff {rel:.4e} "
        f"(out_ref.shape={tuple(out_ref.shape)}, out_ours.shape={tuple(out_ours.shape)})"
    )


def test_gate_order_z_only() -> None:
    """Set W_iz=ones, W_ir=W_in=zeros (all hidden weights and biases zero);
    nn.GRU and ours must agree that only the z-gate's sigmoid fires.

    Companion to ``test_gate_order_r_only`` — swaps which input-side gate is
    active. Together the two tests pin the order of W_ir vs W_iz in the
    translation helper.
    """
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        cell.W_ir.zero_()
        cell.W_iz.fill_(1.0)
        cell.W_in.zero_()
        cell.W_hr.zero_()
        cell.W_hz.zero_()
        cell.W_hn.zero_()
        for b in (cell.b_ir, cell.b_iz, cell.b_in, cell.b_hr, cell.b_hz, cell.b_hn):
            b.zero_()
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(1, 2, 4)
    h0 = torch.zeros(2, 4)
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, (
        f"z-only rel diff {rel:.4e} "
        f"(out_ref.shape={tuple(out_ref.shape)}, out_ours.shape={tuple(out_ours.shape)})"
    )


def test_n_gate_asymmetry() -> None:
    """Force ``r ~ 0`` by setting ``b_ir`` to large-negative; the n-gate must
    reduce to ``tanh(W_in x + b_in)`` (without the ``r * (W_hn h + b_hn)``
    contribution).

    Both nn.GRU and our cell must agree on the asymmetric placement of r
    inside the tanh — see src/gru_qat/gru_cell.py:11-14 module docstring.
    Many home-grown GRU implementations apply r to the whole n-gate
    pre-activation (including the input branch) and silently lose 1-2%
    accuracy. This test isolates that asymmetry: with r squashed to ~0 by
    the strong negative bias, the only path that produces the correct output
    is the asymmetric one. Note that W_in, W_hn, b_in, b_hn are kept at their
    initialized values on purpose — we want a non-trivial n-gate
    contribution from the input branch to verify it survives intact.
    """
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        # Squash r to ~0: zero W_ir so x doesn't drive r, and slam b_ir
        # large-negative so sigmoid(gate_r) -> 0 regardless of h.
        cell.W_ir.zero_()
        cell.W_hr.zero_()
        cell.b_ir.fill_(-100.0)
        cell.b_hr.zero_()
        # W_in, W_hn, b_in, b_hn stay at their init values — that's the
        # whole point of the test.
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(1, 2, 4)
    h0 = torch.zeros(2, 4)
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, (
        f"n-gate-asymmetry rel diff {rel:.4e} "
        f"(out_ref.shape={tuple(out_ref.shape)}, out_ours.shape={tuple(out_ours.shape)})"
    )


# ----------------------------------------------------------------------------
# Round-trip smoke test (Plan 01-01, Task 2; D-01)
# ----------------------------------------------------------------------------


def test_round_trip_nn_gru_to_cell() -> None:
    """Build an nn.GRU first, copy its weights into a fresh GRULayer via the
    inverse helper, then assert layer outputs match.

    Catches bugs in ``_translate_nn_gru_to_cell`` itself before any
    parametrized grid runs. The grid in Plan 02 uses the cell-to-nn.GRU
    direction; this one-shot test exercises the opposite direction so a bug
    in the inverse helper surfaces here rather than silently passing the
    grid (where it would never be called).
    """
    torch.manual_seed(0)
    gru = nn.GRU(8, 16, num_layers=1, bidirectional=False, batch_first=False)
    layer = _translate_nn_gru_to_cell(gru)

    x = torch.randn(7, 4, 8)  # [T=7, B=4, IN=8]
    h0_3d = torch.zeros(1, 4, 16)  # nn.GRU expects [num_layers, B, H]

    out_ref, hT_ref = gru(x, h0_3d)
    out_ours, hT_ours = layer(x, h0_3d.squeeze(0))

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"round-trip out rel diff {rel:.4e}"

    max_diff_h = (hT_ref.squeeze(0) - hT_ours).abs().max().item()
    rel_h = max_diff_h / max(hT_ref.abs().max().item(), 1e-6)
    assert rel_h < 1e-4, f"round-trip h_T rel diff {rel_h:.4e}"


# ----------------------------------------------------------------------------
# Grid constants for the 75-combo parity grid (Plan 01-02; D-08)
# ----------------------------------------------------------------------------
#
# The full grid is T x B x H = 5 x 3 x 5 = 75 combinations, split into
# FAST_GRID (T in {1, 8, 64}; 45 cases) which runs on every `pytest -q`
# invocation, and SLOW_GRID (T in {512, 1024}; 30 cases) which is gated
# behind `@pytest.mark.slow` and only runs under `pytest -m slow`. The
# B/H grid stays full on both sides per CONTEXT.md D-08.

# Fast grid: T in {1, 8, 64}. Runs on every `pytest -q` invocation.
# 3 x 3 x 5 = 45 cases per family (D-08).
FAST_GRID: list[tuple[int, int, int]] = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]
# Slow grid: T in {512, 1024}. Runs only under `pytest -m slow`.
# 2 x 3 x 5 = 30 cases per family (D-08).
SLOW_GRID: list[tuple[int, int, int]] = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]


# ----------------------------------------------------------------------------
# Forward-output parity tests (Plan 01-02, Task 1; REF-01)
# ----------------------------------------------------------------------------
#
# Test family split per D-09: forward-output parity is its OWN parametrized
# function (and OWN _slow sibling), distinct from h_T parity. If the forward
# output drifts but h_T is fine, the bug is in the per-step output write or
# in the time-loop's `outputs.append(h)` ordering; if h_T drifts but forward
# is fine, the bug is in the final-step or in the return-tuple's second
# element. Fusing the two assertions into one function would lose that
# diagnostic signal.


@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_forward_matches_nn_gru(T: int, B: int, H: int) -> None:
    """Forward output parity vs torch.nn.GRU across the fast grid (T in {1,8,64}).

    Uses the cell -> nn.GRU translation helper from Plan 01-01. Both
    implementations get ``h0=None`` (default zero-h0); the h0 != 0 case is
    Plan 01-04's territory. Relative-error idiom with the 1e-6 denominator
    floor and 1e-4 tolerance — see TESTING.md "Relative-error reporting"
    and PATTERNS.md "Core parity-test body pattern".
    """
    torch.manual_seed(0)
    IN = max(H, 1)  # keep input_size tied to H so the grid stays compact

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    out_ref, _ = gru(x)
    out_ours, _ = layer(x)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"out rel diff {rel:.4e} (T={T},B={B},H={H})"


@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_GRID)
def test_layer_forward_matches_nn_gru_slow(T: int, B: int, H: int) -> None:
    """Forward output parity vs torch.nn.GRU across the slow grid (T in {512, 1024}).

    Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    so default ``pytest -q`` doesn't pay the long-T cost. Same 1e-4
    relative tolerance — long sequences shouldn't accumulate drift past
    that under ``set_float32_matmul_precision('highest')``.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    out_ref, _ = gru(x)
    out_ours, _ = layer(x)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"out rel diff {rel:.4e} (T={T},B={B},H={H})"


# ----------------------------------------------------------------------------
# Final-hidden-state (h_T) parity tests (Plan 01-02, Task 2; REF-04)
# ----------------------------------------------------------------------------
#
# D-09 enforces splitting h_T parity into its OWN parametrized function,
# separate from forward-output parity above. If the forward output is fine
# but h_T drifts, the bug is in the final-step write or in how the layer's
# return-tuple's second element is produced; if both fail in lockstep, the
# bug is in the per-step math. Fusing the two would lose that signal.
#
# Shape detail: nn.GRU returns h_n with shape [num_layers=1, B, H]; our
# GRULayer returns h_T with shape [B, H] (no leading num_layers axis).
# Compare via ``hT_ref.squeeze(0)`` vs ``hT_ours``. The denominator floor
# uses ``hT_ref.abs().max()`` (equivalent under squeeze of a size-1 dim,
# reads more directly).


@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_h_T_matches_nn_gru(T: int, B: int, H: int) -> None:
    """h_T parity: distinct test family from forward-output parity so a
    final-step bug surfaces alone. Both nn.GRU's h_n and our h_T are the
    hidden state after T steps (D-09).

    Discards the per-step output (that's the forward-parity test's
    territory). Shape adapter: ``hT_ref`` is ``[1, B, H]`` (nn.GRU's
    ``[num_layers, B, H]``), our ``hT_ours`` is ``[B, H]``; compare via
    ``hT_ref.squeeze(0)``.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    _, hT_ref = gru(x)
    _, hT_ours = layer(x)

    max_diff = (hT_ref.squeeze(0) - hT_ours).abs().max().item()
    rel = max_diff / max(hT_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"h_T rel diff {rel:.4e} (T={T},B={B},H={H})"


@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_GRID)
def test_layer_h_T_matches_nn_gru_slow(T: int, B: int, H: int) -> None:
    """h_T parity across the slow grid (T in {512, 1024}).

    Identical body to the fast variant; gated behind ``@pytest.mark.slow``.
    A long-T h_T drift that the fast grid wouldn't catch would surface here.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    _, hT_ref = gru(x)
    _, hT_ours = layer(x)

    max_diff = (hT_ref.squeeze(0) - hT_ours).abs().max().item()
    rel = max_diff / max(hT_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"h_T rel diff {rel:.4e} (T={T},B={B},H={H})"


# ----------------------------------------------------------------------------
# Backward / gradient parity tests (Plan 01-03, Task 1; REF-03)
# ----------------------------------------------------------------------------
#
# Third test family per D-09: backward parity is its OWN parametrized function
# (and OWN _slow sibling), distinct from forward-output and h_T parity. If the
# forward passes but the backward fails, the bug is in the autograd graph for
# the backward step (e.g. the n-gate's `r * gh_n` derivative), not the
# forward math. Splitting backward into its own family means a gradient bug
# surfaces with a test id that points at exactly which gradient drifted (the
# `{name}` token in the assertion message — see the per-param loop below).
#
# The audit philosophy: "bwd is where bugs hide." The recent fix cluster in
# DEVELOPMENT.md (butterfly OOB at last program, dWh/dbh accumulator slabs,
# cross-CTA fence) all surfaced in backward passes. This family is the
# layer-level analog of the per-Triton-kernel backward-parity tests at
# tests/test_triton_diagonal.py:299-339 — six weight grads (cat'd against
# nn.GRU's [3H, IN] / [3H, H] / [3H] layouts), both bias families, dx, and
# dh_0, all compared via the same < 1e-4 relative-error idiom.
#
# Shape detail: nn.GRU's h_0 has shape [num_layers=1, B, H], so h0_ref.grad
# has shape [1, B, H]. Our GRULayer's h_0 has shape [B, H], so h0_ours.grad
# has shape [B, H]. Compare via `h0_ref.grad.squeeze(0)` vs `h0_ours.grad`.
#
# Detach-clone idiom: x_ours = x_ref.detach().clone().requires_grad_(True)
# (and same for h0) — each implementation owns its own autograd graph. Reusing
# x_ref as the input to GRULayer would build one tape across both graphs and
# the second `.backward(g)` would crash.


@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_backward_matches_nn_gru(T: int, B: int, H: int) -> None:
    """Backward parity: all six weight grads (cat'd against nn.GRU's [3H, IN]
    / [3H, H] layouts), both bias grads, dx, and dh_0 match nn.GRU autograd
    to < 1e-4 relative across the fast grid.

    Separate from forward parity per D-09: if forward passes but backward
    fails, the bug is in the autograd graph for the backward step (e.g. the
    n-gate's ``r * gh_n`` derivative), not the forward math. The ``{name}``
    token in the failure message identifies which of {dx, dh_0, dW_ih, dW_hh,
    db_ih, db_hh} drifted — the single most diagnostic thing in this test.

    Uses a shared random ``g = torch.randn_like(out_ref)`` so both autograd
    graphs see the same upstream gradient signal — every output element
    contributes independently, which is more discriminating than
    ``out.sum().backward()``.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    # Two separate requires_grad=True leaf tensors so each implementation
    # owns its own autograd graph. Detach-clone is mandatory — sharing the
    # same leaf would build one tape across both and the second
    # `.backward(g)` would crash on a second-backward through a graph that
    # was already freed. Note h0 is squeezed BEFORE the clone so the clone's
    # storage matches GRULayer's [B, H] shape and `.grad` accumulates with
    # that shape directly.
    x_ref = torch.randn(T, B, IN, requires_grad=True)
    h0_ref = torch.zeros(1, B, H, requires_grad=True)  # nn.GRU's [1, B, H]
    x_ours = x_ref.detach().clone().requires_grad_(True)
    h0_ours = h0_ref.detach().squeeze(0).clone().requires_grad_(True)  # [B, H]

    out_ref, _ = gru(x_ref, h0_ref)
    out_ours, _ = layer(x_ours, h0_ours)

    # Shared downstream gradient — same g sent into both autograd graphs.
    # randn_like(out_ref) is fine because out_ref and out_ours have the
    # same shape (forward parity passes; that's REF-01's job, gated by the
    # preceding test family).
    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out_ours.backward(g)

    # Build cell-side cat tensors to match nn.GRU's concatenated [3H, *]
    # / [3H] layouts. Gate order is (r, z, n) on both sides — same order
    # as the forward translation in `_translate_cell_to_nn_gru`, so the
    # gradients land in the same rows.
    cell = layer.cell
    our_W_ih = torch.cat([cell.W_ir.grad, cell.W_iz.grad, cell.W_in.grad], dim=0)
    our_W_hh = torch.cat([cell.W_hr.grad, cell.W_hz.grad, cell.W_hn.grad], dim=0)
    our_b_ih = torch.cat([cell.b_ir.grad, cell.b_iz.grad, cell.b_in.grad])
    our_b_hh = torch.cat([cell.b_hr.grad, cell.b_hz.grad, cell.b_hn.grad])

    # Per-gradient relative-error loop. The `{name}` token in the failure
    # message points at exactly which of the six gradient tensors drifted —
    # if a backward bug surfaces, the bd issue title should include this
    # name (e.g. `test_layer_backward_matches_nn_gru[1-1-1] dW_hh drift`).
    for name, ref_t, our_t in [
        ("dx", x_ref.grad, x_ours.grad),
        ("dh_0", h0_ref.grad.squeeze(0), h0_ours.grad),
        ("dW_ih", gru.weight_ih_l0.grad, our_W_ih),
        ("dW_hh", gru.weight_hh_l0.grad, our_W_hh),
        ("db_ih", gru.bias_ih_l0.grad, our_b_ih),
        ("db_hh", gru.bias_hh_l0.grad, our_b_hh),
    ]:
        rel = (ref_t - our_t).abs().max().item() / max(
            ref_t.abs().max().item(), 1e-6
        )
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H})"


@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_GRID)
def test_layer_backward_matches_nn_gru_slow(T: int, B: int, H: int) -> None:
    """Backward parity across the slow grid (T in {512, 1024}).

    Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    so default ``pytest -q`` doesn't pay the long-T autograd cost (backward
    through 1024 timesteps is the longest single autograd graph in the
    audit). A long-T backward drift that the fast grid wouldn't catch would
    surface here — if drift is uniform across the grid, it's a math bug;
    if it's scale-dependent (large-H, large-T), it indicates accumulated
    numerical drift that informs Phase 6 (edge sweeps) rather than
    reopening Phase 1.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x_ref = torch.randn(T, B, IN, requires_grad=True)
    h0_ref = torch.zeros(1, B, H, requires_grad=True)
    x_ours = x_ref.detach().clone().requires_grad_(True)
    h0_ours = h0_ref.detach().squeeze(0).clone().requires_grad_(True)

    out_ref, _ = gru(x_ref, h0_ref)
    out_ours, _ = layer(x_ours, h0_ours)

    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out_ours.backward(g)

    cell = layer.cell
    our_W_ih = torch.cat([cell.W_ir.grad, cell.W_iz.grad, cell.W_in.grad], dim=0)
    our_W_hh = torch.cat([cell.W_hr.grad, cell.W_hz.grad, cell.W_hn.grad], dim=0)
    our_b_ih = torch.cat([cell.b_ir.grad, cell.b_iz.grad, cell.b_in.grad])
    our_b_hh = torch.cat([cell.b_hr.grad, cell.b_hz.grad, cell.b_hn.grad])

    for name, ref_t, our_t in [
        ("dx", x_ref.grad, x_ours.grad),
        ("dh_0", h0_ref.grad.squeeze(0), h0_ours.grad),
        ("dW_ih", gru.weight_ih_l0.grad, our_W_ih),
        ("dW_hh", gru.weight_hh_l0.grad, our_W_hh),
        ("db_ih", gru.bias_ih_l0.grad, our_b_ih),
        ("db_hh", gru.bias_hh_l0.grad, our_b_hh),
    ]:
        rel = (ref_t - our_t).abs().max().item() / max(
            ref_t.abs().max().item(), 1e-6
        )
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H})"


# ----------------------------------------------------------------------------
# h_0 != 0 parity tests (Plan 01-04, Task 1; REF-02)
# ----------------------------------------------------------------------------
#
# Fourth and final D-09 test family: random initial hidden state. The first
# three families (forward, h_T, backward) all run with h_0 defaulted to zeros;
# this one explicitly constructs a random h_0 and threads it through both
# implementations. Purpose: a time-loop that special-cases ``h0=None`` (e.g.
# initializing differently on the None-branch than on the explicit-tensor
# branch) can pass the zero-h0 tests by accident — the random-h0 family is
# the only way to surface that. Per REF-02 + CONTEXT D-09.
#
# Per CONTEXT.md Specifics (line 117): this family asserts BOTH ``out`` and
# ``h_T`` in the SAME test — the isolation is "h_0 != 0", not the family
# split. Splitting it into out-only and h_T-only siblings would create 8 grid
# families instead of 4 (D-09 explicitly rejects that — fwd vs h_T family
# split is the relevant axis only for the zero-h0 grids).
#
# Shape contract: GRULayer takes h0 as [B, H]; nn.GRU takes h0 as
# [num_layers=1, B, H]. Build the [1, B, H] tensor first (the nn.GRU shape)
# then ``.squeeze(0)`` to the [B, H] shape we hand to GRULayer. ``squeeze(0)``
# returns a view with shared storage, which is safe here because the test is
# forward-only (no in-place writes, no autograd state shared between the two
# calls).
#
# No autograd in this family: random-h0 backward is implicitly covered by
# Plan 01-03's ``test_layer_backward_matches_nn_gru`` (which sets ``dh_0``
# parity at zero-h0; the bwd graph is the same path regardless of the h0
# value used in the forward). If a random-h0-specific backward bug existed,
# the gradient test would have caught it via the ``dh_0`` slot. No need to
# duplicate the autograd machinery here.
#
# Per-name assertion loop pattern: same idiom as the backward test, but the
# triples list has only two entries (out + h_T). The ``h0=rand`` tag in the
# failure message distinguishes this family from the zero-h0 fwd / h_T
# grids — pytest test id alone says "T=8,B=4,H=64", we want the failure
# context to scream "...AND h0 was random" so the bd-issue title is
# unambiguous on a glance.


@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_with_random_h0_matches_nn_gru(T: int, B: int, H: int) -> None:
    """h_0 != 0 parity over the fast grid.

    Asserts both ``out`` and ``h_T`` in the same test — this is the h_0 != 0
    isolation, not a fwd-vs-h_T family split (D-09 + CONTEXT Specifics).
    Catches initialization bugs where a path special-cases ``h_0=None`` but
    mishandles a real tensor. The ``h0_3d`` (nn.GRU shape) and ``h0_2d``
    (GRULayer shape) views share storage via ``.squeeze(0)`` — both
    implementations see the same initial values, modulo the leading
    ``num_layers`` axis.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    h0_3d = torch.randn(1, B, H)  # nn.GRU shape: [num_layers=1, B, H]
    h0_2d = h0_3d.squeeze(0)      # GRULayer shape: [B, H] — view, shared storage

    out_ref, hT_ref = gru(x, h0_3d)
    out_ours, hT_ours = layer(x, h0_2d)

    for name, ref_t, our_t in [
        ("out", out_ref, out_ours),
        ("h_T", hT_ref.squeeze(0), hT_ours),
    ]:
        rel = (ref_t - our_t).abs().max().item() / max(
            ref_t.abs().max().item(), 1e-6
        )
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H},h0=rand)"


@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_GRID)
def test_layer_with_random_h0_matches_nn_gru_slow(T: int, B: int, H: int) -> None:
    """h_0 != 0 parity across the slow grid (T in {512, 1024}).

    Identical body to the fast variant; gated behind ``@pytest.mark.slow``.
    A long-T accumulation drift in the random-h0 path that the fast grid
    wouldn't catch would surface here — the initial-state influence
    propagates through 512+ recurrent steps, and any subtle asymmetry between
    the two implementations' h0 handling compounds over T.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    h0_3d = torch.randn(1, B, H)
    h0_2d = h0_3d.squeeze(0)

    out_ref, hT_ref = gru(x, h0_3d)
    out_ours, hT_ours = layer(x, h0_2d)

    for name, ref_t, our_t in [
        ("out", out_ref, out_ours),
        ("h_T", hT_ref.squeeze(0), hT_ours),
    ]:
        rel = (ref_t - our_t).abs().max().item() / max(
            ref_t.abs().max().item(), 1e-6
        )
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H},h0=rand)"
