# Phase 1: Reference-path parity vs nn.GRU - Pattern Map

**Mapped:** 2026-05-13
**Files analyzed:** 1 new test file (+ 2 conditional source files for failing-test-induced fixes)
**Analogs found:** 1 exact + 4 role-match / 1 (only `tests/test_layer_parity.py` is in scope to create)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tests/test_layer_parity.py` (new) | test (pytest module) | request-response (build → forward → compare) | `tests/test_parity.py` | exact (parametrize style, weight-copy helper, relative-error assertion, Identity-quantizer pattern) |
| `src/gru_qat/gru_cell.py` (only if a parity test fails) | model | request-response | self (no analog; fix in-place) | n/a — fix scope only |
| `src/gru_qat/gru_layer.py` (only if a parity test fails) | model | request-response (time loop) | self (no analog; fix in-place) | n/a — fix scope only |

Secondary analogs used for cross-cutting patterns:
- `tests/test_triton_diagonal.py` — relative-error idiom, `_make_<kind>_layer` helper style, `set_float32_matmul_precision` preamble.
- `tests/test_triton_monarch.py` — `_build_gi_from_cell` style for module-level underscore helpers; tightest fp32-Identity tolerance (rel < 1e-5).
- `tests/test_butterfly_dispatch.py` — per-batch error inspection (`rel_per_b`) for diagnosing which batch is the outlier.
- `tests/test_qat_smoke.py` — `@pytest.mark.slow` convention.
- `.planning/codebase/TESTING.md` — full conventions reference (TF32 setup, parity tolerance tiers, helper conventions, "no conftest fixtures").

## Pattern Assignments

### `tests/test_layer_parity.py` (new) — test, request-response

**Primary analog:** `tests/test_parity.py` (cell-level parity vs nn.GRUCell at `< 1e-5`).
**Secondary analogs:** `tests/test_triton_diagonal.py` (relative-error idiom, module helpers), `tests/test_qat_smoke.py` (slow marker).

**Module docstring + imports pattern** (copy from `tests/test_parity.py:1-15`):
```python
"""Layer-parity tests — Phase 1 audit.

Validates that GRULayer with all quantizers set to Identity (use_triton=False,
dense, no structure_hidden) matches torch.nn.GRU(num_layers=1, bidirectional=
False, batch_first=False) on (out, h_T) forward, on the six weight gradients
plus dx and dh_0 backward, and on h_0 != 0 initial state. Tolerance < 1e-4
across a T x B x H = 5 x 3 x 5 = 75-combo grid.

If this fails, the unroll math (or its time-loop orchestration) is wrong and
every later phase's reference is contaminated.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS
```

Notes:
- `from __future__ import annotations` is required (CONVENTIONS.md "Type Annotations").
- No `pytest.importorskip` — reference path is CPU-runnable (see CONTEXT D-07/D-08).
- No `# noqa: E402` — no module-level skip means no out-of-order imports to suppress.

**TF32 / precision preamble** (override from `tests/test_triton_diagonal.py:108`; use `"highest"` not `"high"` per CONTEXT D-07):
```python
# Module-level: we audit math, not TF32. "highest" forces ieee-754 fp32 matmul,
# so the only drift is algorithm, not arithmetic mode. Diverges from the
# Triton kernel tests (which use "high" to test the kernel under realistic
# conditions) — that's intentional.
torch.set_float32_matmul_precision("highest")
```
And per-test:
```python
torch.manual_seed(0)
```
(See `tests/test_triton_diagonal.py:78,107,128,168,201,251,277,303` for the per-test seed convention; we keep the seed at the top of each test, not at module scope, so each test is independent — TESTING.md "Bench-Style Smoke Tests vs. Correctness Tests".)

**Translation helper pattern** (model on `tests/test_parity.py:18-44` `_copy_weights`, but invert direction: cell → nn.GRU is the primary, plus inverse for round-trip):

```python
def _translate_cell_to_nn_gru(layer: GRULayer) -> nn.GRU:
    """Build a torch.nn.GRU(num_layers=1, bidirectional=False, batch_first=False)
    whose weights and biases reproduce ``layer`` exactly.

    Per PyTorch docs (https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
    gate order is (r, z, n) for both sides, matching gru_cell.py's W_ir/W_iz/W_in
    family. Translation is just torch.cat along axis 0:

        weight_ih_l0 = cat([W_ir, W_iz, W_in], dim=0)        # [3H, IN]
        weight_hh_l0 = cat([W_hr, W_hz, W_hn], dim=0)        # [3H, H]
        bias_ih_l0   = cat([b_ir, b_iz, b_in])               # [3H]
        bias_hh_l0   = cat([b_hr, b_hz, b_hn])               # [3H]

    The n-gate asymmetry (r_t * (W_hn h + b_hn) inside the tanh) is preserved
    by this layout because both sides apply r_t identically — see
    src/gru_qat/gru_cell.py:1-15 module docstring.
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
    """Inverse of _translate_cell_to_nn_gru — used only by the round-trip smoke
    test. Direct mirror of tests/test_parity.py:18-44 _copy_weights, but at
    the layer level."""
    layer = GRULayer(
        gru.input_size, gru.hidden_size, recipe=PRESETS["fp32"],
        batch_first=False, gate_layout="split",
    )
    cell = layer.cell
    Wir, Wiz, Win = gru.weight_ih_l0.chunk(3, dim=0)
    Whr, Whz, Whn = gru.weight_hh_l0.chunk(3, dim=0)
    bir, biz, bin_ = gru.bias_ih_l0.chunk(3)
    bhr, bhz, bhn = gru.bias_hh_l0.chunk(3)
    with torch.no_grad():
        cell.W_ir.copy_(Wir); cell.W_iz.copy_(Wiz); cell.W_in.copy_(Win)
        cell.W_hr.copy_(Whr); cell.W_hz.copy_(Whz); cell.W_hn.copy_(Whn)
        cell.b_ir.copy_(bir); cell.b_iz.copy_(biz); cell.b_in.copy_(bin_)
        cell.b_hr.copy_(bhr); cell.b_hz.copy_(bhz); cell.b_hn.copy_(bhn)
    return layer
```

Notes:
- Underscore-prefixed, module-level, fully typed (CONVENTIONS + TESTING.md "No fixtures via conftest").
- Built directly on top of `_copy_weights` at `tests/test_parity.py:18-44` (chunk + copy_).
- D-03 / D-05: the docstring carries the PyTorch GRU formula link.

**fp32-Identity layer-builder pattern** (mirror of `_make_diagonal_layer` at `tests/test_triton_diagonal.py:35-48`, but dense + `use_triton=False`):

```python
def _make_dense_fp32_layer(input_size: int, hidden_size: int) -> GRULayer:
    """fp32 dense reference layer: Identity quantizers, no structure, no Triton.
    This is the path Phase 1 audits."""
    return GRULayer(
        input_size, hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
        # no structure_input / structure_hidden -> dense
        # use_triton defaults to "auto" -> resolves to False because not eligible
    )
```

Reference for `PRESETS["fp32"]`: `src/gru_qat/quantizers.py:284-289` — `weight=bits=32`, `input_act=bits=32`, `hidden=bits=32` (all Identity).

**Parametrize pattern for the 75-combo grid** (model on `tests/test_parity.py:47-54`, but with the larger grid + slow split per D-08):

```python
# Fast grid: T in {1, 8, 64} (always runs).
FAST_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]
# Slow grid: T in {512, 1024} (pytest -m slow).
SLOW_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]


@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_forward_matches_nn_gru(T: int, B: int, H: int) -> None:
    ...
```

Plus a slow-marked sibling:
```python
@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_GRID)
def test_layer_forward_matches_nn_gru_slow(T: int, B: int, H: int) -> None:
    ...
```

(Slow-marker convention is `tests/test_qat_smoke.py:88` `@pytest.mark.slow`. Registered in `pyproject.toml:44-45` per TESTING.md.)

**Core parity-test body pattern** (compose `tests/test_parity.py:55-70` cell-style with the relative-error idiom from `tests/test_triton_diagonal.py:91-93`):

```python
@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_forward_matches_nn_gru(T: int, B: int, H: int) -> None:
    torch.manual_seed(0)
    IN = max(H, 1)  # keep input_size tied to H so the grid stays compact

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    h0 = torch.zeros(1, B, H)  # nn.GRU expects [num_layers, B, H]

    out_ref, hT_ref = gru(x, h0)
    out_ours, hT_ours = layer(x, h0.squeeze(0))

    # Relative-error idiom — tests/test_triton_diagonal.py:120-121
    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"out rel diff {rel:.4e} (T={T},B={B},H={H})"

    max_diff_h = (hT_ref.squeeze(0) - hT_ours).abs().max().item()
    rel_h = max_diff_h / max(hT_ref.abs().max().item(), 1e-6)
    assert rel_h < 1e-4, f"h_T rel diff {rel_h:.4e} (T={T},B={B},H={H})"
```

Notes:
- The relative-error idiom is preferred (CONTEXT "Claude's Discretion"): clearer failure messages than `torch.allclose`, with the offending shape in the assertion.
- `1e-6` floor on the denominator matches `tests/test_triton_diagonal.py:120` exactly.

**Backward / gradient parity pattern** (no direct in-repo analog at the layer level, but compose the autograd-grad pattern from `tests/test_triton_diagonal.py:299-339`):

```python
@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_backward_matches_nn_gru(T: int, B: int, H: int) -> None:
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x_ref = torch.randn(T, B, IN, requires_grad=True)
    h0_ref = torch.zeros(1, B, H, requires_grad=True)
    out_ref, _ = gru(x_ref, h0_ref)
    g = torch.randn_like(out_ref)
    out_ref.backward(g)

    x_ours = x_ref.detach().clone().requires_grad_(True)
    h0_ours = h0_ref.detach().squeeze(0).clone().requires_grad_(True)
    out_ours, _ = layer(x_ours, h0_ours)
    out_ours.backward(g)

    # dx and dh_0 ----------------------------------------------------------
    for name, ref_t, our_t in [
        ("dx",   x_ref.grad,             x_ours.grad),
        ("dh_0", h0_ref.grad.squeeze(0), h0_ours.grad),
    ]:
        rel = (ref_t - our_t).abs().max().item() / max(ref_t.abs().max().item(), 1e-6)
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H})"

    # Weight grads via cat-stack translation -------------------------------
    cell = layer.cell
    our_W_ih = torch.cat([cell.W_ir.grad, cell.W_iz.grad, cell.W_in.grad], dim=0)
    our_W_hh = torch.cat([cell.W_hr.grad, cell.W_hz.grad, cell.W_hn.grad], dim=0)
    our_b_ih = torch.cat([cell.b_ir.grad, cell.b_iz.grad, cell.b_in.grad])
    our_b_hh = torch.cat([cell.b_hr.grad, cell.b_hz.grad, cell.b_hn.grad])

    for name, ref_t, our_t in [
        ("dW_ih", gru.weight_ih_l0.grad, our_W_ih),
        ("dW_hh", gru.weight_hh_l0.grad, our_W_hh),
        ("db_ih", gru.bias_ih_l0.grad,   our_b_ih),
        ("db_hh", gru.bias_hh_l0.grad,   our_b_hh),
    ]:
        rel = (ref_t - our_t).abs().max().item() / max(ref_t.abs().max().item(), 1e-6)
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H})"
```

(Six weight gradients + bias + dx + dh_0 per CONTEXT spec; mirrors the per-param loop at `tests/test_triton_diagonal.py:325-339`.)

**h_T parity pattern** (split from forward parity per CONTEXT D-09 — failure messages need to point at the right family):

```python
@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_h_T_matches_nn_gru(T: int, B: int, H: int) -> None:
    """h_T parity: distinct from out parity so a final-step bug surfaces alone."""
    ... # same body as forward test but assert only h_T, not out
```

**h_0 != 0 parity pattern** (random initial state — fourth test family per CONTEXT D-09):

```python
@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_with_random_h0_matches_nn_gru(T: int, B: int, H: int) -> None:
    torch.manual_seed(0)
    IN = max(H, 1)
    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    h0_3d = torch.randn(1, B, H)          # nn.GRU shape
    h0_2d = h0_3d.squeeze(0)              # GRULayer shape

    out_ref, hT_ref = gru(x, h0_3d)
    out_ours, hT_ours = layer(x, h0_2d)

    # Both out and h_T checks in one test — this is the h_0 != 0 isolation,
    # not the family-split.
    for name, ref_t, our_t in [("out", out_ref, out_ours), ("h_T", hT_ref.squeeze(0), hT_ours)]:
        rel = (ref_t - our_t).abs().max().item() / max(ref_t.abs().max().item(), 1e-6)
        assert rel < 1e-4, f"{name} rel diff {rel:.4e} (T={T},B={B},H={H},h0=rand)"
```

**Micro-test pattern** (CONTEXT D-04 — three focused gate-ordering / n-gate-asymmetry checks; NOT parametrized; run before the grid). Loose analog: `tests/test_parity.py:73-103` `test_cell_with_zero_hidden`, `test_cell_with_zero_input`, `test_cell_with_large_magnitude` (one-shot edge-case smoke tests).

```python
def test_gate_order_r_only() -> None:
    """Set W_ir=ones, W_iz=W_in=zeros; nn.GRU and ours must agree that only
    the r-gate's sigmoid fires. If the cell's gate order is wrong, the grid
    tests will still pass with the translation helper compensating — this
    micro-test isolates the gate-order assumption."""
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        cell.W_ir.fill_(1.0); cell.W_iz.zero_(); cell.W_in.zero_()
        cell.W_hr.zero_(); cell.W_hz.zero_(); cell.W_hn.zero_()
        for b in (cell.b_ir, cell.b_iz, cell.b_in, cell.b_hr, cell.b_hz, cell.b_hn):
            b.zero_()
    gru = _translate_cell_to_nn_gru(layer)
    x = torch.randn(1, 2, 4); h0 = torch.zeros(2, 4)
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)
    rel = (out_ref - out_ours).abs().max().item() / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4


def test_gate_order_z_only() -> None:
    ...  # same template, swap W_iz <-> W_ir


def test_n_gate_asymmetry() -> None:
    """Force r ~ 0 by setting W_ir and b_ir to large-negative; the n-gate
    must reduce to tanh(W_in x + b_in). Both nn.GRU and our cell must agree
    on the asymmetric placement of r inside the tanh — see
    src/gru_qat/gru_cell.py:11-14 module docstring."""
    ...
```

**Round-trip smoke test** (CONTEXT D-01 — one-shot, not parametrized):

```python
def test_round_trip_nn_gru_to_cell() -> None:
    """Build an nn.GRU first, copy its weights into a fresh GRULayer via
    the inverse helper, then assert layer outputs match. Catches bugs in
    _translate_nn_gru_to_cell itself."""
    torch.manual_seed(0)
    gru = nn.GRU(8, 16, num_layers=1, bidirectional=False, batch_first=False)
    layer = _translate_nn_gru_to_cell(gru)
    x = torch.randn(7, 4, 8); h0 = torch.zeros(1, 4, 16)
    out_ref, hT_ref = gru(x, h0)
    out_ours, hT_ours = layer(x, h0.squeeze(0))
    assert (out_ref - out_ours).abs().max().item() < 1e-4
    assert (hT_ref.squeeze(0) - hT_ours).abs().max().item() < 1e-4
```

## Shared Patterns

### Module-level underscore helpers (no `conftest.py`)
**Source:** TESTING.md "No fixtures via conftest" (line 198-200); convention enforced across `tests/test_triton_diagonal.py:35,51`, `tests/test_triton_monarch.py:35,54`, `tests/test_butterfly_dispatch.py:36`, `tests/test_parity.py:18`.
**Apply to:** All helpers in `tests/test_layer_parity.py` (`_translate_*`, `_make_dense_fp32_layer`). Module-level, single underscore prefix, fully typed.

### Relative-error assertion idiom
**Source:** `tests/test_triton_diagonal.py:120-121`, `tests/test_triton_monarch.py:99-101`.
```python
max_diff = (ref - tri).abs().max().item()
rel = max_diff / max(ref.abs().max().item(), 1e-6)
assert rel < TOL, f"<name> rel diff {rel:.4e}"
```
**Apply to:** All four parity test families (forward, h_T, backward, h_0≠0) and the three micro-tests. The `1e-6` floor on the denominator is non-negotiable — prevents division by near-zero on degenerate grids.

### Per-batch error inspection (use only on failure-debugging passes)
**Source:** `tests/test_butterfly_dispatch.py:206-214`.
```python
rel_per_b = (
    (tri_out - ref_out).abs().amax(dim=(0, 2))
    / ref_out.abs().amax().clamp(min=1e-6)
)
assert rel_per_b.max().item() < TOL, (
    f"worst batch={rel_per_b.argmax().item()}, rel by batch={rel_per_b.tolist()}"
)
```
**Apply to:** Not used in the default tests — keep the simple max-rel idiom above. If a grid test fails, the *fix-commit* author may copy this pattern into the failing test's assertion to localize the offending batch (per CONTEXT D-10 commit-A discipline).

### `torch.manual_seed(0)` per test
**Source:** TESTING.md "Bench-Style Smoke Tests vs. Correctness Tests" (line 136); pattern at `tests/test_parity.py:58`, `tests/test_triton_diagonal.py:78,107,128,168,201,251,277,303`.
**Apply to:** Every test in the file that uses randomness. Reset at the top of the test body, not at module scope, so tests are independent.

### `torch.set_float32_matmul_precision`
**Source for default ("high"):** `tests/test_triton_diagonal.py:108`, all kernel-parity tests.
**Override for Phase 1 ("highest"):** CONTEXT D-07. Module-level call near the imports, not per-test — we're auditing math, so TF32 is off everywhere in this file.
**Apply to:** Module top of `tests/test_layer_parity.py` only. Don't touch global state in other files.

### `@pytest.mark.slow` for long-T cases
**Source:** `tests/test_qat_smoke.py:88` (registered in `pyproject.toml:44-45`).
**Apply to:** Each of the four test families gets a `_slow` sibling parametrized over `SLOW_GRID` (T ∈ {512, 1024}). Per CONTEXT D-08, fast grid runs everywhere; slow runs under `pytest -m slow`.

### Two-commit failing-test-before-fix discipline (CONTEXT D-10..12)
**Source:** No in-codebase analog — this is a workflow rule, not a code pattern. Documented in CONTEXT.
**Apply to:** Any parity failure discovered during Phase 1.
1. Commit A: new failing test only, no `src/` changes. Capture `pytest --tb=short` tail.
2. `bd create` per finding, title = test function name; `bd update <id> --notes <pytest-tail>`.
3. Commit B: fix in `src/gru_qat/gru_cell.py` or `src/gru_qat/gru_layer.py` (no test changes), same test now passes. `bd close <id>` after CI green.
4. **Never** use `@pytest.mark.xfail` — silent in `pytest -q`, defeats the audit signal (D-12).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (Cell ↔ nn.GRU translation helper at the layer level) | helper | — | No existing layer-level analog; we extend the cell-level `_copy_weights` pattern from `tests/test_parity.py:18-44` by chunking/concatenating across the (r,z,n) gates. The pattern is novel for this repo but mechanically identical to the cell case. |
| `nn.GRU` (used as ground truth) | external dep | — | Not yet referenced anywhere in `src/` or `tests/` (only `nn.GRUCell` is, at `tests/test_parity.py`). Phase 1 introduces the first use. |

No `src/` files are listed as "no analog" because **Phase 1 plans no speculative `src/` edits** — only fix-commits land changes, and a fix's analog is the file it lives in (CONTEXT D-10).

## Metadata

**Analog search scope:**
- `tests/` (all test files) — read `test_parity.py`, `test_triton_diagonal.py`, `test_triton_monarch.py`, `test_butterfly_dispatch.py`, `test_qat_smoke.py`.
- `src/gru_qat/gru_cell.py` (weight names + module docstring on n-gate asymmetry).
- `src/gru_qat/gru_layer.py` (forward signature, fast-dispatch eligibility).
- `src/gru_qat/quantizers.py:284-289` (`PRESETS["fp32"]` definition).
- `.planning/codebase/TESTING.md` (full conventions reference).

**Files scanned:** 9 (5 test files, 3 src modules, 1 conventions doc).
**Pattern extraction date:** 2026-05-13.
