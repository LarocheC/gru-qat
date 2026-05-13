# Phase 2: Triton fast-path parity vs reference - Pattern Map

**Mapped:** 2026-05-13
**Files analyzed:** 4 new strict-tier test files + 4 existing files (realistic-tier tightening candidates) + 2 special regression tests + 1 static canary
**Analogs found:** 4 exact (TF32-tier siblings) / 4 + 1 exact (regression analog at `tests/test_triton_scan.py:202`)

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `tests/test_triton_scan_strict.py` (new) | test (pytest module) | request-response (build → kernel call → compare) | `tests/test_triton_scan.py` | exact — TF32 sibling, same `_ref_layer` + `gru_scan_forward` / `gru_scan` invocation shape |
| `tests/test_triton_diagonal_strict.py` (new) | test (pytest module) | request-response (gi → Triton diag fwd/bwd → PyTorch diag ref) | `tests/test_triton_diagonal.py` | exact — Stage-B/C pattern already < 1e-4 at line 121, < 1e-3 at line 194; lift to < 1e-5 under `'highest'` |
| `tests/test_triton_monarch_strict.py` (new) | test (pytest module) | request-response (gi+`Wh_struct` → Triton monarch → PyTorch monarch) | `tests/test_triton_monarch.py` | exact — already has `nblocks` parametrize; strict tier swaps TF32 'high' for 'highest' |
| `tests/test_triton_butterfly_strict.py` (new) | test (pytest module) | request-response (gi+twiddles → Triton butterfly → CUDA-op per-step) | `tests/test_butterfly_dispatch.py` | exact — same `extract_butterfly_twiddles` flow; strict tier targets `gru_scan_butterfly_forward_triton` / `gru_scan_butterfly_backward_triton` |
| `tests/test_triton_scan_strict.py` → `test_autotune_dWh_dbh_zero_init_across_configs` (TRI-05, D-23) | regression test | request-response (force >1 autotune config, compare across runs) | `tests/test_triton_scan.py:202-215` (existing slab-zero regression, single-config) | role-match — same target bug, different mechanism (multi-config rotation vs single-config) |
| `tests/test_triton_scan_strict.py` → `test_persistent_kernel_deterministic` (TRI-06, D-24) | regression test | request-response (50× same input → bit-equal output) | none in tree — first cross-CTA determinism test | no analog (see "No Analog Found") |
| `tests/test_triton_scan_strict.py` → static `.cv` grep canary (D-25) | regression test | file-I/O (read source, regex) | none in tree — first static-source-grep test | no analog (see "No Analog Found") |
| `tests/test_triton_scan.py` (modify; D-13 tightening) | test (existing) | request-response | self | n/a — tolerance constants only |
| `tests/test_triton_diagonal.py` (modify; D-13 tightening) | test (existing) | request-response | self | n/a — tolerance constants only |
| `tests/test_triton_monarch.py` (modify; D-13 tightening) | test (existing) | request-response | self | n/a — tolerance constants only |
| `tests/test_butterfly_dispatch.py` (modify; D-13 tightening) | test (existing) | request-response | self | n/a — tolerance constants only |

Secondary analogs cited for shared / cross-cutting patterns:

- `tests/test_layer_parity.py` (Phase 1 output) — the `'highest'` module-level precision preamble and `FAST_GRID` / `SLOW_GRID` split idiom. Strict-tier files copy the precision preamble; the grid shape is per-kernel (D-16) not shared.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-PATTERNS.md` — Phase 1 pattern map; per-kernel relative-error idiom with `1e-6` floor, `torch.manual_seed(0)` per test, two-commit failing-test-before-fix discipline (carried as D-27).
- `.planning/codebase/TESTING.md` — `cuda_only` per-file definition, `pytest.importorskip("triton")` + `# noqa: E402`, "no conftest fixtures", relative-error idiom.

---

## Pattern Assignments

### `tests/test_triton_scan_strict.py` (new) — strict-tier dense kernel parity

**Primary analog:** `tests/test_triton_scan.py` (entire file).
**Secondary analog:** `tests/test_layer_parity.py:33` for the `set_float32_matmul_precision('highest')` preamble.

**Imports + module preamble** (model on `tests/test_triton_scan.py:1-23`, but with the strict-tier precision flag from `tests/test_layer_parity.py:28-33`):

```python
"""Strict-tier parity tests for the dense Triton scan kernel — Phase 2 audit.

Validates ``gru_scan`` / ``gru_scan_persistent`` (and their fwd/bwd helpers)
against the Phase 1 reference path (``GRULayer(use_triton=False, dense,
Identity quantizers)``) at the strict tier:

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 1e-5     # absolute, not relative

Diverges from ``tests/test_triton_scan.py`` (which runs under 'high' /
TF32 with looser bounds) — that's the realistic-deployment tier. Both
files coexist; this file does NOT loosen the existing one.

Also hosts the named regression tests for:
  - TRI-05 (autotune dWh/dbh slab init across multiple configs)
  - TRI-06 (cross-CTA determinism — 50-run bit-identical assertion)
  - D-25 (static .cv cache-modifier grep canary)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

triton = pytest.importorskip("triton")

from gru_qat.gru_layer import GRULayer  # noqa: E402
from gru_qat.quantizers import QuantizerConfig, QuantRecipe  # noqa: E402
from gru_qat.triton_kernels.scan import (  # noqa: E402
    gru_scan,
    gru_scan_forward,
    gru_scan_forward_persistent,
    gru_scan_backward_persistent,
    gru_scan_persistent,
    _gru_scan_backward_pytorch,
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (test_triton_scan.py) uses 'high' to exercise the kernel under
# deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)
```

**`_ref_layer` helper** — **copy verbatim from `tests/test_triton_scan.py:30-44`**. Same fp32-Identity fused+pre_batch GRULayer (the kernel takes the post-input-projection `gi` directly, so reference shape matches). Per D-18, import or duplicate; duplicate here because it's only ~15 lines.

**Shape grid** (per D-16 dense):

```python
# T x B x H grid. Fast set runs on every `pytest -q`; slow set under -m slow.
FAST_DENSE_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 27 cases
SLOW_DENSE_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases
```

**Core strict-tier fwd test pattern** (compose `tests/test_triton_scan.py:115-139` for the kernel-call body with the absolute-error idiom per CONTEXT "Established Patterns" callout: strict tier uses `< 1e-5 abs`, not the realistic-tier relative form):

```python
@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)
def test_scan_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """gru_scan_forward must match the reference GRULayer to < 1e-5 absolute
    under 'highest' precision. fp32 IEEE matmul on both sides → algorithmic
    drift only."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)
        assert w.Wh_cat is not None and w.bh_cat is not None
        triton_out = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

    max_diff = (ref_out - triton_out).abs().max().item()
    # Strict tier: absolute error under IEEE fp32 matmul.
    # Realistic-tier sibling (tests/test_triton_scan.py:139) uses < 5e-3
    # under TF32 — that's correct for its regime; not loosened by us.
    assert max_diff < 1e-5, f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
```

(Slow sibling `@pytest.mark.slow` over `SLOW_DENSE_GRID`. Same body. Same naming convention as `tests/test_layer_parity.py:365-366`.)

**Core strict-tier bwd test pattern** (compose `tests/test_triton_scan.py:144-215` autograd-through-`gru_scan` body with absolute-error gates):

```python
@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)
def test_scan_bwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton autograd gradients must match PyTorch autograd through the
    reference layer to < 1e-5 absolute on x, h0, Wh_cat, bh_cat."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H

    ref_layer = _ref_layer(IN, H).to(device)
    x = torch.randn(T, B, IN, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref_layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    w = ref_layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
    out.float().pow(2).sum().backward()

    ref_dWh_cat = torch.cat(
        [ref_layer.cell.W_hr.grad, ref_layer.cell.W_hz.grad, ref_layer.cell.W_hn.grad],
        dim=0,
    )
    ref_dbh_cat = torch.cat(
        [ref_layer.cell.b_hr.grad, ref_layer.cell.b_hz.grad, ref_layer.cell.b_hn.grad],
    )
    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
        ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
        ("bh_cat", ref_dbh_cat, bh_cat.grad),
    ]:
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 1e-5, f"{name} max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
```

**TRI-05 regression (autotune dWh/dbh)** — analog: `tests/test_triton_scan.py:202-215` (single-config slab-zero regression). Phase 2's variant forces multiple autotune configs to be evaluated in one process:

```python
@cuda_only
def test_autotune_dWh_dbh_zero_init_across_configs() -> None:
    """Regression for TRI-05 (commit c001a8a): the autotuned backward
    kernel allocates dWh/dbh partial accumulators per program and must
    zero them on entry. Pre-fix, a stale slab from autotune-config A
    leaked into config B's accumulator, producing dWh/dbh off by ~O(0.1).

    Existing test at tests/test_triton_scan.py:202-215 catches the bug
    on a single autotune config; this variant forces multiple candidate
    configs to run in one process by clearing the JIT cache between two
    calls with different (T, B) inputs (autotune key=['T', 'B'] —
    see src/gru_qat/triton_kernels/scan.py:732). If the slab-zero fix
    regresses, the second run's gradients diverge from reference.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Two shapes that hit different autotune buckets per the key=['T','B']
    # config at src/gru_qat/triton_kernels/scan.py:732,893. Both must
    # match reference to < 1e-5 abs (strict tier).
    for T, B, H in [(16, 16, 64), (32, 32, 64)]:
        # ... build gi/h0/Wh/bh, run gru_scan + reference, compare grads
        # at < 1e-5 abs. If the second iteration diverges and the first
        # passed, the autotune-config-rotation bug has returned.
```

Note: the strict tier means the assertion is **absolute < 1e-5**, not the relative `< 1e-1` at `tests/test_triton_scan.py:215`. This is a tighter regression bound enabled by `'highest'`.

**TRI-06 regression (cross-CTA determinism)** — no in-repo analog (see "No Analog Found"). Pattern is novel for the repo. Skeleton:

```python
@cuda_only
def test_persistent_kernel_deterministic() -> None:
    """Regression for TRI-06: the persistent fwd/bwd kernels use
    atomic_add(sem='release') + atomic_add(0, sem='acquire') for cross-CTA
    visibility — see src/gru_qat/triton_kernels/scan.py:184-203 and the
    comment block warning that ``cache_modifier='.cv'`` is NOT a fence
    substitute. The pre-fix code (relaxed atomics + .cv load) produced
    output that was *mostly* correct but drifted by ~0.2 absolute on
    some [t>=1, batch, hidden] cells depending on CTA schedule order —
    i.e. non-deterministic.

    This test runs gru_scan_persistent 50 times on bit-identical inputs
    and asserts torch.equal across all 50 outputs (and h_T). If any
    run diverges, the release/acquire pattern has regressed.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, B, H = 64, 16, 128

    gi = torch.randn(T, B, 3 * H, device=device).contiguous()
    h0 = torch.randn(B, H, device=device).contiguous()
    Wh = torch.randn(3 * H, H, device=device).contiguous() * 0.1
    bh = torch.randn(3 * H, device=device).contiguous() * 0.1

    out0 = gru_scan_persistent(gi, h0, Wh, bh)
    for i in range(1, 50):
        out_i = gru_scan_persistent(gi, h0, Wh, bh)
        assert torch.equal(out0, out_i), (
            f"persistent kernel run {i} diverged from run 0 — "
            f"cross-CTA fence may have regressed. "
            f"max abs diff = {(out0 - out_i).abs().max().item():.4e}"
        )
```

Notes:
- `torch.equal` (not allclose) is the strict-tier determinism gate — matches the `TESTING.md:22` rule that "tensor equality uses `torch.equal` (strict) or `torch.allclose(...)`."
- Inputs allocated outside the loop; tensors are NOT re-randomized between runs (CTA scheduling is the only varying factor).
- Use `gru_scan_persistent` (not `gru_scan`) because the persistent kernel is the one with the cross-CTA barrier — the autotune path doesn't have inter-CTA dependencies.

**D-25 static `.cv` grep canary** — no in-repo analog (see "No Analog Found"). Skeleton:

```python
def test_no_cv_cache_modifier_in_scan_source() -> None:
    """Static canary: ``cache_modifier='.cv'`` must NOT appear inside any
    triton.jit kernel body in src/gru_qat/triton_kernels/scan*.py.

    The .cv cache modifier was historically misused as a cross-CTA fence
    substitute; see src/gru_qat/triton_kernels/scan.py:192-199. The
    fix uses atomic_add(sem='release'/'acquire'); this test guards
    against re-introduction.

    Documentation/comment mentions of '.cv' are fine (they explain why
    not to use it). Only direct uses inside @triton.jit-decorated
    function bodies are forbidden — see DEVELOPMENT.md:131-143 for the
    full rationale.

    NOTE: cache_modifier='.cv' currently appears in the BACKWARD kernel
    (scan.py:431, scan.py:625) as the dh_acc fresh-load mechanism. The
    static check must scope to forward-kernel context only, OR — per
    CONTEXT D-25 — assert the count matches the known-safe call sites
    (2 in the bwd kernel) and fail if a 3rd appears. Planner: clarify
    with user whether scope is "no .cv anywhere" or "no new .cv beyond
    the documented bwd dh_acc sites."
    """
    import pathlib
    src_dir = pathlib.Path(__file__).parent.parent / "src" / "gru_qat" / "triton_kernels"
    forbidden = 'cache_modifier=".cv"'
    # ... walk scan*.py, count occurrences, baseline = 2 known bwd sites
```

Notes:
- Uses pathlib (no shell out, no subprocess) — matches CONVENTIONS.md type-discipline preference for pure Python.
- The CONTEXT D-25 wording says "does NOT appear inside `src/gru_qat/triton_kernels/scan*.py`" — but the source currently has 2 legitimate uses in the bwd kernel. **Planner decision needed**: either (a) tighten the canary to assert count == 2 with comments documenting which sites, or (b) replace the bwd `.cv` loads with the release/acquire pattern first (separate bd issue). Recommend (a) as the strict-tier guard, leaving (b) as a deferred refactor.

---

### `tests/test_triton_diagonal_strict.py` (new) — strict-tier diagonal kernel parity

**Primary analog:** `tests/test_triton_diagonal.py` (entire file).

**Imports + preamble** — copy `tests/test_triton_diagonal.py:11-32` verbatim, but:
- Replace `tests/test_triton_diagonal.py:108` per-test `torch.set_float32_matmul_precision("high")` with a module-level `torch.set_float32_matmul_precision("highest")` near the imports (see strict-tier preamble in scan_strict above).
- Module docstring documents: "Strict-tier sibling of test_triton_diagonal.py. Stage B and Stage C at < 1e-5 abs under 'highest' precision. The realistic-tier file's Stage A (cell-vs-PyTorch-reference) is locked at < 1e-5 per `tests/test_triton_diagonal.py:93` and not duplicated here — that's algebraic equality between two PyTorch paths, already strict-tier."

**`_make_diagonal_layer` and `_build_gi_from_cell` helpers** — **copy verbatim from `tests/test_triton_diagonal.py:35-67`**. Per D-18, duplicate (small, < 30 LOC).

**Shape grid** (per D-16 diagonal; note H ∈ {1, 2, 8, 64, 512} — tiny H is supported by the diagonal kernel because there's no matmul on the hidden side):

```python
FAST_DIAG_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]  # 45 cases
SLOW_DIAG_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]  # 30 cases
```

**Core strict-tier fwd / bwd test bodies** — direct mirrors of `tests/test_triton_diagonal.py:102-121` (fwd) and `tests/test_triton_diagonal.py:165-194` (bwd), but:
1. Drop the per-test `set_float32_matmul_precision("high")` (covered at module scope).
2. Replace the relative-error idiom (`max_diff / max(ref.abs().max().item(), 1e-6)`) with **absolute** `(ref - tri).abs().max().item() < 1e-5`. This is the strict tier's signature divergence from realistic tier per CONTEXT "Established Patterns" callout.

```python
@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DIAG_GRID)
def test_diagonal_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton diagonal forward must match the PyTorch reference to < 1e-5
    absolute under 'highest'. Diagonal has no hidden-side matmul, so
    arithmetic drift comes only from the per-step nonlinearities — both
    paths should be bit-stable in fp32 IEEE."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    ref = gru_scan_diagonal_forward_pytorch(gi, h0, Wh_diag, bh_cat)
    tri = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-5, f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
```

(Per `tests/test_triton_diagonal.py:106` docstring: "Diagonal recurrence has no matmul on the hidden side, so the only floating-point noise comes from the input-side gi (already provided as a tensor here) and the per-step nonlinearities — both bit-stable. Tight tolerance." This justifies < 1e-5 abs at strict tier — the realistic-tier file already passes < 1e-4 at line 121.)

---

### `tests/test_triton_monarch_strict.py` (new) — strict-tier monarch kernel parity

**Primary analog:** `tests/test_triton_monarch.py` (entire file).

**Imports + preamble** — copy `tests/test_triton_monarch.py:1-32` verbatim, but module-level `set_float32_matmul_precision("highest")`. Module docstring same template as diagonal_strict.

**`_make_monarch_layer` and `_build_gi_from_cell` helpers** — **copy verbatim from `tests/test_triton_monarch.py:35-77`**.

**Shape grid** (per D-16 monarch — note `nblocks ∈ {2, 4, 8}`):

```python
FAST_MONARCH_GRID = [
    (T, B, H, nblocks)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0  # Monarch requires H divisible by nblocks
]  # ~81 cases (some filtered)
SLOW_MONARCH_GRID = [
    (T, B, H, nblocks)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0
]  # ~54 cases
```

(D-16 says ~27 fast / 27 slow; the actual count depends on the planner's grid pruning. The exact `parametrize("T,B,H,nblocks", FAST_MONARCH_GRID)` decision is Claude's discretion per CONTEXT.)

**Core strict-tier fwd / bwd bodies** — mirror `tests/test_triton_monarch.py:106-127` (fwd) and `tests/test_triton_monarch.py:215-248` (bwd), with the same TF32 → strict transformation as diagonal_strict. Monarch's hidden-side matmul means strict tier WILL stress the kernel's tl.dot reduction order; per D-14, if any (T, B, H, nblocks) combo fails < 1e-5 abs that's a finding (separate bd issue, no `@pytest.mark.xfail` per D-27).

---

### `tests/test_triton_butterfly_strict.py` (new) — strict-tier butterfly kernel parity

**Primary analog:** `tests/test_butterfly_dispatch.py` (entire file).

**Imports + preamble** — copy `tests/test_butterfly_dispatch.py:1-33` verbatim, but module-level `set_float32_matmul_precision("highest")`. Includes the file-local `cuda_only` definition at `tests/test_butterfly_dispatch.py:31-33` ("butterfly dispatch path is CUDA-only" — preserve the reason string variant).

**`_make_layer` helper** — **copy verbatim from `tests/test_butterfly_dispatch.py:36-48`**. (Note: this file's helper takes `hidden_bits` instead of `nblocks`; for strict tier we always pass `hidden_bits=32` since Phase 2 is fp32-Identity per CONTEXT.)

**Shape grid** (per D-16 butterfly — H must be power of 2):

```python
FAST_BFLY_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)  # all powers of 2; D-16 says these
]  # 27 cases
SLOW_BFLY_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases
```

**Module docstring (top, after imports)** — Phase 2 references the existing OOB regression but does NOT duplicate per D-22:

```python
"""Strict-tier parity tests for the Butterfly Triton kernel — Phase 2 audit.

[... usual strict-tier blurb ...]

Note: the per-program scratch-OOB regression for the butterfly fwd kernel
(commit d8218d4, finding TRI-04) is covered at
``tests/test_butterfly_dispatch.py:164`` (test_butterfly_triton_forward_scratch_oob_regression).
That test runs at (T=16, B=32, H=512) under TF32 with < 5e-2 rel; this strict
file does NOT duplicate it (D-22). Phase-exit verification confirms that test
still passes; if it regresses, the bug surfaces there, not here.
"""
```

**Core strict-tier fwd test** — mirror `tests/test_butterfly_dispatch.py:133-160` (the `test_butterfly_triton_forward_matches_per_step` body), but:
1. Replace `< 2e-2` rel with `< 1e-5` abs.
2. Replace the per-test `set_float32_matmul_precision("high")` with module-scope `"highest"`.
3. Reference path stays `gru_scan_butterfly` (CUDA-op per-step path) — butterfly has no pure-PyTorch reference distinct from the kernel under test, so the CUDA-op path serves as ground truth. (This is consistent with TESTING.md noting that butterfly tests compare "the per-step CUDA-op path" as reference.)

**Core strict-tier bwd test** — mirror `tests/test_butterfly_dispatch.py:218-315` (the autograd-vs-Triton-bwd body with the custom `ref_scan` closure on `butterfly_multiply` from `torch_structured`). Same TF32→strict swap. Note this is a more elaborate body than the other three strict files because the butterfly reference is built from `butterfly_multiply` directly rather than a PyTorch closed-form helper.

---

### Realistic-Tier Tightening Candidates (D-13 second bullet)

Per CONTEXT D-13: existing realistic-tier tolerances are tightened to `< 1e-4` rel **where the kernel can pass** at TF32. Below is a per-file inventory of current tolerance constants and a "tightenable / not-tightenable" classification based on the kernel's documented arithmetic regime and the docstrings cited inline.

**Hard rule:** any test whose docstring (or inline comment) explains why the bound is loose **is not tightenable** — its rationale is part of the test contract. Only tighten when the docstring is silent or the rationale is clearly conservative.

#### `tests/test_triton_scan.py` (dense)

| Line | Test | Current bound | Documented rationale | Tightenable? |
|------|------|---------------|----------------------|--------------|
| `:75` | `test_triton_forward_persistent_matches_default` | `rel < 5e-2` | `:74` "TF32 in both paths but different accumulation orders → looser bound." | **No.** Comment documents the bound — TF32 reduction-order is the real limit. |
| `:112` | `test_triton_backward_persistent_matches_pytorch` | `rel < 1e-1` (all 4 grads) | None inline. | **Conditionally yes.** Try `< 1e-2` per name; if `dWh` / `dbh` fail, keep at `1e-1` with new comment. Bwd-grad TF32 noise compounds across T × 3 gates; the looser bound MAY be load-bearing. |
| `:139` | `test_triton_forward_matches_pytorch` | `max_diff < 5e-3` | `:137-138` "TF32 input precision in tl.dot — ~10-bit mantissa per matmul. Drift accumulates across 3 matmuls per step + T steps + nonlinearities." | **No.** Docstring explicitly documents the bound. |
| `:215` | `test_triton_backward_matches_pytorch` | `rel < 1e-1` (`x`, `h0`, `Wh_cat`, `bh_cat`) | `:149-152` "TF32 has ~10-bit mantissa and gradient magnitudes compound across T timesteps and three matmuls per step." Plus `:202-205` "dWh_cat / dbh_cat parity catches a class of autotune-related bugs" (the slab-zero regression). | **No.** Bound is intentionally conservative as a slab-zero regression detector. |
| `:280` | `test_triton_qat_persistent_matches_pytorch` (fwd) | `fwd_rel < 1e-1` | None inline. | **No.** QAT (in-kernel fake-quant) compounds STE noise on top of TF32; 1e-1 is the documented QAT regime per TESTING.md:125-127. |
| `:288` | `test_triton_qat_persistent_matches_pytorch` (grads) | `rel < 2e-1` | None inline. | **No.** Same as above. |
| `:362` | `test_triton_qat_matches_pytorch` (fwd) | `fwd_rel < 1e-1` | `:358-359` "per-step fake-quant noise plus TF32 matmul noise compounds across T timesteps." | **No.** Documented. |
| `:371` | `test_triton_qat_matches_pytorch` (grads) | `rel < 2e-1` | None inline. | **No.** Same as :288. |

**Net tightening candidates for `test_triton_scan.py`:** only line 112 (`test_triton_backward_persistent_matches_pytorch`), possibly to `< 1e-2`. Test on CUDA before committing; if any of `(dgi, dh0, dWh, dbh)` fails, revert with a new docstring sentence per D-20.

#### `tests/test_triton_diagonal.py`

| Line | Test | Current bound | Documented rationale | Tightenable? |
|------|------|---------------|----------------------|--------------|
| `:93` | `test_diagonal_pytorch_forward_matches_cell` (Stage A) | `rel < 1e-5` | None inline. | **Already strict.** Don't touch — this is the algebraic-equality gate per TESTING.md:117. Strict file does NOT duplicate. |
| `:121` | `test_diagonal_triton_forward_matches_pytorch` (Stage B fp32) | `rel < 1e-4` | `:104-106` "no matmul on the hidden side, so the only floating-point noise comes from the input-side gi ... and the per-step nonlinearities — both bit-stable. Tight tolerance." | **Maybe.** Already tight; try `< 1e-5` rel. If passes, this overlaps the strict file's bound — keep both (different precision regimes). |
| `:156` | `test_diagonal_triton_qat_forward_matches_pytorch` | `rel < 1e-3` | `:153-155` "Diagonal has no matmul, so torch.round and tl.rint should agree bit-for-bit ... quant_h_in/out drift across the T-step recurrence is the only noise source." | **Maybe.** Try `< 1e-4`. If passes, commit. |
| `:194` | `test_diagonal_triton_backward_matches_pytorch` | `rel < 1e-3` | None inline. | **Yes, candidate.** Try `< 1e-4`. Same reasoning as forward Stage B. |
| `:239` | `test_diagonal_triton_qat_backward_matches_pytorch` | `rel < 1e-2` | `:238` "STE rounding can flip mask bits at boundaries → looser tol than fp32." | **No.** STE rationale documented. |
| `:268` | `test_diagonal_dispatch_matches_per_step` | `rel < 1e-4` | None inline. | Already at < 1e-4 floor — borderline. Try `< 1e-5`; expect to fail due to dispatch path's TF32 input projection. |
| `:331, :339` | `test_diagonal_dispatch_grad_matches_per_step` | `rel < 1e-3` (x, h0, params) | None inline. | **Yes, candidate.** Try `< 1e-4`. |

**Net tightening candidates for `test_triton_diagonal.py`:** lines 121, 156, 194, 268, 331, 339. Each tightening is an atomic commit per D-20.

#### `tests/test_triton_monarch.py`

| Line | Test | Current bound | Documented rationale | Tightenable? |
|------|------|---------------|----------------------|--------------|
| `:101` | `test_monarch_pytorch_forward_matches_cell` (Stage A) | `rel < 1e-5` | None inline. | **Already strict.** Don't touch. |
| `:127` | `test_monarch_triton_forward_matches_pytorch` | `rel < 5e-3` | `:126` "TF32 matmul + T-step compounding." | **No.** Documented. |
| `:162` | `test_monarch_triton_qat_forward_matches_pytorch` | `rel < 1e-1` | None inline. | **No.** QAT regime. |
| `:210` | `test_monarch_triton_qat_backward_matches_pytorch` | `rel < 1e-1` | None inline. | **No.** QAT regime. |
| `:248` | `test_monarch_triton_backward_matches_pytorch` | `rel < 5e-2` | None inline. | **Conditionally yes.** Try `< 1e-2`; if fails, revert with docstring sentence. |
| `:287-288` | `test_grulayer_use_triton_matches_pytorch_path` | `rel < 5e-2` (out, hT) | None inline. | **Conditionally yes.** Try `< 5e-3`. |
| `:404, :409, :414` | `test_monarch_pytorch_backward_matches_cell` | `rel < 1e-4` (dh0, dWh, dbh) | None inline. | **Yes, candidate.** Try `< 1e-5` — this is a CPU PyTorch-vs-PyTorch comparison, no TF32 in play. Likely passes. |

**Net tightening candidates for `test_triton_monarch.py`:** lines 248, 287, 288, 404, 409, 414.

#### `tests/test_butterfly_dispatch.py`

| Line | Test | Current bound | Documented rationale | Tightenable? |
|------|------|---------------|----------------------|--------------|
| `:74-75` | `test_butterfly_dispatch_matches_per_step` | `rel < 1e-1` | `:71-73` "Triton-kernel path uses different rounding order from the CUDA-op per-step path; tolerance loose enough to absorb log_H stages × T timesteps of accumulated noise." | **No.** Documented. |
| `:160` | `test_butterfly_triton_forward_matches_per_step` | `rel < 2e-2` | `:158-159` "CUDA op vs Triton kernel — different rounding order on log_H * T stages plus TF32 in the recurrence; ~0.5-1% drift is normal." | **No.** Documented. |
| `:210` | `test_butterfly_triton_forward_scratch_oob_regression` | `rel < 5e-2` | `:200-205` "OOB-corruption bug was order ~1.0+ relative, so a 5% threshold cleanly distinguishes 'fixed' from 'regressed'." | **No.** Documented as a regression-bound, not a parity-bound. |
| `:315` | `test_butterfly_triton_backward_matches_autograd` | `rel < 5e-2` | None inline. | **Conditionally yes.** Try `< 1e-2`. |
| `:340` | `test_butterfly_grulayer_triton_path_matches_per_step` | `rel < 5e-3` | None inline. | **Conditionally yes.** Try `< 1e-3`. |
| `:405` | `test_butterfly_triton_qat_forward_matches_per_step` | `rel < 1e-1` | None inline. | **No.** QAT regime. |
| `:459` | `test_butterfly_extract_and_gru_scan_directly` | `rel < 1e-1` | `:457-458` "use_triton=True now routes through the Triton kernel; manual_out uses the CUDA-op per-step path. Different rounding order, loose tolerance." | **No.** Documented. |

**Net tightening candidates for `test_butterfly_dispatch.py`:** lines 315, 340.

---

## Shared Patterns

### `'highest'` module-level precision preamble
**Source:** `tests/test_layer_parity.py:28-33` (Phase 1 output).
**Apply to:** All four new strict-tier files at module scope, immediately after imports and `cuda_only` definition (or before — order doesn't matter since `set_float32_matmul_precision` is global state, not collected at test time).
```python
# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (test_triton_<kind>.py) uses 'high' to exercise the kernel under
# deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")
```
This is the **signature distinguishing feature** between strict and realistic tier files. Document the divergence in each strict-tier file's module docstring.

### Absolute-error assertion (strict tier)
**Source:** Phase 2 CONTEXT "Established Patterns" callout (D-13): "Phase 2 strict-tier uses absolute error at < 1e-5 (since TF32 isn't in play), not relative."
**Apply to:** All strict-tier parity assertions in the four new files.
```python
max_diff = (ref - tri).abs().max().item()
assert max_diff < 1e-5, f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
```
Diverges intentionally from the realistic-tier idiom (relative with `1e-6` floor) — strict tier doesn't need the floor because we're not normalizing.

### Relative-error assertion (realistic tier — used in tightening commits)
**Source:** `tests/test_triton_diagonal.py:120-121`, `tests/test_triton_monarch.py:99-101`, `tests/test_butterfly_dispatch.py:69-71`.
```python
max_diff = (ref - tri).abs().max().item()
rel = max_diff / max(ref.abs().max().item(), 1e-6)
assert rel < TOL, f"<name> rel diff {rel:.4e}"
```
**Apply to:** Realistic-tier files when tightening; do NOT change the idiom shape, only the `TOL` value. Each tightening commits the constant change ONLY (D-20: "two-commit discipline" → tightening commit, then if it fails CI, revert commit with comment).

### `cuda_only` per-file gate
**Source:** Existing convention at `tests/test_triton_scan.py:25-27`, `tests/test_triton_diagonal.py:30-32`, `tests/test_triton_monarch.py:30-32`, `tests/test_butterfly_dispatch.py:31-33`.
**Apply to:** All four new strict-tier files. Define locally per file (not in `conftest.py` per TESTING.md "No fixtures via conftest"). The reason string can be either "Triton kernel requires CUDA" or the kernel-specific variant — preserve the analog file's variant for consistency.

### `pytest.importorskip` at module top
**Source:** `tests/test_triton_scan.py:12`, `tests/test_triton_monarch.py:17`, `tests/test_butterfly_dispatch.py:19`.
**Apply to:**
- `test_triton_scan_strict.py`: `triton = pytest.importorskip("triton")`
- `test_triton_diagonal_strict.py`: `triton = pytest.importorskip("triton")` (the analog file lacks this — diagonal's `_make_diagonal_layer` doesn't go through torch_structured, but the strict file should still skip cleanly on machines without Triton at all).
- `test_triton_monarch_strict.py`: BOTH `pytest.importorskip("triton")` AND `pytest.importorskip("torch_structured")`.
- `test_triton_butterfly_strict.py`: BOTH (same reason as monarch).

Every subsequent module-level import gets `# noqa: E402` per CONTEXT "Established Patterns" callout.

### `torch.manual_seed(0)` per test body
**Source:** TESTING.md "Bench-Style Smoke Tests vs. Correctness Tests" (line 136); used universally in `tests/test_triton_*.py`.
**Apply to:** Top of every test function in all four strict-tier files. Reset per-test (not module scope) so tests are independent.

### Two-commit failing-test-before-fix discipline (carried from Phase 1)
**Source:** Phase 1 PATTERNS.md "Shared Patterns / Two-commit failing-test-before-fix discipline" (lines 393-399); CONTEXT D-27.
**Apply to:** Any kernel-vs-reference parity failure surfaced by the new strict tier.
1. Commit A: failing test on its own, no `src/` changes. Capture `pytest --tb=short` tail.
2. `bd create` with test function name as title; `bd update <id> --notes <pytest-tail>`.
3. Commit B: fix in `src/gru_qat/triton_kernels/scan*.py` (no test changes), test passes. `bd close <id>` after CI green.
4. **Never** `@pytest.mark.xfail` (silent in `pytest -q`, defeats audit signal — D-12 from Phase 1, carried as D-27).

### Realistic-tier tightening commit discipline (D-20)
**Source:** CONTEXT D-20.
**Apply to:** Each tolerance constant tightened in an existing test file.
1. One commit per kernel test file (NOT one commit per constant — that's too granular). E.g., a single commit "tighten realistic-tier tolerances in test_triton_diagonal.py per D-13" that touches lines 121, 156, 194, 268, 331, 339.
2. If CI fails on the tighter bound, immediate revert commit; do NOT loosen incrementally. Re-attempt with a docstring sentence documenting why the tightening was reverted.

### Slow-marker convention
**Source:** `tests/test_qat_smoke.py:88`, `tests/test_layer_parity.py:365-366` (Phase 1 pattern). Registered in `pyproject.toml:44-45`.
**Apply to:** All four strict-tier files. Each kernel's `SLOW_*_GRID` (T ∈ {512, 1024}) is parametrized into a `_slow` sibling test. Default `pytest -q` skips slow; phase-exit GPU run includes `-m slow` per D-26.

```python
@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_DENSE_GRID)
def test_scan_fwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Identical body to the fast variant; gated behind @pytest.mark.slow
    per D-16 (T ∈ {512, 1024})."""
    # ... same body as fast test
```

### Phase 1 LOCKED files (D-28)
**Source:** CONTEXT D-28.
**Apply to:** Verifier asserts `git diff <phase-2-base>..HEAD -- tests/test_parity.py tests/test_layer_parity.py` is empty. Phase 2 commits MUST NOT touch either file. Pattern: do not import helpers FROM these files into the strict-tier kernel files (duplicate per D-18 instead).

---

## No Analog Found

Files / tests with no close in-repo precedent (planner should compose from CONTEXT and RESEARCH instead of expecting a single analog file):

| Test | Role | Data Flow | Why No Analog |
|------|------|-----------|---------------|
| `test_persistent_kernel_deterministic` (TRI-06, D-24) | regression test | request-response, 50× loop | No existing test asserts bit-identical output across N runs of the same kernel. The 50-iteration `torch.equal` pattern is novel; `TESTING.md:22` notes `torch.equal` is used elsewhere only for strict-tier algebraic equalities, not for cross-run determinism. The pattern is mechanically simple but the rationale (cross-CTA fence regression detection) is repo-specific and documented inline in the test docstring + references `DEVELOPMENT.md:131-143` + the source comment block at `src/gru_qat/triton_kernels/scan.py:184-203`. |
| `test_no_cv_cache_modifier_in_scan_source` (D-25) | static / regression test | file-I/O (read source, regex / count) | No existing test does static source inspection. The pattern is mechanically a `pathlib` + string search; the project convention is to use `pathlib` (not `subprocess` / shell-out). Planner: see the **D-25 ambiguity note** in the scan_strict section above — current source has 2 legitimate `.cv` uses in the bwd kernel; the canary either (a) baselines at count == 2 or (b) waits for those sites to be refactored separately. Recommended (a). |
| `test_autotune_dWh_dbh_zero_init_across_configs` (TRI-05, D-23) — variant only | regression test | multi-config rotation | The single-config slab-zero regression at `tests/test_triton_scan.py:202-215` is the closest analog, but the multi-config rotation mechanism (e.g., clearing JIT cache between two different `(T, B)` inputs to force the autotuner to re-tune) has no in-repo precedent. Planner: the simplest implementation is to call `gru_scan` at two `(T, B)` shapes that hit different autotune buckets per `key=['T', 'B']` (`src/gru_qat/triton_kernels/scan.py:732,893`) within one test, asserting both pass < 1e-5 abs. No `triton.runtime.jit.JITFunction.cache.clear()` call needed if both shapes naturally bucket-differ. |

No `src/` files are listed as "no analog" because **Phase 2 plans no speculative `src/` edits** — only fix-commits (per the two-commit discipline) land changes, and a fix's analog is the file where the bug lives.

---

## Metadata

**Analog search scope:**
- `tests/` — read `test_triton_scan.py`, `test_triton_diagonal.py`, `test_triton_monarch.py`, `test_butterfly_dispatch.py`, `test_layer_parity.py` (Phase 1 strict precision preamble + grid split idiom).
- `src/gru_qat/triton_kernels/scan.py` — read autotune configs (`:38-60`), persistent fwd kernel barrier (`:170-208`), persistent bwd grid launch (`:700-729`), autotune key declarations (`:732, 893`), public API entry points (`:1569-1666`).
- `.planning/codebase/TESTING.md` — full conventions reference (TF32 setup, `cuda_only` per-file, `pytest.importorskip` + noqa, no conftest, tolerance tiers).
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-PATTERNS.md` — Phase 1 pattern map (carried forward via CONTEXT D-27, D-28).

**Files scanned:** 8 (5 test files, 1 src kernel module — 4 targeted reads, no full load, 1 conventions doc, 1 prior pattern map).

**Pattern extraction date:** 2026-05-13.
