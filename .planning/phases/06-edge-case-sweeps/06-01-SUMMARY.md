---
phase: 06-edge-case-sweeps
plan: 01
subsystem: testing / edge-case audit
tags: [edge-cases, parity, triton, butterfly, validation, EDG-01, EDG-02, EDG-03, EDG-04]
requires:
  - GRULayer.forward (7-path dispatch)
  - tests/test_layer_parity.py::_translate_nn_gru_to_cell (D-11 import-only)
  - tests/test_triton_monarch_strict.py::_skip_if_monarch_bwd_hw_limit (D-51 import-only)
provides:
  - tests/test_edge_cases.py — 7-path x edge-shape sweep (T=1, B=1, small-H, long-T, T=0/B=0)
  - GRULayer.forward T=0/B=0 ValueError guard
  - structure.py butterfly H<2 ValueError guard
affects:
  - src/gru_qat/gru_layer.py
  - src/gru_qat/structure.py
tech-stack:
  added: []
  patterns:
    - "fail-loud ValueError guard on degenerate input dimensions"
    - "TF32-independent batch-invariance regression (replicated-input contract)"
key-files:
  created:
    - tests/test_edge_cases.py
    - .planning/phases/06-edge-case-sweeps/06-pytest-output.txt
    - .planning/phases/06-edge-case-sweeps/deferred-items.md
  modified:
    - src/gru_qat/gru_layer.py
    - src/gru_qat/structure.py
    - .planning/PROJECT.md
decisions:
  - "T=0/B=0 disposition: all 7 paths raise ValueError naming the offending dimension (D-01); single GRULayer.forward guard"
  - "butterfly H<2 rejected at construction (size-1 factorization undefined)"
  - "_path_tol reuses committed kernel-test contracts: butterfly < 5e-2, dense/monarch/diagonal full-layer < 5e-4, circulant/ldr < 1e-5 — no new bounds"
metrics:
  duration: ~50 min
  completed: 2026-05-15
  tasks: 5
  files-changed: 6
---

# Phase 6 Plan 01: Edge-case Sweeps Summary

A single new `tests/test_edge_cases.py` pins all 7 GRU code paths at boundary
shapes (T=1, B=1, H in {1,2}, long T in {512,1024}, degenerate T=0/B=0). The
T=0/B=0 disposition is now a tested `ValueError` policy. The B=1/small-H sweep
surfaced two real bugs — one fixed in-phase, one deep kernel bug handed off
with a recoverable failing test + beads issue.

## What Shipped

- **`tests/test_edge_cases.py`** (614 lines, 7 test functions): a path × edge-shape
  parametrized sweep over `ALL_PATHS = [reference, dense_triton, diagonal_triton,
  monarch_triton, butterfly_triton, circulant, ldr]`. Covers EDG-01 (T=1 fwd+bwd),
  EDG-02 (B=1 + small-H + butterfly partial-batch-tile), EDG-03 (long-T slow tier),
  EDG-04 (T=0/B=0 ValueError).
- **`GRULayer.forward` T=0/B=0 guard** (`src/gru_qat/gru_layer.py`): a single
  `if seq_len == 0 / batch_size == 0: raise ValueError` after the shape unpack,
  covering all 7 GRULayer-routed paths. Uses `if … raise`, not `assert`.
- **`structure.py` butterfly H<2 guard**: rejects a size-1 butterfly factorization
  at construction (Commit B for the H=1 interpreter-crash bug).
- **PROJECT.md**: the Phase 6 D-01 T=0/B=0 policy logged in Key Decisions.
- **`06-pytest-output.txt`**: UTC-timestamped full-suite artifact.

## Task-by-Task

| Task | Name | Commit(s) | Result |
|------|------|-----------|--------|
| 1 | T=0/B=0 guard + EDG-04 tests + PROJECT.md policy | `7273c19` | 14/14 EDG-04 cases pass |
| 2 | T=1 fwd+bwd sweep (EDG-01) | `0b1bd71` | 13 pass, 1 legitimate monarch-bwd HW-limit skip |
| 3 | B=1 + small-H BLOCK-size sweep (EDG-02) | `eb7242b`, `cca1783`, `d6625cc` | butterfly H=1 crash fixed; butterfly batch-tiling bug surfaced + handed off |
| 4 | Long-T slow-tier drift sweep (EDG-03) | `9dc84b9` | 14/14 slow-tier cases pass |
| 5 | Full-suite run + artifact | `80e8ac3` | artifact captured; baseline lint/type logged |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Missing validation] butterfly H=1 crashed the interpreter**
- **Found during:** Task 3 (B=1/small-H sweep).
- **Issue:** `StructureConfig(kind="butterfly")` with `hidden_size=1` reached
  `torch_structured`'s `butterfly_multiply` CUDA op, which divides by `n//2 == 0`
  and raised a fatal `Floating-point exception` aborting the whole Python process.
  A size-1 butterfly factorization (`log2(1)==0` stages) is mathematically undefined.
- **Fix:** `structure.py:_validate_shapes` now rejects butterfly `in/out < 2` with
  a clear `ValueError` at construction (analogous to the circulant pow-of-2 guard).
- **Two-commit discipline:** Commit A `eb7242b` (failing regression test), bd issue
  `gru-triton-ehf`, Commit B `cca1783` (fix). bd `gru-triton-ehf` closed.
- **Files modified:** `src/gru_qat/structure.py`, `tests/test_edge_cases.py`.

**2. [Tolerance correction] `_path_tol` re-tiered to the committed kernel-test contracts**
- The initial draft used `5e-4` uniformly for all Triton paths. Empirically the
  full-GRULayer diagonal forward routes through the TF32 input-projection GEMM
  (measured ~3e-4 abs drift at B=1, ~7e-8 under `precision('highest')`), so the
  `<1e-5` diagonal-recurrence tier does not apply to the full-layer comparison —
  the PROJECT.md `<5e-4` tl.dot tier does. Butterfly genuinely diverges at the
  1e-2 scale (its `log2(H)` `tl.dot` stages vs the torch_structured FFT per-step
  path); the committed `test_butterfly_dispatch.py` OOB-regression test already
  uses `rel_per_b < 5e-2`. `_path_tol` now reuses those *committed* bounds — no
  new bound invented (D-09 honored).

## Open Findings — Handoff

### bd gru-triton-c2a — butterfly Triton batch-tiling correctness bug (HANDED OFF)

The B=1/small-H sweep surfaced a genuine D-04 kernel bug. The butterfly Triton
forward kernel **violates batch-invariance** at H=512: replicating one B=1 input
across B=33 batch slots, the Triton output for a *periodic stride-8 subset* of
batches (4 batches bit-exact, 4 corrupted, repeating) diverges from the B=1 result
by up to ~6.3e-2 absolute. The per-step PyTorch reference path is batch-invariant
to TF32 (~5e-4). A correct kernel must be batch-invariant. Root cause is in the
`scan_butterfly.py` persistent kernel's batch-tiling / warp-vectorization within
the `BLOCK_B=8` tile.

**Disposition (per Task-3 CONTEXT-BUDGET HANDOFF + D-06):** the recoverable
Commit-A artifacts are landed — a sharp, TF32-independent failing regression test
(`test_butterfly_partial_batch_tile`, RED at B in {7,9,17,33}) and bd issue
`gru-triton-c2a` with the precise stride-8 characterization. The deep kernel fix
(Commit B) is **handed off** — diagnosing a Triton register-aliasing /
warp-vectorization bug in a 4-stage butterfly kernel is the legitimate stopping
point the plan authorizes. **No `@pytest.mark.xfail`** — the test stays RED until
the kernel fix lands. A debugging lead for the next agent: the 4-good/4-bad
pattern within `BLOCK_B=8` with `num_warps=4` suggests a warp-vectorization issue
in the cross-H-lane `partner = offs_h ^ stride_s` butterfly load.

### bd gru-triton-7rj — scan*.py wrapper assert hardening (filed, not fixed)

The four `gru_scan*` Triton wrappers validate shapes with bare `assert` (stripped
under `python -O`). Out of EDG-04 scope per the plan — filed for a future
hardening task, not fixed in Phase 6.

## Final Suite State

- **Non-slow** (`uv run pytest tests/test_edge_cases.py -q`): 78 passed, 3 skipped
  (butterfly H=1 — rejected at construction), **4 failed** — all the documented
  `gru-triton-c2a` butterfly batch-tiling finding.
- **Slow tier** (`-m slow`): 14/14 passed.
- **Regression check**: 238 existing tests across `test_layer_parity.py`,
  `test_triton_diagonal.py`, `test_butterfly_dispatch.py`, `test_calibration.py`
  pass — the T=0/B=0 guard regressed nothing.
- **D-51 locked files**: unmodified (`test_parity.py`, `test_layer_parity.py`,
  `test_structure.py`, the 4 `*_strict.py`).
- **STATE.md / ROADMAP.md**: untouched (orchestrator-owned).
- **lint/type**: plan 06-01's own changes are `ruff`-clean and add zero `mypy`
  errors; the 145-error `mypy` baseline is pre-existing (identical on base commit
  `3bdcf7b`) and logged in `deferred-items.md`.

Artifact: `.planning/phases/06-edge-case-sweeps/06-pytest-output.txt`

## Known Stubs

None.

## Threat Flags

None — the T=0/B=0 and butterfly H<2 guards *reduce* attack surface (fail-loud on
malformed input). No new network/auth/file surface.

## TDD Gate Compliance

Plan type is `execute`, not `tdd`. The one in-phase bug fix (butterfly H=1)
followed the D-04/D-05 two-commit discipline: failing test (`eb7242b`) → bd issue
→ fix (`cca1783`). The handed-off butterfly batch-tiling finding has its Commit-A
failing test landed (`d6625cc`); Commit B is deferred to the kernel-fix handoff.

## Self-Check: PASSED

All created files exist on disk (`tests/test_edge_cases.py`,
`06-pytest-output.txt`, `06-01-SUMMARY.md`, `deferred-items.md`) and all 7
per-task commits (`7273c19`, `0b1bd71`, `eb7242b`, `cca1783`, `d6625cc`,
`9dc84b9`, `80e8ac3`) are present in git history.
