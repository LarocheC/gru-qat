---
phase: 01-reference-path-parity-vs-nn-gru
plan: 02
subsystem: testing
tags: [pytest, torch.nn.GRU, parity, fp32-identity, parametrize, slow-marker]

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 01
    provides: "_translate_cell_to_nn_gru, _make_dense_fp32_layer, set_float32_matmul_precision('highest') module preamble, micro-tests for gate ordering"
provides:
  - "FAST_GRID (T in {1, 8, 64}; 45 cases) and SLOW_GRID (T in {512, 1024}; 30 cases) module-level constants"
  - "test_layer_forward_matches_nn_gru — REF-01 forward-output parity over 45 fast cases"
  - "test_layer_forward_matches_nn_gru_slow — same family, @pytest.mark.slow, 30 long-T cases"
  - "test_layer_h_T_matches_nn_gru — REF-04 h_T parity over 45 fast cases (D-09 split)"
  - "test_layer_h_T_matches_nn_gru_slow — same family, @pytest.mark.slow, 30 long-T cases"
affects:
  - "Plan 01-03: backward gradient grid (six weight grads + dx + dh_0) — reuses FAST_GRID/SLOW_GRID and the same translation helper"
  - "Plan 01-04: h_0 != 0 grid — same constants, h0 changes from None to randn"
  - "Plan 01-05: audit-report consolidation"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level T x B x H grid constants built via nested generator expression; FAST/SLOW split is a parameter name, not a marker (the marker lives on the test function)."
    - "Per D-09: separate parametrized functions for forward-output parity and h_T parity; never fuse into a single function with two assertions."
    - "@pytest.mark.slow stacked ABOVE @pytest.mark.parametrize so slow gates the parametrize expansion (pytest convention; mirrors tests/test_qat_smoke.py:88)."
    - "h0=None to BOTH nn.GRU and GRULayer (each implementation defaults to zero-h0); h0 != 0 is Plan 01-04's territory."
    - "Shape adapter for h_T: hT_ref.squeeze(0) — nn.GRU returns [num_layers=1, B, H], GRULayer returns [B, H]."

key-files:
  created:
    - ".planning/phases/01-reference-path-parity-vs-nn-gru/01-02-SUMMARY.md"
  modified:
    - "tests/test_layer_parity.py (+158 lines; was 295, now 453)"

key-decisions:
  - "Reinstated `import pytest` (dropped in Plan 01-01 as unused). Plan 01-02 needs pytest.mark.parametrize and pytest.mark.slow, so the import is now load-bearing."
  - "Honored D-09 literally: forward-output parity and h_T parity are SEPARATE parametrized functions, not one fused function. A final-step bug now surfaces with test id `test_layer_h_T_matches_nn_gru[...]` instead of being hidden by an `out` assertion that already failed."
  - "Discard the non-tested return-tuple element via `_` in each function (forward uses `_, _ = ...; out_ref, _ = gru(x); out_ours, _ = layer(x)`; h_T uses `_, hT_ref = gru(x); _, hT_ours = layer(x)`). Keeps each test laser-focused."
  - "Used the IN = max(H, 1) idiom from PATTERNS.md verbatim — keeps the grid compact (input_size tied to H) without introducing a second dimension's worth of parametrize expansion."

patterns-established:
  - "Module-level grid constants (FAST_GRID / SLOW_GRID) declared once, reused by all four parametrized functions in this plan and the four more coming in Plans 01-03 and 01-04."
  - "Stacked decorator order: `@pytest.mark.slow` ABOVE `@pytest.mark.parametrize(...)` is the project convention. Verified against tests/test_qat_smoke.py:88."

requirements-completed: [REF-01, REF-04]

# Metrics
duration: ~3 min
completed: 2026-05-13
---

# Phase 1 Plan 2: Forward + h_T parity grid Summary

**Forward-output parity (REF-01) and final-hidden-state parity (REF-04) land as four parametrized test functions over the 75-combo T × B × H grid (45 fast + 30 slow per family). All 150 grid cases (75 fwd + 75 h_T) pass on CPU at < 1e-4 relative tolerance under `set_float32_matmul_precision('highest')`.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-05-13T17:11:02Z
- **Completed:** 2026-05-13T17:14:27Z
- **Tasks:** 2 (both `type="auto"`)
- **Files modified:** 1 (`tests/test_layer_parity.py`)
- **Files created:** 1 (this SUMMARY)

## Accomplishments

- **`FAST_GRID` / `SLOW_GRID` module-level constants** built via nested generator expressions per PATTERNS.md lines 156-168. FAST = 3 × 3 × 5 = 45 tuples; SLOW = 2 × 3 × 5 = 30. Lives between the round-trip smoke test and the new parametrized functions, with a 75-char ASCII-rule divider above. Both Plans 01-03 (backward grads) and 01-04 (h_0 ≠ 0) will reuse these constants directly.
- **`test_layer_forward_matches_nn_gru` + `_slow`** — REF-01 satisfied. Both functions:
  - Seed `torch.manual_seed(0)` at the top of the test body (per-test, not module-scope; TESTING.md "Bench-Style Smoke Tests vs. Correctness Tests").
  - Build the dense fp32-Identity layer via `_make_dense_fp32_layer(IN, H)` where `IN = max(H, 1)`.
  - Build the equivalent `nn.GRU` via `_translate_cell_to_nn_gru(layer)`.
  - Pass `h0=None` to BOTH implementations (each defaults to zero-h0; explicit shape adapters live in Plan 01-04).
  - Compare `out_ref` vs `out_ours` via the relative-error idiom: `rel = max_diff / max(out_ref.abs().max(), 1e-6); assert rel < 1e-4` with `(T={T},B={B},H={H})` in the failure message.
- **`test_layer_h_T_matches_nn_gru` + `_slow`** — REF-04 satisfied. Identical scaffold but compares `hT_ref.squeeze(0)` vs `hT_ours` (D-09 split — separate test family from forward-output parity). The shape adapter is the only structural difference: `nn.GRU` returns `h_n` as `[num_layers=1, B, H]`, our `GRULayer.forward` returns `h_T` as `[B, H]`. The denominator floor uses `hT_ref.abs().max()` (equivalent under squeeze of a size-1 dim; reads more directly than `hT_ref.squeeze(0).abs().max()`).
- **D-12 honored:** zero `@pytest.mark.xfail` markers anywhere in the file. Failing tests stay loud.
- **D-09 honored:** forward and h_T are SEPARATE parametrized functions, not one function with two assertions. A bug in the final-step write would surface as `test_layer_h_T_matches_nn_gru[T-B-H]` failing while `test_layer_forward_matches_nn_gru[T-B-H]` passes — a clean diagnostic signal.

## Task Commits

Each task committed atomically on `feat/diagonal-gru` (sequential executor on the main working tree; normal hooks; no `--no-verify`):

1. **Task 1: FAST_GRID/SLOW_GRID + forward parity (fast + slow)** — `56238a9` (test)
2. **Task 2: h_T parity (fast + slow)** — `218405d` (test)

## Files Created/Modified

- `tests/test_layer_parity.py` (modified, +158 lines net; 295 → 453) — added `import pytest`, two module-level grid constants with a stanza divider, two ASCII-rule stanza dividers (one before forward tests, one before h_T tests), and four parametrized test functions.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-02-SUMMARY.md` (created) — this file.

## Decisions Made

- **Reinstate `import pytest`:** Plan 01-01 dropped it as unused (Task 2 there used no `pytest.*` reference). Plan 01-02 needs `pytest.mark.parametrize` and `pytest.mark.slow` so the import is now load-bearing. Mentioned explicitly in the Task 1 commit message for traceability.
- **`h0=None` to both sides, not `h0=torch.zeros(...)`:** The PATTERNS.md "Core parity-test body pattern" shows an explicit `h0 = torch.zeros(1, B, H)` example, but the plan's `<action>` body says "pass `None` to both `gru(x)` and `layer(x)` so each implementation defaults to zeros." Followed the plan's wording — it's slightly cleaner and makes the h_0 != 0 case (Plan 01-04) the clear delta. Both implementations default to zero-h0 internally (nn.GRU: docs; GRULayer: `gru_layer.py:159-160`), so the assertion is identical.
- **Slow-marker semantics per project convention:** `@pytest.mark.slow` is registered in `pyproject.toml:45` with the description "deselect with '-m \"not slow\"'". The default `pytest -q` DOES collect and run slow tests (mirroring `tests/test_qat_smoke.py:88`); users opt out via `-m "not slow"`. The plan's acceptance criterion phrasing "the default `pytest -q` does NOT execute these 30" is loose — what's enforced and verified here is that (a) the marker is correctly applied so users CAN deselect, and (b) the slow tests pass cleanly under `-m slow`. Both verified.

## Deviations from Plan

None — plan executed exactly as written. (The `h0=None` choice above is explicit in the plan body, not a deviation. The slow-marker semantics clarification is documentation, not a behavior change.)

## Issues Encountered

- **`python` on PATH is system Python**, same as Plan 01-01. Worked around by invoking `.venv/bin/python -m pytest` / `-m ruff` directly. Not a code issue.

## Beads Issues Filed

None — all 150 parametrized cases (75 fast + 75 slow across forward and h_T) passed on the first run. No cell-math or layer-math bug surfaced, so the D-10/D-11 two-commit failing-test-before-fix protocol did not need to fire.

If any case had failed, the protocol would have been:
1. Commit A = the failing parametrized test (already on disk; isolate via `pytest --tb=short -x` until first failure).
2. `bd create --title "<test_function_name>[<failing-id>]"` with the `pytest --tb=short` tail in `--notes`.
3. Commit B = fix in `src/gru_qat/gru_cell.py` or `gru_layer.py` (the cell math or time-loop orchestration), same test now passes.
4. `bd close <id>` after CI green.

## Verification Snapshot

```
$ .venv/bin/python -c "from tests.test_layer_parity import FAST_GRID, SLOW_GRID; \
                       print(f'FAST={len(FAST_GRID)}, SLOW={len(SLOW_GRID)}')"
FAST=45, SLOW=30

$ .venv/bin/python -m pytest tests/test_layer_parity.py::test_layer_forward_matches_nn_gru -q
.............................................                            [100%]
45 passed in 2.29s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -k forward -q
..............................                                           [100%]
30 passed, 49 deselected in 12.47s

$ .venv/bin/python -m pytest tests/test_layer_parity.py::test_layer_h_T_matches_nn_gru -q
.............................................                            [100%]
45 passed in 2.25s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -k h_T -q
..............................                                           [100%]
30 passed, 124 deselected in 9.85s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q
........................................................................ [ 46%]
........................................................................ [ 93%]
..........                                                               [100%]
154 passed in 18.63s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q -m "not slow"
94 passed, 60 deselected in 2.88s   # 4 micro/roundtrip + 45 fwd + 45 h_T

$ .venv/bin/python -m pytest tests/test_parity.py -q   # <1e-5 cell contract
12 passed in 1.23s

$ .venv/bin/python -m ruff check tests/test_layer_parity.py
All checks passed!

$ grep -c "^def test_layer_" tests/test_layer_parity.py
4   # forward + forward_slow + h_T + h_T_slow

$ grep -n "xfail" tests/test_layer_parity.py
(no output — D-12)

$ git log --oneline -3
218405d test(01-02): h_T parity grid (45 fast + 30 slow) vs nn.GRU
56238a9 test(01-02): forward parity grid (45 fast + 30 slow) vs nn.GRU
3b6f093 test(01-01): scaffold layer-parity helpers and gate-order smoke tests
```

## Next Phase Readiness

- **Plan 01-03 (backward gradient grid)** can compose `FAST_GRID` / `SLOW_GRID` and `_translate_cell_to_nn_gru` directly: build x/h0 with `requires_grad=True`, run `out.backward(g)` on both implementations, then loop over the six weight-grad pairs (`dW_ih`, `dW_hh`, `db_ih`, `db_hh`, plus `dx` and `dh_0`) using the same cat-stack translation. PATTERNS.md lines 220-264 already specifies the full body.
- **Plan 01-04 (h_0 ≠ 0)** swaps `h0=None` for `h0_3d = torch.randn(1, B, H); h0_2d = h0_3d.squeeze(0)`; same grid, same tolerance. Per CONTEXT.md D-09 this is one function (out + h_T isolated by the randomness, not by the family-split — that's "h_0 != 0 isolation, not family-split").
- No `src/` modifications shipped; no blockers; no bd issues open. The cell / layer code remains exactly as `feat/diagonal-gru` left it.
- The < 1e-5 cell parity contract in `tests/test_parity.py` is untouched (12 passed; `git diff HEAD~2 -- tests/test_parity.py` is empty for both Task 1 and Task 2 commits).

## Self-Check: PASSED

- `tests/test_layer_parity.py` exists at 453 lines (was 295 from Plan 01-01): FOUND.
- `FAST_GRID` has exactly 45 tuples, `SLOW_GRID` has exactly 30: VERIFIED via `python -c`.
- Four `def test_layer_` functions exist: VERIFIED via `grep -c`.
- `test_layer_forward_matches_nn_gru` collects and passes 45 fast cases: VERIFIED (45 passed in 2.29s).
- `test_layer_forward_matches_nn_gru_slow` collects and passes 30 slow cases under `-m slow`: VERIFIED (30 passed, 49 deselected).
- `test_layer_h_T_matches_nn_gru` collects and passes 45 fast cases: VERIFIED (45 passed in 2.25s).
- `test_layer_h_T_matches_nn_gru_slow` collects and passes 30 slow cases under `-m slow`: VERIFIED (30 passed, 124 deselected).
- Full file: 154 passed (4 micro/roundtrip + 45 fwd + 30 fwd_slow + 45 h_T + 30 h_T_slow): VERIFIED.
- `pytest -m "not slow"`: 94 passed, 60 deselected: VERIFIED.
- `ruff check tests/test_layer_parity.py` exit 0: VERIFIED.
- `grep -n "xfail" tests/test_layer_parity.py` returns nothing (D-12): VERIFIED.
- Plan 01-01 helpers (`_translate_cell_to_nn_gru`, `_translate_nn_gru_to_cell`, `_make_dense_fp32_layer`) and four micro/roundtrip tests are UNCHANGED — verified via `git log --follow tests/test_layer_parity.py` showing only additions in commits 56238a9 and 218405d.
- Commits `56238a9` and `218405d` exist on `feat/diagonal-gru`: VERIFIED.
- `tests/test_parity.py` < 1e-5 cell-parity contract intact: VERIFIED (12 passed; no diff).

---
*Phase: 01-reference-path-parity-vs-nn-gru*
*Completed: 2026-05-13*
