---
phase: 01-reference-path-parity-vs-nn-gru
plan: 01
subsystem: testing
tags: [pytest, torch.nn.GRU, parity, fp32-identity, translation-helpers]

# Dependency graph
requires:
  - phase: 00-init
    provides: "PROJECT.md tolerance tiers (< 1e-4 layer / < 1e-5 cell); REQUIREMENTS.md REF-05; CONTEXT.md D-01..D-12; PATTERNS.md verbatim code excerpts"
provides:
  - "tests/test_layer_parity.py module preamble (set_float32_matmul_precision('highest'))"
  - "Translation helpers: _translate_cell_to_nn_gru, _translate_nn_gru_to_cell"
  - "fp32-Identity layer builder: _make_dense_fp32_layer"
  - "3 gate-order / n-gate-asymmetry micro-tests (r-only, z-only, n-asymmetry)"
  - "1 round-trip smoke test (nn.GRU -> cell direction)"
affects:
  - "Plan 01-02: 75-combo forward parity grid (FAST + SLOW)"
  - "Plan 01-03: 75-combo backward parity grid (six weight grads + dx + dh_0)"
  - "Plan 01-04: h_T parity + h_0 != 0 parity"
  - "Plan 01-05: audit-report consolidation"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level torch.set_float32_matmul_precision('highest') for math-audit tests (diverges from kernel tests using 'high')"
    - "(r, z, n) gate-order translation via torch.cat([W_ir, W_iz, W_in], dim=0) and inverse via chunk(3, dim=0)"
    - "Relative-error idiom < 1e-4 with 1e-6 denominator floor (mirrors tests/test_triton_diagonal.py:120-121)"
    - "torch.manual_seed(0) per-test (not module-scope) so tests are independent"

key-files:
  created:
    - "tests/test_layer_parity.py (295 lines, 3 helpers + 4 tests)"
  modified: []

key-decisions:
  - "Honored D-07: matmul precision pinned to 'highest' module-wide (auditing math, not TF32 mode)."
  - "Honored D-12: zero @pytest.mark.xfail markers — failing tests stay loud."
  - "Dropped Task 1's speculative `import pytest` (with noqa: F401) in the Task 2 commit: no test in this file currently uses the pytest API. If Plan 02 needs parametrize, the import is one line away."
  - "n-gate-asymmetry test uses b_ir = -100 only (W_in/W_hn/b_in/b_hn kept at init) to produce a non-trivial input-branch contribution that proves the asymmetric r placement is preserved."

patterns-established:
  - "Two-direction translation helpers at the layer level: cell->nn.GRU is the primary direction for parametrized grids; nn.GRU->cell is exercised by one round-trip smoke test only (D-01)."
  - "Per-test seed reset (torch.manual_seed(0) at the top of each test body) — pattern lifted from tests/test_triton_diagonal.py and TESTING.md."

requirements-completed: [REF-05]

# Metrics
duration: ~4 min
completed: 2026-05-13
---

# Phase 1 Plan 1: Layer-parity test scaffolding Summary

**Foundation file `tests/test_layer_parity.py` ships with two-direction translation helpers (cell <-> nn.GRU), an fp32-Identity layer builder, three gate-order / n-gate-asymmetry micro-tests, and a round-trip smoke test — all passing on CPU under `set_float32_matmul_precision('highest')`.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-13T17:05:36Z
- **Completed:** 2026-05-13T17:09:09Z
- **Tasks:** 2 (both `type="auto"`)
- **Files modified:** 1 (created `tests/test_layer_parity.py`; no `src/` touches)

## Accomplishments

- `_translate_cell_to_nn_gru(layer) -> nn.GRU` builds a `torch.nn.GRU(num_layers=1, bidirectional=False, batch_first=False)` whose `weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0` reproduce `layer.cell` exactly via `torch.cat([·_r, ·_z, ·_n], dim=0)`. Docstring carries the PyTorch GRU docs URL (D-05) and a cross-reference to `src/gru_qat/gru_cell.py:1-15` for the n-gate asymmetry.
- `_translate_nn_gru_to_cell(gru) -> GRULayer` is the direct layer-level mirror of `tests/test_parity.py:18-44` `_copy_weights` (chunk(3, dim=0) + `with torch.no_grad(): cell.W_ir.copy_(Wir); …`). Honors the `bin_` (trailing underscore) convention because `bin` is a Python builtin.
- `_make_dense_fp32_layer(input_size, hidden_size) -> GRULayer` returns a `GRULayer(recipe=PRESETS["fp32"], gate_layout="split")` with no structure and `use_triton` left at "auto" (resolves to False — the layer is not fast-dispatch eligible). This is the path Plans 02-04 will parametrize.
- `test_gate_order_r_only` / `test_gate_order_z_only`: activate exactly one input-side gate and verify the rest of the cell math reduces correctly. If the cell's internal gate order ever drifts from `(r, z, n)`, the translation helper would silently compensate in the grid — these micro-tests are the only way to surface that.
- `test_n_gate_asymmetry`: forces `r ~ 0` via `b_ir = -100`; keeps `W_in / W_hn / b_in / b_hn` at their `nn.init.uniform_(-1/sqrt(H), 1/sqrt(H))` init values to produce a non-trivial input-branch contribution. Both `nn.GRU` and the cell must agree that the n-gate reduces to `tanh(W_in x + b_in)` — i.e. that `r_t` multiplies only the hidden contribution inside the tanh.
- `test_round_trip_nn_gru_to_cell`: builds an `nn.GRU(8, 16)` first, inverts via `_translate_nn_gru_to_cell`, runs `T=7, B=4` and asserts both `out` and `h_T` agree to `< 1e-4`. Catches inverse-helper bugs that the grid (which only goes cell -> nn.GRU) would never exercise.

## Task Commits

Each task was committed atomically (sequential executor on `feat/diagonal-gru`, no `--no-verify`):

1. **Task 1: scaffold helpers + module preamble** — `786b32c` (test)
2. **Task 2: 3 micro-tests + round-trip smoke** — `3b6f093` (test)

## Files Created/Modified

- `tests/test_layer_parity.py` (created, 295 lines) — module docstring, imports, `torch.set_float32_matmul_precision("highest")` at module scope, three module-level underscore helpers, four test functions.

## Decisions Made

- **Honor D-07 literally:** the matmul-precision call lives at module scope, not inside any test. Phase 1 audits math, not TF32 — same file is meant to fail on algorithm drift, not arithmetic-mode drift.
- **Drop the speculative `import pytest`:** Task 1 initially carried `import pytest  # noqa: F401` as a forward declaration for Task 2; once Task 2 landed without using `pytest.mark.parametrize` / `pytest.raises` / `pytest.fixture`, the import was unused. Removed in the Task 2 commit. Plan 02's parametrized grids will reintroduce it.
- **n-gate-asymmetry test minimality:** chose to zero only `W_ir / W_hr / b_hr` and slam `b_ir = -100` rather than zeroing the full input-side. Keeping `W_in / W_hn / b_in / b_hn` at their default uniform init means the n-gate output is *non-trivial* in both implementations — if either one applied `r` to the input branch (wrong), the assertion would fire. Zeroing everything would only test that `tanh(0) == 0`, which is uninformative.

## Deviations from Plan

None - plan executed exactly as written. (The `import pytest` drop is not a deviation: the plan said "imports — exact order per CONVENTIONS.md" and `pytest` was listed because the original spec assumed it would be needed for Task 2; in the executed Task 2 no `pytest.*` reference was needed, so the import naturally fell out. Ruff would have flagged it as F401 otherwise.)

## Issues Encountered

- **`python` on PATH is system Python, not the project venv.** First `python -c "import tests.test_layer_parity"` invocation hit `ModuleNotFoundError: No module named 'pytest'`. Worked around by invoking `.venv/bin/python` and `.venv/bin/python -m ruff` / `-m pytest` directly. Not a code issue; environment-only.

## Beads Issues Filed

None — all 4 tests passed on the first run. No cell-math bug surfaced, so the D-10/D-11 two-commit protocol did not need to fire. (If any of the micro-tests had failed, Commit A = failing test alone + `bd create --title <test_function_name>` + `bd update <id> --notes <pytest --tb=short tail>`; Commit B = fix in `src/`; `bd close <id>` after CI green. Plan 01-05 will re-verify.)

## Verification Snapshot

```
$ .venv/bin/python -m pytest tests/test_layer_parity.py -q
....                                                                     [100%]
4 passed in 1.81s

$ .venv/bin/python -m ruff check tests/test_layer_parity.py
All checks passed!

$ .venv/bin/python -m pytest tests/test_parity.py -q   # untouched contract
............                                                             [100%]
12 passed in 1.99s

$ grep -c "^def test_" tests/test_layer_parity.py
4

$ grep -n "xfail" tests/test_layer_parity.py
(no output — 0 markers, per D-12)

$ git diff HEAD~2 -- tests/test_parity.py
(empty — < 1e-5 cell parity contract untouched, per PROJECT.md constraint)
```

## Next Phase Readiness

- Helpers are proven self-consistent (round-trip smoke) and gate-order-aware (3 micro-tests). Plans 02-04 can now compose them into the parametrized 75-combo grid (`FAST_GRID` × `SLOW_GRID`) without re-deriving the translation contract.
- No `src/` modifications shipped — the cell / layer code remains exactly as `feat/diagonal-gru` left it. The < 1e-5 cell parity contract in `tests/test_parity.py` is untouched.
- No blockers, no deferred items, no bd issues open against this plan.

## Self-Check: PASSED

- `tests/test_layer_parity.py` exists (295 lines, > 120 min): FOUND.
- `_translate_cell_to_nn_gru`, `_translate_nn_gru_to_cell`, `_make_dense_fp32_layer` are module-level: FOUND (verified via `python -c "import tests.test_layer_parity"`).
- `torch.set_float32_matmul_precision("highest")` at module scope: FOUND (line 33).
- PyTorch GRU docs URL in helper docstring (D-05): FOUND (line 61).
- 4 `def test_` functions, 0 `xfail` markers: VERIFIED.
- `ruff check tests/test_layer_parity.py` exit 0: VERIFIED.
- Commits `786b32c` and `3b6f093` exist on `feat/diagonal-gru`: VERIFIED via `git log --oneline -5`.
- `tests/test_parity.py` `< 1e-5` cell-parity contract intact: VERIFIED (12 passed; `git diff HEAD~2 -- tests/test_parity.py` empty).

---
*Phase: 01-reference-path-parity-vs-nn-gru*
*Completed: 2026-05-13*
