---
phase: 04-quant-on-bit-identity
plan: 03
subsystem: testing
tags: [triton, pytest, quantization, int8, gru, diagonal, monarch, structured-sparsity]

# Dependency graph
requires:
  - phase: 04-quant-on-bit-identity / Plan 04-01
    provides: D-41 recipe + D-42 disposition + QNT-04 fix + _make_dense_layer_quant_int8 + _adversarial_inputs helpers and the dense probe in test_triton_scan_strict.py
provides:
  - "Phase 4 quant-on bit-identity strict-tier sweep for the diagonal kernel: _make_diagonal_layer_quant_int8 helper, _assert_quant_parity helper (byte-identical to D-43 canonical form), _adversarial_inputs helper, QUANT_FAST_GRID (18) / QUANT_SLOW_GRID (9), 4 new test functions (test_diagonal_quant_fwd, _bwd, _fwd_slow, _bwd_slow); 162 parametrized test cases total"
  - "Phase 4 quant-on bit-identity strict-tier sweep for the monarch kernel: _make_monarch_layer_quant_int8(nblocks) helper, byte-identical _assert_quant_parity helper, _adversarial_inputs helper, QUANT_MONARCH_FAST_GRID (54) / SLOW_GRID (27) with H % nblocks == 0 divisibility filter, 4 new test functions; 486 parametrized test cases total"
  - "Disposition-uniform _assert_quant_parity helper present in both files with byte-identical body (D-43 cross-file uniformity for Plan 04-05 verifier)"
affects: [04-04 (already complete, butterfly), 04-05 (verifier audit asserts cross-file uniformity), Phase 5 (calibration lifecycle — this plan's helpers prefigure end-state-identical frozen layers without depending on calibration.py), Phase 7 (audit report)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Disposition-aware _assert_quant_parity helper with strict=True (torch.equal, fwd) / strict=False (abs_diff < h_scale, bwd) — uniform across all four Phase 4 strict files per D-43"
    - "Inline calibrate-then-freeze layer helper: build with mode='min_max' → manually freeze hidden quantizer scales → one forward populates running stats → cell.freeze_quantizers() → returns frozen-INT8 layer end-state-identical to Phase 5's calibrate→freeze without depending on calibration.py"
    - "_build_qgi_from_layer: apply layer.cell.quant_x(x) BEFORE F.linear (D-41 recipe consistency — reference quantizes inside cell.step(), so Triton path must apply same quant before consuming gi)"
    - "Adversarial-class outermost parametrize: cls × shape grid yields pytest IDs like [realistic-8-4-32-2] so the class name surfaces in pytest -ra summary lines without consulting the body"

key-files:
  created: []
  modified:
    - "tests/test_triton_diagonal_strict.py - Phase 4 section appended (+389 lines): _assert_quant_parity, _make_diagonal_layer_quant_int8, _adversarial_inputs, QUANT_FAST/SLOW_GRID, _build_qgi_from_layer, 4 new parametrized test functions (test_diagonal_quant_fwd/_bwd + _slow siblings). Phase 2 fp32 section unchanged."
    - "tests/test_triton_monarch_strict.py - Phase 4 section appended (+407 lines): same helpers (byte-identical _assert_quant_parity per D-43), _make_monarch_layer_quant_int8(nblocks), QUANT_MONARCH_FAST/SLOW_GRID with H % nblocks == 0 filter, 4 new parametrized test functions (test_monarch_quant_fwd/_bwd + _slow siblings). Phase 2 fp32 section unchanged."

key-decisions:
  - "Used cell.quantize_input_weights() (not the realistic-tier _build_gi_from_cell helper) to extract Wi_cat/bi_cat for diagonal/monarch — the helper API is the post-freeze canonical surface and matches how the layer's fast-path dispatch consumes the cell internally"
  - "Helper body of _assert_quant_parity is BYTE-IDENTICAL across both files per D-43 (verified via diff)"
  - "_build_qgi_from_layer is per-file (not shared) per CONTEXT D-47 'per-file extension, no new module' discipline — same body, distinct definitions; matches the per-file _assert_quant_parity copy strategy"

patterns-established:
  - "Disposition-resolved assertion idiom for quant-on Triton/PyTorch parity: strict=True torch.equal for forward (post-INT8-rounding collapses Triton-TF32 and PyTorch-fp32 to same INT8 grid); strict=False abs_diff < h_scale for backward (STE-passthrough fp32 reduction-order drift, bounded by one INT8 step)"
  - "Adversarial-class × shape-grid Cartesian-product parametrize convention for kernel quant-on tests (realistic / near-saturation / large-magnitude × QUANT_*_GRID); failure messages embed cls + T + B + H + (nblocks) for direct triage"

requirements-completed: [QNT-02, QNT-03]

# Metrics
duration: 30min
completed: 2026-05-14
---

# Phase 4 Plan 03: Diagonal + Monarch Kernel Quant-On Strict-Tier Sweep Summary

**Frozen-INT8 quant-on bit-identity parity tests for diagonal + monarch Triton kernels: torch.equal forward + abs_diff < h_scale backward (per D-42 ASYMMETRIC disposition), 648 new parametrized test cases across the two files.**

## Performance

- **Duration:** 30 min
- **Started:** 2026-05-14T11:00Z (approx — orchestrator spawn)
- **Completed:** 2026-05-14T11:30:40Z
- **Tasks:** 2
- **Files modified:** 2 (one per task)

## Accomplishments
- Diagonal kernel Phase 4 quant-on extension: 162 new test cases (18 fast × 3 cls × 2 directions + 9 slow × 3 cls × 2 directions), covering frozen-INT8 per-channel-weight + per-tensor-activation forward AND backward parity at H ∈ {32, 128, 512} against the D-42 asymmetric bound.
- Monarch kernel Phase 4 quant-on extension: 486 new test cases (54 fast × 3 cls × 2 directions + 27 slow × 3 cls × 2 directions), same disposition idiom with the additional nblocks ∈ {2, 4, 8} axis and H % nblocks == 0 divisibility filter.
- Both files share a BYTE-IDENTICAL `_assert_quant_parity` helper body (D-43 cross-file uniformity verified via `diff`).
- Both helpers implement the ACTUAL D-41 recipe (frozen INT8 per-channel weight + per-tensor input_act + per-tensor hidden via inline `cell.freeze_quantizers()`), NOT the looser fp32-weight + frozen-INT8-hidden shortcut from the realistic-tier QAT analogs.
- Test bodies apply `layer.cell.quant_x(x)` BEFORE `F.linear` so the reference and Triton sides both consume the same `gi` (the reference quantizes inside `cell.step()`).
- Smoke-tested on CUDA: `test_diagonal_quant_fwd[realistic-8-1-32]`, `test_monarch_quant_fwd[realistic-8-1-32-2]`, `test_monarch_quant_bwd[realistic-8-1-32-2]`, `test_diagonal_quant_bwd[realistic-8-1-32]` — all PASSED. Validates that the D-42 disposition holds for these kernel + recipe combinations at the smoke shape.

## Task Commits

Each task was committed atomically:

1. **Task 1: Diagonal kernel — Phase 4 section extension** — `592dde5` (test)
2. **Task 2: Monarch kernel — Phase 4 section extension** — `17777bd` (test)

## Files Created/Modified

- `tests/test_triton_diagonal_strict.py` — Phase 4 section appended after the line-302 module-end sentinel. New section contains the D-48 ASCII rule header, disposition rationale comment, `_assert_quant_parity` (D-43 byte-identical), `_make_diagonal_layer_quant_int8` (D-41 recipe via inline freeze), `_adversarial_inputs` (D-46), `QUANT_FAST_GRID` (18) / `QUANT_SLOW_GRID` (9), `_build_qgi_from_layer`, and the 4 test functions `test_diagonal_quant_fwd` / `_bwd` / `_fwd_slow` / `_bwd_slow`. Phase 2 fp32 strict-tier content (lines 1-302) unchanged byte-for-byte (D-52).
- `tests/test_triton_monarch_strict.py` — Phase 4 section appended after the existing slow bwd test. Same helper shape as the diagonal file plus the nblocks parameter on the helper and on the grid constants. `QUANT_MONARCH_FAST_GRID` enforces `H % nblocks == 0` (the divisibility invariant required by `src/gru_qat/structure.py`'s Monarch factor). Phase 2 fp32 monarch content (lines 1-318) unchanged byte-for-byte.

## Decisions Made

- **Helper sharing strategy:** Per-file copies of `_assert_quant_parity`, `_adversarial_inputs`, and `_build_qgi_from_layer` (per CONTEXT D-47 "per-file extension, no new module"). Each copy is byte-identical to the canonical form in `.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md`. Plan 04-05 verifier asserts cross-file uniformity (D-43).
- **Helper API choice:** Used `cell.quantize_input_weights()` (the post-freeze canonical surface) inside `_build_qgi_from_layer`, NOT a hand-rolled `torch.cat` over `cell.quant_W_ir(cell.W_ir)` etc. This matches how the layer's fast-path dispatch consumes the cell internally and avoids duplicating quant logic that's already encapsulated.
- **No autograd wrapper in test bodies:** Direct kernel-call pattern (matches Phase 2 strict-tier shape and the realistic-tier QAT analogs). The autograd `gru_scan_diagonal` / `gru_scan_monarch` paths are exercised in their dispatch-level tests; this plan isolates the kernel pair.

## Deviations from Plan

None - plan executed exactly as written.

The plan body specified the assertion idiom, helper recipe, grid shape, test-function names, and commit-message templates. All were followed verbatim. The D-42 disposition resolution (asymmetric: `torch.equal` fwd + `abs_diff < h_scale` bwd) from Plan 04-01's `checkpoint:human-verify` was already locked into `04-DISPOSITION.md`; this plan consumed it via the byte-identical `_assert_quant_parity` helper.

## Issues Encountered

- **Pre-existing Phase 2 slow-tier failure observed during smoke test:** `test_diagonal_bwd_strict_matches_reference_slow[512-32-64]` (Phase 2, NOT this plan's test) fails on the F-02-02-A dbh accumulator non-associativity drift. The failure is documented in-place in the test's own docstring (lines 245-267 of `tests/test_triton_diagonal_strict.py`) and tracked as a deferred bd issue. Not caused by this plan; not in scope.
- **Bonus finding from 04-DISPOSITION.md (Plan 04-01 probe):** pre-existing Phase 2 strict-tier failures at `test_butterfly_fwd_strict_matches_reference[8-1-32]` (~9.3e-3 vs < 5e-4 bound) and analogous monarch bwd cases (~7.4e-4). Stash-verified pre-existing on Plan 04-01 baseline; tracked in a bd issue, deferred to a future hygiene phase per the orchestrator-level decision logged in `04-DISPOSITION.md` "Bonus finding" section.

## Hygiene Gates (all green)

| Gate | Threshold | Diagonal file | Monarch file |
|------|-----------|---------------|--------------|
| `grep -c "strict=True"` | ≥ 2 | 7 | 7 |
| `grep -c "strict=False"` | ≥ 4 | 11 | 11 |
| `grep -c "xfail"` | == 0 | 0 | 0 |
| `grep -c "quant_x\|freeze_quantizers"` | ≥ 3 | 7 | 6 |
| `ruff check` | clean | ✓ | ✓ |
| New `test_<kind>_quant_(fwd\|bwd)` test items (`--collect-only`) | sweep complete | 162 | 486 |
| Phase 2 fp32 section deletions (`git diff "^-"`) | empty | empty | empty |
| Locked-files diff (`tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py`) | empty | empty | empty |
| `_assert_quant_parity` body cross-file `diff` | byte-identical | identical | identical |
| CUDA smoke test (one fwd + one bwd per kernel) | PASSED | PASSED | PASSED |

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 04-04 (butterfly, parallel) completed independently — commits `02881eb` (test) + `82bf986` (SUMMARY) already landed on the branch by the time this SUMMARY is written.
- Plan 04-02 (dense extension of `tests/test_triton_scan_strict.py`) likewise completed in parallel — commits `eacb553` (test) + `4c172b8` (SUMMARY) already landed.
- All three Wave 2 plans (04-02, 04-03, 04-04) extend their respective strict files with the same D-43 disposition idiom. Cross-file uniformity is now verifier-ready for Plan 04-05.
- Plan 04-05 (audit / phase-exit GPU run) is the next gate. It will run the full 4-kernel quant-on sweep on CUDA hardware and assert cross-file `_assert_quant_parity` uniformity (D-43), Phase 2 fp32 sections unchanged (D-52), locked files unchanged (D-51), and no xfails anywhere (D-50).

## Self-Check: PASSED

- `tests/test_triton_diagonal_strict.py` exists and contains the new section: `[ -f tests/test_triton_diagonal_strict.py ]` → FOUND. `grep -q "def test_diagonal_quant_fwd" tests/test_triton_diagonal_strict.py` → FOUND.
- `tests/test_triton_monarch_strict.py` exists and contains the new section: `[ -f tests/test_triton_monarch_strict.py ]` → FOUND. `grep -q "def test_monarch_quant_fwd" tests/test_triton_monarch_strict.py` → FOUND.
- Commit `592dde5` exists: `git log --oneline --all | grep 592dde5` → FOUND ("test(04-03): diagonal kernel quant-on full sweep per D-49 + D-46").
- Commit `17777bd` exists: `git log --oneline --all | grep 17777bd` → FOUND ("test(04-03): monarch kernel quant-on full sweep with nblocks axis").
- `_assert_quant_parity` body byte-identical across the two files: `diff` returned empty.
- Smoke tests on CUDA: `test_diagonal_quant_fwd[realistic-8-1-32]`, `test_monarch_quant_fwd[realistic-8-1-32-2]`, `test_monarch_quant_bwd[realistic-8-1-32-2]`, `test_diagonal_quant_bwd[realistic-8-1-32]` → all 4 PASSED.

---
*Phase: 04-quant-on-bit-identity*
*Completed: 2026-05-14*
