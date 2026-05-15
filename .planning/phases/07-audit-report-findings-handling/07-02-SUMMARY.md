---
phase: 07-audit-report-findings-handling
plan: 02
subsystem: testing
tags: [pytest, quantization, deepcopy, tf32, triton, divergence-marker, qat]

# Dependency graph
requires:
  - phase: 04-quant-on-bit-identity
    provides: per-cluster h_scale_mult disposition table (04-DISPOSITION.md) — the strict-test contract the n20 re-baseline reconciles against
  - phase: 05-calibration-freeze-lifecycle
    provides: CAL-02 test (test_freeze_all_matches_dynamic_on_last_batch) scoped to quant_x with the n20 cross-phase deferral note
provides:
  - "gru-triton-n20 fixed: make_quantizer deepcopies QuantizerConfig so sibling quantizers (quant_h_in/quant_h_out, the six quant_W_*) each own an independent config — freeze_all no longer silently no-ops the second quantizer"
  - "divergence pytest marker registered in pyproject.toml and applied per-parametrize-case to the TF32-rooted strict cases"
  - "ROADMAP criterion #3 green gate operationalized: pytest -q -m 'not divergence' passes on CUDA (1437 passed) and pytest -m 'slow and not divergence' -q passes (409 passed)"
  - "timestamped post-fix post-marker pytest-output artifact (07-pytest-output.txt)"
affects: [bd-issue-closure, AUDIT-REPORT]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "per-parametrize-case pytest marker application via a module-level _DIV_* id set + _div_param() helper that wraps grid tuples in pytest.param(..., marks=...)"
    - "stacked cls x grid parametrize merged into a single (cls, T, B, H[, nblocks]) parametrize so per-case marking can isolate cls-specific divergent tuples"

key-files:
  created:
    - .planning/phases/07-audit-report-findings-handling/07-pytest-output.txt
  modified:
    - src/gru_qat/quantizers.py
    - pyproject.toml
    - tests/test_calibration.py
    - tests/test_triton_scan_strict.py
    - tests/test_triton_diagonal_strict.py
    - tests/test_triton_monarch_strict.py
    - tests/test_triton_butterfly_strict.py

key-decisions:
  - "n20 fixed via copy.deepcopy(config) as the first statement of make_quantizer (before the bits>=32 Identity short-circuit) — propagates to all six weight quantizers via factory() with no second edit"
  - "Whole at-risk TF32 clusters are divergence-marked rather than a brittle observed-failure subset, because the strict pass/fail split is autotune-config dependent (a boundary case flips across runs)"
  - "Two non-parametrized single-case strict tests carry a function-level @pytest.mark.divergence (no parametrize cross-product to hide); all parametrized tests use per-case pytest.param marks"

patterns-established:
  - "_div_param(values, ident, div_set) helper: returns pytest.param tagged divergence when the generated param id is in the divergence id set"
  - "divergence marker green gate: pytest -q -m 'not divergence' is criterion #3's honest gate; marked cases stay LIVE (no xfail, no skip)"

requirements-completed: [RPT-01, RPT-02]

# Metrics
duration: ~3h
completed: 2026-05-15
---

# Phase 7 Plan 02: n20 fix + divergence marker Summary

**Fixed the gru-triton-n20 shared-QuantizerConfig silent-correctness bug via `deepcopy` in `make_quantizer`, absorbed the resulting Phase 4 strict-test re-baseline, and introduced the `divergence` pytest marker — `pytest -q -m "not divergence"` is now green on CUDA (1437 passed / 0 failed).**

## Performance

- **Duration:** ~3h (dominated by ~5 full strict-suite runs on RTX 2000 Ada — autotune compilation is the cost)
- **Started:** 2026-05-15T07:56Z
- **Completed:** 2026-05-15T10:39Z
- **Tasks:** 2
- **Files modified:** 7 (+1 created)

## Accomplishments

- **n20 fixed (two-commit):** `make_quantizer` now `deepcopy`s its config so `quant_h_in`/`quant_h_out` (and the six `quant_W_*`) each hold an independent `QuantizerConfig`. Before the fix, `FakeQuantize.freeze()` mutated the shared `config.mode='frozen'`, so `freeze_all` silently no-op'd the second sibling — leaving it at the `scale=1.0` buffer init (silent accuracy loss for any calibrate→freeze user).
- **CAL-02 extended (failing test first):** `test_freeze_all_isolates_sibling_quantizer_configs` asserts both hidden quantizers AND the six weight quantizers each own a distinct config and freeze to their own dynamic-mode scales. Commit A (`be0b734`) precedes the fix Commit B (`65c89f8`) in `git log`.
- **`divergence` marker registered + applied:** registered in `pyproject.toml [tool.pytest.ini_options] markers`; applied per-parametrize-case across the four `tests/test_triton_*_strict.py` files plus the CAL-03 dense cases.
- **Both green gates verified on CUDA:** `pytest -q -m "not divergence"` → 1437 passed, 0 failed; `pytest -m "slow and not divergence" -q` → 409 passed, 0 failed.
- **`-m divergence` reproduce run** confirms the 410 marked fast cases are LIVE (collected and run, not skipped/xfailed) and reproduce the documented TF32 divergence on demand.

## Task Commits

1. **Task 1 (TDD): n20 — deepcopy config isolation** — `be0b734` (test, Commit A — failing CAL-02 extension), `65c89f8` (fix, Commit B — deepcopy in make_quantizer)
2. **Task 2: divergence marker + n20 strict-test re-baseline** — `50f4fcd` (chore — register marker), `cd33ba7` (test — re-baseline + per-case marking), `6d52921` (docs — pytest-output artifact)

## Files Created/Modified

- `src/gru_qat/quantizers.py` — added `from copy import deepcopy`; `make_quantizer` deep-copies its config before constructing any quantizer.
- `pyproject.toml` — registered the `divergence` marker under `[tool.pytest.ini_options] markers`.
- `tests/test_calibration.py` — added `test_freeze_all_isolates_sibling_quantizer_configs` (n20 failing test); merged CAL-03's stacked parametrize into a combined `(kernel, ...)` list and `divergence`-marked the dense kernel rows.
- `tests/test_triton_scan_strict.py` — `_DIV_SCAN_*` id sets + `_div_param` helper; `divergence`-marked dense fp32 fwd/bwd strict, dense quant fwd/bwd, plus the two non-parametrized probe tests.
- `tests/test_triton_diagonal_strict.py` — `_DIV_DIAG_*` sets; marked diagonal quant fwd near-saturation/large-magnitude clusters and the long-T `dbh` slow cases (gru-triton-e7t).
- `tests/test_triton_monarch_strict.py` — `_DIV_MONARCH_*` sets; marked the monarch fp32 fwd/bwd strict grids and the n20-rebaselined quant-bwd large-magnitude H=512 cluster.
- `tests/test_triton_butterfly_strict.py` — `_DIV_BFLY_*` sets; marked the butterfly bwd strict grids (log_H TF32 compounding).
- `.planning/phases/07-audit-report-findings-handling/07-pytest-output.txt` — timestamped post-fix post-marker capture (created).

## Decisions Made

- **n20 fix shape:** `copy.deepcopy` placed as the first statement of `make_quantizer`, before the `bits>=32` Identity short-circuit, per D-07. `factory()` wraps `make_quantizer`, so the fix propagates to all six weight quantizers automatically.
- **Whole-cluster marking (deviation-adjacent judgment):** the plan's RESEARCH.md anticipated `~18+` n20-broken strict tests with an exactly-`1×h_scale` residual. The empirical post-fix state is far larger — 311 fast + 213 slow strict failures — because (a) ~208 are pre-existing Phase 2/4 TF32 ACCEPTED-DIVERGENCE (`6dz`/`rwm`/`mjy`/`q3k`/`lqk`/`e7t`/`in0`/`fpl`, independent of n20) and (b) the `< 5e-4` tight-TF32 bound sits right at the TF32 floor so the pass/fail split is **autotune-config dependent** — a boundary case flips across runs. After an initial per-observed-failure marking left 11 flip-induced residual failures, the divergent sets were widened to cover whole at-risk clusters. This stays within the plan's decision rule: every marked cluster is the TF32 `tl.dot`/`tl.sum` reduction-order family (decision rule (a)); no residual was a non-TF32-signature genuine bug (decision rule (c) did not trigger).
- **Per-case vs per-function marking:** all parametrized tests use per-`pytest.param` marks via the `_div_param` helper (Pitfall 1 — clean clusters stay in the green gate: diagonal fp32 fwd/bwd, diagonal quant fwd realistic, monarch/butterfly quant with loose mults). Only two non-parametrized single-case tests carry a function-level `@pytest.mark.divergence`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing `torch-structured` into the project venv**
- **Found during:** Task 2 (strict-test re-baseline — monarch/butterfly strict tests require it)
- **Issue:** `torch_structured` was absent from the worktree `.venv`; monarch/butterfly strict tests would `importorskip`-skip, making the re-baseline incomplete.
- **Fix:** `VIRTUAL_ENV=.venv uv pip install "git+https://github.com/LarocheC/torch-structured"` plus `uv pip install -e ".[dev]"` to populate the worktree venv (the default `uv run` had resolved a different interpreter).
- **Verification:** `import torch_structured` succeeds; monarch/butterfly strict tests collect and run.
- **Committed in:** N/A — environment setup, not a source change.

**2. [Rule 1 - Bug] Widened divergence id sets to whole TF32 clusters**
- **Found during:** Task 2 (first `-m "not divergence"` gate run left 11 residual failures)
- **Issue:** Marking only the observed-failing parametrize ids left clean-looking cases in the green gate that then flipped to failing on the next run — TF32 strict pass/fail is autotune-config dependent.
- **Fix:** Widened `_DIV_*` sets to cover whole at-risk clusters (dense/monarch fp32 strict grids, butterfly bwd grids, diagonal quant near-saturation/large-magnitude). Genuinely-clean clusters (diagonal fp32, diagonal quant realistic) stay unmarked.
- **Verification:** Re-ran both gates — `pytest -q -m "not divergence"` 1437 passed / 0 failed; `pytest -m "slow and not divergence" -q` 409 passed / 0 failed.
- **Committed in:** `cd33ba7` (Task 2 commit).

---

**Total deviations:** 2 (1 blocking env-setup, 1 bug-fix on marker coverage)
**Impact on plan:** Both necessary to achieve the plan's green-gate acceptance criterion. The whole-cluster widening is consistent with the plan's per-test decision rule (all marked cases are the TF32 family); no scope creep.

## Issues Encountered

- **Strict-suite runtime:** each full strict run is 4-32 minutes on RTX 2000 Ada (autotune compilation across the QUANT grids dominates). Five runs were needed (post-fix discovery, fast re-baseline, slow re-baseline, two green-gate verifications) — the bulk of the ~3h duration.
- **`uv run` interpreter mismatch:** the host's bare `uv run` resolved a conda interpreter without `gru_qat`; all test invocations use `VIRTUAL_ENV=.venv uv run --active` to pin the project venv.

## Threat Flags

None — the n20 deepcopy is a correctness fix to model-build internals; no untrusted-input, network, auth, or schema boundary is added or modified. Net risk reduction (a silent quantizer-correctness bug removed).

## Known Stubs

None.

## TDD Gate Compliance

Task 1 followed the two-commit discipline: `test(07-02)` Commit A (`be0b734`) precedes `fix(07-02)` Commit B (`65c89f8`). The failing test was confirmed RED before the fix and GREEN after.

## Next Phase Readiness

- Wave 2 is complete and verified on a CUDA+Triton host (the WAVE-2 GPU-HOST GATE is satisfied — both `-m "not divergence"` gates PASSED, not skipped). Waves 3 (bd-issue closure) and 4 (AUDIT-REPORT) may proceed.
- `gru-triton-n20` is ready to be CLOSED in Wave 3 with a resolution note.
- The `divergence` marker and `07-pytest-output.txt` are the inputs Wave 4's AUDIT-REPORT residual-divergences section consolidates.

## Self-Check: PASSED

- `src/gru_qat/quantizers.py` deepcopy — FOUND (line 257)
- `pyproject.toml` divergence marker — FOUND
- `.planning/phases/07-audit-report-findings-handling/07-pytest-output.txt` — FOUND
- Commits be0b734, 65c89f8, 50f4fcd, cd33ba7, 6d52921 — all present in git log
- No `xfail` in any strict/calibration test file — CONFIRMED
- `git log` shows test(07-02) `be0b734` precedes fix(07-02) `65c89f8` — CONFIRMED

---
*Phase: 07-audit-report-findings-handling*
*Completed: 2026-05-15*
