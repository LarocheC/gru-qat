---
phase: 03-structured-pytorch-fallback-parity
plan: SUMMARY
type: phase-exit
subsystem: structured-pytorch-fallback-parity
status: complete
tags: [phase-exit, audit, structured, parity, circulant, ldr, optional-dep, monkeypatch]
closed_at: 2026-05-14

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    provides: "Detach-clone-twice + shared-g + per-tensor named-failure loop idiom; 'highest' precision preamble; absolute-error assertion convention."
  - phase: 02-triton-fast-path-parity-vs-reference
    provides: "Strict-tier file-naming pattern (test_*_strict.py / test_structure_parity.py); no-xfail discipline; D-28 locked-files contract extended to Phase 3 as D-38."
provides:
  - "Circulant per-step PyTorch path pinned against two independent hand-rolled references (Toeplitz + full-complex FFT) at < 1e-5 abs fwd + bwd."
  - "LDR per-step PyTorch path pinned against an independent hand-rolled slow-Krylov dense reference at < 1e-5 abs fwd + bwd."
  - "Optional-dep failure-mode audit (STR-03): monarch / butterfly / ldr raise clear ImportError with 'torch-structured' install hint when the dep is missing; dense / diagonal / circulant continue to work."
  - "monkeypatch convention codified in .planning/codebase/TESTING.md, scoped to optional-dep failure-mode tests only."
  - "tests/test_structure_parity.py = single home for the entire Phase 3 strict-tier audit (810 lines, 88 fast + 24 slow tests)."
affects: [phase-04-quant-on-bit-identity]

# Tech tracking
tech-stack:
  added:
    - "pytest.MonkeyPatch (narrow exception, optional-dep tests only — Plan 03-03)"
  patterns:
    - "Two-reference cross-check (Plan 03-01): hand-rolled Toeplitz + hand-rolled full-complex FFT verified against each other BEFORE either is compared to production."
    - "External-library-as-spec via Task 0 (Plan 03-02): read torch_structured source as a spec, reconstruct independently in tests, record provenance as a comment block."
    - "torch.fft.fft / torch.fft.ifft (full complex) in test code — first use in repo, genuinely independent of production's rfft path."
    - "Per-section pytest.importorskip mid-file (Plan 03-02): one file can host both deps-free and deps-required tests without skipping the deps-free section."
    - "monkeypatch.setitem(sys.modules, ..., None) (Plan 03-03): canonical idiom for simulating a missing dep when production code does `from <pkg> import ...` directly, bypassing an internal lazy-import helper."

key-files:
  created:
    - "tests/test_structure_parity.py (810 lines total — 286 lines Plan 03-01, +384 lines Plan 03-02, +140 lines Plan 03-03)"
    - ".planning/phases/03-structured-pytorch-fallback-parity/03-SUMMARY.md (this file)"
  modified:
    - ".planning/codebase/TESTING.md (Mocking section scoped — Plan 03-03)"
    - "src/gru_qat/structure.py: UNCHANGED across all Phase 3 commits (no production findings)"

requirements-completed: [STR-01, STR-02, STR-03]

# Metrics
plans-completed: 3
plans-total: 3
waves: 3
test-files-touched: 1
test-files-locked: 3
new-test-cases: 112  # 88 fast + 24 slow (was 0 at phase start; file is new)
bd-issues-opened-during-phase: 0
duration: ~1h (3 plans @ ~5-25min each)
completed: 2026-05-14
---

# Phase 3 SUMMARY: Structured PyTorch Fallback Parity

**Pinned Circulant + LDR per-step PyTorch paths against independent hand-rolled
references at worst-case 2.62e-6 abs (circulant bwd, H=512) and 1.67e-6 abs
(LDR fwd, H=512). Audited torch-structured optionality: all three optional-dep
kinds raise clear ImportError, all three local-impl kinds work without the dep.
Zero production findings.**

## Audit Verdict

**PASS.**

| Requirement | Outcome | Plan | Worst datum                                   |
| ----------- | ------- | ---- | --------------------------------------------- |
| STR-01      | PASS    | 03-01 | Circulant fwd 2.27e-6 / bwd 2.62e-6 (H=512)  |
| STR-02      | PASS    | 03-02 | LDR fwd 1.67e-6 / bwd 1.31e-6 (G leaf, H=512) |
| STR-03      | PASS    | 03-03 | 6/6 new tests GREEN; no production fix needed |

All bounds at < 1e-5 abs with ~4-13x headroom. No `xfail` markers. No locked-file
modifications. Zero `bd` issues opened during the phase.

## Wave Outcomes

### Wave 1 — Plan 03-01 (Circulant quant-off parity)

- Created `tests/test_structure_parity.py` (286 lines): module docstring, `'highest'` preamble, `_build_toeplitz_from_kernel` + `_circulant_via_fft` helpers, FAST/SLOW shape grids, 5 test functions (3 fast + 2 slow siblings, 27 fast + 6 slow parametrized cases).
- **Two-reference cross-check** introduced: hand-rolled Toeplitz form and hand-rolled full-complex FFT form verified against each other (self-consistency tier) BEFORE either compared to production. Catches reference-math bugs before they masquerade as production-path bugs.
- Forward parity: `_CirculantLinear.forward(x)` vs `x @ C.T` — worst 2.27e-6 abs at H=512.
- Backward parity (autograd-grad): `kernel_c` gradient match — worst 2.62e-6 abs at H=512.
- Two auto-fixed test-construction issues (both Rule 1, both in tests/, neither a production bug): (a) production-side leaf via `layer.col.copy_()` not `nn.Parameter(c_prod)` assignment so `.grad` lands correctly; (b) `g` scaled by `1/sqrt(B*H)` so gradient magnitudes stay O(1) and the absolute bound stays meaningful at H ≥ 128.
- D-37 two-commit protocol NOT invoked — `src/gru_qat/structure.py` `_CirculantLinear` unchanged.
- Commits: `987c770` (Task 1 skeleton + helpers), `c8beb6d` (Task 2 tests), `938675b` (Plan SUMMARY).

### Wave 2 — Plan 03-02 (LDR quant-off parity)

- Extended `tests/test_structure_parity.py` from 286 to 670 lines: comment block recording the Task 0 external-library spec read, per-section `pytest.importorskip("torch_structured")` + warnings filter, `_build_ldr_matrix_from_factors` helper (slow Krylov form, Python-loop), FAST_LDR_GRID + SLOW_LDR_GRID, 5 test functions (1 micro + 2 fwd fast/slow + 2 bwd fast/slow). 27 fast + 9 slow LDR cases per family.
- **External-library-as-spec via Task 0** introduced: Task 0 was a read-only step ("locate torch_structured on disk, read `structured/layers.py` and `structured/krylov.py`, record findings as a comment block"). Treating the external library as a canonical spec — not as a black-box oracle to replay — forced clear thinking about the displacement-rank formula and the transpose convention.
- **Micro-validation before parametrized grid**: a single non-parametrized test at (H=8, rank=2) locks the transpose convention `M = sum_i K_A(G[i]) @ K_B(H[i]).T` (`.T` on `K_B`, not `K_A`) BEFORE the 27-case fast grid runs. Without this, a transpose flip would surface as 27 simultaneous red boxes with no diagnostic.
- Forward parity: `_LDRLinear(LDRSubdiagonal(H, r))(x)` vs `x @ M.T` — worst 1.67e-6 abs (H=512, rank=1).
- Backward parity: 4-leaf detach-clone-twice (`subd_A`, `subd_B`, `G`, `H`) — worst 1.31e-6 abs on the `G` leaf (H=512).
- One auto-fixed sequencing issue (Rule 3, not a bug): re-ordered `import torch.nn as nn` from Task 1 → Task 2 to keep ruff green after Task 1 commit.
- D-37 NOT invoked — neither `_LDRLinear` nor `torch_structured` upstream needed changes.
- Commits: `d8b2068` (Task 0+1 spec read + helper + micro), `6489cc3` (Task 2 parametrized fwd + bwd), `68e4cef` (Plan SUMMARY).

### Wave 3 — Plan 03-03 (STR-03 optional-dep failure-mode audit, this plan)

- Extended `tests/test_structure_parity.py` from 670 to 810 lines: added STR-03 section with comment block documenting the mocking strategy matrix, `_raise_missing_torch_structured` helper, 3 new test functions (6 parametrized cases total):
    1. `test_missing_torch_structured_raises_clear_error[monarch]` — GREEN
    2. `test_missing_torch_structured_raises_clear_error[butterfly]` — GREEN
    3. `test_missing_ldr_raises_clear_error` — GREEN
    4. `test_local_impls_work_without_torch_structured[dense]` — GREEN
    5. `test_local_impls_work_without_torch_structured[diagonal]` — GREEN
    6. `test_local_impls_work_without_torch_structured[circulant]` — GREEN
- Two mocking idioms used per D-34:
    * `monkeypatch.setattr("gru_qat.structure._import_torch_structured", _raise_missing_torch_structured)` — covers monarch, butterfly, and the local-impl negative controls (dense/diagonal/circulant).
    * `monkeypatch.setitem(sys.modules, "torch_structured[...].layers", None)` — used **only** for LDR because its branch in `src/gru_qat/structure.py:160-172` bypasses `_import_torch_structured` with a direct `from torch_structured.structured.layers import LDRSubdiagonal`. Setting `sys.modules[name] = None` is Python's documented "this module is known absent" marker and forces ImportError on the next `from` import.
- Updated `.planning/codebase/TESTING.md` "Mocking" section: replaced the absolute "no monkeypatching" rule with a narrow exception scoped to optional-dep failure-mode tests. Two blessed idioms documented; explicitly NOT extended to logic tests.
- Audit verdict: production code's optional-dep guards are correctly placed. **No `src/` changes were required.**
- D-37 NOT invoked.
- Commits: `82c0b1c` (STR-03 tests), `1945afd` (TESTING.md update), this SUMMARY commit forthcoming.

## STR-01..03 Closure Status

| Issue   | Status | Notes                                                                                          |
| ------- | ------ | ---------------------------------------------------------------------------------------------- |
| STR-01  | CLOSED | Circulant fwd + autograd-bwd matches Toeplitz reference at < 2.62e-6 abs across 12 shapes.     |
| STR-02  | CLOSED | LDR fwd + autograd-bwd matches slow-Krylov reference at < 1.67e-6 abs across 36 (B, H, rank). |
| STR-03  | CLOSED | All optional-dep kinds raise clear ImportError; all local-impl kinds work without the dep.    |

## Tier-by-tier Max Abs Diff Datums

| Plan  | Tier                                            | Cases | Worst max abs diff | Worst shape          | Bound  |
| ----- | ----------------------------------------------- | ----- | ------------------ | -------------------- | ------ |
| 03-01 | Self-consistency (FFT vs Toeplitz)              | 9     | 2.27e-6            | (32, 512)            | < 1e-5 |
| 03-01 | Circulant fwd (production vs Toeplitz)          | 12    | 2.27e-6            | H=512                | < 1e-5 |
| 03-01 | Circulant bwd (autograd-grad on kernel_c)       | 12    | 2.62e-6            | H=512                | < 1e-5 |
| 03-02 | LDR micro (H=8, rank=2)                         | 1     | 1.19e-7            | (4, 8, 2)            | < 1e-5 |
| 03-02 | LDR fwd (production vs slow-Krylov)             | 36    | 1.67e-6            | (32, 512, 1)         | < 1e-5 |
| 03-02 | LDR bwd subd_A                                  | 36    | 4.77e-7            | (32, 512, 1)         | < 1e-5 |
| 03-02 | LDR bwd subd_B                                  | 36    | 4.92e-7            | (4, 512, 1)          | < 1e-5 |
| 03-02 | LDR bwd G                                       | 36    | 1.31e-6            | (4, 512, 1)          | < 1e-5 |
| 03-02 | LDR bwd H                                       | 36    | 1.19e-6            | (4, 512, 1)          | < 1e-5 |
| 03-03 | Optional-dep audit (ImportError + finite-output) | 6     | n/a (boolean gate) | —                    | n/a    |

**Overall worst gap:** 2.62e-6 (circulant backward at H=512). The strict 1e-5 bound has ~4-13x headroom across the entire grid. Backward on the `G` factor (LDR rank-r outer products) consistently shows the largest gap among the 4 LDR leaves, which is expected — `G` participates in every rank-r term and accumulates the most fp32 round-off.

## bd Issue Tally

| bd_id            | requirement | summary                                       | root_cause     | fix_commit | regression_test |
| ---------------- | ----------- | --------------------------------------------- | -------------- | ---------- | --------------- |
| (none)           | —           | —                                             | —              | —          | —               |

- **Opened during Phase 3:** 0.
- **Closed during Phase 3:** 0.
- **Outstanding at phase exit:** 2 (both pre-date Phase 3): `gru-triton-4m6` (Phase 1: pre-existing mypy/ruff debt in `src/gru_qat/*`); `gru-triton-e7t` (Phase 2 F-02-02-A: diagonal `dbh` long-T accumulator non-associativity, accepted at < 2e-5 slow-tier bound).

Phase 3 surfaced **zero** new findings. Both the audit of two PyTorch fallback paths (STR-01, STR-02) and the optional-dep contract (STR-03) confirmed existing production code was correct.

## D-38 Locked-File Integrity Verification

Files locked at the start of Phase 3 (D-38, extending D-28), confirmed unchanged on phase exit:

```bash
$ git diff $(git merge-base HEAD <phase-3-base>)..HEAD -- \
    tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py
$ # (empty output — confirmed clean)
```

Concretely, against the latest Phase 3 commit:

```bash
$ git diff --stat tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py
$ # (no diff lines — files identical to phase-3-base across all 9 phase-3 commits)
```

Locked-file pytest runs on phase exit (CPU-only, all GREEN):

- `pytest tests/test_parity.py -q` → **12 passed**
- `pytest tests/test_layer_parity.py -q -m "not slow"` → **184 passed**, 120 deselected
- `pytest tests/test_structure.py -q` → **20 passed**

## New Patterns Introduced in Phase 3

1. **monkeypatch convention for optional-dep tests (Plan 03-03, D-34).** The codebase had zero `monkeypatch` usage before Phase 3. The previously absolute "no mocking" rule in `.planning/codebase/TESTING.md` would have blocked the only sensible way to prove `torch-structured` is optional. Replaced with a narrow exception (two blessed idioms: `setattr` on internal helper; `setitem(sys.modules, ..., None)` for direct-import paths). Going-forward rule: every new optional dep grows a matching failure-mode test ("if the dep is optional, prove it in a test").

2. **`torch.fft.fft` / `torch.fft.ifft` (full complex) in test code (Plan 03-01).** The only prior `torch.fft.*` use in the repo is `src/gru_qat/structure.py:220-222` (rfft/irfft, production). Plan 03-01 introduces full-complex FFT in tests as the genuinely independent FFT reference — deliberately divergent from production's rfft path so the self-consistency check exercises a different code path.

3. **Two-reference cross-check methodology (Plan 03-01).** Phase 1 and Phase 2 compare a single hand-rolled or library path to production. Phase 3 introduces "verify Toeplitz vs FFT self-consistency BEFORE either is compared to production." Catches reference-math bugs before they masquerade as production-path bugs.

4. **External-library-as-spec via Task 0 (Plan 03-02).** Before writing any LDR test code, Task 0 was a read-only "locate torch_structured on disk and read `structured/layers.py` + `structured/krylov.py` line-by-line, record findings as a comment block in the test file." Treating an external library as a canonical spec (rather than as an oracle to replay) forced clear thinking about the transpose convention (`M = sum_i K_A(G[i]) @ K_B(H[i]).T` — `.T` on `K_B`, not `K_A`) and the parameter shapes. The micro-validation test at (H=8, rank=2) then locked the convention in before the parametrized grid ran.

5. **Per-section `pytest.importorskip` mid-file (Plan 03-02).** Conventional pattern is `importorskip` at module top, which would have skipped all circulant tests on a machine without `torch-structured`. Plan 03-02 places the `importorskip` mid-file, between the circulant section and the LDR section, so one file hosts both deps-free and deps-required tests. STR-03's `test_local_impls_work_without_torch_structured` validates that the circulant section above the `importorskip` does in fact run when the dep is mocked away.

6. **`g` scaled by `1/sqrt(B*H)` for absolute backward bounds (Plan 03-01, reused 03-02).** Unscaled `g = torch.randn(B, H)` drives gradient magnitudes to `O(sqrt(B*H))`, pushing fp32 round-off above 1e-5 at H ≥ 128 — algorithmically correct but exceeding the strict absolute bound. Scaling g by `1/sqrt(B*H)` keeps gradient magnitudes O(1) so `< 1e-5 abs` stays meaningful at large shapes, without giving up the diagnostic-power of the shared-g pattern.

## Hand-off to Phase 4 (Quant-on Bit-Identity)

Phase 3 delivers **quant-off forward + backward parity** for circulant and LDR. Phase 4's scope is **quant-on bit-identity** for the structured kinds — proving that the structured-hidden GEMM with active `FakeQuantize` insertions matches the dense path's `FakeQuantize` semantics under a deterministic recipe.

**What Phase 4 inherits clean from Phase 3:**

- `tests/test_structure_parity.py` (810 lines, 88 fast + 24 slow): GREEN strict-tier audit covering circulant + LDR quant-off + optional-dep contract. Phase 4 should **extend** this file with a new STR-04 section gated by a `--quant-on` test config (or by an active `QuantRecipe` parametrize axis); do not create a parallel file.
- `src/gru_qat/structure.py`: unchanged across all of Phase 3. Production dispatch surface is stable.
- `.planning/codebase/TESTING.md`: monkeypatch convention codified — reuse for any quant-on path with optional-component fallbacks.
- The "external-library-as-spec via Task 0" pattern: applicable any time Phase 4 audits a torch-structured-backed path with fake-quant in the loop.

**Flag for Phase 4 (from `src/gru_qat/structure.py:14-20` module docstring):**

- `_LDRLinear` wraps `torch_structured`'s `LDRSubdiagonal.forward` and does NOT have in-kernel fake-quant — the production design applies fake-quant on the **output** of the structured matmul, before bias. Phase 4 must honor this output-side fake-quant pattern, NOT try to insert fake-quant on the LDR factors directly.

**Known risks Phase 4 needs to budget for:**

- **Circulant fp32 FFT vs naive bit-identity.** `_CirculantLinear` uses `torch.fft.rfft / torch.fft.irfft` (fp32). Bit-identity through FFT is not generally possible without going to integer arithmetic. Phase 4 may need a *different* parity target for circulant (e.g. "matches dense bit-identically when FFT is bypassed by replacing the kernel with an explicit Toeplitz matmul" — Plan 03-01's `_build_toeplitz_from_kernel` is reusable here) or a recipe-specific tolerance gate for the circulant kind.
- **LDR fake-quant interaction with `torch-structured`.** The upstream `LDRSubdiagonal.forward` calls into `subdiag_mult` (FFT-based). Phase 4 must confirm that output-side fake-quant on the wrapper (`_LDRLinear.forward`) is sufficient, or that there's a clean way to gate the upstream FFT path with the quant recipe. If output-only fake-quant suffices (which the module docstring implies), the audit becomes "the wrapper's fake-quant output is bit-identical to a dense path's fake-quant output on the same (W, b) realization."
- **D-38 locked files carry forward unchanged through Phase 4.** `tests/test_parity.py`, `tests/test_layer_parity.py`, `tests/test_structure.py` remain locked. Phase 4 verifier asserts empty diff.

**No-xfail discipline (D-39) carries forward.** Phase 3 introduced zero `xfail` markers. Phase 4 should maintain the discipline — failures land as `bd create` + failing test + fix, not as silently-skipped tests.

## Velocity Datum

- **Net-new test cases added in Phase 3:** 112 (88 fast + 24 slow).
- **Net-new test code lines:** 810 (file was new at phase start).
- **Test cases by plan:**
    - 03-01: 27 fast + 6 slow = 33 (circulant self-consistency + fwd + bwd, fast + slow).
    - 03-02: 55 fast + 18 slow = 73 (LDR micro + fwd + bwd, with per-test parametrize over B × H × rank).
    - 03-03: 6 fast + 0 slow = 6 (STR-03 optional-dep audit, CPU-only, no slow tier).
- **Plans:** 3 (one per requirement: STR-01, STR-02, STR-03).
- **Total commits in Phase 3:** 9 (3 per plan × 3 plans: skeleton/helper + parity tests + plan SUMMARY for plans 1+2; STR-03 tests + TESTING.md + this phase-exit SUMMARY for plan 3).
- **Wall-clock:** ~1h aggregate per commit timestamps. Plan 03-01 was the longest single-task (backward parity test-construction iteration); plans 03-02 and 03-03 went linearly per their `<read_first>` instructions.

## Self-Check: PASSED

- `.planning/phases/03-structured-pytorch-fallback-parity/03-SUMMARY.md` — FOUND (this file).
- `tests/test_structure_parity.py` — FOUND, 810 lines, 112 tests collected (88 fast + 24 slow).
- `.planning/codebase/TESTING.md` — FOUND, "Mocking" section updated; `grep -c "monkeypatch"` returns 3, `grep "No.*monkeypatching"` returns nothing.
- Commit `82c0b1c` (STR-03 tests) — FOUND.
- Commit `1945afd` (TESTING.md update) — FOUND.
- `pytest tests/test_structure_parity.py -m "not slow" -q` — 88 passed.
- `pytest tests/test_structure_parity.py -m slow -q` — 24 passed.
- `pytest tests/test_parity.py -q` — 12 passed.
- `pytest tests/test_layer_parity.py -m "not slow" -q` — 184 passed.
- `pytest tests/test_structure.py -q` — 20 passed.
- `ruff check tests/test_structure_parity.py` — clean.
- `git diff tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` — empty.
- `grep -c "xfail" tests/test_structure_parity.py` — 0.
- `grep -c "monkeypatch" tests/test_structure_parity.py` — multiple (STR-03 idioms).
- STATE.md / ROADMAP.md — NOT modified, as instructed.

---
*Phase: 03-structured-pytorch-fallback-parity*
*Completed: 2026-05-14*
