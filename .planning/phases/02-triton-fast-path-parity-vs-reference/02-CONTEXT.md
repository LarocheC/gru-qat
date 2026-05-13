# Phase 2: Triton fast-path parity vs reference - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Pin every Triton variant (dense, diagonal, monarch, butterfly) fwd+bwd to the reference path at the **strict tier** (`torch.set_float32_matmul_precision('highest')`, < 1e-5 abs) using the Phase 1 reference path (`GRULayer(use_triton=False, Identity quantizers, dense)`) as ground truth. Existing TF32-based kernel tests (`test_triton_*.py`) remain in place as the realistic-deployment layer and are NOT loosened.

Also: add explicit named regression tests for the recent fix cluster — autotune dWh/dbh accumulator slab zeroing (`c001a8a`) and cross-CTA cache-modifier-as-fence determinism (the `.cv` anti-pattern). The butterfly OOB regression (`d8218d4`) already has coverage at `tests/test_butterfly_dispatch.py:206`; Phase 2 verifies it still passes but does not duplicate.

In scope:
- New strict-tier test files: `test_triton_scan_strict.py`, `test_triton_diagonal_strict.py`, `test_triton_monarch_strict.py`, `test_triton_butterfly_strict.py`. Each pairs with the existing TF32-tolerance file.
- fp32 forward + backward parity vs reference at < 1e-5 absolute under `set_float32_matmul_precision('highest')`.
- Per-kernel custom shape grids (see Decisions / D-13).
- Two new named regression tests in `test_triton_scan_strict.py`: autotune-config dWh/dbh + 50-run determinism.

Explicitly NOT in scope for Phase 2:
- Quant-on (non-Identity) parity — Phase 4 owns bit-identity for quant-on.
- Structured PyTorch fallback parity (Circulant, LDR) — Phase 3.
- Calibration + freeze lifecycle — Phase 5.
- Edge cases T=0, B=0, very-tiny H for kernels that don't natively support them — Phase 6.
- Loosening any existing kernel-test tolerance — out of scope.
- TF32 deviation as a finding — when the strict tier passes, TF32 noise level is recorded as expected behavior, not a bug (D-15).
- Per-channel `min_max` observer gap — Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Tolerance + precision policy
- **D-13:** Three test tiers exist after this phase, but Phase 2 itself only **adds** the strict tier and ALSO tightens existing realistic-tier tolerances where they pass at < 1e-4:
  - **Strict tier (new, Phase 2 deliverable):** `torch.set_float32_matmul_precision('highest')` at module scope; assertion `< 1e-5` absolute Triton-vs-reference for fwd+bwd. Lives in `test_triton_<kind>_strict.py`.
  - **Realistic tier (existing files, tightened in Phase 2 where possible):** `torch.set_float32_matmul_precision('high')` (TF32); current 5e-3..1e-1 tolerances tightened to < 1e-4 where the kernel can pass. Any kernel whose realistic-tier tolerance can't be tightened keeps its current bound with a TF32-justification comment.
  - **Permissive tier (existing files, untouched):** existing > 1e-4 tolerances retained where TF32 noise genuinely dominates (e.g., backward across many timesteps × gates). Comments document why.
- **D-14:** The strict tier is the audit-pass criterion for Phase 2's REF-TRI mapping. If a kernel fails strict-tier, that's a finding under D-10 (failing test → bd → fix). If it passes strict and the realistic-tier tightening fails, that's also a finding (separate bd issue, separate fix).
- **D-15:** TF32 noise level on a kernel that already passes strict-tier is **recorded in the SUMMARY**, not flagged as a finding. Justification: TF32 mantissa is ~10 bits; some reduction-order reorderings under autotune produce up to ~1e-3 rel drift; this is expected fp32 matmul behavior, not a kernel bug.

### Shape grid scope (per-kernel custom)
- **D-16:** Each kernel gets a grid tuned to its constraints:
  - **Dense (`test_triton_scan_strict.py`):** T ∈ {1, 8, 64, 512, 1024} × B ∈ {1, 4, 32} × H ∈ {32, 128, 512}. Slow-mark T ∈ {512, 1024}. ~45 combos, ~27 fast.
  - **Diagonal (`test_triton_diagonal_strict.py`):** same T/B as dense × H ∈ {1, 2, 8, 64, 512} (diagonal has no matmul → tiny H works). Slow-mark T ∈ {512, 1024}. Tests H=1, H=2 here; Phase 6 handles them across other kernels.
  - **Monarch (`test_triton_monarch_strict.py`):** T/B same × H ∈ {32, 128, 512} × nblocks ∈ {2, 4, 8}. Slow-mark T ∈ {512, 1024}.
  - **Butterfly (`test_triton_butterfly_strict.py`):** T/B same × H ∈ {32, 128, 512} (must be powers of 2; the butterfly kernel only supports 2^k). Slow-mark T ∈ {512, 1024}.
- **D-17:** No T=0 or B=0 in this grid — Phase 6 owns those. No H ∈ {1, 2} except for diagonal — those are kernel-specific edge cases for Phase 6.
- **D-18:** Reuse existing `_make_<kind>_layer` and `_build_gi_from_cell` helpers from the existing test files. Import them rather than duplicate. If they're not exported, add a thin module-local import or a small per-strict-file copy with a comment explaining why.

### Test file location strategy
- **D-19:** Four new files, one per kernel: `tests/test_triton_scan_strict.py`, `tests/test_triton_diagonal_strict.py`, `tests/test_triton_monarch_strict.py`, `tests/test_triton_butterfly_strict.py`. Each:
  - Opens with `pytest.importorskip("triton")` (and `pytest.importorskip("torch_structured")` for monarch/butterfly) at module top.
  - Defines `cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="...")` per file (existing convention; not registered in pyproject.toml).
  - `torch.set_float32_matmul_precision("highest")` at module scope. This is the marker that distinguishes strict-tier files from the existing realistic-tier ones.
- **D-20:** Strict-tier files do **not** include the realistic-tier tightenings — those land in the existing `test_triton_<kind>.py` files as edits to existing tolerance constants. Two-commit discipline applies: each tightening commit is its own atomic change; if a tightening fails CI, revert to the prior bound with a comment.
- **D-21:** Test function naming: `test_<kind>_fwd_strict_matches_reference`, `test_<kind>_bwd_strict_matches_reference`. Don't fuse fwd + bwd in one test — same pattern as Phase 1 D-09.

### Regression test depth (TRI-04..06)
- **D-22 (TRI-04 butterfly OOB):** Existing regression test at `tests/test_butterfly_dispatch.py:206` is sufficient. Phase 2's `test_triton_butterfly_strict.py` references it (in a comment) but does not duplicate. The phase verification step asserts the existing test still passes; no new test for this finding.
- **D-23 (TRI-05 autotune dWh/dbh):** Add `test_autotune_dWh_dbh_zero_init_across_configs` to `tests/test_triton_scan_strict.py`. Force `@triton.autotune` to evaluate multiple candidate configs (e.g., by clearing the JIT cache between two runs, or by running with `TRITON_DEBUG=1` style config-rotation). Assert `dWh` and `dbh` from the second run match the reference path — the bug was: stale accumulator memory from the first autotune-config run leaked into the second. This test must run on CUDA; `cuda_only`.
- **D-24 (TRI-06 cross-CTA determinism):** Add `test_persistent_kernel_deterministic` to `tests/test_triton_scan_strict.py`. Run `gru_scan_persistent` 50 times on the same `(x, h0, weights)` input on CUDA and assert `torch.equal` (bit-identical) across all 50 outputs (`out` and `h_T`). Pinned by D-15: this test asserts strict-tier determinism, which is independent of TF32 noise (deterministic non-bitwise drift under TF32 is still bit-identical run-to-run because the kernel's reduction order is fixed). If any run diverges, that's a finding — likely re-introduction of relaxed atomics or `.cv` cache-modifier-as-fence. Fast (≤ 30s on a modern GPU).
- **D-25 (regression-test extra guard):** Add a static check in `tests/test_triton_scan_strict.py`: `grep`-style assertion that `cache_modifier=".cv"` does NOT appear inside `src/gru_qat/triton_kernels/scan*.py`. This is a structural canary; the determinism test (D-24) is the dynamic guard.

### CUDA execution plan
- **D-26:** Tests are authored now on the current (CPU-only?) dev machine. `pytest tests/test_triton_*_strict.py -q` must skip cleanly on CPU (verified by the existing `pytest.importorskip("triton")` + `@cuda_only` pattern). Phase-exit verification REQUIRES a CUDA box: the user runs `pytest tests/test_triton_*_strict.py -q` on a GPU machine and reports results back. Plan 02-N (the audit-kickoff plan) explicitly includes this user-action gate as a checkpoint.

### Discipline (carried from Phase 1)
- **D-27:** Two-commit failing-test-before-fix per D-10..12 from Phase 1. No `@pytest.mark.xfail`. bd issue per finding before commit A. Strict-tier tests are the primary audit gate; realistic-tier tightenings follow the same protocol.
- **D-28:** Cell-parity `< 1e-5` in `tests/test_parity.py` AND the new Phase 1 layer-parity contract in `tests/test_layer_parity.py` are LOCKED. Phase 2 cannot loosen them. Verifier will assert `git diff` empty on those files across Phase 2 commits.

### Claude's Discretion
- Exact `pytest.parametrize` id strings (e.g., `"T=8-B=4-H=128"`).
- Whether the strict-tier files share a common preamble helper (e.g., a `tests/_triton_strict_helpers.py` with shared `_make_*_layer_strict` constructors). Default: yes if duplication exceeds ~30 lines; otherwise inline.
- Order of plan execution within Phase 2 (the planner decides — likely dense first since the autotune + determinism regressions live there, then diagonal/monarch/butterfly in any order).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project + phase
- `.planning/PROJECT.md` — milestone scope; tolerance contract (< 1e-5 Triton-vs-reference).
- `.planning/REQUIREMENTS.md` §TRI-01..06 — exact text for the six requirements this phase implements.
- `.planning/ROADMAP.md` §"Phase 2: Triton fast-path parity vs reference" — success criteria.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-VERIFICATION.md` — Phase 1 verifier report confirming the reference path is a trusted ground truth.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-CONTEXT.md` §D-10..12 — two-commit discipline + no-xfail rule (carried forward as D-27).
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-PATTERNS.md` — relative-error idiom, parametrize style, slow-mark convention. Applies to Phase 2 too.

### Codebase
- `src/gru_qat/triton_kernels/scan.py` — dense Triton fwd+bwd; site of TRI-05 (autotune dWh/dbh) + TRI-06 (cross-CTA fence).
- `src/gru_qat/triton_kernels/scan_diagonal.py` — diagonal kernel; smallest, fastest variant.
- `src/gru_qat/triton_kernels/scan_monarch.py` — monarch (block-diag) kernel.
- `src/gru_qat/triton_kernels/scan_butterfly.py` — butterfly kernel; site of TRI-04 (last-program OOB, already-fixed by `d8218d4`).
- `src/gru_qat/gru_layer.py:_forward_fast_dispatch` (line 202) — fast-path dispatcher; reference for how each kernel is invoked.
- `tests/test_triton_scan.py` — existing dense kernel tests (TF32, 5e-3..1e-1 rel). Phase 2 references the `_make_dense_layer` and `_build_gi_from_cell` helpers; reads but does NOT modify the existing tolerances unless D-13 realistic-tier tightening applies and the kernel passes the tighter bound.
- `tests/test_triton_diagonal.py`, `tests/test_triton_monarch.py`, `tests/test_butterfly_dispatch.py` — same role for the other kernels.
- `tests/test_parity.py` — locked < 1e-5 cell parity contract (D-28).
- `tests/test_layer_parity.py` — locked < 1e-4 layer parity contract from Phase 1 (D-28).
- `.planning/codebase/TESTING.md` — TF32 setup, relative-error idiom, marker discipline.
- `.planning/codebase/CONVENTIONS.md` — naming, type discipline, math-significant variable names.

### Anti-patterns
- `DEVELOPMENT.md` §"What the agent should NOT do" — explicit warning against `cache_modifier=".cv"` as cross-CTA fence. Phase 2's D-25 static check guards this directly.
- `src/gru_qat/triton_kernels/scan.py:gru_scan_fwd_persistent_kernel` — comment block documenting the release/acquire `atomic_add(sem=...)` pattern.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`_make_<kind>_layer` helpers** (per existing test file): build a fp32-Identity GRULayer with the right structured config. Phase 2 reuses these for the strict tier by importing or duplicating; either way they avoid duplicating the cell-construction boilerplate.
- **`_build_gi_from_cell(layer, x)`** (per existing test file): reproduces the cell's input projection so both the reference and the Triton kernel see the same `gi` tensor. Required by every Triton parity test.
- **`cuda_only = pytest.mark.skipif(...)`** (per existing test file): file-local CUDA gate. Strict files inherit this pattern.
- **`pytest.importorskip("triton")` and `pytest.importorskip("torch_structured")`** at module top: file-skipped on CPU-only machines. Strict files inherit this pattern.
- **Per-batch error inspection idiom** (`tests/test_butterfly_dispatch.py:206`): useful for diagnosing kernel-specific bugs that only manifest at certain `pid_b`. The TRI-05 autotune-config regression may need this.

### Established Patterns
- **One test file per kernel** in tests/. Phase 2 adds a "_strict" sibling per kernel.
- **`# noqa: E402`** is required after `pytest.importorskip` for the subsequent imports.
- **`torch.set_float32_matmul_precision("high")`** is the existing default at the top of every `test_triton_*.py`. The strict-tier files override to `"highest"`.
- **Relative-error idiom with `1e-6` floor:** `max_diff / max(ref.abs().max(), 1e-6) < TOL`. Phase 2 strict-tier uses **absolute error** at < 1e-5 (since TF32 isn't in play), not relative. Document this divergence from the existing pattern in each strict-tier file's docstring.

### Integration Points
- Strict-tier files are new — no modifications to existing tests in this plan. Only realistic-tier tightening (D-13 second bullet) edits existing test files; each tightening is its own commit per D-27.
- `src/` is untouched unless a strict-tier test surfaces a kernel bug → bd issue + Commit A (failing test) + Commit B (fix in src/).
- The TRI-05 and TRI-06 regression tests are new, in `test_triton_scan_strict.py` even though they probe specific past bugs; this groups them with the dense-kernel strict tests for ease of finding.

</code_context>

<specifics>
## Specific Ideas

- **Three tiers in total after Phase 2:**
  1. Strict — `'highest'`, < 1e-5 abs. New files (this phase).
  2. Realistic — `'high'`, < 1e-4 rel where tightenable, ≥ 5e-3 where TF32 dominates. Existing files (this phase tightens where possible).
  3. Permissive — `'high'`, current ≥ 5e-3 tolerances. Existing files (this phase does not touch).
- **Test count estimate:** ~45 + 75 + 27 + 27 = ~174 new strict-tier parametrized cases across 4 kernels. Plus TRI-05 (1 test) and TRI-06 (1 test). Plus the static `.cv` grep canary. Plus a handful of realistic-tier tightenings. Total Phase 2 net-new tests: ~180.
- **Phase-exit GPU run:** required before phase-close. Documented in Plan 02-N as a `checkpoint:human-verify` task — same pattern as Phase 1's Plan 01-05.

</specifics>

<deferred>
## Deferred Ideas

- **Per-channel `min_max` observer fix-vs-fence** — Phase 4 owns this decision (already noted in STATE.md).
- **Quant-on bit-identity** — Phase 4. Strict-tier files are fp32-Identity only.
- **Circulant + LDR strict-tier parity** — Phase 3 (no Triton kernel; per-step PyTorch only).
- **Edge cases T=0, B=0, H<small> across non-diagonal kernels** — Phase 6.
- **Bench re-validation** — explicitly out of scope per PROJECT.md.
- **A shared `tests/_triton_strict_helpers.py` module** — only create if duplication across the 4 strict files exceeds ~30 lines. Planner / executor's call at write time.
- **TF32-vs-fp32 divergence as a finding** — recorded in SUMMARY only, never a bd issue (D-15).

</deferred>

---

*Phase: 2-triton-fast-path-parity-vs-reference*
*Context gathered: 2026-05-13*
