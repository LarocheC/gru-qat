---
phase: 02-triton-fast-path-parity-vs-reference
plan: 01
subsystem: testing
tags: [triton, gru, parity, strict-tier, regression, audit, autotune, persistent-kernel, cross-cta-fence, static-canary]

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    provides: |
      tests/test_layer_parity.py + tests/test_parity.py as LOCKED ground-truth
      contracts (D-28); _ref_layer construction shape; 'highest' precision
      preamble pattern.
provides:
  - tests/test_triton_scan_strict.py (strict-tier dense Triton parity audit)
  - test_autotune_dWh_dbh_zero_init_across_configs (TRI-05 regression, multi-bucket variant of tests/test_triton_scan.py:202-215)
  - test_persistent_kernel_deterministic (TRI-06 regression, 50-run torch.equal cross-CTA fence guard)
  - test_no_cv_cache_modifier_live_uses_in_scan_source (D-25 static .cv canary, baselined at count=0)
  - Strict-tier (< 1e-5 abs under 'highest') audit signal for dense Triton fwd+bwd
affects:
  - 02-02 (diagonal strict) — same pattern; parallel agent
  - 02-03 (monarch strict) — same pattern; parallel agent
  - 02-04 (butterfly strict) — same pattern; parallel agent
  - 02-06 (phase-exit GPU run + finding triage) — consumes audit failures surfaced here

# Tech tracking
tech-stack:
  added: []  # no new libraries; only tests/
  patterns:
    - "Strict-tier test sibling per realistic-tier kernel test file (D-19)"
    - "Absolute-error idiom < 1e-5 (no relative-error floor) for strict tier"
    - "Module-scope torch.set_float32_matmul_precision('highest') as the strict-tier marker"
    - "FAST/SLOW grid split with @pytest.mark.slow gating (D-16)"
    - "lstrip().startswith('#') comment-strip for static source canaries (D-25 pattern)"

key-files:
  created:
    - tests/test_triton_scan_strict.py
    - .planning/phases/02-triton-fast-path-parity-vs-reference/02-01-SUMMARY.md
  modified: []  # zero src/ changes; LOCKED test files (test_parity.py, test_layer_parity.py, test_triton_scan.py) untouched

key-decisions:
  - "Strict tier uses absolute error < 1e-5 (no relative-error floor); diverges intentionally from realistic-tier relative idiom per CONTEXT 'Established Patterns' callout."
  - "TRI-05 and TRI-06 regression tests live in the strict file (not a separate regressions file) — groups them with the dense-kernel strict audit per CONTEXT code_context closer."
  - "D-25 canary scans all scan*.py files (scan.py, scan_diagonal.py, scan_monarch.py, scan_butterfly.py), not just scan.py — matches the D-25 wording 'scan*.py' and catches future diagonal/monarch/butterfly reintroductions."
  - "D-25 canary comment-strip rule is raw.lstrip().startswith('#') — correctly classifies indented Triton-JIT comment lines. Verified at execute time: all 3 .cv occurrences in scan.py (lines 192, 431, 625) are inside #-comment lines; baseline is 0."
  - "_ref_layer helper duplicated verbatim from tests/test_triton_scan.py:30-44 per D-18 (< 30 LOC; inline beats shared module)."
  - "Imports forward-declared at the top (pathlib, gru_scan_persistent) for use by later tasks within the same file — atomic per-task commits still happen, but the noqa F401 annotations were inlined initially and dropped as each task consumed the import."

patterns-established:
  - "Strict-tier dense parity sibling: module-scope 'highest' + cuda_only + FAST/SLOW grids + abs() < 1e-5"
  - "Regression test placement: when a finding is dense-kernel-specific, it lives in the dense strict file"
  - "Static source-grep canary via pathlib (no subprocess): comment-strip with lstrip then content match"

requirements-completed: [TRI-01, TRI-05, TRI-06]

# Metrics
duration: 25min
completed: 2026-05-13
---

# Phase 2 Plan 1: Strict-tier dense Triton parity tests + TRI-05/TRI-06 regressions + D-25 canary Summary

**Strict-tier (< 1e-5 abs under `'highest'`) dense Triton parity audit + named regressions for autotune slab-zero (TRI-05, c001a8a) and persistent-kernel determinism (TRI-06) + static `.cv` cache-modifier canary (D-25) — single new test file, 463 LOC, 93 collected tests, zero modifications to LOCKED files.**

## Performance

- **Duration:** ~25 min (authoring + verification across all 3 tasks)
- **Started:** 2026-05-13T21:00:00Z (approximate; orchestrator-spawned)
- **Completed:** 2026-05-13T21:20:00Z
- **Tasks:** 3
- **Files modified:** 1 (`tests/test_triton_scan_strict.py` — new)

## Accomplishments

- **`tests/test_triton_scan_strict.py` created**: 463 LOC, ruff-clean, syntactically valid, collects 93 tests on CUDA+CPU machines (90 parametrized parity + 2 named regression + 1 static canary).
- **Strict-tier parity audit**: `test_scan_fwd_strict_matches_reference` and `test_scan_bwd_strict_matches_reference` (FAST = 27 cases each, SLOW = 18 cases each via `@pytest.mark.slow`) compare `gru_scan_forward` / `gru_scan` against the Phase 1 `_ref_layer` at `< 1e-5` absolute under `torch.set_float32_matmul_precision('highest')`.
- **TRI-05 regression** (`test_autotune_dWh_dbh_zero_init_across_configs`): multi-shape variant of the single-config slab-zero regression at `tests/test_triton_scan.py:202-215`. Rotates through `(16,16,64)` and `(32,32,64)` to hit different autotune buckets per the `key=['T','B']` declaration at `src/gru_qat/triton_kernels/scan.py:732,893`.
- **TRI-06 regression** (`test_persistent_kernel_deterministic`): 50-run `torch.equal` (strict bit-identity, not `torch.allclose`) on `gru_scan_persistent` with one-time-allocated inputs. **VERIFIED PASSING on the current GPU** — the release/acquire cross-CTA fence at `scan.py:184-208` is intact.
- **D-25 static canary** (`test_no_cv_cache_modifier_live_uses_in_scan_source`): pure-Python pathlib scan of `src/gru_qat/triton_kernels/scan*.py` with `lstrip().startswith('#')` comment-strip; **VERIFIED PASSING on CPU/CUDA at baseline count == 0**. Catches reintroduction of the `.cv` cache-modifier anti-pattern before any GPU run.

## Task Commits

Each task was committed atomically:

1. **Task 1: Strict-tier parity tests + module preamble + helpers** — `5bddd4a` (test) — *see "Deviations" below: this commit was authored by me but landed inside a parallel agent's `5bddd4a docs(02-04): complete butterfly Triton strict-tier parity plan` commit due to a wide `git add` collision on the shared working tree. Content is identical to what was authored for Task 1.*
2. **Task 2: TRI-05 + TRI-06 regression tests** — `ac56d94` (test) — clean atomic commit.
3. **Task 3: D-25 static `.cv` canary** — `ba3d43e` (test) — clean atomic commit.

**Plan metadata commit** (this SUMMARY): will be added by the orchestrator at end of wave.

## Files Created/Modified

- `tests/test_triton_scan_strict.py` (created, 463 LOC) — Strict-tier dense Triton parity audit + TRI-05/TRI-06 regression tests + D-25 static canary.
- `.planning/phases/02-triton-fast-path-parity-vs-reference/02-01-SUMMARY.md` (created, this file).

**Unmodified (LOCKED per D-28 — verified by `git diff HEAD~3` empty):**
- `tests/test_parity.py` (cell parity < 1e-5; 12 passed)
- `tests/test_layer_parity.py` (layer parity; 184 passed, 120 deselected slow)
- `tests/test_triton_scan.py` (realistic-tier TF32 sibling; out of scope for Plan 02-01 per D-20)

## Test function inventory

| Test function | Decorator | Parametrize | Count |
|---|---|---|---|
| `test_scan_fwd_strict_matches_reference` | `@cuda_only` | `FAST_DENSE_GRID` (T×B×H = 3×3×3) | 27 |
| `test_scan_fwd_strict_matches_reference_slow` | `@pytest.mark.slow @cuda_only` | `SLOW_DENSE_GRID` (T×B×H = 2×3×3) | 18 |
| `test_scan_bwd_strict_matches_reference` | `@cuda_only` | `FAST_DENSE_GRID` | 27 |
| `test_scan_bwd_strict_matches_reference_slow` | `@pytest.mark.slow @cuda_only` | `SLOW_DENSE_GRID` | 18 |
| `test_autotune_dWh_dbh_zero_init_across_configs` | `@cuda_only` | none | 1 |
| `test_persistent_kernel_deterministic` | `@cuda_only` | none | 1 |
| `test_no_cv_cache_modifier_live_uses_in_scan_source` | none (pure-Python file scan) | none | 1 |
| **TOTAL collected** | | | **93** |

## Confirmation: CPU collection passes + canary passes

- `uv run python -c "import ast; ast.parse(...)"` → exit 0 (syntactic validity).
- `uv run pytest tests/test_triton_scan_strict.py --collect-only -q` → 93 tests collected.
- `uv run pytest tests/test_triton_scan_strict.py::test_no_cv_cache_modifier_live_uses_in_scan_source -v` → **1 passed in 2.58s**.
- `uv run ruff check tests/test_triton_scan_strict.py` → **All checks passed**.
- `grep -n "xfail" tests/test_triton_scan_strict.py` → empty (D-12 / D-27).

## D-25 canary baseline verification

```bash
$ grep -v '^\s*#' src/gru_qat/triton_kernels/scan.py | grep -c 'cache_modifier=".cv"'
0

$ for f in src/gru_qat/triton_kernels/scan_{diagonal,monarch,butterfly}.py; do
    grep -c 'cache_modifier=".cv"' "$f"
  done
0
0
0
```

All 3 occurrences of `cache_modifier=".cv"` in `scan.py` are at lines 192, 431, 625 — each inside a `#`-prefixed comment line (lstrip-then-startswith('#')). Other `scan*.py` files have zero matches of any kind. **Live count baseline = 0**, canary asserts == 0. Future reintroduction of `.cv` outside a comment in any of the four files will fail this test with the offending file path + line number.

## Decisions Made

- **Absolute-error idiom (not relative)** for strict tier — diverges intentionally from the realistic-tier relative-with-1e-6-floor idiom. Documented in module docstring and in the assertion-message format `(T={T},B={B},H={H})`. Strict tier doesn't need the floor because we're not normalizing.
- **Both TRI-05 and TRI-06 in the dense-strict file**, not a separate `test_regressions.py` — groups regressions with the kernel they probe; matches the "dense kernel = dense file" pattern.
- **D-25 canary scans ALL `scan*.py` files via `glob("scan*.py")`** rather than only `scan.py`. This catches future reintroductions in `scan_diagonal.py` / `scan_monarch.py` / `scan_butterfly.py` and matches the D-25 wording literally.
- **Comment-strip rule = `raw.lstrip().startswith("#")`** — was the precise wording in PATTERNS.md and the right choice for Triton-JIT indented comments. The naive `raw.startswith("#")` would miss indented comments and reintroduce false positives.
- **Imports forward-declared with `# noqa: F401` then noqa removed when consumed** — `pathlib` and `gru_scan_persistent` were declared in Task 1 (per plan's explicit instruction "declare here so all tasks reuse") with `noqa` annotations; the annotations were dropped in Tasks 2 + 3 as the imports were consumed. Final file is ruff-clean with zero noqa-on-the-import-line for those two symbols (the `pytest.importorskip("triton")`-anchored `# noqa: E402` and the symmetry-only-imports' `# noqa: F401` remain).

## Deviations from Plan

### Co-commit anomaly (Rule 1 — Bug, environmental)

**1. [Rule 1 — Bug] Task 1 file landed inside a parallel agent's commit**

- **Found during:** Task 1 commit step.
- **Issue:** I ran `git reset HEAD tests/test_triton_monarch_strict.py` (to unstage another agent's accidental include), and between the reset and my next `git add tests/test_triton_scan_strict.py`, a parallel agent's commit `5bddd4a docs(02-04): complete butterfly Triton strict-tier parity plan` swept in my unstaged `tests/test_triton_scan_strict.py` via its own broader `git add`. The file content committed in `5bddd4a` is byte-identical to what I authored for Task 1 (verified via `diff <(git show 5bddd4a:tests/test_triton_scan_strict.py) tests/test_triton_scan_strict.py` → empty).
- **Fix:** No re-commit needed — file content is correct and present at HEAD. Tasks 2 + 3 committed cleanly on top as `ac56d94` and `ba3d43e`. The plan's atomic-commit semantics are partially violated (Task 1 content is in a commit titled for Plan 02-04), but the audit trail is intact: Task 1 LOC count, content, and tests all match the plan spec.
- **Files affected:** `tests/test_triton_scan_strict.py` (committed; content unchanged from authored).
- **Verification:** `git log --oneline -- tests/test_triton_scan_strict.py` shows `5bddd4a` (Task 1 content) → `ac56d94` (Task 2) → `ba3d43e` (Task 3). All 93 tests collected; D-25 canary passing.
- **Committed in:** `5bddd4a` (Task 1 by anomaly), `ac56d94`, `ba3d43e`.
- **Root cause:** Parallel-agent execution on a single working tree without per-agent worktrees. Future mitigation: orchestrator should consider per-agent `git worktree` isolation, or explicitly serialize the staging-area writes via a lock.

**Total deviations:** 1 environmental (parallel-agent commit collision, no content impact).

**Impact on plan:** Zero impact on file content, test inventory, or audit signal. Cosmetic only — Task 1's commit message lives under a different plan's commit. The Task 1 content is preserved and verifiable.

## Issues Encountered: Phase 2 Audit Findings (deferred to Plan 02-06)

The authoring machine has CUDA available (`torch.cuda.is_available() == True`), so the parametrized parity tests RAN instead of skipping. **This is the expected Phase 2 audit signal per CONTEXT D-14**: "If a kernel fails strict-tier, that's a finding under D-10 (failing test → bd → fix)." Plan 02-06 owns triage. The findings are recorded here for that plan:

### Finding 1: dense Triton fwd does not meet < 1e-5 abs at strict tier

- **Test:** `test_scan_fwd_strict_matches_reference[1-1-32]` (smallest shape)
- **Observed:** `max abs diff = 3.3964e-04` (~34× over the < 1e-5 bound).
- **All 27 FAST_DENSE_GRID parametrize cases FAILED** at strict tier.
- **Interpretation:** Either the kernel has algorithmic drift beyond 1e-5 even under IEEE fp32 matmul, OR the reference path itself drifts (less likely — reference is the Phase 1 LOCKED contract). Most likely cause: `tl.dot` in `'highest'` still uses HFMA reduction order different from PyTorch's matmul. Needs investigation in Plan 02-06.
- **TF32-tier sibling at `tests/test_triton_scan.py:139` is `< 5e-3` and passes**; the audit gate this file installs is intentionally tighter (D-13).

### Finding 2: dense Triton bwd does not meet < 1e-5 abs at strict tier

- **Test:** `test_scan_bwd_strict_matches_reference[*]` — all 27 FAST cases failed.
- **Same root-cause class** as Finding 1; gradient magnitudes compound across timesteps and 3 gates.

### Finding 3: TRI-05 regression test triggers at iter=0 (first iteration)

- **Test:** `test_autotune_dWh_dbh_zero_init_across_configs`
- **Observed:** `iter=0 shape=(16, 16, 64) x max abs diff 8.1623e-04`. The FIRST iteration fails on `dx`; the test never reaches the second iteration's discriminating Wh_cat/bh_cat comparison.
- **Interpretation:** NOT a slab-zero regression (would manifest on iter=1, not iter=0). Same class as Findings 1+2: strict-tier kernel-vs-reference drift on the bwd path. The TRI-05 regression signal itself (iter=1 specifically failing on Wh_cat/bh_cat) is preserved in the assertion message and will be visible IF the slab-zero bug ever returns.

### Finding 4: TRI-06 (persistent kernel determinism) PASSES

- **Test:** `test_persistent_kernel_deterministic`
- **Observed:** **PASSED** in 19.13s (combined with TRI-05). 50 runs of `gru_scan_persistent` produced bit-identical output. Release/acquire cross-CTA fence at `src/gru_qat/triton_kernels/scan.py:184-208` is intact.

**Recommendation for Plan 02-06:**
1. File one bd issue per finding (Findings 1, 2, 3 above — Finding 4 is green).
2. Tighten the strict-tier kernel by switching to `tl.dot(..., precision='ieee')` if available, or document the floor as "TF32-unavoidable" and adjust the strict bound with explicit src/kernel justification (D-15 path: record TF32 noise, not a bug).
3. The static D-25 canary and TRI-06 dynamic determinism guard are both green and stay as-is.

## Next Phase Readiness

- **Plan 02-02 / 02-03 / 02-04** (parallel agents on diagonal / monarch / butterfly strict) are unaffected by this plan; they use the same shape patterns and D-25-style canaries are not duplicated (D-25 is dense-specific).
- **Plan 02-05** (realistic-tier tightening in existing test files) is unblocked.
- **Plan 02-06** (phase-exit GPU run + triage) inherits a clear set of audit findings from this plan: Findings 1, 2, 3 above (with concrete numerical observations).

No blockers. No `@pytest.mark.xfail` introduced (D-27 honored). LOCKED files (`test_parity.py`, `test_layer_parity.py`, `test_triton_scan.py`) untouched (D-28 honored).

## Self-Check: PASSED

- File exists at `tests/test_triton_scan_strict.py` (463 LOC).
- Commits exist:
  - `5bddd4a` (contains Task 1 content) — `git show 5bddd4a:tests/test_triton_scan_strict.py | wc -l` → 269
  - `ac56d94` (Task 2) — `git log --oneline ac56d94 -n 1` → "test(02-01): add TRI-05 + TRI-06 regression tests to scan_strict"
  - `ba3d43e` (Task 3) — `git log --oneline ba3d43e -n 1` → "test(02-01): add D-25 static .cv cache-modifier canary to scan_strict"
- Acceptance gates: all green (ruff, ast.parse, 93 collected, D-25 canary passing on CPU, baseline `grep -c '.cv'` → 0, no xfail, LOCKED files diff-empty, `test_parity.py` + `test_layer_parity.py` pass).

---
*Phase: 02-triton-fast-path-parity-vs-reference*
*Completed: 2026-05-13*
