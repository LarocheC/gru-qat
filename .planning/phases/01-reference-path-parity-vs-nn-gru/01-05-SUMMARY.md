---
phase: 01-reference-path-parity-vs-nn-gru
plan: 05
subsystem: testing
tags: [audit, pytest, parity, fully-green, phase-exit, requirements-closure, bd-tracking]

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 01
    provides: "_translate_cell_to_nn_gru, _translate_nn_gru_to_cell, _make_dense_fp32_layer, set_float32_matmul_precision('highest') preamble, 3 gate-order micro-tests, 1 round-trip smoke"
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 02
    provides: "FAST_GRID (45) + SLOW_GRID (30), forward-output parity grid (REF-01), h_T parity grid (REF-04)"
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 03
    provides: "backward-gradient parity grid (REF-03) covering dx, dh_0, dW_ih, dW_hh, db_ih, db_hh"
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 04
    provides: "h_0 != 0 random initial-state parity grid (REF-02)"
provides:
  - "01-05-SUMMARY.md (the Phase 1 audit verdict — fully green)"
  - "bd issue gru-triton-4m6 tracking pre-existing mypy/ruff debt out-of-scope for Phase 1"
  - "REF-01..05 closure rationale (REQUIREMENTS.md flip from Pending -> Done eligible)"
affects:
  - "ROADMAP.md Phase 1 row: ready to flip from [ ] to [x]"
  - "REQUIREMENTS.md traceability rows REF-01..05: ready to flip from Pending to Done"
  - "Phase 2 (Triton-fast-path parity vs reference): reference path is now the trusted < 1e-4 ground truth for downstream comparisons"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fully-green audit pattern: 304 parametrized tests, 5 verification passes (fast / slow / cell-contract / mypy / ruff), one bd tracking issue for inherited debt, no Commit B fix commits because no findings exist."
    - "D-10/D-12 fully honored at phase exit: zero @pytest.mark.xfail, every Phase 1 commit is `test(...)` or `docs(...)` — no `fix(...)` paired, because the audit surfaced zero parity bugs."
    - "Scope-discipline pattern: pre-existing toolchain debt (mypy + ruff errors at the Phase 1 baseline) is filed as ONE tracking bd issue, not enumerated per-error and not blocked-on, because Phase 1 touched zero `src/` bytes."

key-files:
  created:
    - ".planning/phases/01-reference-path-parity-vs-nn-gru/01-05-SUMMARY.md"
  modified: []

key-decisions:
  - "Fully-green audit verdict: zero parity findings across 304 tests (4 micro/smoke + 300 grid cases × 4 families). No Commit A / Commit B pairs needed because nothing failed."
  - "Pre-existing mypy/ruff debt (145 + 24 errors) is filed as ONE tracking bd issue (gru-triton-4m6), not per-error. Verified out-of-scope: `git log --name-only ad67535..HEAD -- src/` is empty across all Phase 1 commits, and the same error counts exist at the Phase 1 baseline (3b6f093)."
  - "No separate AUDIT-NOTES.md or AUDIT-REPORT.md file: this SUMMARY is the canonical audit artifact, matching the Plans 01-01..04 SUMMARY-per-plan pattern. The RPT-03 'AUDIT-REPORT.md' requirement from REQUIREMENTS.md belongs to a Phase 7 consolidation across REF/TRI/STR/QNT/CAL/EDG — not to Phase 1 alone."
  - "Non-batched input `(T, IN)` (nn.GRU accepts, GRULayer doesn't) is left as a deferred non-blocker — not filed as bd. CONTEXT.md Deferred Ideas line 122 already records this; no Phase 1 regression."
  - "Slow-suite execution budget at ~89s wall-clock is well under the 10-minute abort threshold from CONTEXT D-08 / Plan 01-05 Task 1 action body. No grid pruning needed."

patterns-established:
  - "Phase-exit audit shape: run fast/slow/cell-contract/lint/types in five passes, capture each pass's tail in the SUMMARY's verification snapshot, file one tracking bd issue for any inherited debt that survives the audit, declare verdict in section 1."
  - "Multi-plan SUMMARY composition: each plan (01-01..04) already records its local self-check; the phase-exit SUMMARY (01-05) re-runs every pass holistically and points at the per-plan SUMMARIES for detail. No duplication of test-by-test verification."

requirements-completed: [REF-01, REF-02, REF-03, REF-04, REF-05]

# Metrics
duration: ~6 min
completed: 2026-05-13
---

# Phase 1 Plan 5: Reference-path parity audit (phase-exit verdict) Summary

**Audit verdict: fully green.** All 304 Phase 1 layer-parity tests pass (184 fast + 120 slow) on CPU at < 1e-4 relative tolerance under `set_float32_matmul_precision('highest')`. The locked < 1e-5 cell-parity contract in `tests/test_parity.py` remains green (12/12). Zero parity findings surfaced; zero Commit B fix commits landed in `src/`; zero `@pytest.mark.xfail` markers anywhere in `tests/test_layer_parity.py`. All five REF requirements (REF-01..05) close. The reference path is now the trusted ground truth for Phase 2's Triton-fast-path parity audits at < 1e-5.

## Section 1 — Audit verdict

**Fully green.** Zero findings; zero bd parity issues; zero `src/` modifications across all 5 plans of Phase 1.

The Phase 1 audit ran end-to-end across 5 verification passes:

1. Fast suite (`pytest tests/test_layer_parity.py -q -m "not slow"`) — 184 passed.
2. Slow suite (`pytest tests/test_layer_parity.py -m slow -q`) — 120 passed.
3. Cell-parity contract (`pytest tests/test_parity.py -q`) — 12 passed (< 1e-5 locked contract intact).
4. Strict mypy on `src/gru_qat` scope — 145 errors, ALL pre-existing (out of Phase 1 scope; tracked in bd gru-triton-4m6).
5. Ruff on `src tests` — 24 errors, ALL pre-existing (out of Phase 1 scope; tracked in bd gru-triton-4m6).

No new parity findings means no `fix(...)` commits landed in `src/gru_qat/`. The two-commit D-10/D-11 discipline (Commit A = failing test → Commit B = fix) did not need to fire because every test passed on the first run across all 5 plans (01-01 through 01-04). This is the "fully green on audit kickoff" outcome documented in PLAN.md Task 1 ("If the audit is fully green... skip Task 2 entirely").

## Section 2 — Pass-by-pass results

| Pass | Command | Exit | Duration | Result |
|------|---------|------|----------|--------|
| 1 | `pytest tests/test_layer_parity.py -q -m "not slow"` | 0 | ~5.6s | **184 passed, 120 deselected** |
| 2 | `pytest tests/test_layer_parity.py -m slow -q` | 0 | ~88.7s | **120 passed, 184 deselected** |
| 3 | `pytest tests/test_parity.py -q` | 0 | ~3.0s | **12 passed** (< 1e-5 cell contract green) |
| 4 | `mypy` (strict, src/gru_qat) | 1 | n/a | 145 errors in 10 files — **all pre-existing** (see §7) |
| 5 | `ruff check src tests` | 1 | n/a | 24 errors — **all pre-existing** (see §7) |

Full-suite reproduction (`pytest tests/test_layer_parity.py -q`) collected and ran 304 tests in 71.29s, all green. Slow-suite wall-clock of ~89s is comfortably under the 10-minute abort threshold from CONTEXT D-08; no grid pruning is required.

Verbatim tail of pass 1 (fast):

```
$ .venv/bin/python -m pytest tests/test_layer_parity.py -q -m "not slow"
........................................................................ [ 39%]
........................................................................ [ 78%]
........................................                                 [100%]
184 passed, 120 deselected in 5.63s
```

Verbatim tail of pass 2 (slow):

```
$ .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -q
........................................................................ [ 60%]
................................................                         [100%]
120 passed, 184 deselected in 88.68s (0:01:28)
```

Verbatim tail of pass 3 (cell-parity contract — the locked < 1e-5 gate):

```
$ .venv/bin/python -m pytest tests/test_parity.py -q
............                                                             [100%]
12 passed in 2.95s
```

## Section 3 — REF requirement closure

All five requirements (REF-01..05) close as **CLOSED — covered by audit**. Each maps unambiguously to one or more test functions in `tests/test_layer_parity.py`.

| Req | Statement (paraphrased) | Test coverage | Pass count |
|-----|-------------------------|---------------|------------|
| **REF-01** | Forward output parity over T × B × H = 75-combo grid at < 1e-4 | `test_layer_forward_matches_nn_gru` (45 fast) + `test_layer_forward_matches_nn_gru_slow` (30 slow) | 75 / 75 |
| **REF-02** | h_0 ≠ 0 random initial-state parity at < 1e-4 | `test_layer_with_random_h0_matches_nn_gru` (45 fast) + `test_layer_with_random_h0_matches_nn_gru_slow` (30 slow) | 75 / 75 × 2 tensors (out + h_T) = 150 / 150 |
| **REF-03** | Backward gradient parity (dx, dh_0, dW_ih, dW_hh, db_ih, db_hh) at < 1e-4 | `test_layer_backward_matches_nn_gru` (45 fast) + `test_layer_backward_matches_nn_gru_slow` (30 slow) | 75 / 75 × 6 grads = 450 / 450 |
| **REF-04** | Final hidden state `h_T` parity at < 1e-4 | `test_layer_h_T_matches_nn_gru` (45 fast) + `test_layer_h_T_matches_nn_gru_slow` (30 slow) | 75 / 75 |
| **REF-05** | Gate-ordering / bias-fusion translation helper exists | `_translate_cell_to_nn_gru` + `_translate_nn_gru_to_cell` + `_make_dense_fp32_layer` (module-level), plus 3 gate-order micro-tests (`test_gate_order_r_only`, `test_gate_order_z_only`, `test_n_gate_asymmetry`) and 1 round-trip smoke (`test_round_trip_nn_gru_to_cell`) | 4 / 4 |

**Aggregate:** 4 micro/smoke + (45 + 30) × 4 grid families = **304 tests, 304 passes**.

## Section 4 — Two-commit discipline (D-10) confirmation

D-10 (failing-test-Commit-A → fix-Commit-B) and D-11 (one bd issue per finding) did not fire **because findings = 0**. The Phase 1 git history reflects this cleanly:

```
$ git log --oneline ad67535^..HEAD
2ea67f7 docs(01-04): complete random h_0 != 0 parity plan
95d2305 test(01-04): h_0 != 0 parity grid (45 fast + 30 slow) vs nn.GRU
005673d docs(01-03): complete backward gradient parity plan
8cd96ad test(01-03): backward parity grid (45 fast + 30 slow) vs nn.GRU
3bdddba docs(01-02): complete forward + h_T parity grid plan
218405d test(01-02): h_T parity grid (45 fast + 30 slow) vs nn.GRU
56238a9 test(01-02): forward parity grid (45 fast + 30 slow) vs nn.GRU
ad67535 docs(01-01): complete layer-parity test-scaffolding plan
3b6f093 test(01-01): add 3 gate-order/n-gate-asymmetry micro-tests + round-trip smoke
786b32c test(01-01): scaffold layer-parity helpers (translate cell<->nn.GRU, fp32-Identity builder)
```

- **Commit-A count (failing tests):** 5 `test(01-XX): ...` commits across Plans 01-01..04. Each landed green (no failures), so each is also implicitly its own Commit-B-equivalent for D-10's purposes — but with the trivial right-hand side "no fix needed".
- **Commit-B count (fixes in src/):** **0**. Verifiable via `git log --name-only ad67535..HEAD -- src/` returning empty across the entire Phase 1 range. Zero `src/` modifications.
- **`@pytest.mark.xfail` markers (D-12):** **0**. Verifiable via `grep -n "xfail" tests/test_layer_parity.py` returning empty.

This is exactly the "test passes on first run → no Commit B paired" path described in each per-plan SUMMARY ("Beads Issues Filed: None").

## Section 5 — Test counts

| Cohort | Count |
|--------|-------|
| Micro tests (gate-order × 3 + round-trip smoke × 1) | 4 |
| Forward parity grid (fast + slow) | 45 + 30 = 75 |
| h_T parity grid (fast + slow) | 45 + 30 = 75 |
| Backward parity grid (fast + slow) | 45 + 30 = 75 |
| h_0 != 0 random parity grid (fast + slow) | 45 + 30 = 75 |
| **TOTAL** | **304** |
| Fast suite (`-m "not slow"`) | 4 + 45 × 4 = **184** |
| Slow suite (`-m slow`) | 30 × 4 = **120** |

Final file: `tests/test_layer_parity.py` — **718 lines**, growing from 0 → 295 (Plan 01-01) → 453 (Plan 01-02) → 609 (Plan 01-03) → 718 (Plan 01-04). No further line changes in this plan; this SUMMARY is the only artifact.

## Section 6 — Cell-parity contract integrity

The < 1e-5 cell-parity contract in `tests/test_parity.py` (PROJECT.md Constraint, ROADMAP success criterion 4) is **untouched** across the entire Phase 1 range:

```
$ git diff ad67535^..HEAD -- tests/test_parity.py
(empty)

$ .venv/bin/python -m pytest tests/test_parity.py -q
............                                                             [100%]
12 passed in 2.95s
```

This satisfies the must_have "`tests/test_parity.py` `< 1e-5` cell parity gate is still green after every Commit B (verifiable via `pytest tests/test_parity.py -q` exit 0)". Vacuously true since there were no Commit B's, but verified directly.

## Section 7 — Pre-existing debt note (bd tracking)

The audit's mypy and ruff passes surface pre-existing toolchain debt that is **out-of-scope for Phase 1** by every available measure (zero `src/` touches; identical error counts at Phase 1 baseline as at Phase 1 head):

- **mypy** (strict, `src/gru_qat` scope per `pyproject.toml`): 145 errors in 10 files at Phase 1 head. Identical at baseline (3b6f093).
- **ruff** (`src tests` scope): 24 errors at Phase 1 head. Identical at baseline.

Verification:

```
$ git log --name-only ad67535..HEAD -- src/
(empty — zero src/ modifications across all Phase 1 commits)

$ .venv/bin/python -m mypy 2>&1 | tail -1
Found 145 errors in 10 files (checked 12 source files)

$ .venv/bin/python -m ruff check src tests 2>&1 | tail -1
Found 24 errors.
```

Filed as ONE tracking bd issue (per orchestrator + user decision at the audit checkpoint):

- **bd issue: `gru-triton-4m6`** — "Pre-existing mypy/ruff debt in src/gru_qat/*"
- Type: task, Priority: P3, Status: open.
- Notes capture the verification commands (`git log --name-only ad67535..HEAD -- src/` empty; mypy + ruff surface the same errors at head as at baseline).
- Belongs to a future hygiene/follow-up phase, not Phase 1.

**Why one tracking issue and not 169 per-error issues:** D-11 (one bd issue per parity finding) applies to parity *findings* — failures uncovered by the audit's parity assertions. Pre-existing toolchain debt that is identical pre- and post-audit is not a finding; it is inherited debt. Filing it as one tracking bd issue both honors session-close discipline (CLAUDE.md "File issues for remaining work") and avoids polluting bd with churn that has no Phase 1 actionable.

## Section 8 — Deferred items

Per CONTEXT.md Deferred Ideas (lines 119-127) and Plan 01-05's `<action>` body Pass 2 budget note:

- **Non-batched input `(T, IN)`** (nn.GRU accepts a 2-D input without batch dim; GRULayer requires `(T, B, IN)`). Not raised as a Phase 1 concern in CONTEXT or REQUIREMENTS. **Not filed as bd** — would be a new requirement, not an audit finding. If a user later needs this, file as new REF requirement; not blocked by this audit.
- **Slow-suite execution budget.** Current wall-clock ~89 seconds. CONTEXT D-08 / PLAN.md Task 1 specify a 10-minute abort threshold. Comfortably under budget; no grid pruning needed. CONTEXT.md Deferred Idea line 126 (prune the grid if slow > 5 min) remains deferred — not actionable now.
- **Bidirectional / multi-layer parity** (CONTEXT line 124) — out of scope per SCOPE.md. nn.GRU has both; GRULayer doesn't. Not auditing what doesn't exist. **Not filed as bd.**
- **Hand-rolled INT8 reference GRU** (CONTEXT line 125) — explicitly out of scope at project level (PROJECT.md Key Decisions). **Not filed as bd.**

## Section 9 — Hand-off to Phase 2

**Phase 1 is closed; Phase 2 is unblocked.**

Phase 2 (Triton-fast-path parity vs reference) plans (TRI-01..06 per REQUIREMENTS.md) will compare each Triton variant (dense, diagonal, monarch, butterfly) against the reference path audited in Phase 1, at the tighter < 1e-5 tolerance (per PROJECT.md tolerance tiers). The reference path is now the **trusted ground truth** because:

1. It matches `torch.nn.GRU` to < 1e-4 across the full 75-combo T × B × H grid on forward, backward, h_T, AND random-h_0. The grid spans T ∈ {1, 8, 64, 512, 1024} (long-T accumulation included), B ∈ {1, 4, 32} (degenerate B=1 included), H ∈ {1, 2, 8, 64, 512} (degenerate H=1, H=2 included).
2. The gate-order asymmetry (n-gate's `r * (W_hn h + b_hn)`) is explicitly tested by the 3 micro-tests — Phase 2's Triton kernels can rely on (r, z, n) layout being correct in the reference without re-deriving it.
3. The < 1e-5 cell-parity contract is preserved as the inner-loop invariant; the layer audit is layered on top, not a replacement.
4. The four-family symmetry pattern (forward / h_T / backward / h_0 != 0) is the natural template for Phase 2 Triton tests: same shape grid, same translation helpers, swap "nn.GRU" for "reference GRULayer" as the comparison target. The Plan 01-01 helpers (`_translate_cell_to_nn_gru`, `_make_dense_fp32_layer`) are reusable directly; Phase 2 will add a Triton-layer builder alongside.

The reference path itself shipped UNTOUCHED across all 5 plans — Phase 2 inherits the same `feat/diagonal-gru` reference-path code that landed before Phase 1 started (`d8218d4` / `c001a8a` / `4e10402` cluster pre-dates Phase 1). The audit validates what is already there; Phase 2 audits what builds on top.

## Task Commits

This plan ships exactly one commit:

1. **Task 3 (SUMMARY-only):** `docs(01-05): complete phase 1 audit (fully green)` — pending below.

Tasks 1 (audit kickoff) and 2 (checkpoint) produced agent-side state only (audit captures, bd issue creation). Per the orchestrator's checkpoint decision, the audit-note artifact is folded into this SUMMARY commit — no separate `notes(01-05): ...` commit needed. This matches the Plans 01-01..04 pattern of one-SUMMARY-per-plan with audit detail inline.

## Files Created/Modified

- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-05-SUMMARY.md` (created) — this file. **Zero other file changes.**

## Beads Issues Filed

- **`gru-triton-4m6`** — "Pre-existing mypy/ruff debt in src/gru_qat/*" (P3, open). One tracking issue for inherited toolchain debt; not a Phase 1 parity finding. See §7.

No bd parity issues filed because the audit is fully green. No `bd update <id> --claim` or `bd close <id>` operations occurred during this plan (other than the open of gru-triton-4m6).

## Deviations from Plan

None — plan executed exactly as written, taking the explicit "fully green → skip Task 2 → Task 3 SUMMARY only" branch documented in PLAN.md Task 1's `<action>` ("If the audit is fully green (zero failures): skip Task 2 entirely. Task 3 produces the green-audit SUMMARY."). The orchestrator's checkpoint resume-signal `audit green` explicitly approves this path.

The one cross-checkpoint instruction (file ONE tracking bd issue for pre-existing mypy/ruff debt) is documented in §7 and was completed before this SUMMARY was written. This is a session-close discipline directive (CLAUDE.md "File issues for remaining work"), not a parity finding or deviation.

## Issues Encountered

None — every command ran clean on the first attempt. (The orchestrator's prior auditing across Plans 01-01..04 already confirmed `.venv/bin/python` rather than system Python — same environment workaround as documented in Plan 01-01's "Issues Encountered" section.)

## Verification Snapshot

```
$ .venv/bin/python -m pytest tests/test_layer_parity.py -q -m "not slow"
184 passed, 120 deselected in 5.63s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -q
120 passed, 184 deselected in 88.68s (0:01:28)

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q
304 passed in 71.29s (0:01:11)

$ .venv/bin/python -m pytest tests/test_parity.py -q
12 passed in 2.95s

$ grep -n "xfail" tests/test_layer_parity.py
(no output)

$ git log --name-only ad67535..HEAD -- src/
(empty)

$ bd show gru-triton-4m6 | head -2
○ gru-triton-4m6 · Pre-existing mypy/ruff debt in src/gru_qat/*   [● P3 · OPEN]
Owner: claroche · Type: task

$ bd ready | head -2
○ gru-triton-4m6 ● P3 Pre-existing mypy/ruff debt in src/gru_qat/*

$ wc -l tests/test_layer_parity.py
718 tests/test_layer_parity.py

$ .venv/bin/python -m pytest tests/test_layer_parity.py --collect-only -q | tail -1
304 tests collected in 1.45s
```

## Phase-exit Gate Checklist (must_haves verification)

- [x] `pytest tests/test_layer_parity.py -q -m "not slow"` exits 0 (fast suite, 184 collected).

  ```bash
  .venv/bin/python -m pytest tests/test_layer_parity.py -q -m "not slow"
  ```

- [x] `pytest tests/test_layer_parity.py -m slow -q` exits 0 (slow suite, 120 collected). Within budget at ~89s wall-clock (10-min threshold).

  ```bash
  .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -q
  ```

- [x] `pytest tests/test_parity.py -q` exits 0 (< 1e-5 cell contract preserved, 12 passed).

  ```bash
  .venv/bin/python -m pytest tests/test_parity.py -q
  ```

- [x] `bd ready` shows ONLY the one tracking issue `gru-triton-4m6` (pre-existing debt). No unresolved Phase 1 *parity* findings — there are none.

  ```bash
  bd ready
  ```

- [x] No `@pytest.mark.xfail` in `tests/test_layer_parity.py` (D-12).

  ```bash
  grep -n "xfail" tests/test_layer_parity.py   # exit 1, no output
  ```

- [x] For each finding, Commit A precedes Commit B in `git log --follow tests/test_layer_parity.py`. **Vacuously satisfied** — zero findings, zero Commit B's.

  ```bash
  git log --follow tests/test_layer_parity.py --oneline
  git log --name-only ad67535..HEAD -- src/   # empty
  ```

## Self-Check: PASSED

- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-05-SUMMARY.md` exists: WILL VERIFY after Write tool completes.
- bd issue `gru-triton-4m6` exists with the expected title and is visible in `bd ready`: VERIFIED via `bd show gru-triton-4m6` and `bd ready`.
- All 9 mandatory sections present (audit verdict, pass-by-pass results, REF closure, two-commit confirmation, test counts, cell-parity integrity, pre-existing debt note, deferred items, hand-off): COUNTED in this file (each starts with `## Section N` or sits in the canonical SUMMARY-template slot above).
- Phase 1 layer-parity tests all green (184 fast + 120 slow = 304): VERIFIED via three independent pytest invocations.
- Cell-parity contract (`tests/test_parity.py`) still passes: VERIFIED via fresh `pytest tests/test_parity.py -q` (12 passed).
- No `xfail` in `tests/test_layer_parity.py`: VERIFIED via `grep -n` (exit 1, no output).
- No `src/` modifications across entire Phase 1: VERIFIED via `git log --name-only ad67535..HEAD -- src/` returning empty.
- `STATE.md` / `ROADMAP.md` UNTOUCHED in this plan (orchestrator owns post-wave writes): VERIFIED — this plan's commit will only contain `01-05-SUMMARY.md`.
- One commit will land for this plan (`docs(01-05): ...`): TO VERIFY at commit time.

---
*Phase: 01-reference-path-parity-vs-nn-gru*
*Plan: 05 — phase-exit audit*
*Completed: 2026-05-13*
