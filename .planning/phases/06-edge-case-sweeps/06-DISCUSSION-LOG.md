# Phase 6: Edge-case sweeps - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 6-edge-case-sweeps
**Areas discussed:** Empty inputs (T=0/B=0), Bug disposition, Coverage matrix density, Test file organization, Locked-file conflict resolution

---

## Empty inputs (T=0 / B=0)

| Option | Description | Selected |
|--------|-------------|----------|
| Raise ValueError (fail-loud) | Every path raises ValueError naming the offending dimension; matches kernel reality (0-grid can't launch) | ✓ |
| Return empty, shaped output | Paths return correctly-shaped zero-size tensors; permissive, needs pre-launch guards | |
| Split: empty out for reference, error for Triton | Reference returns empty, Triton raises; honest but inconsistent API | |

**User's choice:** Raise ValueError (fail-loud)
**Notes:** Locked policy logged in PROJECT.md. Resolves success criterion #4's "either … OR" to the ValueError branch for all 7 paths.

---

## Bug disposition

| Option | Description | Selected |
|--------|-------------|----------|
| Fix in-phase (Phase 4/5 discipline) | Failing test → bd issue → fix → passing test, all within Phase 6 | ✓ |
| Fix shallow, defer deep | Fix guard/helper bugs in-phase; defer deep kernel rewrites to Phase 7 | |
| File all, defer all to Phase 7 | Phase 6 surfaces only; all fixes in Phase 7 | |

**User's choice:** Fix in-phase (Phase 4/5 discipline)
**Notes:** Accepts that a deep BLOCK-size kernel fix can enlarge the phase. No `@pytest.mark.xfail` — a deferred red test would have no clean disposition under the no-xfail rule.

---

## Coverage matrix density

| Option | Description | Selected |
|--------|-------------|----------|
| All 7 paths × all edge dims, realistic inputs only | Edge cases test SHAPE handling; Phase 4 already covered adversarial numerics | ✓ |
| All paths × all dims × 3 adversarial classes | Full cross-product; large test count, slow-tier heavy | |
| Triton paths full, per-step paths basic-shapes only | Circulant/LDR get only T=1/B=1/H-small | |

**User's choice:** All 7 paths × all edge dims, realistic inputs only
**Notes:** Uniform coverage — circulant/LDR per-step paths get the same edge dimensions as Triton paths (rejected the reduced-subset option).

---

## Test file organization

| Option | Description | Selected |
|--------|-------------|----------|
| Single tests/test_edge_cases.py | All edge sweeps in one new file, parametrized | (revised choice) |
| Extend each per-variant test file | Co-locate edge cases with each kernel's tests | ✓ (initial) |
| Let the planner decide | No locked preference | |

**User's choice:** Initially "Extend each per-variant test file" — revised to "Single tests/test_edge_cases.py" after the locked-file conflict surfaced.
**Notes:** See the follow-up area below.

---

## Locked-file conflict resolution

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: Triton extend, rest in new file | 4 Triton paths extend unlocked files; reference + circulant/LDR in new file | |
| All in new tests/test_edge_cases.py | Every path's edge cases in one new file | ✓ |
| Lift D-51 lock for Phase 6 | Unlock test_parity.py / test_layer_parity.py / test_structure.py | |

**User's choice:** All in new tests/test_edge_cases.py
**Notes:** Per-variant extension collided with D-51 — the reference path (`test_parity.py`, `test_layer_parity.py`) and circulant/LDR (`test_structure.py`) homes are all locked since Phase 1. A single new file sidesteps the conflict and matches the Phase 5 single-file precedent.

---

## Claude's Discretion

- Exact `@pytest.mark.parametrize` grid structure for path × shape.
- Import-from-strict-files vs local layer factories — decided by coupling analysis (locked files may be imported from, not edited).
- Physical placement of the T=0/B=0 `ValueError` guard (layer vs kernel-wrapper) — policy locked, placement delegated.
- HW-limit skips (`_skip_if_monarch_bwd_hw_limit`-style) vs genuine BLOCK-assumption bugs.
- Plan count — single file suggests a single race-free plan, but a path-group split is allowed if no two plans write `test_edge_cases.py` concurrently.

## Deferred Ideas

- Adversarial × edge-shape cross-product — intentional non-goal (Phase 4 owns adversarial numerics).
- `AUDIT-REPORT.md` — Phase 7 deliverable.
- Performance/bench re-validation at small shapes — v2 (PERF-01/02).
- `step_triton(x_t, h)` streaming kernel — out of scope per SCOPE.md.
