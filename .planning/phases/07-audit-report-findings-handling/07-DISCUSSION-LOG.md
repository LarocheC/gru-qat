# Phase 7: Audit report + findings handling - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 7-audit-report-findings-handling
**Areas discussed:** Finding triage framework, Green gate (criterion #3), Lint/type gate, gru-triton-n20 bug, Q1/Q2 conflict reconciliation

---

## Finding triage framework (the 14 open bd issues)

| Option | Description | Selected |
|--------|-------------|----------|
| Three buckets: FIX / ACCEPTED-DIVERGENCE / DEFER-v2 | Tractable bugs FIXed, TF32 family accepted, HW/process classified | (effective outcome) |
| Fix everything in Phase 7 | Includes input_precision='ieee' TF32 rewrites | ✓ (initial) |
| Defer all unresolved to v2 | Pure documentation, nothing fixed | |

**User's choice:** Initially "Fix everything" — reconciled (see Q1/Q2 conflict below) to: fix all *genuine tractable* bugs, accept the irreducible TF32 family as documented divergence.
**Notes:** "Fix everything" was bounded by the reconciliation — the irreducible TF32 cases and the e0l hardware limit are explicitly NOT code-fixed.

---

## Green gate — ROADMAP criterion #3 (`pytest -q` must pass)

| Option | Description | Selected |
|--------|-------------|----------|
| `divergence` marker; gate runs `-m "not divergence"` | Mark known TF32-divergent cases; gate excludes them; marked tests stay live as documentation | ✓ |
| Re-baseline strict-test bounds to the accepted disposition | Adjust tolerances so tests pass green | |
| Scope criterion #3 to the core suite only | Strict-tier excluded from the gate entirely | |

**User's choice:** `divergence` marker; gate runs `pytest -q -m "not divergence"`.
**Notes:** Marked tests stay collected and run (no xfail, no skip) — `pytest -m divergence` reproduces the divergence on demand. The operationalization is documented in AUDIT-REPORT.md.

---

## Lint/type gate — mypy/ruff (gru-triton-4m6, ~145 mypy + ~23 ruff errors)

| Option | Description | Selected |
|--------|-------------|----------|
| Fix src/gru_qat only; document test-helper baseline | mypy + ruff src green; test debt accepted | |
| Fix everything — src and tests fully green | Clear all errors including test files | ✓ |
| Accept the whole baseline, document only | Fix nothing, reframe criterion #3 | |

**User's choice:** Fix everything — src and tests fully green.
**Notes:** `mypy` is config-scoped to `src/gru_qat` (strict), so "tests green" = ruff-clean tests, not mypy-strict tests — no conflict with the existing tests-not-mypy-strict convention.

---

## gru-triton-n20 (shared QuantizerConfig instance — silent-correctness bug)

| Option | Description | Selected |
|--------|-------------|----------|
| Fix in Phase 7, absorb the strict-test re-baseline | deepcopy fix + re-baseline the ~18 affected Phase 4 strict tests | ✓ |
| Accept-divergence: document + recommend, defer fix to v2 | Ship the known bug, document the fix | |

**User's choice:** Fix in Phase 7, absorb the strict-test re-baseline.
**Notes:** The deepcopy fix changes quantizer scales and breaks Phase 4 strict tests' bit-identity contract; that re-baseline is in-scope Phase 7 work, not a regression.

---

## Q1/Q2 conflict reconciliation

| Option | Description | Selected |
|--------|-------------|----------|
| Fix all genuine bugs; `divergence`-mark only the irreducible TF32 cases | Self-consistent reading | ✓ |
| Truly fix everything — do the input_precision='ieee' rewrites | No marker needed; massive scope; overrides Option C | |
| Divergence marker wins — fix only tractable bugs | Same end state as option 1, framed document-heavy | |

**User's choice:** Fix all genuine bugs; `divergence`-mark only the irreducible TF32 cases.
**Notes:** Q1 ("fix everything") and Q2 ("divergence marker") could not both hold literally — a truly-fixed TF32 family leaves nothing to mark. Also surfaced: `e0l` is a hardware limit (not code-fixable), `u00` is a process finding (no code bug). The reconciliation is the self-consistent reading and matches PROJECT.md's locked Option C decision.

---

## Claude's Discretion

- Per-issue FIX-vs-ACCEPTED-DIVERGENCE call for the ambiguous backward-drift issues (mjy, lht, e7t, fpl) — needs root-cause inspection.
- Per affected strict test in the n20 re-baseline: re-baseline the bound vs `divergence`-mark it.
- Plan/wave structure; `divergence` marker granularity (per-function vs per-parametrize-case).

## Deferred Ideas

- `input_precision="ieee"` TF32-elimination kernel rewrites → v2.
- `gru-triton-e0l` Monarch-bwd kernel-tiling redesign for consumer-GPU SMEM → v2.
- ACT-01 / ACT-02 (per-channel min_max observer, LSQ/PACT) → v2 (already in REQUIREMENTS.md v2).
- PERF-01 / PERF-02 bench re-validation → v2.
