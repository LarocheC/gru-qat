# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-13)

**Core value:** Every code path that claims to compute a GRU must produce numerically equivalent output to `torch.nn.GRU` (under matched recipe), and any deviation must be a tested, documented, intentional one — not a silent drift.
**Current focus:** Phase 1 — Reference-path parity vs nn.GRU

## Current Position

Phase: 1 of 7 (Reference-path parity vs nn.GRU)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-05-13 — Roadmap created, 28 v1 requirements mapped to 7 phases

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| — | — | — | — |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- Init: Baseline is `torch.nn.GRU` (cuDNN); gate-ordering / bias-fusion quirks live in test-helper layer, not in reference-path code.
- Init: Reference PyTorch path is ground truth for Triton/structured parity (no third baseline).
- Init: Forward + backward parity both required (recent fix cluster shows bwd is where bugs hide).
- Init: Tiered tolerance — < 1e-5 for cell + Triton-vs-reference, < 1e-4 for layer-vs-nn.GRU, bit-identical for quant-on.
- Init: Fix in-milestone (each finding → failing test → beads issue → fix → audit ends green).

### Pending Todos

None yet.

### Blockers/Concerns

None yet. Phase 4 (Quant-on) will require a decision on per-channel `min_max` observer: fix vs. fence — log to PROJECT.md when phase enters planning.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none — milestone init)* | | | |

## Session Continuity

Last session: 2026-05-13
Stopped at: Roadmap created, ready for `/gsd-plan-phase 1`
Resume file: None
