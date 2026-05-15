# Phase 6: Edge-case sweeps - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Pin every GRU code path at boundary shapes — proving that small, degenerate, and long inputs either produce correct output or fail with a clear, tested error.

The 7 paths in scope (the goal's enumeration):
1. Reference path (`GRULayer(use_triton=False)`, dense)
2. Dense Triton (`scan.py`)
3. Diagonal Triton (`scan_diagonal.py`)
4. Monarch Triton (`scan_monarch.py`)
5. Butterfly Triton (`scan_butterfly.py`)
6. Circulant per-step PyTorch fallback
7. LDR per-step PyTorch fallback

Edge dimensions swept (success criteria #1–#4):
- **T=1** — single timestep, fwd + bwd (EDG-01)
- **B=1** and **H ∈ {1, 2}** — small-shape BLOCK-assumption failure modes (EDG-02)
- **T ∈ {512, 1024}** — long-sequence accumulated-drift check, `@pytest.mark.slow` (EDG-03)
- **T=0 / B=0** — empty-input disposition (EDG-04)

**Not in scope:** new kernels, new structured kinds, performance work, adversarial *numerical* coverage (Phase 4 already swept the 3 adversarial classes — Phase 6 edge cases test SHAPE handling, see D-07).
</domain>

<decisions>
## Implementation Decisions

### A. Empty-input disposition (T=0 / B=0)

- **D-01:** Every path raises `ValueError` on `T=0` or `B=0` — fail-loud, no empty-output passthrough. The message MUST name the offending dimension (`T` or `B`).
  - **Rationale:** An empty sequence/batch is almost always a caller bug; embedded/deploy contexts want it surfaced, not silently absorbed. Triton kernels cannot launch a 0-size grid anyway, so a uniform fail-loud policy matches kernel reality and avoids an inconsistent per-path API. Aligns with the existing error-handling convention (`ValueError` with the offending field name — see `gru_cell.py:107`, `structure.py:79`).
- **D-02:** This policy is logged in `PROJECT.md` as the locked Phase 6 disposition for success criterion #4 (the criterion's "either … OR" is now resolved to the ValueError branch for all 7 paths).
- **D-03:** If a path *currently* hangs, NaNs, or raises an unclear/wrong error at `T=0`/`B=0`, that is a finding — handle it under D-05 (fix in-phase). The guard may live at the `GRULayer` level (reference + dispatch) and/or the Triton wrapper entry (`scan*.py` wrappers already assert `is_cuda`/`dtype` — add the dimension guard alongside). Planner decides placement; the *policy* (ValueError, names the dim) is locked.

### B. Bug disposition when an edge sweep surfaces a real bug

- **D-04:** CONCERNS.md predicts BLOCK-size-assumption failures at `B=1`, `H ∈ {1, 2}` (butterfly `B % BLOCK_B`, monarch non-pow2 `BLKSZ`, persistent-grid limits). When an edge sweep surfaces a real bug, **fix it in Phase 6** — do not defer.
- **D-05:** Follow the inherited D-37/D-50 two-commit discipline: failing test (Commit A) → `bd create` issue → fix (Commit B) → passing test. **No `@pytest.mark.xfail`** anywhere in the Phase 6 surface.
- **D-06:** Consequence accepted: a deep BLOCK-size kernel fix can enlarge the phase. That is the chosen tradeoff — Phase 6 is an audit phase; surfacing-without-fixing would leave the no-xfail rule with no clean disposition for red tests. Phase 7 (`AUDIT-REPORT.md`) is the *reporting* phase, not a deferred-fix dumping ground.

### C. Coverage matrix density

- **D-07:** All 7 paths × all edge dimensions, **realistic inputs only** (a single input class). Edge cases test *shape* handling — Phase 4 already swept the 3 adversarial numerical classes (`realistic` / `near-saturation` / `large-magnitude`). Do NOT cross-product edge shapes against adversarial classes.
- **D-08:** Coverage is uniform across all 7 paths — circulant and LDR per-step paths get the *same* edge dimensions as the Triton paths (T=1, B=1, H∈{1,2}, T∈{512,1024}, T=0/B=0), not a reduced subset. (User explicitly rejected the "per-step paths basic-shapes only" option.)
- **D-09:** Long-T tests (`T ∈ {512, 1024}`) are `@pytest.mark.slow` per success criterion #3. Tolerance: layer-vs-`nn.GRU` < 1e-4 for the reference path; Triton-vs-reference per the PROJECT.md tier (< 1e-5 non-`tl.dot` / < 5e-4 `tl.dot`). Reuse the tiered tolerances — do not invent new bounds.

### D. Test file organization

- **D-10:** All edge sweeps go in a **single new `tests/test_edge_cases.py`**, parametrized over path × shape. (User initially preferred per-variant extension, then revised after the locked-file conflict surfaced — see D-11.)
  - **Rationale:** Matches the Phase 5 single-file precedent (`test_calibration.py` held the whole lifecycle story). Keeps the edge-case story navigable in one place. A single file also means a single plan with no parallel-write race surface (the F-04-05-D / `gru-triton-u00` race that Phases 4–5 mitigated).
- **D-11:** Per-variant extension was rejected because it collides with the D-51 locked files: the reference path's natural homes (`test_parity.py`, `test_layer_parity.py`) and the circulant/LDR home (`test_structure.py`) are all locked since Phase 1. Only the 4 Triton base files (`test_triton_scan.py`, `test_triton_diagonal.py`, `test_triton_monarch.py`, `test_butterfly_dispatch.py`) were unlocked — not enough to host all 7 paths. A new file sidesteps the conflict entirely.

### Claude's Discretion (delegated to planner / executor)

- Exact `@pytest.mark.parametrize` structure for the path × shape grid.
- Whether to import shared helpers from Phase 4 strict files or build minimal local layer factories — decide by import-coupling analysis (note `test_parity.py`/`test_layer_parity.py`/`test_structure.py` are locked and may NOT be edited; importing *from* them is fine).
- Where the `T=0`/`B=0` `ValueError` guard physically lives (layer vs kernel-wrapper) — policy is locked (D-01), placement is not.
- `_skip_if_monarch_bwd_hw_limit`-style skips for any shape the RTX 2000 Ada kernels genuinely cannot launch (inherited from Phase 4) — but a HW-limit *skip* is distinct from a *bug*; a true BLOCK-assumption bug must be fixed (D-04), not skipped.
- Plan count: a single file *suggests* a single plan (race-free), but the planner may split by path-group if task count warrants — provided no two plans write `test_edge_cases.py` concurrently.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope + requirements
- `.planning/ROADMAP.md` § Phase 6 — phase goal + 5 success criteria
- `.planning/REQUIREMENTS.md` EDG-01 / EDG-02 / EDG-03 / EDG-04 — locked requirements with verification stubs
- `.planning/PROJECT.md` — core value, tolerance tiers, constraints; D-02 logs the T=0/B=0 policy here

### Known fragility (drives the BLOCK-size sweeps)
- `.planning/codebase/CONCERNS.md` — butterfly `B % BLOCK_B` partial-tile OOB (lines ~58–62, ~226), monarch non-pow2 `BLKSZ` pad-to-pow2 fragility (~132), persistent-grid `cdiv(B,block_b)*cdiv(H,block_oh) <= sm_count` deadlock (~111–114, ~156), per-channel `min_max` known-broken path (~232)
- `.planning/codebase/TESTING.md` — existing test coverage map and gaps
- `.planning/codebase/ARCHITECTURE.md` — path dispatch + fast-path eligibility rules

### Implementation surfaces
- `src/gru_qat/gru_layer.py` — `GRULayer.forward`, `_forward_fast_dispatch` (fast-path eligibility at ~line 100); natural home for the reference/dispatch-level `T=0`/`B=0` guard
- `src/gru_qat/gru_cell.py` — existing `ValueError` shape-validation convention to mirror (e.g. line 107)
- `src/gru_qat/structure.py` — circulant (`_CirculantLinear`) + LDR (`_LDRLinear`) per-step paths; existing `ValueError` convention (line ~79)
- `src/gru_qat/triton_kernels/scan.py`, `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py` — Triton wrappers assert `is_cuda`/`dtype`; the dimension guard goes alongside

### Phase 4/5 reusable test infrastructure (import only — files are effectively locked for new additions)
- `tests/test_triton_scan_strict.py` — `_adversarial_inputs`, layer factories, `_assert_quant_parity` (Phase 6 uses realistic inputs only per D-07, but the layer factories may still be useful)
- `tests/test_triton_diagonal_strict.py`, `tests/test_triton_monarch_strict.py` (`_skip_if_monarch_bwd_hw_limit`), `tests/test_triton_butterfly_strict.py`
- `tests/test_calibration.py` — Phase 5 single-file precedent for organizing a multi-path parametrized sweep

### D-51 locked files — MUST NOT be edited (Phase 6 may import from them, not modify)
- `tests/test_parity.py`, `tests/test_layer_parity.py`, `tests/test_structure.py`
- The 4 `tests/test_triton_*_strict.py` strict files

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_adversarial_inputs(cls, T, B, H)` and the per-kernel layer factories in the Phase 4 strict files — Phase 6 needs only the `realistic` class (D-07) but the factories build correctly-configured layers per path.
- `_skip_if_monarch_bwd_hw_limit(T, B, H, nblocks)` — inherited skip for shapes the monarch bwd kernel cannot compile on RTX 2000 Ada. Distinguish HW-limit skip (legitimate) from a BLOCK-assumption bug (must fix, D-04).
- `tests/test_calibration.py` — the Phase 5 template for a single-file path × shape parametrized sweep with `cuda_only` + `pytest.importorskip("triton")` gating.
- Existing `ValueError` shape-validation pattern (`gru_cell.py`, `structure.py`) — the T=0/B=0 guard mirrors this exactly (message names the offending dimension).

### Established Patterns
- Triton tests gate with `cuda_only` mark + `pytest.importorskip("triton")`; reference + circulant/LDR per-step tests run CPU-side and need no CUDA.
- Tiered tolerance per PROJECT.md Constraints — Phase 6 reuses, never invents (D-09).
- `torch.set_float32_matmul_precision("high")` is set in numeric-tolerance tests so reference and test paths share a TF32 regime.
- Single-plan-per-shared-file to avoid the F-04-05-D parallel-write race (Phases 4–5 mitigation).

### Integration Points
- New file `tests/test_edge_cases.py` — the only new source artifact; parametrized over all 7 paths × edge shapes.
- T=0/B=0 `ValueError` guards may add a few lines to `gru_layer.py` and/or the `scan*.py` wrappers (and possibly `structure.py`) — these are the only `src/gru_qat/**` changes Phase 6 expects, and only if the sweep proves a path currently lacks the guard.

</code_context>

<specifics>
## Specific Ideas

- Edge cases prove *shape* robustness, not numerical adversariality — Phase 4 owns the adversarial numerical sweep, Phase 6 owns the boundary-shape sweep. Keep the two concerns separate (D-07).
- The `B % BLOCK_B != 0` partial-last-tile corner (butterfly OOB fix `d8218d4` shipped *without* a regression test, per CONCERNS.md) is a specific must-cover case — `B=1` is the extreme of that family; the planner should also consider the CONCERNS.md-suggested `B ∈ {1,3,5,7,9,17,33}` butterfly sweep.
- "Fix in-phase" (D-04) means Phase 6 may legitimately end up touching kernel code — that is expected and accepted, not scope creep.

</specifics>

<deferred>
## Deferred Ideas

| Idea | Why deferred | Suggested phase |
|---|---|---|
| Adversarial × edge-shape cross-product | Phase 4 already covered adversarial numerics; Phase 6 is shape-only by decision D-07 | Not planned — intentional non-goal |
| `AUDIT-REPORT.md` summarizing the full audit | That is Phase 7's deliverable; Phase 6 only files bd issues for its own findings | Phase 7 |
| Per-channel `min_max` observer broken path (CONCERNS.md ~232) | Resolved in Phase 4 (QNT-04 / bd:gru-triton-x15 closed) — not a Phase 6 concern | Done (Phase 4) |
| Performance / bench re-validation at small shapes | Correctness-only milestone; bench is explicitly out of scope per PROJECT.md | v2 (PERF-01/02) |
| `step_triton(x_t, h)` streaming kernel for T=1 | Streaming bypasses Triton by design; per-launch overhead may exceed per-step Python cost | Out of scope per SCOPE.md |

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 6-Edge-case sweeps*
*Context gathered: 2026-05-15*
