---
phase: 01-reference-path-parity-vs-nn-gru
plan: 04
subsystem: testing
tags: [pytest, torch.nn.GRU, parity, fp32-identity, parametrize, slow-marker, h_0-nonzero, random-h0]

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 01
    provides: "_translate_cell_to_nn_gru, _make_dense_fp32_layer, set_float32_matmul_precision('highest') module preamble"
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 02
    provides: "FAST_GRID (45 tuples), SLOW_GRID (30 tuples), pytest import"
provides:
  - "test_layer_with_random_h0_matches_nn_gru — REF-02 random initial hidden state parity over 45 fast cases × 2 tensors (out + h_T)"
  - "test_layer_with_random_h0_matches_nn_gru_slow — same family, @pytest.mark.slow, 30 long-T cases × 2 tensors"
affects:
  - "Plan 01-05: audit-report consolidation — all four D-09 parametrized families (fwd, h_T, bwd, h0!=0) now exist over the full 75-combo grid"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Random initial hidden state shape adapter: h0_3d = torch.randn(1, B, H) is the nn.GRU shape; h0_2d = h0_3d.squeeze(0) is the GRULayer shape — view (shared storage), safe for forward-only tests."
    - "Per-name assertion loop with `h0=rand` tag in the failure message — distinguishes this family from the zero-h0 fwd / h_T grids that share the same T,B,H pytest id space; bd-issue titles are unambiguous on a glance."
    - "Both `out` and `h_T` asserted in the SAME test — D-09 isolation is the h_0!=0 axis, not the fwd-vs-h_T family split (CONTEXT Specifics line 117 explicitly rejects creating 8 grid families instead of 4)."

key-files:
  created:
    - ".planning/phases/01-reference-path-parity-vs-nn-gru/01-04-SUMMARY.md"
  modified:
    - "tests/test_layer_parity.py (+109 lines; was 609, now 718)"

key-decisions:
  - "Honored D-09 + CONTEXT Specifics literally: both `out` and `h_T` are asserted in the SAME test for the h_0!=0 family. The isolation axis here is 'random h_0', not 'forward vs final-hidden-state' — that split is already done by Plan 01-02's fwd vs h_T pair on the zero-h0 grids. Splitting random-h0 into two siblings would have created 8 grid families against the 4 specified by D-09."
  - "Shape adapter via `.squeeze(0)` view (shared storage), not `.clone()`: the test is forward-only and neither call writes in-place to h0. A clone is unnecessary; a view is faster and removes a degree of freedom where the two implementations might diverge on a separately-stored tensor (they can't, but the simpler code says so visibly)."
  - "`h0=rand` token in the assertion failure message: pytest test ids alone read 'test_layer_with_random_h0_matches_nn_gru[T-B-H]'; the tag in the assertion is redundant by 1 bit but doubles as a grep handle for `bd ready` / `pytest --tb=short` output post-failure — distinguishes this family at a glance from the zero-h0 forward / h_T families that share the same {T,B,H} parametrize id space."
  - "No autograd machinery in this family: random-h0 backward is implicitly covered by Plan 01-03's `test_layer_backward_matches_nn_gru` via the `dh_0` slot. The backward graph for a forward pass is the same path regardless of the h_0 value used in the forward; if a random-h0-specific bwd bug existed, the gradient test would have caught it. Duplicating the detach-clone idiom + 6-grad loop here would add 50 lines and zero diagnostic power."

patterns-established:
  - "Random-h_0 parity body: torch.manual_seed(0) → IN=max(H,1) → _make_dense_fp32_layer(IN, H) → _translate_cell_to_nn_gru(layer) → x = torch.randn(T,B,IN) → h0_3d = torch.randn(1,B,H), h0_2d = h0_3d.squeeze(0) → both implementations consume their shape → per-name loop over [(out), (h_T)] with `h0=rand` failure tag."
  - "All four D-09 parametrized families (fwd, h_T, bwd, h0!=0) now use the same scaffold. The h0!=0 family is the smallest delta from the fwd/h_T scaffold (just swap h0 construction + assert both); the bwd family is the largest delta (detach-clone + 6 grads). The symmetry makes the file easy to extend in future audits."

requirements-completed: [REF-02]

# Metrics
duration: ~3.5 min
completed: 2026-05-13
---

# Phase 1 Plan 4: Random initial hidden state (h_0 != 0) parity Summary

**Random initial hidden state parity (REF-02) lands as two parametrized test functions over the 75-combo T × B × H grid (45 fast + 30 slow), comparing both `out` and `h_T` between `GRULayer` and `torch.nn.GRU` at < 1e-4 relative tolerance when `h_0` is a random tensor. All 150 random-h0 assertions (75 cases × 2 tensors) pass on CPU under `set_float32_matmul_precision('highest')`. All four D-09 parametrized families (fwd, h_T, bwd, h_0!=0) now exist over the full grid; Phase 1 audit body is complete.**

## Performance

- **Duration:** ~3.5 min
- **Started:** 2026-05-13T17:23:07Z
- **Completed:** 2026-05-13T17:26:39Z
- **Tasks:** 1 (`type="auto"`)
- **Files modified:** 1 (`tests/test_layer_parity.py`)
- **Files created:** 1 (this SUMMARY)

## Accomplishments

- **`test_layer_with_random_h0_matches_nn_gru` (FAST_GRID, 45 cases)** — REF-02 satisfied on the fast grid. Per case, the test:
  1. Seeds `torch.manual_seed(0)` and builds the dense fp32-Identity layer + matching nn.GRU via the existing Plan 01-01 helpers.
  2. Constructs `x = torch.randn(T, B, IN)` where `IN = max(H, 1)`.
  3. Constructs `h0_3d = torch.randn(1, B, H)` (nn.GRU's `[num_layers=1, B, H]` shape) and `h0_2d = h0_3d.squeeze(0)` (GRULayer's `[B, H]` shape — a view, shared storage with h0_3d, safe for this forward-only test).
  4. Runs `(out_ref, hT_ref) = gru(x, h0_3d)` and `(out_ours, hT_ours) = layer(x, h0_2d)`.
  5. Loops over the two `(name, ref_t, our_t)` triples — `out` and `h_T` (the latter with `hT_ref.squeeze(0)` shape-adapt) — asserting `rel < 1e-4` with `(T={T},B={B},H={H},h0=rand)` in every failure message.
- **`test_layer_with_random_h0_matches_nn_gru_slow` (SLOW_GRID, 30 cases)** — REF-02 satisfied on the slow grid (T ∈ {512, 1024}). Identical body, decorated with `@pytest.mark.slow` above `@pytest.mark.parametrize` per the file's locked decorator order. Long-T random-h0 is where any subtle accumulation drift in the h0 propagation would surface — the initial-state influence compounds over 512+ recurrent steps and any asymmetry between the two implementations' h0 handling would show up here even if it survives T=64 in the fast grid.
- **D-09 honored a fourth and final time:** random-h0 parity is its OWN parametrized function family, AND inside that family both `out` and `h_T` are asserted together — because the isolation axis here is "random h_0", not "forward vs final-hidden-state". Splitting would create 8 grid families instead of the 4 specified by D-09.
- **D-12 honored:** zero `@pytest.mark.xfail` markers anywhere in the file.
- **No `src/` modifications.** All 150 assertions (75 cases × 2 tensors) pass on the first run on CPU. The cell's manual unroll correctly threads a non-zero initial hidden state into the time loop — no "h0=None default special-case" bug, no off-by-one initialization drift.

## Task Commits

Sequential executor on `feat/diagonal-gru` (normal hooks; no `--no-verify`):

1. **Task 1: h_0 != 0 parity (fast + slow)** — `95d2305` (test)

## Files Created/Modified

- `tests/test_layer_parity.py` (modified, +109 lines net; 609 → 718) — added one ASCII-rule stanza divider, an extended comment block on the random-h0 philosophy / shape contract / why-no-autograd, and two parametrized test functions. No changes to imports, no new helpers, no touching of Plan 01-01 helpers, Plan 01-02 grid constants / fwd / h_T tests, or Plan 01-03 backward tests.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-04-SUMMARY.md` (created) — this file.

## Decisions Made

- **Both `out` and `h_T` in one test, not two:** Plan body + CONTEXT Specifics + D-09 are explicit on this — the h_0!=0 family's isolation axis is the random initial state, not the per-tensor split. Plan 01-02 already isolated fwd vs h_T at zero-h0; doing the same here would duplicate that signal without adding new diagnostic power. If a random-h0 bug somehow affected only `h_T` and not `out` (or vice versa), the named-failure-loop's `name` token would still identify which one drifted from inside the same test.
- **`squeeze(0)` view, not `.clone()`:** The test is forward-only; no in-place writes; no autograd state shared between the two calls. The view-based shape adapter is mechanically equivalent to a separately-stored clone for this test's purposes and is one line shorter. If a future audit ever extends this to a random-h0 backward test, a `.clone()` would be required for the detach-requires_grad pattern (since the two graphs would otherwise share a leaf) — but that's out of scope here per the no-autograd-machinery decision below.
- **`h0=rand` tag in the failure message:** Redundant by 1 bit with the test function name (which already says "with_random_h0"), but doubles as a grep handle for `bd ready` listings and pytest tail output. Without it, a `bd ready` row reading "test_layer_with_random_h0_matches_nn_gru[T=64-B=4-H=512]" and a hypothetical `test_layer_forward_matches_nn_gru[T=64-B=4-H=512]` failure would have to be visually distinguished by the test function name only — the tag lets the assertion message itself scream "AND h0 was random" so triage is one-shot.
- **No autograd machinery for random-h0:** The plan's `<action>` body is explicit that random-h0 backward is implicitly covered by Plan 01-03's `test_layer_backward_matches_nn_gru` via the `dh_0` slot (the backward graph for a forward pass is the same path regardless of the h_0 value used in the forward). Duplicating the detach-clone idiom + 6-grad loop here would add ~50 lines and zero new diagnostic signal. If a random-h0-specific backward bug existed, the existing `dh_0` parity assertion at zero-h0 would have caught it via the chain rule (or, equivalently, by inspection: the same autograd Function objects are constructed regardless of the leaf value).

## Deviations from Plan

None — plan executed exactly as written. (The `squeeze(0)` view choice, the `h0=rand` tag, the absence of autograd, and the both-assertions-in-one-test pattern all match the plan's `<action>` body and CONTEXT D-09 + Specifics line 117 verbatim.)

## Issues Encountered

- **`python` on PATH is system Python**, same as Plans 01-01 through 01-03. Worked around by invoking `.venv/bin/python -m pytest` / `-m ruff` directly. Not a code issue; documented in Plan 01-01's SUMMARY.

## Beads Issues Filed

None — all 150 random-h0 assertions (75 cases × 2 tensors per case) passed on the first run. No initialization bug or h_0-propagation drift surfaced in the reference path, so the D-10/D-11 two-commit failing-test-before-fix protocol did not need to fire.

If any case had failed (e.g. `out` drift at T=1024 with h_0 = randn — an accumulation-into-the-recurrence bug), the protocol would have been:

1. **Commit A:** The failing parametrized test (already on disk; isolate via `pytest --tb=short -x -k random_h0` until first failure, capture the offending T/B/H and the tensor `{name}`).
2. **`bd create --title "test_layer_with_random_h0_matches_nn_gru[T-B-H] out drift (h0=rand)"`** with the `pytest --tb=short` tail in `--notes`. The bd title encodes both the test id, the tensor name, and the `h0=rand` discriminator — actionable in one line.
3. **Commit B:** Fix in `src/gru_qat/gru_layer.py` (most likely the `if h0 is None: h0 = x.new_zeros(...)` branch versus the explicit-h0 path producing subtly different initial conditions — see `gru_layer.py:159-160`). Same test now passes.
4. **`bd close <id>`** after CI green.

## Verification Snapshot

```
$ .venv/bin/python -m pytest tests/test_layer_parity.py::test_layer_with_random_h0_matches_nn_gru -q
.............................................                            [100%]
45 passed in 2.28s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -k random_h0 -q
..............................                                           [100%]
30 passed, 274 deselected in 9.97s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q -m "not slow"
184 passed, 120 deselected in 6.78s
# 4 micro/roundtrip + 45 fwd + 45 h_T + 45 bwd + 45 h0rand = 184

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q
304 passed in 66.40s (0:01:06)
# Full file: 4 micro/roundtrip + 75 fwd + 75 h_T + 75 bwd + 75 h0rand = 304

$ .venv/bin/python -m pytest tests/test_layer_parity.py --collect-only -q | tail -5
tests/test_layer_parity.py::test_layer_with_random_h0_matches_nn_gru_slow[1024-32-8]
tests/test_layer_parity.py::test_layer_with_random_h0_matches_nn_gru_slow[1024-32-64]
tests/test_layer_parity.py::test_layer_with_random_h0_matches_nn_gru_slow[1024-32-512]

304 tests collected in 1.21s

$ .venv/bin/python -m pytest tests/test_parity.py -q   # <1e-5 cell contract
12 passed in 1.31s

$ .venv/bin/python -m ruff check tests/test_layer_parity.py
All checks passed!

$ grep -c "def test_layer_with_random_h0" tests/test_layer_parity.py
2

$ grep -n "xfail" tests/test_layer_parity.py
(no output — D-12)

$ wc -l tests/test_layer_parity.py
718 tests/test_layer_parity.py

$ git log --oneline -3
95d2305 test(01-04): h_0 != 0 parity grid (45 fast + 30 slow) vs nn.GRU
005673d docs(01-03): complete backward gradient parity plan
8cd96ad test(01-03): backward parity grid (45 fast + 30 slow) vs nn.GRU
```

## Drift Analysis

All 75 random-h0-grid cases pass on both `out` and `h_T`; no drift surfaced. If drift had appeared, the meaningful question (analog of Plan 01-03's drift-analysis section) would have been whether it was:

- **Uniform across the grid** → a structural h_0-propagation bug in the time loop (e.g. the `if h0 is None: h0 = x.new_zeros(...)` branch in `gru_layer.py:159-160` constructing a tensor with a different dtype / device / strides than what the user-provided h_0 would have). Would surface equally at T=1 and T=1024. **Action:** Fix in `src/gru_qat/gru_layer.py` per D-10; regression test stays.
- **Scale-dependent (large-T, large-H only)** → accumulation drift in the random-h0 path that doesn't appear in zero-h0 because the initial state has nonzero magnitude that compounds through the recurrence. Would NOT appear in the zero-h0 fwd / h_T grids. **Action:** Either tighten the per-shape tolerance per Phase 6 edge-sweep precedent, or carve out the offending corner as a known-accepted divergence in the AUDIT-REPORT. Does NOT reopen Phase 1.

Neither happened. The reference path threads the random initial state through the recurrence with the same numerical behavior as `nn.GRU` to < 1e-4 across all 75 corners of the T × B × H grid.

## Next Phase Readiness

- **All four D-09 parametrized families now exist** in `tests/test_layer_parity.py`: forward, h_T, backward, h_0!=0. Each has a fast (45-case) and slow (30-case) sibling; total grid coverage is `4 × 75 = 300` parametrized cases plus 4 micro/roundtrip tests = 304 tests on disk.
- **Phase 1 audit body is complete from the test-coverage side.** Plan 01-05 (audit-report consolidation) can now reference the full set of test families and grid counts without anything still in flight.
- **No `src/` modifications** shipped; no blockers; no bd issues open. The cell / layer code remains exactly as `feat/diagonal-gru` left it before this plan started — the entire Phase 1 audit has not surfaced a single reference-path bug.
- **< 1e-5 cell parity contract in `tests/test_parity.py` is untouched** (12 passed; `git diff HEAD~1 -- tests/test_parity.py` is empty for commit `95d2305`).
- **The four-family symmetry pattern (fwd / h_T / bwd / h_0!=0)** is the natural template for future audits at other layer levels (e.g. Triton-vs-reference parity in Phase 2, where the same four families would compare a Triton-fast-path layer against this reference-path layer at < 1e-5).

## Self-Check: PASSED

- `tests/test_layer_parity.py` exists at 718 lines (was 609 from Plan 01-03): FOUND.
- Two `def test_layer_with_random_h0` functions exist: VERIFIED via `grep -c` (= 2).
- `test_layer_with_random_h0_matches_nn_gru` collects and passes 45 fast cases: VERIFIED (45 passed in 2.28s).
- `test_layer_with_random_h0_matches_nn_gru_slow` collects and passes 30 slow cases under `-m slow`: VERIFIED (30 passed, 274 deselected in 9.97s).
- Full `pytest -m "not slow"`: 184 passed (4 micro/roundtrip + 4 fast families × 45): VERIFIED.
- Full `pytest`: 304 passed (4 + 4 families × 75): VERIFIED.
- `ruff check tests/test_layer_parity.py` exit 0: VERIFIED.
- `grep -n "xfail" tests/test_layer_parity.py` returns nothing (D-12): VERIFIED.
- Plan 01-01 helpers (`_translate_cell_to_nn_gru`, `_translate_nn_gru_to_cell`, `_make_dense_fp32_layer`), four micro/roundtrip tests, Plan 01-02 grid constants (FAST_GRID=45, SLOW_GRID=30) and four fwd/h_T grid functions, and Plan 01-03's two backward grid functions are all UNCHANGED — verified by inspection of `git diff HEAD~1 -- tests/test_layer_parity.py` showing only additions after the slow backward test.
- Commit `95d2305` exists on `feat/diagonal-gru`: VERIFIED via `git log --oneline -3`.
- `tests/test_parity.py` < 1e-5 cell-parity contract intact: VERIFIED (12 passed).
- No deletions in the commit: VERIFIED via `git diff --diff-filter=D --name-only HEAD~1 HEAD` returning empty.

---
*Phase: 01-reference-path-parity-vs-nn-gru*
*Completed: 2026-05-13*
