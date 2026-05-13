---
phase: 01-reference-path-parity-vs-nn-gru
plan: 03
subsystem: testing
tags: [pytest, torch.nn.GRU, parity, fp32-identity, parametrize, slow-marker, autograd, backward, gradients]

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 01
    provides: "_translate_cell_to_nn_gru, _make_dense_fp32_layer, set_float32_matmul_precision('highest') module preamble"
  - phase: 01-reference-path-parity-vs-nn-gru
    plan: 02
    provides: "FAST_GRID (45 tuples), SLOW_GRID (30 tuples), pytest import"
provides:
  - "test_layer_backward_matches_nn_gru — REF-03 backward parity over 45 fast cases × 6 gradient tensors (dx, dh_0, dW_ih, dW_hh, db_ih, db_hh)"
  - "test_layer_backward_matches_nn_gru_slow — same family, @pytest.mark.slow, 30 long-T cases × 6 gradient tensors"
affects:
  - "Plan 01-04: h_0 != 0 grid — same constants, same translation helper, but h0 changes from None to randn"
  - "Plan 01-05: audit-report consolidation — backward parity now part of the audit signal across the full 75-combo grid"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-gradient relative-error loop pattern (analog of tests/test_triton_diagonal.py:325-339): list of (name, ref_t, our_t) triples, iterate with `assert rel < 1e-4, f\"{name} rel diff ...\"` so failure messages point at exactly which of the six gradients drifted."
    - "Detach-clone idiom for two-graph autograd parity: `x_ours = x_ref.detach().clone().requires_grad_(True)` so each implementation owns its own autograd graph; reusing the same leaf would build one tape across both and the second `.backward()` would crash."
    - "h0 squeeze-then-clone shape adapter: `h0_ours = h0_ref.detach().squeeze(0).clone().requires_grad_(True)` ensures `.grad` accumulates with [B, H] shape, not [1, B, H]."
    - "Shared upstream gradient `g = torch.randn_like(out_ref)` sent into both `out.backward(g)` calls — fairer than `out.sum().backward()` (every output element contributes independently)."
    - "Cell-side cat-stack translation for grads mirrors the forward helper: `our_W_ih = torch.cat([cell.W_ir.grad, cell.W_iz.grad, cell.W_in.grad], dim=0)`, etc., so gradients land in the same rows as nn.GRU's `weight_ih_l0.grad` ([3H, IN])."

key-files:
  created:
    - ".planning/phases/01-reference-path-parity-vs-nn-gru/01-03-SUMMARY.md"
  modified:
    - "tests/test_layer_parity.py (+156 lines; was 453, now 609)"

key-decisions:
  - "Honored D-09 literally for a third time: backward parity is a SEPARATE parametrized function from forward-output and h_T parity. If forward passes but backward fails, the test id (`test_layer_backward_matches_nn_gru[T-B-H]`) points unambiguously at the autograd graph, not the forward math. The per-gradient `{name}` token in the assertion message gives a second axis of diagnostic precision — failure surfaces which of the six gradient tensors drifted."
  - "Used `randn_like` over `ones_like` / `sum` for the shared upstream gradient `g`: every output element contributes independently to the gradient signal, making the test more discriminating against bugs that only affect specific spatial / temporal positions in the unroll."
  - "Squeezed h0 BEFORE the clone so `h0_ours.grad` has shape `[B, H]` directly (matches GRULayer's API), not `[1, B, H]` followed by a squeeze on the grad side. Avoids an asymmetric squeeze that would have to be undone."
  - "Did NOT call `.zero_grad()` anywhere — both implementations start with `.grad = None` and the first backward populates them. `.zero_grad()` on the layer would lose the cell's grads (cell is a child module; layer.zero_grad() recurses). The whole-suite pattern of `torch.manual_seed(0)` + fresh `_make_dense_fp32_layer(...)` per test ensures grad state is fresh on every parametrize iteration."

patterns-established:
  - "Backward-parity test pattern: build two graphs with detach-clone, share `g = randn_like(out_ref)`, loop over (name, ref, ours) triples, name-the-gradient in the assertion message. Reusable verbatim for any future layer-level autograd-parity test."
  - "All four parametrized families (fwd, fwd_slow, h_T, h_T_slow, bwd, bwd_slow) now use the same scaffold: torch.manual_seed(0) → _make_dense_fp32_layer(IN=max(H,1), H) → _translate_cell_to_nn_gru(layer) → forward both → assert. Plan 01-04 will extend this pattern by swapping the h0 construction line for the random-h0 case."

requirements-completed: [REF-03]

# Metrics
duration: ~4 min
completed: 2026-05-13
---

# Phase 1 Plan 3: Backward gradient parity grid Summary

**Backward / gradient parity (REF-03) lands as two parametrized test functions over the 75-combo T × B × H grid (45 fast + 30 slow), comparing all six gradient tensors (`dx`, `dh_0`, `dW_ih`, `dW_hh`, `db_ih`, `db_hh`) between `GRULayer.backward()` and `torch.nn.GRU.backward()` at < 1e-4 relative tolerance. All 450 gradient-parity assertions (75 cases × 6 gradients) pass on CPU under `set_float32_matmul_precision('highest')`.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-13T17:16:25Z
- **Completed:** 2026-05-13T17:20:40Z
- **Tasks:** 1 (`type="auto"`)
- **Files modified:** 1 (`tests/test_layer_parity.py`)
- **Files created:** 1 (this SUMMARY)

## Accomplishments

- **`test_layer_backward_matches_nn_gru` (FAST_GRID, 45 cases)** — REF-03 satisfied on the fast grid. Per case, the test:
  1. Seeds `torch.manual_seed(0)` and builds the dense fp32-Identity layer + matching nn.GRU.
  2. Constructs two `requires_grad=True` leaf pairs via the detach-clone idiom (`x_ref` / `x_ours`; `h0_ref` / `h0_ours`), with h0 squeezed BEFORE the clone so `h0_ours.grad` has shape `[B, H]` directly.
  3. Runs forward on both implementations.
  4. Sends the SAME shared random `g = randn_like(out_ref)` into both `out.backward(g)` calls — fair gradient comparison.
  5. Builds cell-side cat tensors for the four weight/bias-grad pairs (gate order `(r, z, n)` mirrors the forward translation helper).
  6. Loops over six `(name, ref_t, our_t)` triples — `dx`, `dh_0`, `dW_ih`, `dW_hh`, `db_ih`, `db_hh` — asserting `rel < 1e-4` with the `{name}` token in every failure message.
- **`test_layer_backward_matches_nn_gru_slow` (SLOW_GRID, 30 cases)** — REF-03 satisfied on the slow grid (T ∈ {512, 1024}). Identical body, decorated with `@pytest.mark.slow` above `@pytest.mark.parametrize` per the file's locked decorator order convention. Backward through 1024 timesteps is the longest single autograd graph in the audit; passing here confirms no accumulated numerical drift in the backward pass at long-T even under `set_float32_matmul_precision('highest')`.
- **D-09 honored a third time:** backward parity is a SEPARATE parametrized function from forward-output and h_T parity. The grid test id pinpoints the test family, and the `{name}` token in the assertion message pinpoints the gradient — two axes of diagnostic precision.
- **D-12 honored:** zero `@pytest.mark.xfail` markers in the file.
- **No `src/` modifications.** All 75 cases × 6 gradients = 450 gradient-parity assertions pass on the first run on CPU. The cell's manual unroll's autograd graph (and the time loop's `outputs.append(h)` / `torch.stack` plumbing) is correct end-to-end against `torch.nn.GRU`'s autograd at < 1e-4 across the full grid.

## Task Commits

Sequential executor on `feat/diagonal-gru` (normal hooks; no `--no-verify`):

1. **Task 1: Backward parity (fast + slow)** — `8cd96ad` (test)

## Files Created/Modified

- `tests/test_layer_parity.py` (modified, +156 lines net; 453 → 609) — added one ASCII-rule stanza divider, an extended comment block on backward-bug philosophy / shape detail / detach-clone idiom, and two parametrized test functions. No changes to imports, no new helpers, no touching the Plan 01-01 helpers or Plan 01-02 grid constants.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-03-SUMMARY.md` (created) — this file.

## Decisions Made

- **Shared `g = randn_like(out_ref)` over `out.sum().backward()`:** Explicit in the plan's `<action>` body. The random `g` is more discriminating: every output element contributes independently, so a bug that only affects specific time/batch positions surfaces, whereas `sum().backward()` would average it out.
- **No `.zero_grad()` between forward and backward:** Both implementations start with `.grad = None` on a fresh `_make_dense_fp32_layer(...)` per parametrize iteration. `.zero_grad()` on the layer would recurse into the child cell and nuke the cell's gradients — exactly the data this test is comparing against. The fresh-layer-per-iteration pattern (already established by Plans 01-01 and 01-02) makes grad state isolation automatic.
- **Squeeze h0 BEFORE the clone, not after:** `h0_ours = h0_ref.detach().squeeze(0).clone().requires_grad_(True)` — squeezing produces a `[B, H]` view, the clone gives that view its own storage, and `.grad` then accumulates with `[B, H]` shape. The plan's `<action>` body specifies this order explicitly; doing it the other way would produce a `[1, B, H]`-shaped `.grad` that needs a squeeze on every comparison, asymmetric with our cell's API.
- **Cat order for grads matches forward translation:** `[W_ir.grad, W_iz.grad, W_in.grad]` along axis 0 mirrors `_translate_cell_to_nn_gru`'s forward cat `[W_ir, W_iz, W_in]`. Both sides see `(r, z, n)` order on the `weight_ih_l0` / `weight_hh_l0` tensors, so the gradients land in the same rows. If the forward helper's gate order ever changes, the backward helper must change in lockstep (caught by Plan 01-01's gate-order micro-tests).

## Deviations from Plan

None — plan executed exactly as written. (The decorator order, detach-clone idiom, six-grad triple list, `{name}` token in assertion messages, and 1e-6 denominator floor all match the plan's `<action>` body and PATTERNS.md lines 220-264 verbatim.)

## Issues Encountered

- **`python` on PATH is system Python**, same as Plans 01-01 and 01-02. Worked around by invoking `.venv/bin/python -m pytest` / `-m ruff` directly. Not a code issue; documented in Plan 01-01's SUMMARY.

## Beads Issues Filed

None — all 450 gradient-parity assertions (75 cases × 6 gradients) passed on the first run. No backward-graph bug surfaced in the reference path, so the D-10/D-11 two-commit failing-test-before-fix protocol did not need to fire.

If any case had failed (e.g. `dW_hh` drift at T=512), the protocol would have been:

1. **Commit A:** The failing parametrized test (already on disk; isolate via `pytest --tb=short -x` until first failure, capture the offending T/B/H and the gradient `{name}`).
2. **`bd create --title "test_layer_backward_matches_nn_gru[T-B-H] dW_hh drift"`** with the `pytest --tb=short` tail in `--notes`. The bd title encodes both the test id and the gradient name — actionable in one line.
3. **Commit B:** Fix in `src/gru_qat/gru_cell.py` or `gru_layer.py` (most likely the n-gate's asymmetric `r * gh_n` derivative path, where PyTorch's autograd graph for the in-place-style update could have subtly diverged from the explicit unroll's). Same test now passes.
4. **`bd close <id>`** after CI green.

## Verification Snapshot

```
$ .venv/bin/python -m pytest tests/test_layer_parity.py::test_layer_backward_matches_nn_gru -q
.............................................                            [100%]
45 passed in 4.07s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -m slow -k backward -q
..............................                                           [100%]
30 passed, 199 deselected in 44.21s

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q -m "not slow"
139 passed, 90 deselected in 5.76s
# 4 micro/roundtrip + 45 fwd + 45 h_T + 45 bwd = 139

$ .venv/bin/python -m pytest tests/test_layer_parity.py -q
229 passed in 57.65s
# Full file: 4 micro/roundtrip + 75 fwd + 75 h_T + 75 bwd = 229

$ .venv/bin/python -m pytest tests/test_parity.py -q   # <1e-5 cell contract
12 passed in 1.31s

$ .venv/bin/python -m ruff check tests/test_layer_parity.py
All checks passed!

$ grep -c "def test_layer_backward" tests/test_layer_parity.py
2

$ grep -n "xfail" tests/test_layer_parity.py
(no output — D-12)

$ wc -l tests/test_layer_parity.py
609 tests/test_layer_parity.py

$ git log --oneline -3
8cd96ad test(01-03): backward parity grid (45 fast + 30 slow) vs nn.GRU
3bdddba docs(01-02): complete forward + h_T parity grid plan
218405d test(01-02): h_T parity grid (45 fast + 30 slow) vs nn.GRU
```

## Drift Analysis

All 75 backward-grid cases pass; no drift surfaced. If drift had appeared, the plan's `<output>` block asked for a note on whether it was uniform or scale-dependent (large-H, large-T):

- **Uniform across the grid** → indicates a structural math bug in the autograd graph (e.g. the n-gate's `r * gh_n` derivative wired wrong, or a sign error in `(1 - z) * n + z * h`'s gradient). Surfaces equally at T=1 and T=1024, at H=1 and H=512. **Action:** Fix in `src/gru_qat/gru_cell.py` (per the D-10 protocol); regression test stays.
- **Scale-dependent (large-H, large-T only)** → indicates accumulated numerical drift, not a math bug. The reference path is correct, but the comparison against `nn.GRU`'s cuDNN-fused backward (which uses different intermediate accumulation order) drifts past 1e-4 at the extreme corner of the grid. **Action:** Either tighten the per-shape tolerance per Phase 6 edge-sweep precedent (`tests/test_butterfly_dispatch.py:206-214` per-batch error inspection), or carve out the offending corner as a known-accepted divergence in the AUDIT-REPORT. Does NOT reopen Phase 1.

Neither happened. Reference path's autograd is structurally aligned with `nn.GRU`'s and survives the < 1e-4 bar at T=1024 / H=512 / B=32.

## Next Phase Readiness

- **Plan 01-04 (h_0 ≠ 0)** can swap the h0 construction (`torch.zeros(1, B, H)` → `torch.randn(1, B, H)`) and reuse the same translation helper, the same FAST_GRID/SLOW_GRID, the same per-test scaffold. Per D-09 this is one function comparing both `out` and `h_T` simultaneously (the randomness is the isolation, not the family-split).
- **No `src/` modifications** shipped; no blockers; no bd issues open. The cell / layer code remains exactly as `feat/diagonal-gru` left it before this plan started.
- **< 1e-5 cell parity contract in `tests/test_parity.py` is untouched** (12 passed; `git diff HEAD~1 -- tests/test_parity.py` is empty for commit `8cd96ad`).
- **Recent backward-fix cluster** (`d8218d4` butterfly OOB, `c001a8a` dWh/dbh accumulator slabs, `4e10402` diagonal variant) operates at the Triton-kernel layer, not the reference path. This plan exercises the reference path's autograd — which is `torch.autograd` walking the manual unroll, not a Triton bwd kernel. Those fixes' regression tests live in their respective `tests/test_triton_*.py` files and are unaffected.

## Self-Check: PASSED

- `tests/test_layer_parity.py` exists at 609 lines (was 453 from Plan 01-02): FOUND.
- Two `def test_layer_backward` functions exist: VERIFIED via `grep -c` (= 2).
- `test_layer_backward_matches_nn_gru` collects and passes 45 fast cases: VERIFIED (45 passed in 4.07s).
- `test_layer_backward_matches_nn_gru_slow` collects and passes 30 slow cases under `-m slow`: VERIFIED (30 passed, 199 deselected in 44.21s).
- Full `pytest -m "not slow"`: 139 passed (4 micro/roundtrip + 45 fwd + 45 h_T + 45 bwd): VERIFIED.
- Full `pytest`: 229 passed (4 + 75 + 75 + 75): VERIFIED.
- `ruff check tests/test_layer_parity.py` exit 0: VERIFIED.
- `grep -n "xfail" tests/test_layer_parity.py` returns nothing (D-12): VERIFIED.
- Plan 01-01 helpers (`_translate_cell_to_nn_gru`, `_translate_nn_gru_to_cell`, `_make_dense_fp32_layer`) and four micro/roundtrip tests, and Plan 01-02 grid constants (FAST_GRID=45, SLOW_GRID=30) and four fwd/h_T grid functions UNCHANGED — verified by `git diff HEAD~1 -- tests/test_layer_parity.py | grep '^-'` showing only the trailing-whitespace boundary line (no deletions of prior code).
- Commit `8cd96ad` exists on `feat/diagonal-gru`: VERIFIED via `git log --oneline -3`.
- `tests/test_parity.py` < 1e-5 cell-parity contract intact: VERIFIED (12 passed; `git diff HEAD~1 -- tests/test_parity.py` returns empty).

---
*Phase: 01-reference-path-parity-vs-nn-gru*
*Completed: 2026-05-13*
