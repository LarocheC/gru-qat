---
phase: 04-quant-on-bit-identity
plan: 04
subsystem: testing
tags: [butterfly, triton, quant-on, int8, frozen, fake-quant, qat, parity]

# Dependency graph
requires:
  - phase: 04-quant-on-bit-identity
    provides: "D-42 ASYMMETRIC disposition (fwd torch.equal / bwd < h_scale), QNT-04 _update_observer fix, _make_dense_layer_quant_int8 idiom blueprint"
provides:
  - "Butterfly Triton kernel quant-on parity suite (fwd + bwd) under the actual D-41 frozen INT8 recipe"
  - "_make_butterfly_layer_quant_int8 wrapper around existing _make_layer (with config.bits + qmin/qmax retune to land the actual D-41 recipe — _make_layer's default bits=32 is NOT INT8)"
  - "_assert_quant_parity helper byte-identical to D-43 idiom (uniform across Plans 04-02..04)"
  - "Dual-layer state-sharing assertion (quant_h_in.scale, quant_h_out.scale, quant_W_ir.scale, quant_struct_Wh_r.scale) — catches load_state_dict buffer-propagation regressions BEFORE parity assertion (mitigation T-04-16)"
  - "Adversarial input coverage on butterfly: 18 fast × 3 cls × 2 directions = 108 fast cases + 54 slow cases"
affects:
  - "Plan 04-05 (audit kickoff + CUDA GPU run + phase-exit SUMMARY)"
  - "Phase 5 (calibration lifecycle) — frozen-INT8 + structured-hidden layer build pattern reusable for calibrate->freeze round-trip tests"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wrap-existing-helper + post-construction quantizer retune (config.bits + config.mode + qmin/qmax recompute) — lets butterfly Phase 4 helper reuse the Phase 2 _make_layer rather than duplicating the StructureConfig(kind=butterfly) layer construction"
    - "Dual-layer comparator via load_state_dict for kernels without a pure-PyTorch reference (butterfly): both layers built by the same helper, only use_triton flag differs"
    - "Early state-sharing assertion separates buffer-propagation regressions from kernel-divergence findings"
    - "Per-grad assertions over named_parameters() inside the bwd test body — gives butterfly twiddles (struct_Wh_*) named coverage without hard-coding parameter names"

key-files:
  created:
    - ".planning/phases/04-quant-on-bit-identity/04-04-SUMMARY.md"
  modified:
    - "tests/test_triton_butterfly_strict.py — appended ~466 lines Phase 4 section"

key-decisions:
  - "Wrap _make_layer rather than duplicate (per plan directive + threat T-04-15 mitigation). _make_layer produces bits=32 weight + bits=32 input_act + bits=hidden_bits hidden — NOT INT8. The wrapper retunes config.bits to 8 on weight/input_act AND recomputes the qmin/qmax instance attrs (cached at FakeQuantize.__init__, not auto-recomputed when config.bits changes). Documented in helper docstring."
  - "Also retune quant_struct_Wh_r/_z/_n (the actually-used hidden-side output quantizers on butterfly's structured-hidden path) in addition to the unused quant_W_hr/_hz/_hn placeholders. Without this, the actual D-41 recipe would not land on the structured hidden side — the unused quant_W_hr quantizers are present but never observe data."
  - "State-sharing assertion targets quant_struct_Wh_r.scale on the hidden side (not quant_W_hr.scale, which is a never-observed placeholder in structure_hidden mode whose frozen scale would be uninformative)."
  - "Test uses GRULayer.forward() rather than gru_scan_butterfly_forward_triton directly. layer.forward() runs quant_x internally per D-41, so the input-quantization-before-linear order is enforced by the layer — no extra quant_x call in the test body."

patterns-established:
  - "wrapper-with-retune helper: when an existing helper produces close-but-not-exact recipe shape, wrap it then mutate config + recompute cached qmin/qmax — avoids the duplication risk T-04-15 flags"
  - "state-sharing precheck before parity assertion: catches load_state_dict regressions explicitly rather than letting them surface as confusing parity blowups"
  - "per-grad parity via named_parameters() loop with _assert_quant_parity per param: scales to butterfly's variable twiddle parameter count without hard-coding names"

requirements-completed:
  - "QNT-02 (butterfly fwd contribution; full requirement closed alongside Plans 04-02 dense + 04-03 diagonal/monarch)"
  - "QNT-03 (butterfly bwd contribution)"

# Metrics
duration: 8min
completed: 2026-05-14
---

# Phase 4 Plan 04-04: Butterfly Kernel Quant-on Parity Suite Summary

**Butterfly Triton kernel quant-on parity audit suite at H ∈ {32, 128, 512}: dual-layer comparator (use_triton=True vs False via load_state_dict), early state-sharing assertion on hidden + per-channel + structured-output quantizer scales, _assert_quant_parity per D-43 (Result A fwd / Result B bwd), three adversarial classes (realistic / near-saturation / large-magnitude) × fast + slow grids.**

## Performance

- **Duration:** ~8 min (470s)
- **Started:** 2026-05-14T11:16:41Z
- **Completed:** 2026-05-14T11:24:36Z
- **Tasks:** 1 (single-file extension)
- **Files modified:** 1 (`tests/test_triton_butterfly_strict.py`)

## Accomplishments
- Phase 4 section appended to `tests/test_triton_butterfly_strict.py` (466 new lines).
- `_make_butterfly_layer_quant_int8(H, *, use_triton, h_scale=0.02)` wraps the existing `_make_layer(H, use_triton, hidden_bits=8)`; retunes weight + input_act quantizers (including `quant_struct_Wh_r/_z/_n` on the structured hidden side) to `bits=8, axis=0, mode='min_max'` with `qmin/qmax` recomputed; runs ONE inline calibration forward; calls `cell.freeze_quantizers()`; hidden quantizers manually frozen at `h_scale`.
- `_assert_quant_parity` byte-identical to 04-DISPOSITION.md (D-43 uniformity).
- Dual-layer state-sharing assertion (`_assert_state_sharing`) on four buffers (`quant_h_in.scale`, `quant_h_out.scale`, `quant_W_ir.scale` per-channel input weight, `quant_struct_Wh_r.scale` structured hidden output) — runs early in every test body before parity check, surfacing buffer-propagation regressions (T-04-16) clearly.
- Per-grad assertions: `dx`, `dh0`, AND every `named_parameters()` entry (butterfly twiddles via `struct_Wh_*` modules, hidden biases `b_hr/_hz/_hn`) named in failure messages.
- Adversarial input coverage: 3 classes (`realistic` / `near-saturation` / `large-magnitude`) × 18-case `QUANT_FAST_GRID` × 2 directions = 108 fast cases; `_slow` siblings over `QUANT_SLOW_GRID` (`T=512`) = 54 slow cases.

## Task Commits

1. **Task 1: Butterfly kernel — Phase 4 section extension** — `02881eb` (test)

## Files Created/Modified
- `tests/test_triton_butterfly_strict.py` — Phase 4 section appended (`_assert_quant_parity`, `_make_butterfly_layer_quant_int8`, `_adversarial_inputs`, `QUANT_FAST_GRID`, `QUANT_SLOW_GRID`, `_assert_state_sharing`, `_run_butterfly_quant_fwd_case`, `_run_butterfly_quant_bwd_case`, four parametrized test functions). Phase 2 fp32 sections + `_assert_grad_close` helper unchanged (D-52).
- `.planning/phases/04-quant-on-bit-identity/04-04-SUMMARY.md` — this file.

## Decisions Made
- **Wrap `_make_layer` AND retune `config.bits` + `qmin/qmax`:** the existing `_make_layer` produces `bits=32` weight + `bits=32` input_act (it was authored for Phase 2 fp32-Identity tests). Phase 4's D-41 recipe requires `bits=8`. Mutating `config.bits` post-construction is not enough — `qmin/qmax` are cached on the `FakeQuantize` instance at `__init__` (per `src/gru_qat/quantizers.py:74`) and do not auto-recompute. The helper therefore mutates both `config.bits` AND re-derives `qmin/qmax = q._qrange(bits, symmetric)`. Recorded explicitly in the helper docstring so future maintainers don't miss the cached-attr foot-gun.
- **Retune `quant_struct_Wh_r/_z/_n` in addition to `quant_W_hr/_hz/_hn`:** for butterfly (`structure_hidden=butterfly`), the hidden-side weight quantizers actually used are the `quant_struct_Wh_*` ones (output of the structured per-step layer, per `src/gru_qat/gru_cell.py:334-336`). The `quant_W_hr/_hz/_hn` quantizers exist but are unused placeholders (per `gru_cell.py:196-204`). Plan's "skip any that are None" directive does not cover this case (the placeholders are not None, just unused). To land the actual D-41 recipe on every used quantizer, the wrapper retunes BOTH sets.
- **State-sharing assertion target on hidden side is `quant_struct_Wh_r.scale`, not `quant_W_hr.scale`:** the placeholder `quant_W_hr` never observes data, so its frozen scale would remain at the initial value (1.0) regardless of state sharing — uninformative. Asserting on `quant_struct_Wh_r.scale` (which DOES observe data via the inline calibration forward) exercises the buffer-propagation path on a non-trivial value.
- **No xfails (D-50):** the test bodies assert strict / tight-INT8 bounds as the D-42 disposition specifies. If CUDA execution surfaces butterfly-specific divergences outside those bounds, Plan 04-05 audit dispositions via D-37 (file bd issue, decide disposition revision or kernel fix).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Retune `config.bits` and recompute `qmin/qmax` on wrapped quantizers**
- **Found during:** Task 1 (writing the `_make_butterfly_layer_quant_int8` helper)
- **Issue:** The plan body directs the helper to "override quantizer modes to produce the ACTUAL D-41 recipe (weight/input_act → `mode='min_max'`; hidden → `mode='frozen'` with manually-set `h_scale`)". The existing `_make_layer(H, use_triton, hidden_bits=8)` produces a recipe with `bits=32` for weight + `bits=32` for input_act + `bits=hidden_bits=8` for hidden, all in default `mode='dynamic'`. Mode override alone leaves `bits=32` on weight/input_act — that is NOT the D-41 recipe (`bits=8` everywhere). Without also retuning `config.bits` to 8 AND recomputing the cached `qmin/qmax` (per `src/gru_qat/quantizers.py:74`), the helper would land a `bits=32 frozen` recipe, not the `bits=8 frozen` D-41 recipe.
- **Fix:** Added a `_retune_weight(q)` closure that sets `config.bits=8`, `config.axis=0`, `config.symmetric=True`, `config.mode='min_max'`, AND recomputes `q.qmin, q.qmax = q._qrange(...)`. Same retune applied to `quant_x` (per-tensor). Hidden quantizers similarly get `config.bits=8` + recompute + manual `scale = h_scale`.
- **Files modified:** `tests/test_triton_butterfly_strict.py` (inside helper body — single commit with Task 1).
- **Verification:** helper-docstring `Recipe` block enumerates bits/axis/symmetric/mode for each quantizer; `_assert_state_sharing` later confirms per-channel `quant_W_ir.scale` and `quant_struct_Wh_r.scale` both propagate via `load_state_dict` (would only be informative if bits were correctly retuned).
- **Committed in:** `02881eb` (Task 1 commit)

**2. [Rule 2 - Missing Critical Functionality] Retune `quant_struct_Wh_r/_z/_n` (structured-hidden output quantizers) in addition to `quant_W_hr/_hz/_hn`**
- **Found during:** Task 1
- **Issue:** Plan body directs override on `quant_W_ir/_iz/_in/_hr/_hz/_hn` only. For butterfly, `structure_hidden=StructureConfig(kind='butterfly')` causes `quant_W_hr/_hz/_hn` to be unused placeholders (per `src/gru_qat/gru_cell.py:196-204` — quantizers created but never observe data) — the actually-used hidden-side weight quantizers are `quant_struct_Wh_r/_z/_n` (output of the per-step structured layer, per `gru_cell.py:334-336`). Without retuning the `quant_struct_Wh_*` ones, the helper would only land D-41 partially: dense input-side weights + input_act + hidden activation in INT8, but the butterfly structured hidden output quantizers would remain at `bits=32, mode='dynamic'`.
- **Fix:** Added a second loop over `quant_struct_Wh_r/_z/_n` calling the same `_retune_weight(q)` closure. `cell.freeze_quantizers()` then freezes them along with everything else (it walks `self.modules()` per `gru_cell.py:497-505`).
- **Files modified:** `tests/test_triton_butterfly_strict.py` (helper body).
- **Verification:** state-sharing assertion targets `quant_struct_Wh_r.scale` — if it weren't retuned to bits=8 + frozen via the calibration pass, its scale would not be a meaningful frozen value to assert on.
- **Committed in:** `02881eb`

**3. [Rule 1 - Bug recovery: accidental cross-agent file inclusion in commit]**
- **Found during:** post-commit verification (`git show --stat HEAD`)
- **Issue:** The first commit attempt (`bfced20`) accidentally included `tests/test_triton_scan_strict.py` (which contains parallel agent 04-02's pending uncommitted Phase 4 work). Root cause unclear — `git add tests/test_triton_butterfly_strict.py` was the only stage call before the commit, and `git status --short` between staging and commit showed only the butterfly file in the index. The `.beads/hooks/pre-commit` hook is wired in but the script body (a thin wrapper around `bd hooks run pre-commit`) should not auto-stage. Hypothesis (unconfirmed): a bd hook side-effect or a stale staging-area condition not visible via `git status --short`.
- **Fix:** `git reset --soft HEAD~1` (preserve working tree; both files now staged), `git restore --staged tests/test_triton_scan_strict.py` (unstage scan_strict), then `git commit --only tests/test_triton_butterfly_strict.py -m "..."` (explicit single-path commit). Recovery commit is `02881eb`. The accidental commit `bfced20` was never pushed to remote, so the reset is local-only and does not impact other agents or branches.
- **Files modified:** none beyond the original task scope — the surgery operated on git state, not file contents. `tests/test_triton_scan_strict.py` returned to its pre-commit working-tree state (still pending for agent 04-02 to commit).
- **Verification:** `git show --stat HEAD` now shows only `tests/test_triton_butterfly_strict.py | 466 insertions(+)`; `git status --short` shows the parallel-agent's scan_strict + STATE.md / config.json / `04-02-SUMMARY.md` as unstaged/untracked (their commits, not mine).
- **Committed in:** `02881eb` (the recovery commit IS the task commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 2 — recipe-correctness; 1 Rule 1 — git-state recovery for parallel-agent isolation)
**Impact on plan:** Deviations 1+2 are recipe-correctness gaps in the plan body's directive vs the actual codebase shape — both are essential for the helper to produce the D-41 recipe the rest of the plan assumes. Deviation 3 is recovery from a self-inflicted parallel-isolation breach (root cause unclear); the recovery surgery preserves all parallel-agent work and lands a clean butterfly-only commit. No scope creep.

## Issues Encountered

- **Pre-existing PATTERNS doc inaccuracy:** `.planning/phases/04-quant-on-bit-identity/04-PATTERNS.md:446-447` claims `_make_layer` produces `bits=8 dense weight + bits=8 input_act + bits=hidden_bits hidden` — actually `bits=32` for weight + input_act. The doc was authored assuming a different `_make_layer` body than what exists in `tests/test_triton_butterfly_strict.py:86-100`. Plan body's wrap-then-override-mode-only directive was derived from the same incorrect assumption. Fix-forward via Deviation 1 (retune `config.bits` + recompute `qmin/qmax`).
- **CUDA-side test failures are expected audit signal:** when run on CUDA, `test_butterfly_quant_fwd` / `_bwd` surface divergence beyond D-42 disposition bounds (`torch.equal` fails for butterfly fwd at e.g. `cls=realistic, T=8, B=1, H=32` with `max_abs_diff = 8e-02 = 4 × h_scale`). This is consistent with the plan's `<bd_workflow>` directive: "If butterfly quant-on surfaces a finding outside the asymmetric disposition bounds, D-37 protocol." The disposition was empirically derived from a dense probe only; butterfly's log_H per-stage `tl.dot`-vs-IEEE matmul drift compounds beyond the dense one-INT8-step bound. Plan 04-05 (audit kickoff + CUDA GPU run) is the proper venue for D-37 file-and-disposition; Plan 04-04's deliverable is the audit instrument, not the audit verdict.
- **Bonus finding from D-42 already captured:** the disposition file notes pre-existing Phase 2 strict-tier butterfly fwd failures at `[8-1-32]` (max diff ~9.3e-3 vs <5e-4 bound) — orthogonal to Phase 4 scope, tracked in a bd issue per the disposition doc.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 04-04 deliverable is in place (Task 1 committed as `02881eb`).
- Plan 04-05 (audit kickoff + CUDA GPU run + phase-exit SUMMARY) is the next plan in Wave 3. It runs the full quant-on suite (Plans 04-02 dense + 04-03 diagonal + monarch + 04-04 butterfly) on CUDA and dispositions any per-kernel findings via D-37. Expected findings on butterfly (per the CPU smoke run above): per-grad divergence beyond `h_scale` bound on some shape/cls combinations; possibly fwd divergence beyond `torch.equal` for higher `H`. Plan 04-05 either:
  1. Files a bd issue + revises D-42 to a per-kernel disposition (e.g., butterfly fwd → `< h_scale` instead of `torch.equal`), OR
  2. Pursues kernel `input_precision='ieee'` change (out of test-only scope; deferred to Phase 7 per CONTEXT D-42).
- No blockers — STATE.md / ROADMAP.md updates are deferred to the orchestrator (per parallel-executor protocol).

## Self-Check: PASSED

Verified before declaring complete:
- ✓ `tests/test_triton_butterfly_strict.py` modified (466 insertions in commit `02881eb`).
- ✓ Commit `02881eb` exists: `git log --oneline -1` returns the test(04-04) commit.
- ✓ `.planning/phases/04-quant-on-bit-identity/04-04-SUMMARY.md` written (this file).
- ✓ `grep -c "strict=True" tests/test_triton_butterfly_strict.py` = 4 (≥ 2 required).
- ✓ `grep -c "strict=False" tests/test_triton_butterfly_strict.py` = 6 (≥ 4 required).
- ✓ `grep -c xfail tests/test_triton_butterfly_strict.py` = 0 (D-50 compliance).
- ✓ `grep -c "freeze_quantizers" tests/test_triton_butterfly_strict.py` = 2 (≥ 1 required).
- ✓ Wrapper calls `_make_layer(H, use_triton=use_triton, hidden_bits=8)` (≥ 1 occurrence; threat T-04-15 mitigated).
- ✓ `git diff HEAD~1 HEAD -- tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py tests/test_butterfly_dispatch.py` empty (D-51 locked files + D-22 OOB regression untouched).
- ✓ Phase 2 fp32 sections + `_assert_grad_close` helper byte-identical (only Phase 4 section appended).
- ✓ `pytest.importorskip("torch_structured")` preserved at module top.
- ✓ `ruff check tests/test_triton_butterfly_strict.py` — all checks passed.
- ✓ `pytest tests/test_triton_butterfly_strict.py --collect-only -q | grep test_butterfly_quant_ | wc -l` = 162 (= 27 shape cases × 3 classes × 2 directions, matches grid arithmetic).
- ✓ Commit modifies ONLY `tests/test_triton_butterfly_strict.py` (single-file, 466 insertions — parallel-agent isolation honored after recovery in Deviation 3).

---
*Phase: 04-quant-on-bit-identity*
*Completed: 2026-05-14*
