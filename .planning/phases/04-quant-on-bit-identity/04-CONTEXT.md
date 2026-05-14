# Phase 4: Quant-on bit-identity - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Pin every Triton variant (dense, diagonal, monarch, butterfly) to the reference path under an **active frozen INT8 recipe** (per-channel weight + per-tensor activation), at the strictest tier the empirical probe permits (`torch.equal` if achievable; tight-INT8-grid-bounded otherwise per the disposition gate below). Plus fix QNT-04 — the per-channel `min_max` observer broken-axis-reduction bug from Phase 1's known-gap note.

In scope:
- Frozen INT8 recipe applied to each Triton kernel + matching reference path.
- Bit-identity (or tight-INT8-grid) parity for fwd `(out, h_T)` AND bwd gradients across kernels.
- Three adversarial input classes per kernel: realistic, near-saturation, large-magnitude.
- QNT-04 fix in `src/gru_qat/quantizers.py:_update_observer` with regression test via D-37 two-commit protocol.
- Empirical probe at Plan 04-01 to determine the actual bit-identity bound before authoring the rest of the suite.

Explicitly NOT in scope for Phase 4:
- Structured PyTorch fallbacks (circulant, LDR) quant-on — they have no Triton path; per-step PyTorch quant-on tests are covered by existing `test_structure.py` int8-QAT smoke layer.
- Calibration lifecycle — Phase 5 owns `calibrate` → `freeze_all` → Triton round-trip.
- Edge cases (T=0, B=0, H ∈ {1, 2}) for quant-on — Phase 6.
- LSQ / PACT learnable activation scales (ACT-02 v2 deferred item).
- Bias quantization, LUT sigmoid/tanh — Phase 6+ scope per SCOPE.md.

</domain>

<decisions>
## Implementation Decisions

### Bit-identity strategy (D-41..43)
- **D-41:** Plan 04-01 is an **empirical probe**. Build a small fixed-shape test (T=8, B=4, H=64 dense, INT8 per-channel weight + per-tensor activation, frozen recipe, `set_float32_matmul_precision('highest')`). Run reference and dense Triton; check `torch.equal(out_ref, out_triton)` AND per-gradient `torch.equal` on `(dx, dh_0, dWh, dbh)`. Capture the result.
- **D-42:** Plan 04-01 ends with a `checkpoint:human-verify`. The probe result determines the disposition:
  - **Result A: bit-identity holds.** Plans 04-02..04 use `torch.equal` as the assertion. Triton paths under quant-on become genuinely bit-identical to reference. The Phase 2 disposition (TF32 OK for fp32 Identity) does NOT carry into quant-on because the post-quant rounding dominates.
  - **Result B: bit-identity fails.** Surface findings to orchestrator/human. Options on the table at the checkpoint: (a) tight-INT8-grid bound `abs_diff < scale_h * 1` (one INT8 step); (b) `tl.dot(input_precision="ieee")` kernel change per Phase 2 Option B (out-of-scope-y, requires src/ kernel modifications); (c) defer disposition to Phase 7 audit report. Default recommendation at the checkpoint: **(a) tight-INT8-grid**.
- **D-43:** Whatever disposition is chosen at the Plan 04-01 checkpoint becomes the **uniform contract** for Plans 04-02..04 (dense, diagonal, monarch, butterfly quant-on). Don't mix per-kernel — pick one bound and apply it.

### QNT-04 per-channel min_max observer (D-44..45)
- **D-44:** **Fix.** Rewrite `FakeQuantize._update_observer` in `src/gru_qat/quantizers.py:135-146` to do per-axis reduction when `axis is not None`. Use `torch.amin(x, dim=other_dims)` / `torch.amax(x, dim=other_dims)` where `other_dims` is the tuple of dims to reduce over (all except `self.axis`). Running stats become per-channel tensors of shape `[channels]` instead of scalars.
- **D-45:** **Failing-test-before-fix per D-37.** Plan 04-X (likely 04-05 or folded into 04-01's plan) writes Commit A: a new test in `tests/test_quantizers.py` that exercises the per-channel min_max path on a tensor with distinct per-channel ranges (e.g., channel 0 in [-1, 1], channel 1 in [-10, 10]); asserts running_min / running_max are PER-CHANNEL tensors with the expected values. The current code produces scalar running_min / running_max — the test fails. Commit B: the fix in `quantizers.py`. The test passes. `bd create` per finding (one issue for QNT-04 / ACT-01 closure). bd issue closes after Commit B and CI green. Existing `tests/test_quantizers.py` is NOT a locked file — extending it is fine.

### Adversarial input coverage (D-46)
- **D-46:** Three classes per kernel per direction (fwd, bwd):
  - **Realistic:** `torch.randn(T, B, IN) * 0.5` (scaled to fit INT8 dynamic range).
  - **Near-saturation:** values right at the INT8 boundary. Construct as `torch.linspace(-0.99, 0.99, T*B*IN).reshape(T, B, IN) * scale_x_max` where `scale_x_max` is the activation quantizer's max representable value before clipping.
  - **Large-magnitude:** `torch.randn(T, B, IN) * 5` (forces clipping; tests that both reference and Triton clip identically).
- Each adversarial class is parametrized over a smaller-than-Phase-2 grid (see D-49). Per-test failure messages include the class name ("realistic" | "near-saturation" | "large-magnitude") for easy triage.

### Test file location (D-47..48)
- **D-47:** **Extend** Phase 2's `tests/test_triton_<kind>_strict.py` files with a new `## Quant-on (Phase 4)` section per file. Reuses Phase 2's FAST/SLOW grid constants and `_make_<kind>_layer` helpers — no duplication.
- **D-48:** New section header per file:
  ```python
  # ----------------------------------------------------------------------
  # Phase 4: Quant-on bit-identity (frozen INT8 per-channel weight +
  #                                  per-tensor activation)
  # Tolerance: per D-42 disposition (resolved at Plan 04-01 checkpoint)
  # ----------------------------------------------------------------------
  ```
- The `_make_<kind>_layer_quant_int8` helper (one per file) builds a frozen-INT8 layer using the canonical recipe. Underscore-prefixed per existing convention.

### Shape grid (D-49)
- **D-49:** Smaller grid than Phase 2 (bit-identity is binary):
  - `T ∈ {8, 64}` × `B ∈ {1, 4, 32}` × `H ∈ {32, 128, 512}` = 18 fast cases per kernel.
  - `T ∈ {512}` slow tier (optional; mark `@pytest.mark.slow`) = 9 slow cases per kernel.
  - For monarch: additional `nblocks ∈ {2, 4, 8}` axis.
  - For butterfly: H restricted to powers of 2 ({32, 128, 512} already meets).
  - For diagonal: include H ∈ {1, 2, 8} from Phase 2 — NO, omit. Phase 4 isn't an edge-case sweep (Phase 6).
  - 3 adversarial classes × 18 fast cases = 54 fast cases per kernel × 4 kernels = ~216 fast tests. Plus 27 slow per kernel × 4 = 108. Total ~324 quant-on tests.

### Discipline (carried forward)
- **D-50:** D-10..12 / D-27 / D-37 two-commit failing-test-before-fix discipline. No `@pytest.mark.xfail`. `bd create` per finding.
- **D-51:** D-28 / D-38 locks: `tests/test_parity.py`, `tests/test_layer_parity.py`, `tests/test_structure.py` UNCHANGED. Verifier asserts `git diff` empty.
- **D-52:** Phase 2's `tests/test_triton_<kind>_strict.py` files ARE in scope for editing (D-47 extends them). The Phase 2 fp32 sections within those files MUST remain unchanged — extension only.
- **D-53:** `tests/test_quantizers.py` is in scope for the QNT-04 test (D-45). NOT a locked file.

### Phase 2 disposition does NOT directly carry into Phase 4
- **D-54:** The Phase 2 Option C / TF32 / `< 5e-4` disposition was specifically for **fp32 Identity-quantizer Triton parity**. Quant-on is empirically different — the INT8 post-quant rounding may dominate over TF32 matmul drift. Plan 04-01's probe is the decisive evidence. Don't assume bit-identity fails; don't assume it holds. Measure.

### CUDA execution gate (D-55)
- **D-55:** Tests author now (CPU-only OK; quant-on tests are `cuda_only`-gated). CUDA box required for Plan 04-01 probe AND for the final phase-exit GPU run. Plan 04-05 (or wherever the audit-kickoff lives) carries the `checkpoint:human-verify` for the final GPU validation, similar to Phase 2's 02-06.

### Claude's Discretion
- Exact `pytest.parametrize` id strings for the three adversarial classes.
- Whether to fold QNT-04 into one of the existing plans (likely Plan 04-01 since it owns the probe + initial scope-setting) or split into its own plan.
- Plan count: 4 or 5 plans depending on QNT-04 folding decision.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project + phase
- `.planning/PROJECT.md` — Constraints (quant-on tier = bit-identical); Key Decisions (Phase 2 Option C disposition).
- `.planning/REQUIREMENTS.md` §QNT-01..04 — the 4 requirements for this phase.
- `.planning/ROADMAP.md` §"Phase 4: Quant-on bit-identity" — success criteria.
- `.planning/phases/01-CONTEXT.md` §D-10..12 — two-commit discipline carry-forward.
- `.planning/phases/02-CONTEXT.md` §D-13..28 — Phase 2 disposition (TF32 reality); D-25 `.cv` canary; D-27/D-28 carry-forward.
- `.planning/phases/02-SUMMARY.md` — Phase 2 close + Option C rationale.
- `.planning/phases/03-SUMMARY.md` — Phase 3 close + hand-off notes mentioning circulant fp32 FFT risk and LDR output-side fake-quant risk for Phase 4.
- `.planning/phases/01-VERIFICATION.md`, `02-VERIFICATION.md`, `03-VERIFICATION.md` — verifier reports.

### Codebase
- `src/gru_qat/quantizers.py:135-146` — `FakeQuantize._update_observer` (the QNT-04 broken method).
- `src/gru_qat/quantizers.py:178` — `FakeQuantizePerChannel` class.
- `src/gru_qat/gru_layer.py:_extract_h_quant_params` (~line 28) — pulls frozen quant params from cell for Triton in-kernel fake-quant.
- `src/gru_qat/gru_cell.py:CellWeights` — frozen-INT8 weight bag.
- `src/gru_qat/triton_kernels/scan.py`, `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py` — kernels with optional in-kernel fake-quant.
- `tests/test_triton_scan_strict.py`, `_diagonal_strict.py`, `_monarch_strict.py`, `_butterfly_strict.py` — Phase 2 strict-tier files that Plan 04 extends per D-47.
- `tests/test_quantizers.py` — extend with QNT-04 regression test per D-45.
- `tests/test_parity.py` + `tests/test_layer_parity.py` + `tests/test_structure.py` — LOCKED per D-28/D-38/D-51.

### External
- PyTorch INT8 quantization reference (for adversarial-input boundary calculation): https://docs.pytorch.org/docs/stable/quantization.html

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`_make_<kind>_layer` helpers** in each Phase 2 strict file (and original TF32-tier file): Phase 4 adds a sibling `_make_<kind>_layer_quant_int8` per file that takes the same shape args + builds a frozen-INT8 recipe.
- **FAST/SLOW grid constants** from Phase 2 strict files: Phase 4 uses a smaller grid per D-49 (separate constants `QUANT_FAST_GRID`, `QUANT_SLOW_GRID` to avoid confusion with the fp32 grids).
- **Relative-error idiom + per-tensor failure messages**: Phase 1/2 PATTERNS. Phase 4 uses `torch.equal` (binary) or `abs_diff` (tight bound per disposition), but the failure-message scaffolding stays the same.
- **`monkeypatch` pattern** from Phase 3 D-34: not needed for Phase 4 (no missing-dep tests here).
- **`bd create` + Commit A/B discipline** from Phase 1/2: carry forward unchanged.

### Established Patterns
- **One section per phase per kernel test file** (added by extension, not new files). Section headers use the ASCII rule-divider pattern from CONVENTIONS.md.
- **`cuda_only` + `pytest.importorskip("triton")`** at file top — already present from Phase 2.
- **Adversarial test naming**: `test_<kind>_quant_fwd_realistic`, `test_<kind>_quant_fwd_near_saturation`, `test_<kind>_quant_fwd_large_magnitude`, plus `_bwd_` variants.

### Integration Points
- **Per-kernel strict file extensions** (D-47): NO new files. Each Phase 2 strict file grows by ~150-200 lines.
- **`tests/test_quantizers.py` extension** for QNT-04 regression: small, well-bounded.
- **`src/gru_qat/quantizers.py:_update_observer` fix** (D-44): the only `src/` modification expected in Phase 4 (and it's a fix-commit, paired with a failing test commit per D-45).
- **Plan 04-01 probe** is the gate: its checkpoint:human-verify resolves D-42 disposition before Plans 04-02..04 are written. If the planner chunks the planning, only Plan 04-01 should be authored in detail; Plans 04-02..04 can be sketched-and-filled-in after the probe result.

</code_context>

<specifics>
## Specific Ideas

- **Probe shape (D-41):** T=8, B=4, H=64 dense; matches the smallest realistic-but-non-tiny shape that exercises the quant + matmul pipeline.
- **Empirical probe gradients:** check `torch.equal` on `dx`, `dh_0`, `dWh`, `dbh` independently. If even one fails, that's a finding (could indicate a deterministic-but-non-bit-identical bwd op like a `scatter_add` reduction order).
- **Three adversarial classes (D-46):** docstring per kernel section explains what each class is supposed to expose. Realistic = baseline. Near-saturation = INT8 rounding boundary correctness. Large-magnitude = clipping correctness.
- **QNT-04 regression test (D-45):** the failing test should exercise a tensor where running_min / running_max DIFFER per channel. Current scalar-reduction code produces a single global min/max; per-channel code produces a `[channels]`-shaped tensor. The test asserts `running_min.ndim > 0 and running_min.shape == (channels,)` AND `running_min[0] != running_min[1]` for a constructed input.

</specifics>

<deferred>
## Deferred Ideas

- **`tl.dot(input_precision="ieee")` kernel change** — surfaced as Phase 2 Option B; still deferred. May come up again at Plan 04-01 checkpoint if the probe fails AND the user wants to pursue genuine bit-identity through the kernel.
- **LSQ / PACT learnable activation scales (ACT-02)** — v2.
- **Bias quantization, LUT sigmoid/tanh** — Phase 6+ per SCOPE.md.
- **Quant-on for structured PyTorch fallbacks (circulant, LDR)** — no Triton path; existing `test_structure.py` int8-QAT smoke layer is enough. If a user later wants strict quant-on parity for them, file a separate audit issue.
- **Group_size + min_max combination** — if Plan 04-X's QNT-04 test surfaces other broken-axis combinations, document but don't fix unless it blocks the phase.

</deferred>

---

*Phase: 4-quant-on-bit-identity*
*Context gathered: 2026-05-14*
