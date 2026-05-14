# Phase 4: Quant-on bit-identity - Discussion Log

> **Audit trail only.** Decisions are captured in CONTEXT.md.

**Date:** 2026-05-14
**Phase:** 4-quant-on-bit-identity
**Areas discussed:** Bit-identity feasibility, QNT-04 fix vs fence, Adversarial input scope, Test file location + grid scope

---

## Bit-identity feasibility under Phase 2 TF32 disposition

### Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Empirical probe FIRST, then plan tolerance | Plan 04-01 probe → checkpoint resolves the bound | ✓ |
| Assume bit-identity holds | Plan everything at torch.equal; risk ALL tests fail | |
| Pre-emptively use < 1 INT8 ULP | Pragmatic; gives up strict contract | |

**User's choice:** Empirical probe first. Captured as D-41..43.

### Disposition if probe fails

| Option | Description | Selected |
|--------|-------------|----------|
| Surface checkpoint:human-verify | Like Phase 2 Option C — pick at the time | ✓ |
| Auto-fall-back to < 1 INT8 ULP | Mechanical | |
| Halt phase | Most cautious | |

**User's choice:** Checkpoint. Captured as D-42.

---

## QNT-04 — per-channel min_max observer

### Fix or fence?

| Option | Description | Selected |
|--------|-------------|----------|
| Fix — implement per-channel reduction | Rewrite _update_observer; closes ACT-01 | ✓ |
| Fence — raise NotImplementedError | Simpler; defers to v2 | |
| Both — fix bug AND validate other unsupported combinations | Most thorough | |

**User's choice:** Fix. Captured as D-44.

### Regression test before fix?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — Commit A failing test before Commit B fix (D-37) | Strongest audit trail | ✓ |
| No — just fix it | Faster; less rigorous | |

**User's choice:** Yes per D-37. Captured as D-45.

---

## Adversarial input scope

| Option | Description | Selected |
|--------|-------------|----------|
| Three classes per kernel | Realistic + near-saturation + large-magnitude | ✓ |
| Just near-saturation | One class | |
| Full sweep (+ exact-boundary + zeros + NaN + Inf) | Most thorough | |

**User's choice:** Three classes. Captured as D-46.

---

## Test file location + grid scope

### File location

| Option | Description | Selected |
|--------|-------------|----------|
| Extend Phase 2's test_triton_<kind>_strict.py | Reuse helpers + constants | ✓ |
| New test_triton_<kind>_quant.py per kernel | Cleanest separation; duplicates | |
| Single consolidated file | Compact; breaks per-kernel pattern | |

**User's choice:** Extend Phase 2's files. Captured as D-47..48.

### Grid

| Option | Description | Selected |
|--------|-------------|----------|
| Smaller grid (T∈{8,64}×B∈{1,4,32}×H∈{32,128,512}) | Bit-identity is binary | ✓ |
| Reuse Phase 2's full grids | Consistency; longer GPU run | |

**User's choice:** Smaller grid. Captured as D-49.

---

## Claude's Discretion

- Plan count: 4 or 5 plans depending on QNT-04 folding decision.
- Exact pytest.parametrize id strings for adversarial classes.
- Whether to author Plans 04-02..04 in detail OR sketch-and-fill-in after Plan 04-01 probe resolves D-42 disposition. Planner decides.

## Deferred Ideas

- `tl.dot(input_precision="ieee")` kernel change — Phase 2 Option B still on the table as a Plan 04-01 checkpoint option.
- LSQ/PACT (ACT-02) — v2.
- Bias quantization, LUT — Phase 6+.
- Quant-on for circulant/LDR — no Triton path; existing smoke tests sufficient.
- Group_size + min_max combination — fix only if Plan 04-X surfaces it as blocking.
