# Phase 2: Triton fast-path parity vs reference - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-13
**Phase:** 2-triton-fast-path-parity-vs-reference
**Areas discussed:** Tolerance + precision policy, Shape grid scope, Test file location, TRI-04..06 regression test depth

---

## Tolerance + precision policy

### Precision strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Forced fp32 strict ('highest', < 1e-5) | New tests under 'highest'; existing TF32 tests untouched. | |
| Adopt existing TF32 tolerances as Phase 2 contract | Loosen PROJECT.md to 5e-3 rel under 'high'. | |
| Both — add strict tier + tighten existing where possible | Three tiers: strict ('highest', < 1e-5), realistic ('high', < 1e-4 where possible), permissive ('high', existing where TF32 dominates). | ✓ |

**User's choice:** Both — three tiers. Captured as D-13/D-14.

### TF32 verdict (when strict passes but realistic fails tighter bound)

| Option | Description | Selected |
|--------|-------------|----------|
| Record TF32 noise; not a finding | Math-pass under 'highest' = audit pass. TF32 drift is expected fp32 behavior. | ✓ |
| Treat as a finding | Investigate every TF32 divergence. | |

**User's choice:** Record TF32 noise; not a finding. Captured as D-15.

---

## Shape grid scope for Triton tests

### Grid strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Per-kernel custom grids | Each kernel gets a grid tuned to its constraints (H ranges, nblocks for monarch, power-of-2 for butterfly). | ✓ |
| Same 75-combo grid as Phase 1 | Maximum consistency; some kernels error out on unsupported shapes. | |
| Compact strict grid (3×3×3 = 27) | Minimum coverage; edge cases deferred to Phase 6. | |

**User's choice:** Per-kernel custom grids. Captured as D-16/D-17/D-18.

### CUDA execution plan

| Option | Description | Selected |
|--------|-------------|----------|
| Author tests now; user runs on CUDA box | `cuda_only` decorator; CPU run skips cleanly. User must run on GPU before phase-exit. | ✓ |
| Need CUDA access during the phase | Halt the phase until CUDA confirmed. | |
| Author + dry-run check on CPU only | `pytest --collect-only`; defer GPU run. | |

**User's choice:** Author now, user runs on CUDA box. Captured as D-26.

---

## Test file location

### Where strict-tier tests live

| Option | Description | Selected |
|--------|-------------|----------|
| One new file per kernel: test_triton_<kind>_strict.py | Four new files paired with the existing test files. Clear separation. | ✓ |
| Single new file: test_triton_strict_parity.py | One file covering all four kernels. Monolithic. | |
| Extend the existing test_triton_<kind>.py files | Mix strict + realistic in one file. Per-test precision toggling is fragile. | |

**User's choice:** One new file per kernel. Captured as D-19/D-20/D-21.

### Bundling tiers in the new files

| Option | Description | Selected |
|--------|-------------|----------|
| Strict only — realistic tightenings live in existing files | Cleanest separation; each new file has one job. | ✓ |
| Bundle three tiers in the new files | Per-kernel "everything" file. Mixes concerns. | |
| Strict only; defer realistic tightening | Phase 2 = pure strict; tighten existing as a follow-up phase. | |

**User's choice:** Strict only; realistic tightenings go into existing files. D-20 captures this.

---

## TRI-04..06 regression test depth

### TRI-04 (Butterfly OOB)

| Option | Description | Selected |
|--------|-------------|----------|
| Existing coverage is sufficient | `tests/test_butterfly_dispatch.py:206` already locks the fix. | ✓ |
| Add a strict-tier version | Same B-sweep at < 1e-5; belt + suspenders. | |

**User's choice:** Existing coverage sufficient. Captured as D-22.

### TRI-05 (Autotune dWh/dbh)

| Option | Description | Selected |
|--------|-------------|----------|
| New named test in test_triton_scan_strict.py | Force multiple autotune configs; assert second run matches reference. | ✓ |
| Skip — rely on broader grid | Trust grid to retrigger. Risky because the bug was data-dependent. | |

**User's choice:** New named test. Captured as D-23.

### TRI-06 (Cross-CTA `.cv` determinism)

| Option | Description | Selected |
|--------|-------------|----------|
| 50-run determinism test | torch.equal across 50 runs; fast (≤ 30s on GPU). | ✓ |
| 10-run smoke (lighter) | Same idea, fewer iterations. Less likely to surface intermittent issues. | |
| Skip — trust static grep audit | Comment in source + `cache_modifier=".cv"` static check. No runtime test. | |

**User's choice:** 50-run determinism. Captured as D-24. Plus a static grep canary as D-25 — best of both.

---

## Claude's Discretion

- Exact pytest.parametrize id strings (e.g., "T=8-B=4-H=128").
- Whether to factor out a shared `tests/_triton_strict_helpers.py`. Default: yes if duplication > ~30 lines.
- Plan execution order within Phase 2. Default: dense first (autotune + determinism live there), then diagonal/monarch/butterfly in any order.
- Whether to use `torch.allclose(..., atol=1e-5, rtol=0)` or compute `(a-b).abs().max() < 1e-5` directly. The latter gives better failure messages.

## Deferred Ideas

- Per-channel `min_max` observer fix-vs-fence (Phase 4).
- Quant-on bit-identity for Triton paths (Phase 4).
- Circulant + LDR (no Triton kernel; Phase 3).
- Edge cases for non-diagonal kernels at H=1, H=2, T=0, B=0 (Phase 6).
- Bench re-validation (out of scope).
- Shared strict helpers module — defer the decision to executor write time.
