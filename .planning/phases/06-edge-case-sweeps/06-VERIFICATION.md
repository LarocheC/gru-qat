---
phase: 06-edge-case-sweeps
verified: 2026-05-15T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  note: initial verification
---

# Phase 6: Edge-case Sweeps Verification Report

**Phase Goal:** Every path (reference, dense Triton, diagonal Triton, monarch Triton, butterfly Triton, circulant per-step, LDR per-step) survives T=1, B=1, H∈{1,2}, T∈{512,1024}, and T=0/B=0 with either correct output or a clear tested error.
**Verified:** 2026-05-15
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | T=1 fwd+bwd sweep for every path, parity at the path's tolerance tier (EDG-01, SC#1) | ✓ VERIFIED | `test_t1_forward_parity` + `test_t1_backward_parity`, parametrized over all 7 `ALL_PATHS` at `(1,4,8)`; tiers via `_path_tol` reuse PROJECT.md bounds (ref <1e-4, diag/dense/monarch <5e-4, butterfly <5e-2, circ/ldr <1e-5). Live run: 82 passed. |
| 2 | B=1 + H∈{1,2} sweep for every path, CONCERNS.md BLOCK-size failure modes (EDG-02, SC#2) | ✓ VERIFIED | `test_b1_small_h_parity` grid contains literal `(8,1,1)`,`(8,1,2)` for all 7 paths; `test_butterfly_partial_batch_tile` at B∈{1,3,5,7,9,17,33}, H=512 — the partial-tile sweep. Surfaced 2 real bugs, both fixed in-phase. |
| 3 | T∈{512,1024} long-sequence tests `@pytest.mark.slow`, drift within tier-A tolerance (EDG-03, SC#3) | ✓ VERIFIED | `test_long_t_drift` decorated `@pytest.mark.slow`, parametrized `T∈[512,1024]` × 7 paths. Live `-m slow` run: 14 passed, 71 deselected. |
| 4 | T=0/B=0 raise a clear ValueError naming the offending dimension on every path; no NaN/hang; policy logged in PROJECT.md (EDG-04, SC#4) | ✓ VERIFIED | `gru_layer.py:169-176` — `if seq_len == 0: raise ValueError(f"...T={seq_len}")` and matching `batch_size == 0` → `B`. Uses `if…raise`, not `assert`. Single guard after line-158 unpack covers all 7 GRULayer-routed paths. `test_t0_b0_raises_valueerror` × 7 paths × 2 dims. PROJECT.md line 125 Key Decisions row logs the policy. |
| 5 | Any mismatch → failing test → bd issue → in-phase fix; no `@pytest.mark.xfail` (SC#5, D-04/D-05) | ✓ VERIFIED | `gru-triton-ehf` (butterfly H=1 crash) Commit A `eb7242b` → Commit B `cca1783`, bd CLOSED. `gru-triton-c2a` (butterfly batch-invariance) Commit A `d6625cc` → Commit B `6d09571`, bd CLOSED. Zero `xfail` tokens in `test_edge_cases.py` or modified src files. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_edge_cases.py` | 7-path × edge-shape parametrized sweep, ≥250 lines, `def test_t1_` | ✓ VERIFIED | 614 lines; 7 test functions; `ALL_PATHS` has all 7 entries; `test_t1_forward_parity`/`test_t1_backward_parity` present. Imported, collected, executed (82 pass). |
| `.planning/PROJECT.md` | Logged T=0/B=0 ValueError disposition, contains `T=0` | ✓ VERIFIED | Line 125 Key Decisions row "Phase 6 D-01 (T=0/B=0 disposition)" — all 7 paths raise ValueError naming the dim. |
| `06-pytest-output.txt` | Timestamped pytest-output artifact | ✓ VERIFIED | Exists, non-empty (26 KB). NOTE: artifact is STALE (pre-c2a-fix) — live re-run is authoritative. |
| `src/gru_qat/gru_layer.py` | T=0/B=0 ValueError guard | ✓ VERIFIED | Guard at lines 169-176, `if…raise`, names T/B. |
| `src/gru_qat/structure.py` | butterfly H<2 ValueError guard (in-phase ehf fix) | ✓ VERIFIED | `_validate_shapes` lines 107-109: `if in_features < 2 or out_features < 2: raise ValueError("butterfly requires in/out >= 2…")`. |
| `src/gru_qat/triton_kernels/scan_butterfly.py` | c2a barrier fix landed | ✓ VERIFIED | 9 `tl.debug_barrier()` calls; commit `6d09571` "add intra-CTA barriers between butterfly stages". |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `test_edge_cases.py` | `test_layer_parity.py` | `importlib.import_module("test_layer_parity")` | ✓ WIRED | Line 58-59 — import-only, D-11 honored; `_translate_nn_gru_to_cell` accessed. |
| `test_edge_cases.py` T=0/B=0 tests | `GRULayer.forward` guard | `pytest.raises(ValueError, match=...)` | ✓ WIRED | Line 376 `with pytest.raises(ValueError, match=bad_dim)`; 14 EDG-04 cases pass against the live guard. |
| `test_edge_cases.py` monarch bwd | `test_triton_monarch_strict.py` | `_skip_if_monarch_bwd_hw_limit` import | ✓ WIRED | `_maybe_skip_monarch_bwd` line 391; 1 legitimate HW-limit SKIP observed. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full edge-case suite (non-slow) | `uv run pytest tests/test_edge_cases.py -q` | 82 passed, 3 skipped, 0 failed (107s) | ✓ PASS |
| Long-T slow tier | `uv run pytest tests/test_edge_cases.py -m slow -q` | 14 passed, 71 deselected (117s) | ✓ PASS |
| No xfail in src | `grep xfail src/gru_qat/{gru_layer,structure,triton_kernels/scan_butterfly}.py` | NONE | ✓ PASS |
| Locked files unmodified | `git diff --name-only 3bdcf7b..HEAD` filtered for locked files | NO LOCKED FILES MODIFIED | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EDG-01 | 06-01-PLAN.md | T=1 single-timestep fwd+bwd for every path | ✓ SATISFIED | `test_t1_forward_parity`/`test_t1_backward_parity`, 7 paths, live pass. |
| EDG-02 | 06-01-PLAN.md | B=1 + H∈{1,2} for every path, BLOCK-size modes | ✓ SATISFIED | `test_b1_small_h_parity` + `test_butterfly_partial_batch_tile`; surfaced & fixed ehf + c2a. |
| EDG-03 | 06-01-PLAN.md | T∈{512,1024} long-sequence parity, `slow` marked | ✓ SATISFIED | `test_long_t_drift` `@pytest.mark.slow`, 14 pass. |
| EDG-04 | 06-01-PLAN.md | T=0/B=0 clear tested error, no NaN/hang | ✓ SATISFIED | `GRULayer.forward` guard + `test_t0_b0_raises_valueerror`; policy in PROJECT.md. |

All 4 EDG IDs declared in PLAN frontmatter `requirements:` and verified. No orphaned requirements — REQUIREMENTS.md maps exactly EDG-01..04 to Phase 6.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No debt markers (TBD/FIXME/XXX), no stubs, no xfail in Phase 6 surface | ℹ️ Info | Clean. SUMMARY "Known Stubs: None" confirmed. |

The 3 SKIPs in the live suite are all legitimate and accounted for:
- 2× `test_edge_cases.py:485` — butterfly H=1 parametrize cases, correctly skipped because H=1 is rejected at construction; the dedicated `test_butterfly_h1_raises_valueerror` covers the contract.
- 1× monarch bwd HW-limit (`gru-triton-e0l`, RTX 2000 Ada SMEM constraint) — a hardware limit, not a coverage gap.

### Discrepancy: SUMMARY.md vs live codebase

The SUMMARY.md "Open Findings — Handoff" section claims `gru-triton-c2a` was HANDED OFF with the kernel fix deferred and 4 tests RED. This is **stale** — the SUMMARY was written at commit `f4096d8`, before the c2a fix commit `6d09571`. Verified against the live codebase:
- c2a fix `6d09571` landed (9 `tl.debug_barrier()` in `scan_butterfly.py`).
- bd `gru-triton-c2a` is CLOSED.
- Debug session `.planning/debug/butterfly-batch-invariance-c2a.md` status `resolved`.
- Live re-run: 82 passed, 0 failed — the 4 previously-RED `test_butterfly_partial_batch_tile` cases are now GREEN.

Per goal-backward methodology, the live codebase is authoritative over SUMMARY narrative. The phase goal IS achieved — every path survives the edge sweep with correct output or a clear tested error, and no `@pytest.mark.xfail` remains anywhere.

### Decisions Honored (CONTEXT.md D-01..D-11)

| Decision | Status | Evidence |
|----------|--------|----------|
| D-01 ValueError on T=0/B=0 naming the dim | ✓ | `gru_layer.py:169-176` |
| D-02 policy logged in PROJECT.md | ✓ | PROJECT.md:125 |
| D-03 guard at GRULayer.forward | ✓ | Single guard after line-158 unpack |
| D-04 fix-in-phase | ✓ | ehf + c2a both fixed in-phase |
| D-05 two-commit discipline, no xfail | ✓ | Commit A/B pairs for both bugs; zero xfail |
| D-06 deep kernel fix accepted | ✓ | c2a kernel fix landed (`6d09571`) |
| D-07 realistic inputs only | ✓ | Single `torch.randn` input class |
| D-08 uniform 7-path coverage | ✓ | `ALL_PATHS` has 7 entries; circulant/ldr full grid |
| D-09 tolerances reused from PROJECT.md | ✓ | `_path_tol` reuses committed bounds, no new |
| D-10 single test_edge_cases.py | ✓ | One new file, 614 lines |
| D-11 import-only from locked test_layer_parity.py | ✓ | `importlib.import_module`, locked file unchanged |

### Human Verification Required

None. All success criteria are programmatically verifiable and were verified by live test execution on a CUDA + Triton host.

### Gaps Summary

No gaps. All 5 ROADMAP Phase 6 success criteria are met, all 4 EDG requirements satisfied, all 11 CONTEXT.md decisions honored. Both bugs surfaced by the edge sweep (`gru-triton-ehf`, `gru-triton-c2a`) were fixed in-phase with the D-04/D-05 two-commit discipline and their bd issues CLOSED. `gru-triton-7rj` (scan*.py wrapper `assert` hardening) is correctly filed-not-fixed — it is explicitly out of EDG-04 scope (D-01 scope is `GRULayer.forward`-routed calls). The D-51 locked files are unmodified. No `@pytest.mark.xfail` anywhere. Live test suite: 82 passed + 14 slow passed, 0 failed.

---

_Verified: 2026-05-15_
_Verifier: Claude (gsd-verifier)_
