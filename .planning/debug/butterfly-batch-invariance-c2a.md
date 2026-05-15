---
status: resolved
trigger: "butterfly Triton forward kernel violates batch-invariance at H=512 (bd gru-triton-c2a) — persistent kernel produces different output for identical per-batch inputs depending on batch index, period-8 cycle (batches 0-3 exact, 4-7 corrupted, repeat)"
created: 2026-05-15T00:00:00Z
updated: 2026-05-15T00:00:00Z
---

## Current Focus

hypothesis: RESOLVED — see Resolution.
test: —
expecting: —
next_action: —

## Symptoms

expected: The butterfly Triton forward kernel produces batch-invariant output — replicating one B=1 input across B batches yields bit-exact (or TF32-tier) identical output for every batch slot. The per-step PyTorch reference path (use_triton=False) IS batch-invariant.
actual: Triton output diverges per batch slot in a PERIODIC stride-8 cycle — batches 0-3 bit-exact (dev=0.0), batches 4-7 corrupted (~6e-2 abs), 8-11 exact, 12-15 corrupted, ... up to 6.3e-2 absolute deviation at H=512.
errors: tests/test_edge_cases.py::test_butterfly_partial_batch_tile[7], [9], [17], [33] FAIL (AssertionError on the batch-invariance contract). [1], [3], [5] PASS.
reproduction: uv run pytest tests/test_edge_cases.py::test_butterfly_partial_batch_tile -q on a CUDA host. Or: butterfly GRULayer (H=512, T=16, fp32-Identity), forward on x1=[T,1,H], then on x1.repeat(1,33,1) — corrupted batch slots diverge from the B=1 result.
started: Surfaced by Phase 6 plan 06-01 Task 3 (EDG-02 B=1 + small-H BLOCK-size sweep). The butterfly OOB fix d8218d4 shipped without a B%BLOCK_B partial-tile regression test — CONCERNS.md predicted this failure mode.

## Eliminated
<!-- APPEND only -->
- timestamp: 2026-05-15 — HYPOTHESIS REJECTED: "period-8 batch-tiling indexing bug (wrong offset/mask/stride for lanes 4-7 of a BLOCK_B=8 tile)". With block_b=1 (each batch is its own program, NO batch tiling at all) the corruption STILL occurs (batches 1/6/7 corrupt for B=8). Identical single-row programs producing divergent output rules out any batch-offset / batch-mask / batch-stride error. local_b / mask_b / offs_b indexing in the kernel is correct.

## Evidence
<!-- APPEND only -->
- timestamp: 2026-05-15 — Per-batch deviation probe (replicate B=1 input across B slots, H=512, T=16): B=4 -> all 0.0; B=8 -> all 8 lanes nonzero (growing 5e-3..1.2e-1); B=12 -> lanes 0-3 = 0.0, 4-7 nonzero, 8-11 = 0.0; B=16 -> all 16 nonzero. Pattern is NOT "second half of every tile" — it tracks warp layout, not tile layout.
- timestamp: 2026-05-15 — block_b sweep at B=8: block_b=1 -> batches 1,6,7 corrupt; block_b=2 -> 1,4,5 corrupt; block_b=4 -> 1..7 corrupt; block_b=8 -> 4..7 corrupt. Corruption persists at block_b=1 => not a batch-tiling bug.
- timestamp: 2026-05-15 — num_warps / num_stages sweep at B=8 (decisive): num_warps=1 -> ALL ZERO (batch-invariant); num_warps=2 -> ALL ZERO; num_warps=4 -> corrupt; num_warps=8 -> corrupt (worst ~6e-2); num_stages=2/3 also corrupt. The bug is gated entirely by num_warps >= 4 => classic intra-CTA missing-barrier race on the warp-shared scratch buffer.

## Resolution

root_cause: The butterfly persistent forward (and backward) kernel in src/gru_qat/triton_kernels/scan_butterfly.py routes the running per-gate state through a per-program scratch buffer in global memory between butterfly stages. Each stage loads `offs_h` and its XOR-`partner` position, applies the 2x2 twiddle, and stores back to `offs_h`. That scratch buffer is shared across ALL warps of the CTA, and a position's `partner` is frequently owned by a DIFFERENT warp than the one that wrote it in the previous stage. Triton only orders loads/stores within a single warp — there was NO CTA-wide barrier / memory fence between consecutive stages. With num_warps >= 4 (the default num_warps=4), one warp's stage-s store was not guaranteed visible to another warp's stage-(s+1) `partner` load, so stages read stale scratch. The result is a warp-layout-dependent (hence batch-index-dependent) corruption that vanishes at num_warps in {1,2} and is independent of block_b. The misleading "period-8 / lanes-4-7" signature in the original bug report was an artifact of one particular B/warp-layout combination, not a batch-tiling indexing error.
fix: Inserted `tl.debug_barrier()` (CTA-wide barrier + memory fence) between every butterfly stage in scan_butterfly.py: after the scratch copy-in and after each stage's per-gate store loop in the forward kernel; and after the state[g,0] seed, after each forward-recompute stage, after the d_state seed, and after each reverse-stage `d_old` store in the backward kernel. This guarantees all warps' stage-s scratch writes are visible before any warp issues a stage-(s+1) cross-warp partner load. No batch-indexing code was changed (it was already correct). Pre-existing ruff warnings (unused `import math`, dead `contrib_dd`/`contrib_dp` in the backward kernel) were left untouched as out-of-scope.
verification: tests/test_edge_cases.py::test_butterfly_partial_batch_tile — all 7 parametrized cases (B in {1,3,5,7,9,17,33}) GREEN (was 3 RED). Full regression: tests/test_edge_cases.py + tests/test_butterfly_dispatch.py -m "not slow" -> 86 passed, 3 skipped, 0 failed (no regressions). Manual batch-invariance probe (B in {4,8,12,16}, H=512) now reports worst per-batch deviation 0.0 everywhere.
files_changed: ["src/gru_qat/triton_kernels/scan_butterfly.py"]
