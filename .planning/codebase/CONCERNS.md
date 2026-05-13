# Codebase Concerns

**Analysis Date:** 2026-05-13

## Tech Debt

**`min_max` observer broken for per-channel activation quant:**
- Issue: `FakeQuantize._update_observer` (`src/gru_qat/quantizers.py:135-146`) does a global scalar reduction (`x.detach().min()`, `x.detach().max()`) regardless of `config.axis`. Per-channel running stats are not accumulated.
- Files: `src/gru_qat/quantizers.py:135-146`
- Impact: Per-channel activation quant with `mode="min_max"` produces an incorrect (single-scalar) scale. Currently masked because the fast paths use per-tensor + frozen-post-calibration; per-channel weight quant uses `dynamic` mode where the bug doesn't fire.
- Fix approach: Reshape `x` so the channel axis is preserved, run `amin/amax` over the other dims with `keepdim=True`. One-method change. EMA update needs to be vectorized as well. See `DEVELOPMENT.md` "Phase 1 known gap" and the `TODO(phase=4)` at line 136.

**`triton_kernels/__init__.py` is dead-letter stub:**
- Issue: Top-of-file says "NOTHING IN THIS FILE RUNS YET. It is an interface design." `triton_gru_cell` raises `NotImplementedError("phase=5")` (`src/gru_qat/triton_kernels/__init__.py:91-104`). The actual Phase 5 code shipped under `scan.py` / `scan_monarch.py` / `scan_butterfly.py` / `scan_diagonal.py` with a totally different shape (multi-step persistent kernels, not single-step). The interface here describes a single-step cell that does not exist.
- Files: `src/gru_qat/triton_kernels/__init__.py:1-104`
- Impact: Misleading docs. A new contributor reading this would think the Triton path is unimplemented or believes the entrypoint is `triton_gru_cell`.
- Fix approach: Replace contents with a one-liner that re-exports `is_available()` plus pointers to `scan.py` / `scan_monarch.py` / `scan_butterfly.py` / `scan_diagonal.py`. Delete the `triton_gru_cell` stub.

**`TODO(phase=5): TritonGRULayer` placeholder in `gru_layer.py`:**
- Issue: Comment at `src/gru_qat/gru_layer.py:305-308` describes a non-existent `TritonGRULayer` class. The actual integration was done via `use_triton` flag + `_forward_fast_dispatch` on the existing `GRULayer`. The comment is stale.
- Files: `src/gru_qat/gru_layer.py:305-308`
- Impact: Misleading — implies a separate layer subclass is the intended path.
- Fix approach: Delete the comment.

**`TODO(phase=5): export_int_weights()` in `gru_cell.py`:**
- Issue: Comment at `src/gru_qat/gru_cell.py:507-509` promises an export helper that returns int tensors + scales + zero points for the inference kernel. Phase 5 shipped without it; the Triton path consumes fake-quant floats in-kernel rather than true int tensors.
- Files: `src/gru_qat/gru_cell.py:507-509`
- Impact: No documented export path to TFLite/ONNX/embedded — the SCOPE.md "we produce quantized weights and a reference int kernel suitable for porting" promise has no implementation.
- Fix approach: Either implement the export, or file a Phase 6 ticket and delete the comment so it's not confused with shipped work.

**`TODO(phase=3): LSQ gradient scaling` in `ste.py`:**
- Issue: Comment at `src/gru_qat/ste.py:84-88` describes LSQ (Learnable Step-size Quantization) gradient scaling. `learnable_scale` flag is plumbed in `QuantizerConfig` (`src/gru_qat/quantizers.py:59`) but never read anywhere.
- Files: `src/gru_qat/ste.py:84-88`, `src/gru_qat/quantizers.py:59`
- Impact: User can set `QuantizerConfig(learnable_scale=True)` with zero effect. Silent no-op.
- Fix approach: Either implement (subclass `STERound` with grad-scaled step, opt-in via flag in `FakeQuantize.forward`), or remove the flag from `QuantizerConfig` to avoid the lie. Listed as open question #3 in `DEVELOPMENT.md`.

**Stale-by-comment `# TODO(phase=2): test_simulator_parity` in tests:**
- Issue: `tests/test_quantizers.py:88-90` skips `test_simulator_parity` with `@pytest.mark.skip(reason="phase=2 — requires simulator import")`. SCOPE.md §"Relationship to existing codebase" claims `FakeQuantize.forward()` must produce bit-identical output to the simulator's `quantize_dequantize()` and that parity is tested in `tests/test_simulator_parity.py` (Phase 2). That test file doesn't exist, and the skipped marker is the only trace of the contract.
- Files: `tests/test_quantizers.py:88-90`
- Impact: SCOPE.md's bit-identity contract with the simulator is unenforced. Drift can land silently.
- Fix approach: Either ship the simulator parity test (preferred — SCOPE.md is explicit about the requirement) or downgrade the SCOPE claim.

**Dead `noqa: F841` loads in butterfly backward kernel:**
- Issue: `src/gru_qat/triton_kernels/scan_butterfly.py:495-497` loads `bhr` / `bhz` / `bhn` and immediately silences the unused-variable warning. These three lines are dead work on every backward step.
- Files: `src/gru_qat/triton_kernels/scan_butterfly.py:495-497`
- Impact: Three wasted global loads per batch tile per backward call. Minor perf footgun.
- Fix approach: Delete the loads.

**`_USE_TRITON_BACKWARD` toggle is a runtime module-level constant:**
- Issue: `src/gru_qat/triton_kernels/scan.py:1515` exposes `_USE_TRITON_BACKWARD = True` as a top-level switch with no setter / context manager. Toggling it requires editing the file.
- Files: `src/gru_qat/triton_kernels/scan.py:1515-1566`
- Impact: Debugging workflow ("flip to PyTorch backward and compare") requires source edits, which makes ad-hoc bisection painful and can leak into commits.
- Fix approach: Lift to an env var (`GRU_QAT_TRITON_BACKWARD=0`) or a context manager.

## Known Bugs

**Recent fix: butterfly scratch/state OOB at last program (`d8218d4`):**
- Symptoms: Out-of-bounds writes when `B % BLOCK_B != 0` and the last program tile had partial `mask_b`. Scratch buffer was being written without `mask_b` at certain points.
- Files: `src/gru_qat/triton_kernels/scan_butterfly.py` (forward and backward kernels)
- Trigger: Batch sizes not divisible by `block_b` (default 8).
- Status: Fixed in commit `d8218d4` (very recent). Suggests the masking story across `scan_butterfly.py`'s many gather/scatter passes was fragile and not exhaustively covered by tests; another corner is plausible.
- Audit suggestion: Run the butterfly fwd/bwd at `B=1, 3, 5, 7, 9, 17, 33` against the PyTorch reference; the test matrix in `tests/test_butterfly_dispatch.py` doesn't sweep these.

**Recent fix: zero dWh/dbh accumulator slabs in autotuned bwd kernel (`c001a8a`):**
- Symptoms: `@triton.autotune` reuses the same `dWh_partial` / `dbh_partial` tensors across trial configs. Each trial accumulates into the prior trial's result, so the chosen config's output was wrong until the slabs were zeroed inside the kernel.
- Files: `src/gru_qat/triton_kernels/scan.py:975-1004` (per-program zeroing loop with detailed comment), wrapper at `scan.py:1370-1375`.
- Trigger: Only visible with `@triton.autotune`; manual config selection didn't trip it.
- Status: Fixed by in-kernel zeroing. Same class of bug could lurk in the monarch / butterfly autotuned paths if they ever gain `@triton.autotune` decorators — the explicit comment at `scan.py:975-981` is a load-bearing warning.

**Recent fix: persistent forward kernel barrier was using relaxed atomics (`0e26193`):**
- Symptoms: Non-deterministic ~0.2 absolute drift on `gru_scan_persistent` outputs at `t >= 1`. Earlier code used `atomic_add` without `sem="release"` + `tl.load(cache_modifier=".cv")` for the spin-wait. Cache modifier looked like a fence substitute, but the CUDA memory model doesn't guarantee data visibility after the post-increment counter without an acquire fence.
- Files: `src/gru_qat/triton_kernels/scan.py:184-203` (forward), `scan.py:608-617` (backward), explicit warning in DEVELOPMENT.md §"What the agent should NOT do".
- Trigger: Persistent kernels only; depends on CTA scheduling order.
- Status: Fixed via release/acquire `atomic_add(sem=...)` pattern. **Carries forward as a documented anti-pattern** — DEVELOPMENT.md and the in-kernel comments both warn future contributors not to substitute `.cv` cache modifier for an acquire fence.

## Security Considerations

**No untrusted-input attack surface:**
- This is a library, not a service. No HTTP/network surface, no user-input deserialization. Threat model is "the user is running the code on their own machine on their own data."
- Files: N/A
- Current mitigation: N/A (out of scope).
- Recommendations: N/A.

**Asserts used for shape validation in hot kernel wrappers:**
- Risk: `assert h0.shape == (B, H)` etc. across `src/gru_qat/triton_kernels/scan*.py` (38 assert statements total). Python `-O` strips assertions. A user running `python -O` would get a raw kernel call with mis-shaped tensors, which can lead to OOB reads/writes inside the kernel.
- Files: `src/gru_qat/triton_kernels/scan.py:233-235, 660-664, 1349-1353, 1665-1667`; `scan_monarch.py:88-90, 328-329, 772`; `scan_butterfly.py:365, 371, 374`; `scan_diagonal.py:90, 233, 491`; `gru_cell.py:449`.
- Current mitigation: Default Python (no `-O`) runs the asserts.
- Recommendations: Promote shape checks to `if … raise ValueError(...)` for the user-facing wrappers (`gru_scan`, `gru_scan_persistent`, `gru_scan_monarch`, `gru_scan_butterfly`, `gru_scan_diagonal`). Internal kernel-side asserts can stay.

## Performance Bottlenecks

**Reference PyTorch path is intentionally slow:**
- Problem: `GRULayer` with `use_triton=False` runs a Python time loop over `step()` / `step_structured()` / `step_with_gi()`. At `(T=64, B=32, H=512)` fp32 the dense compile-step variant is 38.7 ms vs cuDNN's 4.4 ms (8.8× slower). See bench table in `DEVELOPMENT.md` lines 168-179.
- Files: `src/gru_qat/gru_layer.py:170-200`, `src/gru_qat/gru_cell.py:351-462`.
- Cause: Manual per-timestep Python loop; cuDNN is a fused C++ kernel. Manual unroll is REQUIRED for QAT — that's the whole reason this library exists (cuDNN has no fake-quant hooks).
- Improvement path: **DO NOT optimize the reference path.** SCOPE.md §"Don't optimize the reference path." and DEVELOPMENT.md §"What the agent should NOT do" both forbid it. Speed lives in `triton_kernels/*`. Anyone trying to make the reference faster is making a wrong-layer fix.

**Circulant and LDR structured kinds have no Triton kernel:**
- Problem: `structure_hidden=StructureConfig(kind="circulant"|"ldr")` falls back to the slow per-step PyTorch path. `_fast_dispatch_eligible` is false for these kinds (`src/gru_qat/gru_layer.py:99-105`).
- Files: `src/gru_qat/gru_layer.py:99-105`, `src/gru_qat/structure.py:207-247`.
- Cause: Circulant uses `torch.fft.rfft` per step; LDR uses `torch_structured.structured.layers.LDRSubdiagonal` per step. Neither has a multi-step persistent kernel.
- Improvement path: Write `scan_circulant.py` / `scan_ldr.py` following `scan_monarch.py` / `scan_butterfly.py` templates. Documented as a deferred Phase 5+ upgrade path in `DEVELOPMENT.md` §"Adding a new structured kind".

**Butterfly backward kernel uses global memory between stages:**
- Problem: Triton register tensors can't do dynamic gather/scatter, so each butterfly stage in `scan_butterfly.py` reads from and writes back to per-program scratch in global memory (`src/gru_qat/triton_kernels/scan_butterfly.py:17-27`). Backward kernel additionally stores `LOG_H_PAD+1` per-stage state snapshots per gate.
- Files: `src/gru_qat/triton_kernels/scan_butterfly.py:417-903`.
- Cause: Fundamental Triton language limitation. L2 absorbs some of the cost.
- Improvement path: Bench at `H=512`: butterfly train is 20.3 ms vs dense Triton 8.8 ms (per `DEVELOPMENT.md`). At small `H` the launch-fusion win still beats the global-memory tax, but butterfly is slower than Monarch nblocks=8 (2.0 ms) by a wide margin. Acceptable tradeoff for the parameter count (4.6K vs 32K per gate at H=512).

**Persistent grid is capped at SM count — deadlock guard, not perf:**
- Problem: `scan.py:251-254` and `scan.py:682-685` raise `RuntimeError` if `cdiv(B, block_b) * cdiv(H, block_oh) > sm_count`. Spin-wait barrier deadlocks otherwise (a CTA that never gets resident-on-SM never increments the counter, and waiters never finish).
- Files: `src/gru_qat/triton_kernels/scan.py:249-254, 680-685`.
- Cause: Fundamental constraint of cross-CTA persistent kernels with spin-wait barriers.
- Improvement path: Raise `BLOCK_B` / `BLOCK_OH` at large `(B, H)`. For genuinely larger shapes, fall back to the autotune (non-persistent) path. Documented in the wrapper raise message.

## Fragile Areas

**Cross-CTA spin-wait barriers — release/acquire pattern is load-bearing:**
- Files: `src/gru_qat/triton_kernels/scan.py:184-203` (fwd), `scan.py:608-617` (bwd), `scan_monarch.py:274-279` (fwd), `scan_monarch.py:730-734` (bwd).
- Why fragile: The `tl.atomic_add(barrier_ptr + t, 0, sem="acquire")` no-op-add is the ONLY way to get an acquire fence on the spin-load — `tl.load` doesn't accept a memory order. Future Triton releases might add `tl.load(sem=…)`, but until then any contributor who "simplifies" the spin-wait to `tl.load(barrier_ptr + t)` or `tl.load(cache_modifier=".cv")` reintroduces non-deterministic drift.
- Safe modification: **Do not touch.** If you must, run `test_triton_forward_persistent_matches_default` and `test_triton_backward_persistent_matches_pytorch` 100× — drift only fires on some CTA scheduling orders and may not reproduce on a single run. The comments in `scan.py:184-203` and DEVELOPMENT.md §"What the agent should NOT do" are explicit warnings.
- Test coverage: `tests/test_triton_scan.py:49-112` tests persistent vs default and persistent vs PyTorch backward. Tolerance `rel < 5e-2` for forward, `rel < 1e-1` for backward. Tight enough to catch the original `.cv` bug (which was ~0.2 absolute), but not exhaustively run under stress.

**Butterfly kernel masking with `H_PAD = next_pow2(H)`:**
- Files: `src/gru_qat/triton_kernels/scan_butterfly.py:182-203` (fwd), `scan_butterfly.py:484-499` (bwd).
- Why fragile: Two interleaved index spaces — dense tensors (h0, gi, out, bh, dh0, dgi, dout, dh_acc, dbh) are sized at `H`, scratch / twiddle / dtwiddle_partial are at `H_PAD`. `mask_h = offs_h < H` is required on every load/store touching dense tensors; missing it produces silent OOB reads or stomps memory. The recent `d8218d4` "butterfly scratch/state OOB at last program" fix is in this exact class of bug.
- Safe modification: When editing the butterfly kernels, identify which tensor a pointer points into (dense `H` vs padded `H_PAD`) and apply the matching mask. Re-run `test_butterfly_dispatch.py` at non-pow2 `H` (the test parametrization covers some but is not exhaustive).
- Test coverage: `tests/test_butterfly_dispatch.py` parametrizes some non-pow2 shapes. Recent OOB fix shipped without an explicit regression test for `B % BLOCK_B != 0` at the partial last tile.

**Monarch kernel autotile + non-pow2 `BLKSZ`:**
- Files: `src/gru_qat/triton_kernels/scan_monarch.py:160-280` (fwd kernel masking), `_pick_tile` in `scan_monarch.py:284-295`.
- Why fragile: Monarch matmul block size `BLKSZ` may be non-pow2 (`9355fc6 scan_monarch: support non-pow2 BLKSZ via pad-to-pow2 + mask`). Tile sizes are auto-picked by `_pick_tile`. Two recent commits (`9e347bc`, `9355fc6`) had to add pad-to-pow2 + mask logic; another tile-size combo could regress.
- Safe modification: When changing `_pick_tile` heuristics, sweep `BLKSZ ∈ {32, 64, 96, 128, 256}` against the PyTorch reference. `tests/test_triton_monarch.py` covers some shapes but not exhaustively.
- Test coverage: `tests/test_triton_monarch.py` has shape sweeps; tolerances are `rel < 5e-3` (fwd vs ref), `rel < 1e-4` (bwd vs ref) — tight enough to catch silent drift.

**`FakeQuantize` granularity dispatch is by subclass, not by flag:**
- Files: `src/gru_qat/quantizers.py:68-225`.
- Why fragile: The SCOPE.md §3 design says granularity is parameterized by `(axis, group_size, symmetric, bits)` flags rather than a class hierarchy. The implementation uses both: configs carry the flags, but actual dispatch goes through three subclasses (`FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`) selected in `make_quantizer`. DEVELOPMENT.md §"What the agent should NOT do" explicitly forbids collapsing them into one class.
- Safe modification: When adding a new granularity, subclass `FakeQuantize` and override `_compute_scale_zp`. Do NOT add an `if/else` branch to an existing subclass.
- Test coverage: `tests/test_quantizers.py` exercises all three concrete classes.

**`bf16 around fake-quant` is a closed experiment:**
- Files: documented in `DEVELOPMENT.md:14-18` and open-question #2.
- Why fragile: Anyone re-evaluating "let's try bf16 around the quant ops to save GEMM cost" will repeat the same fp32↔bf16 cast tax problem that already killed the experiment at our shapes.
- Safe modification: Strict dtype discipline — every fake-quant op preserves input dtype; every internal float op runs fp32 unless the caller opts into autocast. See `CLAUDE.md` "Conventions" section.

**`pre_batch_input=True` changes activation-quant behavior:**
- Files: `src/gru_qat/gru_cell.py:409-434` (`input_projection`).
- Why fragile: With `pre_batch_input=True`, `quant_x` runs ONCE over the whole `[T, B, in]` sequence rather than per-step. For per-tensor dynamic mode this means a single scale across the whole sequence — a meaningful behaviour change vs. per-step quant. Closer to the eventual inference kernel behaviour, but a switch that silently changes accuracy.
- Safe modification: Document in code (already done in the docstring). Tests should cover both `pre_batch_input` values.
- Test coverage: Both paths are exercised but the difference in dynamic-mode scale isn't asserted.

## Scaling Limits

**Persistent kernel grid capped at SM count:**
- Current capacity: `cdiv(B, block_b) * cdiv(H, block_oh) <= sm_count` (e.g. 24 on RTX 2000 Ada). With default `block_b=8, block_oh=128`, supports up to `B*H/(8*128) <= 24` programs.
- Limit: Beyond that, the persistent kernel raises `RuntimeError`. See `scan.py:249-254, 680-685`.
- Scaling path: Larger tile sizes, or fall back to the autotune (non-persistent) path. Documented in raise message.

**Sequence length `T` known at launch:**
- Current capacity: Any T that fits in memory (forward saves `out: [T, B, H]`; backward saves intermediate scratch).
- Limit: T must be known at launch time. Streaming inference (`step(x_t, h)` called by user iteratively) bypasses the Triton kernels entirely and runs the per-step PyTorch path. Listed as open question #5 in `DEVELOPMENT.md`.
- Scaling path: For streaming, accept the per-step slowdown — the kernel is designed for full-sequence training, not streaming.

**`extract_diagonal_factors` etc. consume `cell._hidden_dense`:**
- Files: `src/gru_qat/triton_kernels/scan_diagonal.py:43`, `scan_monarch.py:extract_monarch_factors`, `scan_butterfly.py:61`.
- Reach into private cell state (`_hidden_dense`, `struct_Wh_r`, etc.). If `gru_cell.py` ever renames or restructures the per-gate structured submodules, the kernel extractors break silently (no abstract interface separating them).
- Scaling path: Define a `StructuredHiddenWeights` protocol that all four structured kinds implement (`get_factors() -> tuple[Tensor, ...]`), have `gru_cell.py` expose it.

## Dependencies at Risk

**`torch-structured` is a private repo, no PyPI release:**
- Risk: `pyproject.toml:13-15` documents install via `uv pip install git+https://github.com/LarocheC/torch-structured`. Repo is the user's personal library, not a maintained package. Hash-pinned? No — installed at HEAD.
- Files: `src/gru_qat/structure.py:60-69`, `pyproject.toml:13-15`.
- Impact: A breaking change to `torch_structured.monarch.blockdiag_linear.BlockdiagLinear`, `torch_structured.Butterfly`, or `torch_structured.structured.layers.LDRSubdiagonal` silently breaks structured-mode training. The lazy import keeps dense users isolated, but anyone with `pip install 'gru-qat[structured]'` is exposed.
- Migration plan: Pin a specific commit hash in install instructions. Or vendor the small subset of `torch_structured` actually used (only three classes are referenced).

**`triton>=2.2` is the listed minimum:**
- Risk: `pyproject.toml:11` declares `triton = ["triton>=2.2"]`. Commit `9f00cb4 triton 3.1 compat: move libdevice path, drop unsupported cache modifier` shows the codebase already had to chase API changes between Triton versions. `tl.extra.cuda.libdevice.rint`, `tl.atomic_add(sem=...)`, `cache_modifier=".cv"` are all Triton-internal APIs that have moved or been deprecated in past releases.
- Files: `pyproject.toml:11`; usage of `tl.extra.cuda.libdevice` is widespread across `triton_kernels/*`.
- Impact: A Triton 4.x release could break the kernels. No CI matrix testing across Triton versions.
- Migration plan: Add a pinned upper bound (`triton>=3.0,<4.0`) once breakage is observed. Track Triton release notes for `tl.extra.cuda.libdevice` and `atomic_add(sem=...)` API stability.

**Optional `triton` extra but kernels are imported by default at use site:**
- Risk: `gru_layer.py:221, 231, 245` does inline `from gru_qat.triton_kernels.scan_diagonal import ...` only when `use_triton=True` is active. Good. But `tests/test_triton_*.py` use `pytest.importorskip("triton")` at module top, so on a non-CUDA / non-Triton machine those tests are skipped silently — green CI without ever exercising the Triton path.
- Files: `tests/test_triton_scan.py:12`, `tests/test_triton_monarch.py`, etc.
- Impact: If your CI runner doesn't have a GPU, the Triton tests don't run and you can ship broken kernels.
- Mitigation: Requires GPU CI runner. Documented in `CLAUDE.md` "Triton tests skip automatically when CUDA is unavailable."

## Missing Critical Features

**No int weight export / inference runtime:**
- Problem: SCOPE.md §"What this is not" promises "we produce quantized weights and a reference int kernel suitable for porting to TFLite/ONNX Runtime / a custom embedded runtime, but we don't target those backends directly." No `export_int_weights()` function exists. `TODO(phase=5)` at `gru_cell.py:507-509` says it's "deferred until kernel layout is fixed" — kernel layout has been fixed for multiple structured kinds, so the deferral reason no longer holds.
- Blocks: Any downstream deployment workflow that consumes `gru-qat` output for embedded inference.
- Fix approach: Implement `export_int_weights()` returning a dict per layer with `{ "W_ir": int_tensor, "scale_W_ir": Tensor, "zp_W_ir": Tensor, ... }`. Match the simulator layout (`SCOPE.md §"Relationship to existing codebase"`).

**No LUT sigmoid/tanh for int inference:**
- Problem: SCOPE.md §6 "Sigmoid/tanh stay in float during QAT … the inference kernel will substitute LUTs (Phase 6)." Phase 6 is "Not started" per `DEVELOPMENT.md:164-166`.
- Blocks: True integer-only inference. Currently the Triton kernels still call `tl.sigmoid` / `tl.extra.cuda.libdevice.tanh` in fp32 inside the kernel.
- Fix approach: Phase 6 work. Out of scope for the QAT-training workflow but required for embedded.

**No LSQ (Learnable Step-size Quantization):**
- Problem: `learnable_scale` flag exists in `QuantizerConfig` (`quantizers.py:59`), `TODO(phase=3)` at `ste.py:84` describes the implementation, but no code reads the flag. Activation scales rely on min_max + freeze after calibration.
- Blocks: Better int4 accuracy at aggressive bit widths (LSQ is the standard for int4 PTQ/QAT).
- Fix approach: Implement `LSQRound` autograd Function, add `FakeQuantizeLSQ` subclass, register in `make_quantizer`. Listed as open question #3 in `DEVELOPMENT.md`.

**No LSTM / bidirectional support:**
- Problem: Single-direction, single-layer GRU only. SCOPE.md §"What this is not" calls out LSTM as out-of-scope; bidirectional is also excluded. Stacking two `GRULayer`s for a 2-layer GRU works; bidirectional requires a wrapper.
- Blocks: Anyone needing GRU with reverse direction or LSTM.
- Fix approach: Bidirectional is a thin wrapper. LSTM would be a separate cell (four gates, plus cell state quantizer). Documented as "additive Phase" in SCOPE.md.

**No streaming inference path through Triton:**
- Problem: `GRULayer.forward` requires full sequence. Streaming (`step(x_t, h)` called by user step-at-a-time) works through `GRUCellQuant.forward` but bypasses Triton entirely. Listed as open question #5 in `DEVELOPMENT.md`.
- Blocks: Real-time inference at any throughput.
- Fix approach: A separate `step_triton(x_t, h)` kernel — but the per-launch overhead may exceed the per-step Python overhead for T=1, defeating the point.

## Test Coverage Gaps

**Simulator parity unenforced:**
- What's not tested: SCOPE.md says `FakeQuantize.forward()` must produce bit-identical output to the (external) simulator's `quantize_dequantize()`. `tests/test_simulator_parity.py` doesn't exist; the placeholder skip is in `tests/test_quantizers.py:90`.
- Files: `tests/test_quantizers.py:88-90`.
- Risk: Silent drift between QAT training and the inference simulator the project is supposed to be parity-tested against.
- Priority: High (it's an explicit SCOPE contract).

**Butterfly partial-batch-tile OOB:**
- What's not tested: `B % BLOCK_B != 0` corner. The `d8218d4` fix went in without a parametrized regression test.
- Files: `tests/test_butterfly_dispatch.py`.
- Risk: Re-regression when butterfly masking is touched.
- Priority: Medium.

**Per-channel `min_max` activation observer:**
- What's not tested: No test covers the known-broken global-reduction path in `quantizers.py:135-146`. The bug is untested AND unblocked because no fast path uses per-channel activation min_max.
- Files: `tests/test_quantizers.py`.
- Risk: When someone fixes the observer, there's no test to confirm the fix.
- Priority: Low (until someone wants per-channel activation calibration).

**Persistent kernel barrier under stress:**
- What's not tested: Single-run parity tests (`tests/test_triton_scan.py:49-112`) catch the original `.cv` cache-modifier bug, but a more subtle barrier bug (e.g. a missed release on a rare exit path) wouldn't reliably fire on a single run.
- Files: `tests/test_triton_scan.py`.
- Risk: Future barrier modifications may pass single-shot tests but fail under sustained CTA-scheduling pressure.
- Priority: Low (current pattern is stable, just hard to test).

**Bench scripts are not exercised by `pytest`:**
- What's not tested: `bench/bench_layer.py`, `bench/bench_triton_fwd.py`, `bench/bench_triton_train.py` are runnable scripts; no `test_bench.py` runs them in smoke-test mode (T=1, B=1).
- Files: `bench/*.py`.
- Risk: A breaking change to `GRULayer`'s constructor signature can land green and silently break the bench (which is the canonical source for `DEVELOPMENT.md`'s perf table).
- Priority: Low.

**`Circulant` / `LDR` kinds have no Triton parity test:**
- What's not tested: No `tests/test_triton_circulant.py` or `tests/test_triton_ldr.py` — these kinds don't have Triton kernels yet. `tests/test_structure.py` covers the per-step PyTorch path.
- Files: N/A — kernels don't exist.
- Risk: If someone adds a Triton kernel for these kinds, there's no existing test scaffolding.
- Priority: Low (depends on whether circulant/LDR ever get kernels).

---

*Concerns audit: 2026-05-13*
