# DEVELOPMENT.md — Agent Handoff

This document is for the implementing agent. Read `SCOPE.md` first.

## Working agreement

- Environment: `uv` (see `pyproject.toml`). `uv sync` to bootstrap.
- Strict typing: all public functions annotated; `mypy --strict` on the
  `gru_qat` package.
- Strict dtype discipline: never silently upcast. Every fake-quant op
  preserves input dtype. Every internal float op runs in `torch.float32`
  unless the caller has explicitly opted into fp16/bf16.
- Each phase has an exit test in `tests/`. Do not start phase N+1 until
  phase N tests pass.
- One PR per phase. Commit message format: `phase N: <verb> <object>`.

## File map

```
src/gru_qat/
  __init__.py          public API surface
  ste.py               Straight-Through Estimator autograd functions
  quantizers.py        FakeQuantize module + observers
  calibration.py       observer collection / freezing for inference
  gru_cell.py          GRUCellQuant — single-step, fully unrolled
  gru_layer.py         GRULayer — multi-step, hidden-state carry
  triton_kernels/
    __init__.py        (Phase 5) Triton GRU cell

tests/
  test_ste.py
  test_quantizers.py
  test_parity.py            cell parity vs torch.nn.GRUCell at fp32
  test_simulator_parity.py  fake-quant matches inference simulator
  test_qat_smoke.py         end-to-end QAT trains on toy task
  test_triton_parity.py     (Phase 5)
```

## Phased plan

### Phase 0 — bootstrap (target: 1 hour)

1. `uv sync` works.
2. `pytest -q` runs (zero tests pass; many skipped).
3. CI lint passes.

**Exit test**: `pytest --collect-only` succeeds without import errors.

### Phase 1 — STE and quantizer primitives (target: 1 day)

Implement `ste.py` and `quantizers.py`. The skeletons in this repo show the
contracts; fill in the bodies.

Required quantizers, all sharing the `FakeQuantize` base:
- per-tensor symmetric / asymmetric
- per-channel symmetric (axis configurable)
- per-group symmetric (group_size along axis)

For each, support `bits ∈ {2, 3, 4, 8}`.

Observer modes:
- `dynamic` — scale recomputed each forward from current tensor (training)
- `min_max` — running min/max stats updated during forward
- `frozen` — uses stored `scale`/`zero_point`, no stats update (inference)

**Exit tests**:
- `test_quantizers.py::test_roundtrip_no_clip` — values inside qrange
  reconstruct with error ≤ scale/2.
- `test_quantizers.py::test_per_channel_independent` — scales per channel
  differ when input rows have different magnitudes.
- `test_quantizers.py::test_group_axis` — group_size=K returns ceil(N/K)
  scales along axis.
- `test_simulator_parity.py` — bit-identical to existing simulator's
  `quantize_dequantize()` for matched configs.

### Phase 2 — GRU cell with fp32 parity (target: 1 day)

Implement `GRUCellQuant.forward()` with all quantizers set to `Identity`.
Validate it matches `torch.nn.GRUCell` exactly.

**Exit test**: `test_parity.py::test_cell_matches_torch_gru_cell` — max abs
diff < 1e-5 over 100 random `(input, hidden, weight)` triples, including
edge cases (h=0, x=0, large magnitudes).

This phase exists to lock down the unroll math before we layer
quantization on top. If parity fails here, every later test is meaningless.

### Phase 3 — fake-quant insertion in the cell (target: 2 days)

Replace `Identity` quantizers with real `FakeQuantize` modules. Insertion
points (mark each with a comment in `gru_cell.py`):

1. Input activation `x_t`
2. Hidden state `h_{t-1}` (read side)
3. Weights `W_ir, W_iz, W_in, W_hr, W_hz, W_hn`
4. (Optional, gated by flag) gate pre-activations before sigmoid/tanh
5. Output hidden state `h_t` (write side)

Bias remains fp32. Sigmoid/tanh remain fp32.

**Exit tests**:
- `test_qat_smoke.py::test_no_op_quant_matches_fp32` — when all quantizers
  are pass-throughs, output == fp32 path.
- `test_qat_smoke.py::test_int8_per_channel_close_to_fp32` — INT8
  per-channel weight quant + per-tensor activation quant on a single cell
  evaluation: max relative error < 5%.
- `test_qat_smoke.py::test_swap_granularity_no_code_change` — same model,
  swap weight quantizer from per-channel to per-group(64) by changing one
  factory argument; both run, both produce sensible output.

### Phase 4 — multi-step layer + calibration (target: 2 days)

`GRULayer` wraps the cell, loops over time. Two paths:
- training: dynamic scales for activations, learnable scales for
  weights (LSQ).
- calibration: observers running on real data to record activation ranges,
  then frozen for inference.

Hidden state quantizer **must** transition from dynamic (training) to
frozen (inference); otherwise the inference kernel cannot be written
because step `t+1` doesn't know step `t`'s scale.

**Exit tests**:
- `test_qat_smoke.py::test_layer_trains_to_baseline` — synthetic seq2seq
  regression. INT8 QAT model converges to within 1% MSE of fp32 baseline.
- `test_qat_smoke.py::test_calibration_freezes_scales` — after
  `calibrate(loader)` then `freeze()`, scales stop updating.

### Phase 5 — Triton kernel (target: 4 days)

See `triton_kernels/__init__.py` for the kernel-level contract. The cell
is one `triton.jit` function consuming quantized weight tiles + fp16/bf16
activations and producing fp16/bf16 hidden state.

Quantization scheme is selected by the *Python wrapper* that picks the
kernel variant; the kernel itself is parameterized over `BITS`,
`GROUP_SIZE`, `SYMMETRIC` as `tl.constexpr`.

Variants in priority order:
1. fp16 weights, fp16 acts (baseline; matches cuDNN regime, validates loop
   structure)
2. int8 per-channel weights, fp16 acts
3. int4 per-group weights, fp16 acts (the actual prize)
4. int8 weights, int8 acts (rare but useful for embedded)

**Exit tests**:
- `test_triton_parity.py` — Triton output matches the PyTorch fake-quant
  path within fp16 tolerance for each variant.
- benchmark script reports throughput vs cuDNN at `(batch, hidden) ∈
  {(1,128), (16,256), (64,512)}`.

### Phase 6 — int activations and LUT nonlinearities (optional)

For full integer inference. Out of scope for the QAT side; needed only for
embedded deployment.

## Upgrade pathways

The skeleton is designed so each of these is a *localized* change.

### Adding a new quantization scheme

1. Subclass `FakeQuantize` in `quantizers.py`. Override `_compute_scale_zp`.
2. Add a factory entry in `quantizers.QUANTIZER_FACTORIES`.
3. Pass the factory name into `GRUCellQuant(weight_quantizer=...)`.

No changes to `gru_cell.py` or `gru_layer.py`.

### Switching from STE-round to a different gradient estimator

Edit `ste.py`. The round op is wrapped in `STERound.apply`; replace it
with `LSQRound.apply` or similar. Quantizers that need a learnable step
size already pass through `STERound`; gradient flow is contained.

### Fusing the three input gates into one GEMM

In `gru_cell.py`, replace the three `F.linear(x, W_i*)` calls with a single
`F.linear(x, W_i)` where `W_i = cat([W_ir, W_iz, W_in], dim=0)`. Then split
the output along dim 1. Per-channel weight quant survives this trivially
because the concat is along the per-channel axis. Per-tensor weight quant
*does not* — guard with `assert weight_quantizer.axis is not None` if
you take this path.

### Targeting a different hardware backend (CUTLASS, IREE, embedded)

The Triton kernel is the reference for the integer math. Port the inner
loop. The Python `GRULayer` doesn't change; only the
`triton_kernels/__init__.py` dispatcher does.

### Adding LSTM later

Mostly copy-paste of `gru_cell.py` with four gates instead of three and a
cell state quantizer. The quantizer infrastructure is unchanged.

## What the agent should NOT do

- Do not rewrite the existing simulator's `quant_primitives.py`. Import
  from it; match its conventions.
- Do not optimize the PyTorch reference path. Its job is to be slow,
  obvious, and correct. Speed lives in Triton.
- Do not add quantization to bias, sigmoid, or tanh in the reference path
  without an explicit ticket. These are deliberate omissions.
- Do not collapse `FakeQuantize` granularities into a single class with
  `if/else` branches. Subclassing keeps the dispatch flat and the kernel
  variants tractable.

## Open questions for the human (Clément)

These are decisions to make before Phase 3, not before Phase 0:

1. LSQ vs PACT vs static observers for activation scales? Default in
   skeleton is min-max observer with optional LSQ flag.
2. Hidden state bits — same as activations, or separate? Skeleton allows
   separate; defaulting them to equal would simplify the API.
3. Is bias-fp32 acceptable for the target deployment, or do we need
   bias-int32 (quantized to weight_scale × act_scale)? Affects export, not
   QAT. Defer to Phase 6.
4. Streaming inference frame size — fixed or variable? Fixed simplifies
   the Triton kernel substantially.
