# gru-qat — Pluggable QAT and Quantized Inference for GRU

## What this is

A small library that owns the GRU cell so we can apply arbitrary quantization
schemes (per-tensor / per-channel / per-group / fine-grained) at training time
(QAT) and at inference. The cell is manually unrolled in PyTorch so every
quantization insertion point is explicit and replaceable.

The reference path is pure PyTorch with fake-quant. The accelerated path is
Triton (Phase 5+). Both share the same `Quantizer` interface, so a quantization
recipe defined for QAT can be evaluated against the inference kernel without
re-specifying it.

## What this is not

- Not a wrapper around `cudnn` or `torch.nn.GRU`. cuDNN's GRU is a closed
  fused kernel with no hooks; we cannot insert fake-quant into it. We replace
  it.
- Not a general RNN framework. Only GRU. LSTM is out of scope; if needed it's
  an additive Phase.
- Not a full PTQ toolkit. Calibration support exists for activation
  scales (needed because hidden state ranges drift), but weight PTQ
  (GPTQ/AWQ) is out of scope — those are upstream of this library and produce
  weights we consume.
- Not a deployment runtime. We produce quantized weights and a reference int
  kernel suitable for porting to TFLite/ONNX Runtime / a custom embedded
  runtime, but we don't target those backends directly.

## Key design decisions (and the reasoning)

### 1. Manual unroll, not `nn.GRU`

`torch.nn.GRU` dispatches to cuDNN. We write `GRUCellQuant` and a `GRULayer`
that loops over time in Python. Slower at fp32, but every fake-quant insertion
is a line of code we can change.

### 2. `Quantizer` is an `nn.Module`, not a function

It needs to hold state: learnable step size (LSQ), running min/max observers,
or frozen calibrated scales. Subclasses differ only in how they compute
`(scale, zero_point)` from the input tensor and their own state.

### 3. Granularity is a parameter, not a class hierarchy

`FakeQuantize(axis=None)` → per-tensor.
`FakeQuantize(axis=0)` → per-channel (output rows of weights).
`FakeQuantize(axis=0, group_size=64)` → per-group along axis 0.

This keeps the kernel-side dispatch on a small number of orthogonal flags
(`axis`, `group_size`, `symmetric`, `bits`) rather than on subclass identity.

### 4. Gates are split, not fused

cuDNN concatenates `W_ir | W_iz | W_in` into one GEMM. We keep them separate
so each gate can carry its own activation scale. Per-channel weight
quantization works either way; per-tensor weight quantization is much better
with split gates because the three gate matrices have different value
distributions. Fused-gate optimization is a Phase 5 toggle.

### 5. Hidden state quantization is first-class

The hidden state is both the output of step `t` and an input to step `t+1`. Its
range can drift over long sequences and dominates accuracy loss in
streaming inference. We quantize it explicitly through a dedicated
`hidden_quantizer` whose scale is calibrated (not dynamic) for inference.

### 6. Sigmoid/tanh stay in float during QAT

Real int inference uses LUTs; for QAT they are not where accuracy is lost,
and float nonlinearities keep gradients clean. The inference kernel will
substitute LUTs (Phase 6).

## Out-of-scope / explicit non-goals

- LSTM, vanilla RNN, bidirectional layers
- Mixed-precision (fp16 act × int4 weight) GEMM in the reference path —
  reference is fake-quant only; mixed-precision lives in Triton/CUTLASS
- Distributed training hooks
- Quantization of bias terms (kept fp32 — standard practice and doesn't
  affect accuracy meaningfully)
- ONNX export (downstream concern — we provide weight + scale + zp tensors
  in a documented layout)

## Success criteria

- **Parity**: `GRUCellQuant` with no-op quantizers matches `torch.nn.GRUCell`
  to `< 1e-5` on random inputs (validates the manual unroll).
- **QAT convergence**: on a synthetic regression task, INT8 per-channel weight
  + per-tensor activation QAT trains to within 1% of fp32 baseline.
- **Quantizer swap**: switching a single argument (`group_size=None → 64`)
  changes the scheme end-to-end with no other code changes.
- **Triton ≥ 0.7× cuDNN throughput** at hidden_size ≥ 256, batch ≥ 16, fp16
  weights. (Target, not blocking.)

## Relationship to existing codebase

This library reuses the quantization primitives from the inference simulator
(`quant_primitives.py`, `quantized_ops.py`). Specifically:

- `FakeQuantize.forward()` in QAT mode must produce bit-identical output to
  the simulator's `quantize_dequantize()` for the same `(scale, zp, qmin,
  qmax)`. Tested in `tests/test_simulator_parity.py` (Phase 2).
- The eventual inference path emits weights in the simulator's layout so
  `compare.py` can be reused.
