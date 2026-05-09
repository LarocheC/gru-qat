# gru-qat

Pluggable QAT and quantized inference for GRU.

- **Why**: cuDNN's GRU is a closed kernel; we cannot insert fake-quant. To
  do QAT with arbitrary quantization granularities (per-channel, per-group,
  fine-grained int4) we own the cell.
- **What**: a manually-unrolled GRU cell where every quantizable quantity
  is a `FakeQuantize` module that can be swapped without touching the cell
  code. Reference path is pure PyTorch; accelerated path is Triton (Phase 5).

## Read first

1. [`SCOPE.md`](./SCOPE.md) — what's in, what's out, key design decisions.
2. [`DEVELOPMENT.md`](./DEVELOPMENT.md) — phased plan, file map, tests,
   upgrade pathways. Read this before writing code.

## Quick start

```bash
uv sync
pytest tests/test_ste.py tests/test_quantizers.py    # Phase 1
pytest tests/test_parity.py                          # Phase 2
pytest tests/test_qat_smoke.py                       # Phase 3+4
```

## Status

Skeleton with working contracts and inline TODOs marking each phase.
Phase 1 and 2 should pass once subclass bodies are completed; Phase 3+4
require additional wiring described in `DEVELOPMENT.md`.
