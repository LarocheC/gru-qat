<!-- refreshed: 2026-05-13 -->
# Architecture

**Analysis Date:** 2026-05-13

## System Overview

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                            Public API surface                             │
│                          `src/gru_qat/__init__.py`                        │
│  GRUCellQuant · GRULayer · FakeQuantize* · QuantRecipe · StructureConfig  │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              GRULayer                                     │
│                       `src/gru_qat/gru_layer.py`                          │
│   forward(x, h0) → time-loop driver                                       │
│   `_forward_fast_dispatch` picks Triton variant from StructureConfig.kind │
│   `calibrate()` temporarily disables use_triton so per-step path runs     │
└──────┬─────────────────────────────────────────────────────────┬──────────┘
       │ reference path (per-step)                               │ fast path (one launch / T)
       ▼                                                         ▼
┌──────────────────────────┐        ┌────────────────────────────────────────┐
│      GRUCellQuant        │        │       triton_kernels/scan*.py          │
│  `src/gru_qat/gru_cell.py`│       │   one persistent kernel per (variant,  │
│  step / step_with_gi /   │        │   fwd|bwd); covers all T timesteps     │
│  step_structured         │        │   ┌────────────┬──────────────────┐    │
│  6 weight quantizers +   │        │   │ scan.py    │ dense            │    │
│  3 activation quantizers │        │   │ scan_diag  │ diagonal         │    │
│  + optional 3 gate quants│        │   │ scan_monarch│ block-diag      │    │
└──────┬───────────────────┘        │   │ scan_butter│ butterfly        │    │
       │                            │   └────────────┴──────────────────┘    │
       ▼                            └────────────────────────────────────────┘
┌──────────────────────────┐        ┌────────────────────────────────────────┐
│      FakeQuantize        │        │           STE primitives               │
│  `src/gru_qat/quantizers.py`│ ←──│        `src/gru_qat/ste.py`            │
│  Identity / PerTensor /  │        │  STERound · STEClamp · fake_quant_ste  │
│  PerChannel / PerGroup   │        └────────────────────────────────────────┘
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐        ┌────────────────────────────────────────┐
│ structured-matrix layers │        │           Calibration                  │
│ `src/gru_qat/structure.py`│       │     `src/gru_qat/calibration.py`       │
│ _DiagonalLinear,         │        │  calibrate(module, loader, n_batches)  │
│ _CirculantLinear,        │        │  freeze_all(module)                    │
│ BlockdiagLinear (monarch)│        └────────────────────────────────────────┘
│ Butterfly, LDR (extern)  │
└──────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public API | Re-exports the user-facing classes/types | `src/gru_qat/__init__.py` |
| `GRULayer` | Multi-step driver; chooses reference vs. Triton fast path; owns calibration lifecycle | `src/gru_qat/gru_layer.py` |
| `GRUCellQuant` | Manually-unrolled single GRU step with explicit fake-quant insertion points | `src/gru_qat/gru_cell.py` |
| `CellWeights` | Bag of fake-quantized weights computed once per forward, reused across T steps | `src/gru_qat/gru_cell.py` |
| `FakeQuantize` hierarchy | Pluggable observers + quantize/dequantize; per-tensor / per-channel / per-group / Identity | `src/gru_qat/quantizers.py` |
| `QuantRecipe` / `QuantizerConfig` / `PRESETS` | Declarative bundle of quantizer configs for a recipe | `src/gru_qat/quantizers.py` |
| STE primitives | `STERound`, `STEClamp`, `fake_quant_ste` — sole place gradient is faked | `src/gru_qat/ste.py` |
| `StructureConfig` + `make_structured_linear` | Factory for diagonal / Monarch / circulant / butterfly / LDR `nn.Linear` substitutes; lazy-imports `torch_structured` | `src/gru_qat/structure.py` |
| `calibrate` / `freeze_all` | Switch activation quantizers to `min_max`, run forwards, then lock scales | `src/gru_qat/calibration.py` |
| Dense Triton scan | Persistent + autotune fwd/bwd kernels; `gru_scan_persistent`, `gru_scan` | `src/gru_qat/triton_kernels/scan.py` |
| Diagonal Triton scan | Elementwise-recurrence persistent kernel (no cross-CTA barrier, `h` in registers) | `src/gru_qat/triton_kernels/scan_diagonal.py` |
| Monarch Triton scan | Block-diagonal matmul per gate per timestep; persistent | `src/gru_qat/triton_kernels/scan_monarch.py` |
| Butterfly Triton scan | log_H stages of strided 2×2 mixing per gate; persistent | `src/gru_qat/triton_kernels/scan_butterfly.py` |
| Triton kernels `__init__` | Phase-5 design notes; `is_available()` helper; placeholders | `src/gru_qat/triton_kernels/__init__.py` |

## Pattern Overview

**Overall:** Two parallel execution paths sharing one `Quantizer` interface.

- **Reference path** (pure PyTorch, deliberately slow): time-unrolled GRU cell with `FakeQuantize` modules at every documented insertion point. Used for fp32 parity testing, QAT correctness, calibration (observers must fire per step), and as the executable spec for the Triton kernels.
- **Fast path** (Triton persistent kernels): one launch covers all T timesteps for fwd or bwd. Selected at `GRULayer` construction via `use_triton`/`structure_hidden`. Reads frozen per-tensor symmetric scales from the matching cell quantizers and applies fake-quant inside the kernel.

**Key Characteristics:**
- Manual unroll over `nn.GRU` so every quantization insertion point is explicit and replaceable (see `SCOPE.md` §1).
- `FakeQuantize` is an `nn.Module` — holds observer / frozen-scale state, not a pure function.
- Granularity is a parameter `(axis, group_size, symmetric, bits)`, not a class hierarchy — subclass identity controls only `_compute_scale_zp`.
- Calibration lifecycle is explicit: dynamic → min_max (calibration) → frozen (inference).
- Structured hidden weights are orthogonal to quantization; `StructureConfig.kind` swaps the H×H GEMM without touching the cell math.
- Triton fast path requires fused-gate layout, dense input side, and (for in-kernel fake-quant) frozen per-tensor symmetric hidden quantizers.

## Layers

**Public API:**
- Purpose: Stable user-facing entry points re-exported from one module
- Location: `src/gru_qat/__init__.py`
- Contains: `GRUCellQuant`, `GRULayer`, `FakeQuantize*`, `QuantRecipe`, `QuantizerConfig`, `Identity`, `PRESETS`, `STERound`, `STEClamp`, `StructureConfig`, `make_structured_linear`
- Depends on: every sibling module under `gru_qat/`
- Used by: tests/, bench/, downstream callers

**Layer / multi-step driver:**
- Purpose: Iterate the cell over the time dimension; dispatch to reference vs. Triton fast path
- Location: `src/gru_qat/gru_layer.py`
- Contains: `GRULayer.forward`, `_forward_fast_dispatch`, `calibrate`, `freeze`, `_extract_h_quant_params`
- Depends on: `gru_cell`, `quantizers`, `structure`, `triton_kernels.scan_*`
- Used by: user code, tests, bench harness

**Cell / single step:**
- Purpose: Manually-unrolled GRU step with all fake-quant insertion points
- Location: `src/gru_qat/gru_cell.py`
- Contains: `GRUCellQuant`, `CellWeights`, `step`, `step_with_gi`, `step_structured`, `quantize_weights`, `quantize_input_weights`, `freeze_quantizers`
- Depends on: `quantizers`, `structure`
- Used by: `GRULayer`, tests

**Quantization:**
- Purpose: Pluggable fake-quant modules + STE autograd
- Location: `src/gru_qat/quantizers.py`, `src/gru_qat/ste.py`
- Contains: `FakeQuantize` base, `Identity`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`, `QuantizerConfig`, `QuantRecipe`, `PRESETS`, `make_quantizer`, `STERound`, `STEClamp`, `fake_quant_ste`
- Depends on: `torch.autograd`, `torch.nn`
- Used by: `gru_cell`, `gru_layer`, `calibration`

**Structured matrices:**
- Purpose: Drop-in `nn.Linear` substitutes for the H×H hidden GEMM
- Location: `src/gru_qat/structure.py`
- Contains: `StructureConfig`, `make_structured_linear`, `_DiagonalLinear`, `_CirculantLinear`, `_ButterflyLinear`, `_LDRLinear`
- Depends on: `torch_structured` (lazy, optional) for Monarch / Butterfly / LDR; local impls for Diagonal / Circulant
- Used by: `gru_cell`, Triton extract_* helpers

**Calibration:**
- Purpose: Run forwards in observer mode and freeze activation scales
- Location: `src/gru_qat/calibration.py`
- Contains: `calibrate(module, loader, n_batches)`, `freeze_all(module)`
- Depends on: `quantizers.FakeQuantize`
- Used by: `GRULayer.calibrate`, user code post-training

**Triton kernels:**
- Purpose: Persistent multi-step fwd/bwd kernels for each structured kind that has Triton support
- Location: `src/gru_qat/triton_kernels/scan.py` (dense), `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py`
- Contains: `@triton.jit` fwd/bwd kernels, `torch.autograd.Function` wrappers, `gru_scan*`, `extract_*_factors` helpers
- Depends on: `triton`, CUDA, frozen-scale buffers on the cell
- Used by: `GRULayer._forward_fast_dispatch`, dedicated Triton tests

## Data Flow

### Primary Request Path — reference (per-step) forward

1. User calls `GRULayer.forward(x, h0)` (`src/gru_qat/gru_layer.py:139`).
2. `batch_first` transpose if needed; allocate `h0` if `None`.
3. Branch on `self.use_triton and x.is_cuda`. Reference branch continues here; fast branch jumps to `_forward_fast_dispatch`.
4. Choose the per-step body (`step` / `step_with_gi` / `step_structured`), optionally wrapped by `torch.compile` (`gru_layer.py:127-137`).
5. (Dense) `cell.quantize_weights()` runs all 6 weight FakeQuantize modules once and returns a `CellWeights` bag (`gru_cell.py:240`).
6. (Optional, `pre_batch_input=True`) `cell.input_projection(x, w)` runs `quant_x` + one `F.linear` over `[T, B, in]` producing `gi: [T, B, 3H]` (`gru_cell.py:409`).
7. For `t in range(T)`: call the body — quant_x → F.linear (input+hidden) → optional gate_act fake-quant → sigmoid/tanh in fp32 → `(1-z)*n + z*h` → `quant_h_out` (`gru_cell.py:351`).
8. Stack outputs, reverse `batch_first` transpose, return `(out, h_T)`.

### Fast Path — Triton dispatch

1. `GRULayer.forward` detects `use_triton and x.is_cuda` and calls `_forward_fast_dispatch(x, h0)` (`gru_layer.py:202`).
2. `xq = self.cell.quant_x(x)` runs once on the full `[T, B, in]` tensor.
3. `Wi_cat, bi_cat = cell.quantize_input_weights()` concatenates the three input-side weights along axis 0 (`gru_cell.py:274`).
4. `gi = F.linear(xq, Wi_cat, bi_cat)` produces `[T, B, 3H]` via cuBLAS.
5. `_extract_h_quant_params` pulls `(scale, qmin, qmax)` from `cell.quant_h_in` / `cell.quant_h_out` if they are frozen + per-tensor + symmetric; otherwise `None` disables in-kernel fake-quant (`gru_layer.py:28`).
6. Dispatch on `self._dispatch_kind` (`"diagonal" | "monarch" | "butterfly"`): call `extract_<kind>_factors(cell)` and `gru_scan_<kind>(gi, h0, Wh_*, bh_cat, h_in_quant, h_out_quant)`.
7. The corresponding `triton_kernels/scan_<kind>.py` autograd `Function` launches a single persistent kernel for the whole T-step recurrence (fwd) and one matching kernel for the bwd half.
8. Return `(out, out[-1])`, optional `batch_first` transpose.

### Quantization data flow inside a step

1. `quant_x(x_t)` → quantized input activation.
2. `quant_h_in(h_{t-1})` → quantized hidden activation (only on the matmul side; the carry side uses raw `h` so fp32 parity holds when quantizers are Identity).
3. For each gate g ∈ {r,z,n}: `quant_W_ig(W_ig)`, `quant_W_hg(W_hg)` (dense) or `quant_struct_W*_g(struct_W*_g(.))` (structured) → quantized weights.
4. F.linear contributions sum into a fp32 gate pre-activation.
5. Optional `quant_gate_g` on the pre-activation (Identity by default).
6. `sigmoid` / `tanh` run in fp32 (intentional — bias / sigmoid / tanh are not quantized in the reference path).
7. `h_new = (1-z)*n + z*h_old` in fp32.
8. `quant_h_out(h_new)` → quantized write-side hidden.

### Calibration flow

1. User trains in `dynamic` observer mode (default activation quantizer mode).
2. User calls `GRULayer.calibrate(loader, n_batches)` (`gru_layer.py:269`). Wrapper temporarily sets `self.use_triton = False` so per-step quantizers actually fire.
3. `calibration.calibrate(...)` collects activation FakeQuantize submodules by leaf name, switches each to `mode="min_max"`, resets running stats, runs `n_batches` forwards in `module.eval()` (`calibration.py:31`).
4. User calls `GRULayer.freeze()` → `cell.freeze_quantizers()` → every FakeQuantize transitions `min_max → frozen` (storing `(scale, zero_point)` from running min/max). After this, fast-path dispatch reads stable scales out of `cell.quant_h_in/out`.

**State Management:**
- Quantizer state lives in `FakeQuantize` buffers: `scale`, `zero_point`, `running_min`, `running_max`, `_initialized` (Python attribute).
- Cell weight tensors are `nn.Parameter`s; structured layers hold their own parameter sets.
- `GRULayer.use_triton` is a Python attribute that calibration toggles transiently.
- No global mutable state.

## Key Abstractions

**`FakeQuantize`:**
- Purpose: Pluggable fake-quant op with observer / frozen-scale state
- Examples: `src/gru_qat/quantizers.py:68` (base), `:154` (Identity), `:169` (PerTensor), `:178` (PerChannel), `:192` (PerGroup)
- Pattern: Abstract base with one abstract method `_compute_scale_zp(x)`. Subclasses differ only in scale/zp derivation. Observer modes (`dynamic`/`min_max`/`frozen`) are a config flag, not a subclass.

**`QuantRecipe` / `QuantizerConfig`:**
- Purpose: Declarative bundle that builds all required quantizers
- Examples: `src/gru_qat/quantizers.py:265` (`QuantRecipe`), `:45` (`QuantizerConfig`), `:284` (`PRESETS["fp32" | "int8_per_channel" | "int4_per_group_64"]`)
- Pattern: Dataclasses with `field(default_factory=...)`. `make_quantizer(config)` is the factory entry point; `bits >= 32` short-circuits to `Identity`.

**`CellWeights`:**
- Purpose: Hoist weight quantization out of the time loop
- Examples: `src/gru_qat/gru_cell.py:38`
- Pattern: Dataclass holding the six fake-quantized weight tensors plus optional `Wi_cat`/`Wh_cat`/`bi_cat`/`bh_cat` for fused-gate layout.

**`StructureConfig` + `make_structured_linear`:**
- Purpose: Swap the H×H hidden (or input) GEMM for an `O(n log n)` or `O(n)` parameterization
- Examples: `src/gru_qat/structure.py:34` (`StructureConfig`), `:118` (`make_structured_linear`), `:177` (`_DiagonalLinear`)
- Pattern: Dataclass-driven factory. Kinds: `dense`, `diagonal`, `monarch`, `circulant`, `butterfly`, `ldr`. `torch_structured` is lazy-imported only when a non-local kind is requested.

**STE wrappers:**
- Purpose: Sole location where the QAT gradient story is faked
- Examples: `src/gru_qat/ste.py:15` (`STERound`), `:32` (`STEClamp`), `:59` (`fake_quant_ste`)
- Pattern: `torch.autograd.Function` subclasses with identity / clipped-identity backward. `fake_quant_ste(x, scale, zp, qmin, qmax)` is the canonical building block.

**Triton autograd wrappers:**
- Purpose: Bridge a `triton.jit` kernel pair (fwd, bwd) to PyTorch autograd
- Examples: `src/gru_qat/triton_kernels/scan.py:1518` (`GRUScanFunction`), `:1589` (`GRUScanPersistentFunction`), `scan_diagonal.py:629` (`GRUScanDiagonalFunction`)
- Pattern: `torch.autograd.Function` that saves `(gi, h0, Wh, bh, out)` plus `h_in_quant`/`h_out_quant` tuples; `gru_scan_<kind>(...)` is the public `.apply` wrapper.

## Entry Points

**`GRULayer(...)`:**
- Location: `src/gru_qat/gru_layer.py:49`
- Triggers: User construction in training / inference code
- Responsibilities: Build the cell, decide fast-path eligibility (`structure_input is None and kind in {diagonal, monarch, butterfly} and gate_layout == 'fused'`), optionally wrap per-step body in `torch.compile(mode="default")`.

**`GRULayer.forward(x, h0=None)`:**
- Location: `src/gru_qat/gru_layer.py:139`
- Triggers: Standard PyTorch forward
- Responsibilities: Time-loop or Triton dispatch; returns `(out, h_T)`.

**`GRULayer._forward_fast_dispatch(x, h0)`:**
- Location: `src/gru_qat/gru_layer.py:202`
- Triggers: `self.use_triton and x.is_cuda`
- Responsibilities: Pre-batch input projection, extract structured factors + frozen scales, call the matching `gru_scan_<kind>`.

**`GRUCellQuant.step / step_with_gi / step_structured`:**
- Location: `src/gru_qat/gru_cell.py:351`, `:436`, `:302`
- Triggers: Per-step body inside the layer's time loop
- Responsibilities: One GRU step with all fake-quant insertion points wired.

**`gru_scan_persistent` / `gru_scan_diagonal` / `gru_scan_monarch` / `gru_scan_butterfly_triton`:**
- Location: `src/gru_qat/triton_kernels/scan.py:1624`, `scan_diagonal.py:661`, `scan_monarch.py` (`gru_scan_monarch`), `scan_butterfly.py` (`gru_scan_butterfly_triton`)
- Triggers: Called from `_forward_fast_dispatch`
- Responsibilities: Wrap `torch.autograd.Function` calls that launch the persistent fwd/bwd Triton kernels.

**`calibrate(module, loader, n_batches)` / `freeze_all(module)`:**
- Location: `src/gru_qat/calibration.py:31`, `:129`
- Triggers: Post-QAT-training, before export
- Responsibilities: Switch activation quantizers to `min_max`, run forwards, then freeze scales.

## Architectural Constraints

- **Single-layer, single-direction, GRU only.** Stacking, bidirectionality, and LSTM are out of scope (`SCOPE.md` non-goals).
- **Streaming inference bypasses Triton.** Triton kernels require T at launch time; `cell.step(x_t, h)` works for streaming but loses the fast path.
- **Fast-path eligibility is strict:** input side must be dense, gate layout must be `"fused"`, hidden structure must be one of `{diagonal, monarch, butterfly}`. Any other configuration falls back to the per-step PyTorch path (`gru_layer.py:100`).
- **`pre_batch_input=True` requires `gate_layout="fused"`** and dense input side (`gru_layer.py:65`, `:80`).
- **Fused gate layout requires `recipe.weight.axis=0`** — per-tensor weight quant under fused gates would silently share one scale across all three gate matrices (`gru_cell.py:106`).
- **In-kernel fake-quant requires frozen + per-tensor + symmetric** hidden quantizers (`gru_layer.py:28`). Any other state disables it without error.
- **Cross-CTA visibility in persistent kernels uses release/acquire `atomic_add`.** `tl.load(cache_modifier=".cv")` is NOT a fence substitute and was empirically observed to produce ~0.2 absolute drift before fixed (`DEVELOPMENT.md` §Phase 5; `triton_kernels/scan.py:184-200`).
- **Persistent forward grid must fit on the GPU's SMs concurrently** — spin-wait deadlocks if the scheduler can't run all CTAs at once (`scan.py:70-73`).
- **bf16 around fake-quant was tried and dropped.** Strict dtype discipline: every fake-quant op preserves input dtype; internal float ops run in fp32 unless caller opts into autocast (`DEVELOPMENT.md` §Working agreement).
- **Threading:** single-stream, single-process; Triton kernel CTAs are the only intra-kernel parallelism.
- **Global state:** none; no module-level singletons.
- **Circular imports:** none observed. `calibration.py` imports `quantizers` lazily inside the function body to keep `__init__.py` clean.
- **`torch_structured` is an optional dependency.** Lazy-imported in `structure.py:60`; dense-only users don't need it.

## Anti-Patterns

### Re-quantizing weights inside the time loop

**What happens:** A naive multi-step driver would call the six weight FakeQuantize modules on every timestep.
**Why it's wrong:** Weights are invariant across time. With T=64, six quantizer calls × 64 steps = 384 redundant module calls per forward; dominates int8 training cost.
**Do this instead:** Hoist with `w = self.cell.quantize_weights()` (or `quantize_input_weights()` in fast path) once per forward and reuse — see `gru_layer.py:183` and `gru_cell.py:240`.

### Optimizing the reference (PyTorch) path

**What happens:** Adding fused kernels, custom autograd, or compile tricks inside `gru_cell.py`.
**Why it's wrong:** The reference path's job is to be slow, obvious, and correct — it's the executable spec for the Triton kernels and the parity oracle for `< 1e-5` against `nn.GRUCell`.
**Do this instead:** Add the optimization as a new Triton kernel variant under `src/gru_qat/triton_kernels/` and a new dispatch branch in `GRULayer._forward_fast_dispatch` (`DEVELOPMENT.md` §"What the agent should NOT do").

### Collapsing `FakeQuantize` granularities into one class with if/else

**What happens:** A single `FakeQuantize` class branches on `axis is None`, `group_size is None`, etc.
**Why it's wrong:** Kernel-side dispatch lives on these flags too; flattening the hierarchy moves the branching to runtime and makes the Triton variant matrix harder to reason about.
**Do this instead:** Subclass `FakeQuantize` and override `_compute_scale_zp`. Register via `make_quantizer` (`quantizers.py:236`).

### Quantizing bias, sigmoid, or tanh in the reference path

**What happens:** Adding `quant_b_*` modules or replacing `torch.sigmoid`/`torch.tanh` with quantized stand-ins inside `gru_cell.py`.
**Why it's wrong:** Deliberate omissions. Bias-fp32 doesn't meaningfully affect accuracy; sigmoid/tanh quantization is a deployment concern (LUTs) deferred to Phase 6.
**Do this instead:** File a ticket; treat LUT swap-in as a Phase 6 deployment task, not a QAT change.

### Using `tl.load(cache_modifier=".cv")` as a cross-CTA fence

**What happens:** Relaxed `atomic_add` for the barrier + a `.cv`-modified load to "see" the increment.
**Why it's wrong:** `.cv` invalidates an L1 line but does not provide the acquire semantics required for cross-CTA visibility of *data* writes performed before the counter increment. Non-deterministic ~0.2 absolute drift observed on `out[t>=1]`.
**Do this instead:** `tl.atomic_add(barrier_ptr + t, 1, sem="release")` to increment, then spin on `tl.atomic_add(barrier_ptr + t, 0, sem="acquire")` to read (`triton_kernels/scan.py:184-200`).

### Calling `GRULayer.calibrate` without disabling the Triton fast path

**What happens:** Running calibration while `use_triton=True`; the fast dispatch reads scales directly from quantizer buffers and never calls their `forward`, so observers don't update.
**Why it's wrong:** Running min/max stays at `±inf` and `freeze()` produces garbage scales.
**Do this instead:** Always go through `GRULayer.calibrate(...)`; the wrapper toggles `use_triton` off and back on (`gru_layer.py:289-299`).

### Re-using training stats for calibration

**What happens:** Setting activation quantizers to `min_max` during training and skipping a separate calibration pass.
**Why it's wrong:** Training-time augmentation can shift activation distributions in ways the deployed model never sees (`calibration.py:14-19`).
**Do this instead:** Train in `dynamic` mode, run `calibrate(layer, val_loader, n_batches=64)` on a held-out loader, then `freeze`.

## Error Handling

**Strategy:** Fail fast at construction with `ValueError`; fail fast at runtime with `RuntimeError` for path-mismatch.

**Patterns:**
- Shape / config validation raises `ValueError` at `__init__` time: `gru_layer.py:65, 81, 110`; `gru_cell.py:107`; `structure.py:76-115`.
- Misuse of dense-only helpers in structured mode raises `RuntimeError`: `gru_cell.py:254`, `:284`.
- Optional `torch_structured` raises a descriptive `ImportError` with install hint only when a structured kind that needs it is requested: `structure.py:62-68`.
- Triton kernels assert `is_cuda` and `dtype == float32` at the wrapper entry: `scan.py:1661-1662`.
- Per-tensor symmetric / frozen requirement for in-kernel fake-quant is silently degraded (returns `None`) rather than raising — see `_extract_h_quant_params` (`gru_layer.py:28`).

## Cross-Cutting Concerns

**Logging:** No logging framework. `calibrate(..., verbose=True)` prints progress (`calibration.py:110`).
**Validation:** Constructor-time shape/config checks; assertions inside per-step bodies for fused-layout invariants (`gru_cell.py:371, 449`).
**Authentication:** Not applicable.
**Determinism:** Reference path is fully deterministic. Triton persistent kernels were non-deterministic before the release/acquire fix; now deterministic across CTA-scheduling orders.
**Dtype discipline:** Every fake-quant op preserves input dtype; internal float ops run in fp32 unless the caller explicitly opted into autocast.

---

*Architecture analysis: 2026-05-13*
