# Coding Conventions

**Analysis Date:** 2026-05-13

## Naming Patterns

**Files:**
- `snake_case.py` throughout: `gru_cell.py`, `gru_layer.py`, `quantizers.py`, `ste.py`, `structure.py`, `calibration.py`.
- Triton kernel modules live under `src/gru_qat/triton_kernels/` and use the prefix `scan` for the dense variant, then `scan_<kind>.py` for structured variants: `scan.py`, `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py`.
- Tests mirror source modules one-to-one with the prefix `test_`: `test_ste.py`, `test_quantizers.py`, `test_parity.py`, `test_qat_smoke.py`, `test_calibration.py`, `test_structure.py`, `test_triton_scan.py`, `test_triton_diagonal.py`, `test_triton_monarch.py`, `test_butterfly_dispatch.py`.
- Benches live in `bench/` and use the `bench_` prefix: `bench_layer.py`, `bench_triton_fwd.py`, `bench_triton_train.py`.

**Functions:**
- `snake_case` for top-level and methods: `make_quantizer`, `make_structured_linear`, `fake_quant_ste`, `extract_diagonal_factors`, `gru_scan_diagonal_forward_pytorch`.
- Private/internal helpers prefixed with a single underscore: `_compute_scale_zp`, `_qrange`, `_scale_zp_from_min_max`, `_update_observer`, `_import_torch_structured`, `_validate_shapes`, `_forward_fast_dispatch`, `_extract_h_quant_params`.
- Triton reference/Triton-kernel pairs distinguish backends in the suffix: `gru_scan_<kind>_forward_pytorch` vs. `gru_scan_<kind>_forward_triton`; same for `..._backward_pytorch` / `..._backward_triton`. Example: `src/gru_qat/triton_kernels/scan_diagonal.py:gru_scan_diagonal_forward_pytorch`, `..._forward_triton`.

**Variables:**
- `snake_case` for locals (`gate_r`, `n_input_branch`, `h_carry`, `bh_cat`).
- Math-significant short names match the unrolled GRU equations exactly: `r`, `z`, `n`, `h`, `gi_r`, `gi_z`, `gi_n`, `gh_r`, `gh_z`, `gh_n`. Keep them — `src/gru_qat/gru_cell.py:1` documents the math so the cell math reads like the paper.
- Shape conventions for tensors that recur across the codebase: `T` (time), `B` (batch), `H` (hidden), `IN` (input dim). Used as parameter names in tests/kernels (`tests/test_triton_scan.py:48`, `src/gru_qat/triton_kernels/scan_diagonal.py:88`).
- Weight tensor names follow PyTorch's `W_ir`, `W_iz`, `W_in`, `W_hr`, `W_hz`, `W_hn`; bias names follow `b_ir`, ..., `b_hn`.
- Concatenated "fused" versions: `Wi_cat`, `Wh_cat`, `bi_cat`, `bh_cat` (`src/gru_qat/gru_cell.py:53`).

**Types & classes:**
- `PascalCase` for classes and dataclasses: `GRUCellQuant`, `GRULayer`, `FakeQuantize`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`, `QuantizerConfig`, `QuantRecipe`, `StructureConfig`, `CellWeights`, `STERound`, `STEClamp`, `Identity`.
- `typing.Literal` is used for closed string unions instead of `Enum`: `GateLayout = Literal["split", "fused"]` (`src/gru_qat/gru_cell.py:35`), `ObserverMode = Literal["dynamic", "min_max", "frozen"]` (`src/gru_qat/quantizers.py:37`), `StructuredKind = Literal["dense", "diagonal", "monarch", "circulant", "butterfly", "ldr"]` (`src/gru_qat/structure.py:31`).
- `QuantizerFactory = Callable[[], FakeQuantize]` — type aliases use `PascalCase` and live next to the function that consumes them (`src/gru_qat/quantizers.py:233`).

## Code Style

**Formatting:**
- `ruff` configured in `pyproject.toml`:
  - `line-length = 100`
  - `target-version = "py310"`
- No explicit formatter (Black/isort) — ruff covers both. Run `ruff check src tests`.

**Linting:**
- `ruff` rules use defaults (no rule-set extension declared in `pyproject.toml`).
- Test files that need module imports after a `pytest.importorskip` use `# noqa: E402` to silence "module level import not at top of file" — see `tests/test_structure.py:27`, `tests/test_triton_diagonal.py:17`, `tests/test_triton_monarch.py:19`.
- Triton kernels with intentionally-unused bias loads use `# noqa: F841` (`src/gru_qat/triton_kernels/scan_butterfly.py:495`).

**Type checking:**
- `mypy` is configured with `strict = true` and scoped to `files = ["src/gru_qat"]` (`pyproject.toml:32-35`). Tests and benches are **not** type-checked.
- `python_version = "3.10"`.
- `from __future__ import annotations` is at the top of every src file (`src/gru_qat/ste.py:10`, `quantizers.py:26`, `gru_cell.py:17`, `gru_layer.py:18`, `structure.py:23`, `calibration.py:22`) and every test file. PEP 604 union syntax (`int | None`, `torch.Tensor | None`) is used everywhere — `Optional[...]` does not appear.
- `# type: ignore[override]` is the standard escape hatch for `torch.autograd.Function.forward/backward` (`src/gru_qat/ste.py:24,28,41,53`; `triton_kernels/scan.py:1530`; etc.).
- `# type: ignore[import-not-found]` for the optional `torch_structured` import (`src/gru_qat/structure.py:63`).
- `# type: ignore[arg-type]` for `**kwargs` forwarding in factory constructors (`src/gru_qat/gru_cell.py:491`).

## Import Organization

**Order (observed across `src/`):**
1. `from __future__ import annotations`
2. Standard library (`abc`, `dataclasses`, `typing`, `warnings`)
3. Third-party (`torch`, `torch.nn`, `torch.nn.functional`, `triton`, `triton.language`)
4. Internal (`from gru_qat.<module> import ...`)

**Conventions:**
- `import torch.nn as nn` and `import torch.nn.functional as F` are the canonical aliases — keep them.
- `import triton` and `import triton.language as tl` are kept side-by-side in kernel files (`src/gru_qat/triton_kernels/scan_diagonal.py:27`).
- Internal cross-package imports always use the full path `from gru_qat.<module>` — never relative imports. Example: `from gru_qat.quantizers import (...)` in `src/gru_qat/gru_cell.py:26`.
- Optional / soft dependencies (`torch_structured`) are imported **lazily inside the function that needs them** via `_import_torch_structured()` (`src/gru_qat/structure.py:60`) — never at module top. This keeps dense-only usage from requiring the optional dep.
- Tests defer `gru_qat` imports until **after** `pytest.importorskip("torch_structured")` or `pytest.importorskip("triton")` — see `tests/test_structure.py:25`, `tests/test_triton_monarch.py:17`. The `# noqa: E402` is paired with this pattern.

**Path aliases:**
- None. Imports are flat under the `gru_qat` package root.

## Type Annotations

- Every public function and method has full type annotations on parameters and return type, including `-> None`.
- Tests are also fully annotated (`def test_foo() -> None:`) even though mypy skips them.
- Tensor shapes are documented in docstrings and inline comments using `[T, B, H]`-style shorthand, not encoded in types.
- `torch.Tensor | None` is preferred over `Optional[torch.Tensor]`.
- `tuple[float, int, int] | None` is the canonical shape for "frozen per-tensor symmetric quant params" — see `src/gru_qat/gru_layer.py:29`. Passed positionally to Triton wrappers as `h_in_quant`, `h_out_quant`.

## Error Handling

**Patterns:**
- Validation errors raise `ValueError` with a message containing the offending field name and the constraint: `"fused gate layout requires recipe.weight.axis=0; got axis={...}. ..."` (`src/gru_qat/gru_cell.py:107`), `"diagonal requires square (in == out); got in={...}, out={...}"` (`src/gru_qat/structure.py:79`).
- Mode/state errors (caller asked a dense-only API on a structured cell) raise `RuntimeError`: `"quantize_weights() is dense-only; use the structured forward path ..."` (`src/gru_qat/gru_cell.py:255`).
- Optional-dependency failures raise `ImportError` with an actionable install hint and use `raise ... from e` to preserve the original traceback: `src/gru_qat/structure.py:65-68`.
- `TypeError` is used for unsupported runtime types from user-supplied iterables: `src/gru_qat/calibration.py:105`.
- Internal "this should never happen" branches use `RuntimeError` with the offending value: `src/gru_qat/gru_layer.py:255`.

**Assertions:**
- `assert` is used for invariants that the caller cannot violate without a programming error (e.g., shape sanity checks after construction, internal dispatch contracts): `assert w.Wi_cat is not None and w.Wh_cat is not None` (`gru_cell.py:371`), `assert self.gate_layout == "fused"` (`gru_cell.py:449`).
- Do not use `assert` for input validation that's reachable from public API — use `ValueError`/`RuntimeError`.

**STE specifics:**
- `STEClamp.backward` returns `(grad_output * mask, None, None)` — the `None`s correspond to non-Tensor scalar arguments. This is the autograd convention and must not be "tidied" (`src/gru_qat/ste.py:53`).

## Logging

**Framework:** None. `print()` is used only for benchmark output (`bench/bench_layer.py:232`) and behind a `verbose: bool = False` flag in `calibrate()` (`src/gru_qat/calibration.py:109`).

**Patterns:**
- Library code is silent. The calibration loop is the only place that prints, and only when `verbose=True`.
- No `logging` module usage. If logging is added, it should go through `logging.getLogger("gru_qat")` and stay silent by default.

## Comments & Docstrings

**Module-level docstrings:**
- Every src file opens with a multi-paragraph docstring describing what the file is, what it's for, and the design constraints. Examples:
  - `src/gru_qat/gru_cell.py:1` — explains the GRU math and the asymmetric `r * (W_hn h + b_hn)` step.
  - `src/gru_qat/ste.py:1` — explains why STE lives in this file and where future extensions go.
  - `src/gru_qat/structure.py:1` — explains the optional dependency story.
- Tests open with a similar docstring describing what's being verified and which phase it gates (`tests/test_parity.py:1`, `tests/test_qat_smoke.py:1`).

**Inline comments:**
- Use comments to explain *why* (rationale, gotcha, design choice), not *what*. Heavy use for:
  - Non-obvious math invariants (`gru_cell.py:13-14` on the asymmetric n-gate).
  - Performance rationale (`gru_layer.py:120-126` on why `mode="default"` and not `mode="reduce-overhead"`).
  - Numerical-stability tolerances and the reason they're set (`tests/test_triton_scan.py:137-138`, `tests/test_butterfly_dispatch.py:158-159`).
  - Known broken / deferred work tagged `TODO(phase=N)`: `src/gru_qat/ste.py:84`, `quantizers.py:136`, `gru_cell.py:507`, `gru_layer.py:305`. Phase number indicates the planned phase for the fix.
  - Anti-patterns called out with regression-test framing (`tests/test_butterfly_dispatch.py:163-181` documents the scratch-OOB bug the test guards).

**Section dividers:**
- ASCII rule dividers organize long files into stanzas:
  ```python
  # ----------------------------------------------------------------------
  # Configuration
  # ----------------------------------------------------------------------
  ```
  See `src/gru_qat/quantizers.py:40`, `:149`, `:228`, `:260`. Use the same 75-char rule style.

## Function & Class Design

**Function size:**
- Most functions are short (10-40 lines). The GRU cell `step()` is intentionally longer (~50 lines) because every quantization insertion point is named and commented (`src/gru_qat/gru_cell.py:351`).

**Parameters:**
- Keyword-only arguments via `*,` are used liberally for anything that isn't a primary positional input — see `GRUCellQuant.__init__` (`src/gru_qat/gru_cell.py:93`), `GRULayer.__init__` (`src/gru_qat/gru_layer.py:55`), `make_structured_linear` (`src/gru_qat/structure.py:122`), `calibrate` (`src/gru_qat/calibration.py:33`). This is a hard convention.
- Booleans are always keyword-only.

**Dataclasses for config:**
- `QuantizerConfig`, `QuantRecipe`, `StructureConfig`, `CellWeights` are all `@dataclass`. Use dataclasses for config bags and value objects — not plain dicts or namedtuples.
- `field(default_factory=lambda: ...)` is the canonical mutable-default pattern (`src/gru_qat/quantizers.py:272`).

**Module design:**
- Public API is exported explicitly via `src/gru_qat/__init__.py:21-36` `__all__`.
- Private implementation classes use a leading underscore (`_DiagonalLinear`, `_CirculantLinear`, `_ButterflyLinear`, `_LDRLinear` in `src/gru_qat/structure.py:177`).
- Subclassing is preferred over `if/else` switches for variants that affect dispatch — `FakeQuantize` is the canonical example (4 subclasses, one method override each). **Do not collapse** them into a single class with `if axis is None: ... elif group_size: ...`.

## QAT-Specific Conventions

These rules are documented in `CLAUDE.md` and `DEVELOPMENT.md` and must be preserved:

**Dtype discipline:**
- Every `FakeQuantize.forward` preserves input dtype. Internal float ops run in `torch.float32` unless the caller has explicitly opted into autocast.
- `bf16` around `fake_quant_ste` was tried and dropped — the fp32↔bf16 cast tax around quant/dequant boundaries exceeded the GEMM saving at the relevant shapes. Don't re-introduce it.
- Tests cast the loss to fp32 before the reduction when working with potentially-bf16 outputs: `loss = out.float().pow(2).sum()` (e.g., `tests/test_structure.py:97`, `tests/test_butterfly_dispatch.py:118`).

**What NOT to quantize in the reference path:**
- Bias is fp32 throughout (the reference path; bias-int32 export is a deferred Phase 6 concern).
- `torch.sigmoid` and `torch.tanh` stay fp32 (LUT replacement is also Phase 6).
- These are *deliberate* omissions — don't add fake-quant to them without an explicit ticket.

**Reference path is slow on purpose:**
- `GRUCellQuant` + `GRULayer`'s Python time loop are the readable, obvious, correct reference. Don't optimize them. Speed lives in the Triton kernels (`src/gru_qat/triton_kernels/`).

**FakeQuantize granularity dispatch:**
- The four subclasses (`Identity`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`) keep the kernel-dispatch surface flat. **Do not** collapse them into one class with `if/else` on `axis` / `group_size`. New schemes (LSQ, log-quant, NF4) should subclass `FakeQuantize` and add a factory entry in `make_quantizer` (`src/gru_qat/quantizers.py:236`).

**Parity tolerance for fp32 cell:**
- `GRUCellQuant` with `PRESETS["fp32"]` (Identity quantizers) must match `torch.nn.GRUCell` to `< 1e-5` absolute on the parametrized shapes (`tests/test_parity.py:70`). This is a hard gate — don't loosen it.

**Per-channel `min_max` observer (known gap):**
- `FakeQuantize._update_observer` uses a *global* scalar reduction even when `axis` is set (`src/gru_qat/quantizers.py:135-146`). Per-channel activation quant with `min_max` therefore doesn't accumulate per-channel running stats. Not blocking — per-channel weight quant uses `dynamic` mode where scales are derived from static weights each forward. Documented in `DEVELOPMENT.md` Phase 1 known gap.

**Triton cross-CTA barriers:**
- Use the release/acquire `tl.atomic_add(barrier, ..., sem=...)` pattern, **not** `tl.load(cache_modifier=".cv")`. The `.cv` modifier is not an acquire fence and produced non-deterministic drift (`src/gru_qat/triton_kernels/scan.py`, comment in `gru_scan_fwd_persistent_kernel`; `DEVELOPMENT.md:131-143`).

## Testing-Adjacent Conventions

- Numerical-tolerance tests force matmul precision with `torch.set_float32_matmul_precision("high")` so the reference and test paths share a TF32 regime (`tests/test_triton_scan.py:56`, `tests/test_triton_diagonal.py:108`, etc.).
- Random seeds are set with `torch.manual_seed(0)` at the top of any test that exercises randomness (`tests/test_qat_smoke.py:95`, `tests/test_triton_diagonal.py:78`).
- Relative-error tolerances are spelled out in tests with the reasoning: TF32 mantissa width, compounding across `T` steps, STE rounding boundaries, etc. — see comments in `tests/test_triton_scan.py:137-138`, `tests/test_triton_diagonal.py:238`. Match this style when adding new parity tests.

---

*Convention analysis: 2026-05-13*
