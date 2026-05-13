# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

- `SCOPE.md` — design rationale, non-goals, success criteria.
- `DEVELOPMENT.md` — file map, phase status, bench numbers, upgrade
  pathways, and an explicit "what the agent should NOT do" section.

These two docs are authoritative; this file is a fast-path summary.

## Commands

```bash
uv sync                                # bootstrap
uv pip install -e ".[dev]"             # add pytest, mypy, ruff
uv pip install git+https://github.com/LarocheC/torch-structured  # structured-matrix paths

pytest -q                              # full suite (~100 tests)
pytest tests/test_triton_monarch.py -q # one file
pytest -k "monarch and qat" -q         # one test by name
pytest -m "not slow" -q                # skip slow tests
mypy                                   # strict, src/gru_qat only (see pyproject.toml)
ruff check src tests

python bench/bench_layer.py            # dense train-step bench
python bench/bench_triton_train.py     # train-step bench across variants
```

Triton tests skip automatically when CUDA is unavailable (`pytest.importorskip("triton")` + a `cuda_only` mark).

## Architecture

Single-direction, single-layer GRU written for QAT. cuDNN's GRU is a closed fused kernel with no fake-quant hooks, so the cell is manually unrolled and every quantization insertion point is explicit.

**Reference path (pure PyTorch, slow on purpose):**
`GRUCellQuant` (single step) → `GRULayer` (Python time loop) → `calibrate()` collects activation min/max → `freeze_all()` locks scales for inference. The reference path's job is to be slow, obvious, and correct — speed lives in Triton.

**Fast path (Triton persistent kernels):**
One kernel launch covers all T timesteps for fwd or bwd. `GRULayer._forward_fast_dispatch` picks dense / Monarch / Butterfly based on `structure_hidden`. Cross-CTA visibility uses the release/acquire `atomic_add(sem=...)` pattern — see the explicit warning in `DEVELOPMENT.md` about `cache_modifier=".cv"` not being a fence substitute.

**Structured hidden weights (Phase 5+):**
`StructureConfig(kind=...)` swaps the H×H hidden GEMM for Monarch (block-diagonal), Butterfly (`O(H log H)`), Circulant, or LDR. Monarch and Butterfly have matching Triton kernels; Circulant/LDR fall back to the per-step PyTorch path. Depends on the external `torch-structured` library (lazy-imported in `structure.py`).

**Quantizer design:**
`FakeQuantize` is an `nn.Module` (holds observer / frozen-scale state). Granularity is parameterized by `(axis, group_size, symmetric, bits)`, not class hierarchy. Subclasses differ only in `_compute_scale_zp`. Gates default to `split` so each gate carries its own activation scale; `fused` layout is required for `pre_batch_input=True` and for several Triton paths.

## Conventions

- **Dtype discipline**: every fake-quant op preserves input dtype; internal float ops run fp32 unless the caller explicitly opts into autocast. bf16 around fake-quant was tried and dropped — the cast tax exceeded the GEMM saving at our shapes.
- **Don't quantize bias, sigmoid, or tanh** in the reference path. These are deliberate omissions (LUTs are a deferred Phase 6 concern).
- **Don't optimize the reference path.** Speed lives in Triton.
- **Don't collapse `FakeQuantize` granularities into one class with if/else.** Subclassing keeps kernel dispatch flat.
- **Parity tolerance**: `GRUCellQuant` with Identity quantizers matches `torch.nn.GRUCell` to `< 1e-5`. Don't loosen this.
- Per-channel `min_max` observer is known-broken for activations (uses a global reduction). Not blocking — per-channel weight quant uses `dynamic` mode. See Phase 1 known gap in `DEVELOPMENT.md`.

## Workflow (from parent `/home/claroche/CLAUDE.md`)

This repo uses **bd (beads)** for issue tracking. Run `bd prime` for full workflow. Use `bd ready` / `bd show <id>` / `bd update <id> --claim` / `bd close <id>`. Do NOT use TodoWrite or markdown TODO lists. Session-close protocol (quality gates → push to remote → verify `git status` shows up-to-date) is mandatory per the parent file.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

<!-- GSD:project-start source:PROJECT.md -->
## Project

**gru-triton — Native-PyTorch Parity Audit**

`gru-triton` (aka `gru-qat`) is a single-direction, single-layer GRU built for quantization-aware training (QAT) with a slow reference path in pure PyTorch and a fast path in persistent Triton kernels (dense + diagonal + monarch + butterfly). Structured hidden weights (Monarch, Butterfly, Circulant, LDR) are pluggable via `StructureConfig`. This milestone is a **correctness audit** of the existing implementation against a native PyTorch baseline (`torch.nn.GRU`), with the explicit goal of finding holes, bugs, and mismatches before any further feature work.

**Core Value:** **Every code path that claims to compute a GRU must produce numerically equivalent output to `torch.nn.GRU` (under matched recipe), and any deviation must be a tested, documented, intentional one — not a silent drift.**

### Constraints

- **Baseline**: `torch.nn.GRU` (1 layer, unidirectional, `batch_first=True`) — handles gate ordering / bias fusion quirks at the test-helper layer, not by changing reference-path code.
- **Tolerance tiers**:
  - fp32 Identity-quantizer cell parity: < 1e-5 (existing contract; do not loosen)
  - fp32 reference-layer vs `nn.GRU`: < 1e-4 (allows for accumulation drift over T)
  - Triton vs reference under same recipe: < 1e-5
  - Quant-on (active recipe, deterministic): bit-identical
- **Test framework**: pytest with existing markers (`cuda_only`, `slow`). Long-T parity tests (F3) marked `slow`.
- **Linting / typing**: ruff + mypy strict on `src/gru_qat`. New test helpers in `tests/` are not mypy-strict (matches existing convention).
- **Don't optimize the reference path** — even if A/F surfaces slowness, speed lives in Triton. Reference is correct-by-construction.
- **Don't loosen `< 1e-5`** for Identity quantizer cell parity. If A1 fails at < 1e-4, that's a *new* test for the layer; the cell test stays at < 1e-5.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python `>=3.10` — entire library, tests, and benches. mypy targets `py310`; ruff `target-version = "py310"` (`pyproject.toml`).
- Triton kernel language (`triton.language as tl`) — persistent fwd/bwd kernels in `src/gru_qat/triton_kernels/scan*.py`. Not a separate compiled language, but is the hot path that all speed targets depend on. Kernels use `@triton.jit`, `triton.autotune`, `tl.dot`, `tl.atomic_add(sem="release"|"acquire")`.
- None. No C/C++/CUDA `.cu` files in this repo — the CUDA path is delegated to PyTorch + Triton.
## Runtime
- CPython 3.10+. Lockfile `uv.lock` resolves wheels for cp310/cp311/cp312/cp313 on linux x86_64 / aarch64, macOS arm64, and win_amd64.
- CUDA runtime is required for the fast path and most tests. The PyTorch reference path runs on CPU but is "slow on purpose" (per `SCOPE.md`, `CLAUDE.md`).
- `uv` (Astral). Documented in `CLAUDE.md` and `DEVELOPMENT.md`: `uv sync` to bootstrap; `uv pip install -e ".[dev]"` for dev extras.
- Lockfile: `uv.lock` (present, ~186 KB, committed to repo).
- `hatchling` (`pyproject.toml [build-system]`). Wheel target: `src/gru_qat` (src-layout).
## Frameworks
- `torch>=2.2` — required. Locked at `torch==2.11.0` in `uv.lock`. Provides `nn.Module`, autograd, fake-quant STE machinery, `torch.compile`, and the CUDA wheels (`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-cuda-runtime`, etc., pulled transitively).
- `numpy>=1.24` — locked at `numpy==2.4.4` / `2.2.6` (multi-version resolution).
- `triton = ["triton>=2.2"]` — fast-path runtime. Locked at `triton==3.6.0` in `uv.lock`. Tests gate with `pytest.importorskip("triton")` in `tests/test_triton_*.py`.
- `torch-structured` — NOT on PyPI and NOT declared in `pyproject.toml`. Installed from git: `uv pip install git+https://github.com/LarocheC/torch-structured`. Lazy-imported in `src/gru_qat/structure.py:_import_torch_structured`. See `INTEGRATIONS.md` for details.
- `pytest>=7` — locked at `pytest==9.0.3`. Config in `pyproject.toml [tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-ra"`, custom marker `slow`.
- `pytest-xdist>=3` — locked at `pytest-xdist==3.8.0`. Parallel test runner; not invoked in default `pytest -q`.
- `mypy>=1.7` — locked at `mypy==2.0.0`. Strict mode, scoped to `src/gru_qat` only (`pyproject.toml [tool.mypy]`).
- `ruff>=0.1` — locked at `ruff==0.15.12`. `line-length = 100`, `target-version = "py310"` (`pyproject.toml [tool.ruff]`).
- `hatchling` — wheel builder. Configured at `pyproject.toml [tool.hatch.build.targets.wheel]`.
## Key Dependencies
- `torch==2.11.0` — autograd graph, `nn.Module` ownership of quantizer state, `torch.compile` for the `compile_step=True` path (`gru_layer.py`), all GEMMs in the reference cell (`gru_cell.py`).
- `triton==3.6.0` — `@triton.jit`, `triton.autotune`, `tl.atomic_add` semaphore ops (release/acquire pattern documented in `DEVELOPMENT.md` "Cross-CTA barriers" and `scan.py:gru_scan_fwd_persistent_kernel`).
- `torch-structured` (external, git URL) — Monarch (`ts.monarch.blockdiag_linear.BlockdiagLinear`), Butterfly (`ts.Butterfly`), LDR (`torch_structured.structured.layers.LDRSubdiagonal`). Only required when `StructureConfig.kind` is one of `monarch|butterfly|ldr|circulant` (Circulant uses a thin local impl actually — see `src/gru_qat/structure.py:_CirculantLinear`).
- `nvidia-cublas==13.1.0.3`
- `nvidia-cudnn-cu13==9.19.0.56`
- `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-cupti==13.0.85`, `nvidia-cuda-nvrtc==13.0.88`
- `nvidia-cufft==12.0.0.61`, `nvidia-cufile==1.15.1.6`, `nvidia-curand==10.4.0.35`
- `nvidia-cusolver==12.0.4.66`, `nvidia-cusparse==12.6.3.3`, `nvidia-cusparselt-cu13==0.8.0`
- `nvidia-nccl-cu13==2.28.9`, `nvidia-nvjitlink==13.0.88`, `nvidia-nvshmem-cu13==3.4.5`, `nvidia-nvtx==13.0.85`
- `cuda-bindings==13.2.0`, `cuda-pathfinder==1.5.4`, `cuda-toolkit==13.0.2`
## Configuration
- No `.env` file present. No environment variables read by source code (`grep "os.environ"` in `src/` returns nothing).
- Hardware selection is via `torch.cuda.is_available()` in tests / benches; no env-var override.
- bench scripts read CLI flags only (`argparse` in `bench/bench_layer.py:main`, `bench/bench_triton_train.py:main`).
- `pyproject.toml` is the single source of truth. No `setup.py`, no `setup.cfg`, no `Makefile`, no `tox.ini`.
- `[tool.hatch.build.targets.wheel] packages = ["src/gru_qat"]` — src-layout wheel.
- `[tool.mypy]`: `python_version = "3.10"`, `strict = true`, `files = ["src/gru_qat"]` — tests and benches are intentionally NOT mypy-checked.
- `[tool.ruff]`: `line-length = 100`, `target-version = "py310"`.
- `testpaths = ["tests"]`
- `addopts = "-ra"` (short summary of all non-pass outcomes)
- Custom marker: `slow` (`-m "not slow"` to skip).
- CUDA-only tests gate via the `cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), ...)` idiom defined per-file in `tests/test_triton_*.py`. Triton tests additionally use `pytest.importorskip("triton")`.
## Platform Requirements
- Python 3.10+ (lockfile resolves cp310–cp313).
- `uv` installed (Astral's package manager).
- For Triton path + most tests: NVIDIA GPU + matching CUDA runtime. Bench numbers in `README.md` / `DEVELOPMENT.md` are measured on **RTX 2000 Ada** (sm_89). `scan.py:_AUTOTUNE_CONFIGS_FWD` lists "Configs tuned for sm_89 (Ada). SMEM ~100KB/CTA".
- For structured matrix support: `torch-structured` installed from git URL (see `INTEGRATIONS.md`).
- No deployment target in this repo. `SCOPE.md` explicitly states "Not a deployment runtime."
- Library produces "quantized weights and a reference int kernel suitable for porting to TFLite/ONNX Runtime / a custom embedded runtime, but we don't target those backends directly" (`SCOPE.md`).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- `snake_case.py` throughout: `gru_cell.py`, `gru_layer.py`, `quantizers.py`, `ste.py`, `structure.py`, `calibration.py`.
- Triton kernel modules live under `src/gru_qat/triton_kernels/` and use the prefix `scan` for the dense variant, then `scan_<kind>.py` for structured variants: `scan.py`, `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py`.
- Tests mirror source modules one-to-one with the prefix `test_`: `test_ste.py`, `test_quantizers.py`, `test_parity.py`, `test_qat_smoke.py`, `test_calibration.py`, `test_structure.py`, `test_triton_scan.py`, `test_triton_diagonal.py`, `test_triton_monarch.py`, `test_butterfly_dispatch.py`.
- Benches live in `bench/` and use the `bench_` prefix: `bench_layer.py`, `bench_triton_fwd.py`, `bench_triton_train.py`.
- `snake_case` for top-level and methods: `make_quantizer`, `make_structured_linear`, `fake_quant_ste`, `extract_diagonal_factors`, `gru_scan_diagonal_forward_pytorch`.
- Private/internal helpers prefixed with a single underscore: `_compute_scale_zp`, `_qrange`, `_scale_zp_from_min_max`, `_update_observer`, `_import_torch_structured`, `_validate_shapes`, `_forward_fast_dispatch`, `_extract_h_quant_params`.
- Triton reference/Triton-kernel pairs distinguish backends in the suffix: `gru_scan_<kind>_forward_pytorch` vs. `gru_scan_<kind>_forward_triton`; same for `..._backward_pytorch` / `..._backward_triton`. Example: `src/gru_qat/triton_kernels/scan_diagonal.py:gru_scan_diagonal_forward_pytorch`, `..._forward_triton`.
- `snake_case` for locals (`gate_r`, `n_input_branch`, `h_carry`, `bh_cat`).
- Math-significant short names match the unrolled GRU equations exactly: `r`, `z`, `n`, `h`, `gi_r`, `gi_z`, `gi_n`, `gh_r`, `gh_z`, `gh_n`. Keep them — `src/gru_qat/gru_cell.py:1` documents the math so the cell math reads like the paper.
- Shape conventions for tensors that recur across the codebase: `T` (time), `B` (batch), `H` (hidden), `IN` (input dim). Used as parameter names in tests/kernels (`tests/test_triton_scan.py:48`, `src/gru_qat/triton_kernels/scan_diagonal.py:88`).
- Weight tensor names follow PyTorch's `W_ir`, `W_iz`, `W_in`, `W_hr`, `W_hz`, `W_hn`; bias names follow `b_ir`, ..., `b_hn`.
- Concatenated "fused" versions: `Wi_cat`, `Wh_cat`, `bi_cat`, `bh_cat` (`src/gru_qat/gru_cell.py:53`).
- `PascalCase` for classes and dataclasses: `GRUCellQuant`, `GRULayer`, `FakeQuantize`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`, `QuantizerConfig`, `QuantRecipe`, `StructureConfig`, `CellWeights`, `STERound`, `STEClamp`, `Identity`.
- `typing.Literal` is used for closed string unions instead of `Enum`: `GateLayout = Literal["split", "fused"]` (`src/gru_qat/gru_cell.py:35`), `ObserverMode = Literal["dynamic", "min_max", "frozen"]` (`src/gru_qat/quantizers.py:37`), `StructuredKind = Literal["dense", "diagonal", "monarch", "circulant", "butterfly", "ldr"]` (`src/gru_qat/structure.py:31`).
- `QuantizerFactory = Callable[[], FakeQuantize]` — type aliases use `PascalCase` and live next to the function that consumes them (`src/gru_qat/quantizers.py:233`).
## Code Style
- `ruff` configured in `pyproject.toml`:
- No explicit formatter (Black/isort) — ruff covers both. Run `ruff check src tests`.
- `ruff` rules use defaults (no rule-set extension declared in `pyproject.toml`).
- Test files that need module imports after a `pytest.importorskip` use `# noqa: E402` to silence "module level import not at top of file" — see `tests/test_structure.py:27`, `tests/test_triton_diagonal.py:17`, `tests/test_triton_monarch.py:19`.
- Triton kernels with intentionally-unused bias loads use `# noqa: F841` (`src/gru_qat/triton_kernels/scan_butterfly.py:495`).
- `mypy` is configured with `strict = true` and scoped to `files = ["src/gru_qat"]` (`pyproject.toml:32-35`). Tests and benches are **not** type-checked.
- `python_version = "3.10"`.
- `from __future__ import annotations` is at the top of every src file (`src/gru_qat/ste.py:10`, `quantizers.py:26`, `gru_cell.py:17`, `gru_layer.py:18`, `structure.py:23`, `calibration.py:22`) and every test file. PEP 604 union syntax (`int | None`, `torch.Tensor | None`) is used everywhere — `Optional[...]` does not appear.
- `# type: ignore[override]` is the standard escape hatch for `torch.autograd.Function.forward/backward` (`src/gru_qat/ste.py:24,28,41,53`; `triton_kernels/scan.py:1530`; etc.).
- `# type: ignore[import-not-found]` for the optional `torch_structured` import (`src/gru_qat/structure.py:63`).
- `# type: ignore[arg-type]` for `**kwargs` forwarding in factory constructors (`src/gru_qat/gru_cell.py:491`).
## Import Organization
- `import torch.nn as nn` and `import torch.nn.functional as F` are the canonical aliases — keep them.
- `import triton` and `import triton.language as tl` are kept side-by-side in kernel files (`src/gru_qat/triton_kernels/scan_diagonal.py:27`).
- Internal cross-package imports always use the full path `from gru_qat.<module>` — never relative imports. Example: `from gru_qat.quantizers import (...)` in `src/gru_qat/gru_cell.py:26`.
- Optional / soft dependencies (`torch_structured`) are imported **lazily inside the function that needs them** via `_import_torch_structured()` (`src/gru_qat/structure.py:60`) — never at module top. This keeps dense-only usage from requiring the optional dep.
- Tests defer `gru_qat` imports until **after** `pytest.importorskip("torch_structured")` or `pytest.importorskip("triton")` — see `tests/test_structure.py:25`, `tests/test_triton_monarch.py:17`. The `# noqa: E402` is paired with this pattern.
- None. Imports are flat under the `gru_qat` package root.
## Type Annotations
- Every public function and method has full type annotations on parameters and return type, including `-> None`.
- Tests are also fully annotated (`def test_foo() -> None:`) even though mypy skips them.
- Tensor shapes are documented in docstrings and inline comments using `[T, B, H]`-style shorthand, not encoded in types.
- `torch.Tensor | None` is preferred over `Optional[torch.Tensor]`.
- `tuple[float, int, int] | None` is the canonical shape for "frozen per-tensor symmetric quant params" — see `src/gru_qat/gru_layer.py:29`. Passed positionally to Triton wrappers as `h_in_quant`, `h_out_quant`.
## Error Handling
- Validation errors raise `ValueError` with a message containing the offending field name and the constraint: `"fused gate layout requires recipe.weight.axis=0; got axis={...}. ..."` (`src/gru_qat/gru_cell.py:107`), `"diagonal requires square (in == out); got in={...}, out={...}"` (`src/gru_qat/structure.py:79`).
- Mode/state errors (caller asked a dense-only API on a structured cell) raise `RuntimeError`: `"quantize_weights() is dense-only; use the structured forward path ..."` (`src/gru_qat/gru_cell.py:255`).
- Optional-dependency failures raise `ImportError` with an actionable install hint and use `raise ... from e` to preserve the original traceback: `src/gru_qat/structure.py:65-68`.
- `TypeError` is used for unsupported runtime types from user-supplied iterables: `src/gru_qat/calibration.py:105`.
- Internal "this should never happen" branches use `RuntimeError` with the offending value: `src/gru_qat/gru_layer.py:255`.
- `assert` is used for invariants that the caller cannot violate without a programming error (e.g., shape sanity checks after construction, internal dispatch contracts): `assert w.Wi_cat is not None and w.Wh_cat is not None` (`gru_cell.py:371`), `assert self.gate_layout == "fused"` (`gru_cell.py:449`).
- Do not use `assert` for input validation that's reachable from public API — use `ValueError`/`RuntimeError`.
- `STEClamp.backward` returns `(grad_output * mask, None, None)` — the `None`s correspond to non-Tensor scalar arguments. This is the autograd convention and must not be "tidied" (`src/gru_qat/ste.py:53`).
## Logging
- Library code is silent. The calibration loop is the only place that prints, and only when `verbose=True`.
- No `logging` module usage. If logging is added, it should go through `logging.getLogger("gru_qat")` and stay silent by default.
## Comments & Docstrings
- Every src file opens with a multi-paragraph docstring describing what the file is, what it's for, and the design constraints. Examples:
- Tests open with a similar docstring describing what's being verified and which phase it gates (`tests/test_parity.py:1`, `tests/test_qat_smoke.py:1`).
- Use comments to explain *why* (rationale, gotcha, design choice), not *what*. Heavy use for:
- ASCII rule dividers organize long files into stanzas:
## Function & Class Design
- Most functions are short (10-40 lines). The GRU cell `step()` is intentionally longer (~50 lines) because every quantization insertion point is named and commented (`src/gru_qat/gru_cell.py:351`).
- Keyword-only arguments via `*,` are used liberally for anything that isn't a primary positional input — see `GRUCellQuant.__init__` (`src/gru_qat/gru_cell.py:93`), `GRULayer.__init__` (`src/gru_qat/gru_layer.py:55`), `make_structured_linear` (`src/gru_qat/structure.py:122`), `calibrate` (`src/gru_qat/calibration.py:33`). This is a hard convention.
- Booleans are always keyword-only.
- `QuantizerConfig`, `QuantRecipe`, `StructureConfig`, `CellWeights` are all `@dataclass`. Use dataclasses for config bags and value objects — not plain dicts or namedtuples.
- `field(default_factory=lambda: ...)` is the canonical mutable-default pattern (`src/gru_qat/quantizers.py:272`).
- Public API is exported explicitly via `src/gru_qat/__init__.py:21-36` `__all__`.
- Private implementation classes use a leading underscore (`_DiagonalLinear`, `_CirculantLinear`, `_ButterflyLinear`, `_LDRLinear` in `src/gru_qat/structure.py:177`).
- Subclassing is preferred over `if/else` switches for variants that affect dispatch — `FakeQuantize` is the canonical example (4 subclasses, one method override each). **Do not collapse** them into a single class with `if axis is None: ... elif group_size: ...`.
## QAT-Specific Conventions
- Every `FakeQuantize.forward` preserves input dtype. Internal float ops run in `torch.float32` unless the caller has explicitly opted into autocast.
- `bf16` around `fake_quant_ste` was tried and dropped — the fp32↔bf16 cast tax around quant/dequant boundaries exceeded the GEMM saving at the relevant shapes. Don't re-introduce it.
- Tests cast the loss to fp32 before the reduction when working with potentially-bf16 outputs: `loss = out.float().pow(2).sum()` (e.g., `tests/test_structure.py:97`, `tests/test_butterfly_dispatch.py:118`).
- Bias is fp32 throughout (the reference path; bias-int32 export is a deferred Phase 6 concern).
- `torch.sigmoid` and `torch.tanh` stay fp32 (LUT replacement is also Phase 6).
- These are *deliberate* omissions — don't add fake-quant to them without an explicit ticket.
- `GRUCellQuant` + `GRULayer`'s Python time loop are the readable, obvious, correct reference. Don't optimize them. Speed lives in the Triton kernels (`src/gru_qat/triton_kernels/`).
- The four subclasses (`Identity`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`) keep the kernel-dispatch surface flat. **Do not** collapse them into one class with `if/else` on `axis` / `group_size`. New schemes (LSQ, log-quant, NF4) should subclass `FakeQuantize` and add a factory entry in `make_quantizer` (`src/gru_qat/quantizers.py:236`).
- `GRUCellQuant` with `PRESETS["fp32"]` (Identity quantizers) must match `torch.nn.GRUCell` to `< 1e-5` absolute on the parametrized shapes (`tests/test_parity.py:70`). This is a hard gate — don't loosen it.
- `FakeQuantize._update_observer` uses a *global* scalar reduction even when `axis` is set (`src/gru_qat/quantizers.py:135-146`). Per-channel activation quant with `min_max` therefore doesn't accumulate per-channel running stats. Not blocking — per-channel weight quant uses `dynamic` mode where scales are derived from static weights each forward. Documented in `DEVELOPMENT.md` Phase 1 known gap.
- Use the release/acquire `tl.atomic_add(barrier, ..., sem=...)` pattern, **not** `tl.load(cache_modifier=".cv")`. The `.cv` modifier is not an acquire fence and produced non-deterministic drift (`src/gru_qat/triton_kernels/scan.py`, comment in `gru_scan_fwd_persistent_kernel`; `DEVELOPMENT.md:131-143`).
## Testing-Adjacent Conventions
- Numerical-tolerance tests force matmul precision with `torch.set_float32_matmul_precision("high")` so the reference and test paths share a TF32 regime (`tests/test_triton_scan.py:56`, `tests/test_triton_diagonal.py:108`, etc.).
- Random seeds are set with `torch.manual_seed(0)` at the top of any test that exercises randomness (`tests/test_qat_smoke.py:95`, `tests/test_triton_diagonal.py:78`).
- Relative-error tolerances are spelled out in tests with the reasoning: TF32 mantissa width, compounding across `T` steps, STE rounding boundaries, etc. — see comments in `tests/test_triton_scan.py:137-138`, `tests/test_triton_diagonal.py:238`. Match this style when adding new parity tests.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
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
- **Reference path** (pure PyTorch, deliberately slow): time-unrolled GRU cell with `FakeQuantize` modules at every documented insertion point. Used for fp32 parity testing, QAT correctness, calibration (observers must fire per step), and as the executable spec for the Triton kernels.
- **Fast path** (Triton persistent kernels): one launch covers all T timesteps for fwd or bwd. Selected at `GRULayer` construction via `use_triton`/`structure_hidden`. Reads frozen per-tensor symmetric scales from the matching cell quantizers and applies fake-quant inside the kernel.
- Manual unroll over `nn.GRU` so every quantization insertion point is explicit and replaceable (see `SCOPE.md` §1).
- `FakeQuantize` is an `nn.Module` — holds observer / frozen-scale state, not a pure function.
- Granularity is a parameter `(axis, group_size, symmetric, bits)`, not a class hierarchy — subclass identity controls only `_compute_scale_zp`.
- Calibration lifecycle is explicit: dynamic → min_max (calibration) → frozen (inference).
- Structured hidden weights are orthogonal to quantization; `StructureConfig.kind` swaps the H×H GEMM without touching the cell math.
- Triton fast path requires fused-gate layout, dense input side, and (for in-kernel fake-quant) frozen per-tensor symmetric hidden quantizers.
## Layers
- Purpose: Stable user-facing entry points re-exported from one module
- Location: `src/gru_qat/__init__.py`
- Contains: `GRUCellQuant`, `GRULayer`, `FakeQuantize*`, `QuantRecipe`, `QuantizerConfig`, `Identity`, `PRESETS`, `STERound`, `STEClamp`, `StructureConfig`, `make_structured_linear`
- Depends on: every sibling module under `gru_qat/`
- Used by: tests/, bench/, downstream callers
- Purpose: Iterate the cell over the time dimension; dispatch to reference vs. Triton fast path
- Location: `src/gru_qat/gru_layer.py`
- Contains: `GRULayer.forward`, `_forward_fast_dispatch`, `calibrate`, `freeze`, `_extract_h_quant_params`
- Depends on: `gru_cell`, `quantizers`, `structure`, `triton_kernels.scan_*`
- Used by: user code, tests, bench harness
- Purpose: Manually-unrolled GRU step with all fake-quant insertion points
- Location: `src/gru_qat/gru_cell.py`
- Contains: `GRUCellQuant`, `CellWeights`, `step`, `step_with_gi`, `step_structured`, `quantize_weights`, `quantize_input_weights`, `freeze_quantizers`
- Depends on: `quantizers`, `structure`
- Used by: `GRULayer`, tests
- Purpose: Pluggable fake-quant modules + STE autograd
- Location: `src/gru_qat/quantizers.py`, `src/gru_qat/ste.py`
- Contains: `FakeQuantize` base, `Identity`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`, `QuantizerConfig`, `QuantRecipe`, `PRESETS`, `make_quantizer`, `STERound`, `STEClamp`, `fake_quant_ste`
- Depends on: `torch.autograd`, `torch.nn`
- Used by: `gru_cell`, `gru_layer`, `calibration`
- Purpose: Drop-in `nn.Linear` substitutes for the H×H hidden GEMM
- Location: `src/gru_qat/structure.py`
- Contains: `StructureConfig`, `make_structured_linear`, `_DiagonalLinear`, `_CirculantLinear`, `_ButterflyLinear`, `_LDRLinear`
- Depends on: `torch_structured` (lazy, optional) for Monarch / Butterfly / LDR; local impls for Diagonal / Circulant
- Used by: `gru_cell`, Triton extract_* helpers
- Purpose: Run forwards in observer mode and freeze activation scales
- Location: `src/gru_qat/calibration.py`
- Contains: `calibrate(module, loader, n_batches)`, `freeze_all(module)`
- Depends on: `quantizers.FakeQuantize`
- Used by: `GRULayer.calibrate`, user code post-training
- Purpose: Persistent multi-step fwd/bwd kernels for each structured kind that has Triton support
- Location: `src/gru_qat/triton_kernels/scan.py` (dense), `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py`
- Contains: `@triton.jit` fwd/bwd kernels, `torch.autograd.Function` wrappers, `gru_scan*`, `extract_*_factors` helpers
- Depends on: `triton`, CUDA, frozen-scale buffers on the cell
- Used by: `GRULayer._forward_fast_dispatch`, dedicated Triton tests
## Data Flow
### Primary Request Path — reference (per-step) forward
### Fast Path — Triton dispatch
### Quantization data flow inside a step
### Calibration flow
- Quantizer state lives in `FakeQuantize` buffers: `scale`, `zero_point`, `running_min`, `running_max`, `_initialized` (Python attribute).
- Cell weight tensors are `nn.Parameter`s; structured layers hold their own parameter sets.
- `GRULayer.use_triton` is a Python attribute that calibration toggles transiently.
- No global mutable state.
## Key Abstractions
- Purpose: Pluggable fake-quant op with observer / frozen-scale state
- Examples: `src/gru_qat/quantizers.py:68` (base), `:154` (Identity), `:169` (PerTensor), `:178` (PerChannel), `:192` (PerGroup)
- Pattern: Abstract base with one abstract method `_compute_scale_zp(x)`. Subclasses differ only in scale/zp derivation. Observer modes (`dynamic`/`min_max`/`frozen`) are a config flag, not a subclass.
- Purpose: Declarative bundle that builds all required quantizers
- Examples: `src/gru_qat/quantizers.py:265` (`QuantRecipe`), `:45` (`QuantizerConfig`), `:284` (`PRESETS["fp32" | "int8_per_channel" | "int4_per_group_64"]`)
- Pattern: Dataclasses with `field(default_factory=...)`. `make_quantizer(config)` is the factory entry point; `bits >= 32` short-circuits to `Identity`.
- Purpose: Hoist weight quantization out of the time loop
- Examples: `src/gru_qat/gru_cell.py:38`
- Pattern: Dataclass holding the six fake-quantized weight tensors plus optional `Wi_cat`/`Wh_cat`/`bi_cat`/`bh_cat` for fused-gate layout.
- Purpose: Swap the H×H hidden (or input) GEMM for an `O(n log n)` or `O(n)` parameterization
- Examples: `src/gru_qat/structure.py:34` (`StructureConfig`), `:118` (`make_structured_linear`), `:177` (`_DiagonalLinear`)
- Pattern: Dataclass-driven factory. Kinds: `dense`, `diagonal`, `monarch`, `circulant`, `butterfly`, `ldr`. `torch_structured` is lazy-imported only when a non-local kind is requested.
- Purpose: Sole location where the QAT gradient story is faked
- Examples: `src/gru_qat/ste.py:15` (`STERound`), `:32` (`STEClamp`), `:59` (`fake_quant_ste`)
- Pattern: `torch.autograd.Function` subclasses with identity / clipped-identity backward. `fake_quant_ste(x, scale, zp, qmin, qmax)` is the canonical building block.
- Purpose: Bridge a `triton.jit` kernel pair (fwd, bwd) to PyTorch autograd
- Examples: `src/gru_qat/triton_kernels/scan.py:1518` (`GRUScanFunction`), `:1589` (`GRUScanPersistentFunction`), `scan_diagonal.py:629` (`GRUScanDiagonalFunction`)
- Pattern: `torch.autograd.Function` that saves `(gi, h0, Wh, bh, out)` plus `h_in_quant`/`h_out_quant` tuples; `gru_scan_<kind>(...)` is the public `.apply` wrapper.
## Entry Points
- Location: `src/gru_qat/gru_layer.py:49`
- Triggers: User construction in training / inference code
- Responsibilities: Build the cell, decide fast-path eligibility (`structure_input is None and kind in {diagonal, monarch, butterfly} and gate_layout == 'fused'`), optionally wrap per-step body in `torch.compile(mode="default")`.
- Location: `src/gru_qat/gru_layer.py:139`
- Triggers: Standard PyTorch forward
- Responsibilities: Time-loop or Triton dispatch; returns `(out, h_T)`.
- Location: `src/gru_qat/gru_layer.py:202`
- Triggers: `self.use_triton and x.is_cuda`
- Responsibilities: Pre-batch input projection, extract structured factors + frozen scales, call the matching `gru_scan_<kind>`.
- Location: `src/gru_qat/gru_cell.py:351`, `:436`, `:302`
- Triggers: Per-step body inside the layer's time loop
- Responsibilities: One GRU step with all fake-quant insertion points wired.
- Location: `src/gru_qat/triton_kernels/scan.py:1624`, `scan_diagonal.py:661`, `scan_monarch.py` (`gru_scan_monarch`), `scan_butterfly.py` (`gru_scan_butterfly_triton`)
- Triggers: Called from `_forward_fast_dispatch`
- Responsibilities: Wrap `torch.autograd.Function` calls that launch the persistent fwd/bwd Triton kernels.
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
### Optimizing the reference (PyTorch) path
### Collapsing `FakeQuantize` granularities into one class with if/else
### Quantizing bias, sigmoid, or tanh in the reference path
### Using `tl.load(cache_modifier=".cv")` as a cross-CTA fence
### Calling `GRULayer.calibrate` without disabling the Triton fast path
### Re-using training stats for calibration
## Error Handling
- Shape / config validation raises `ValueError` at `__init__` time: `gru_layer.py:65, 81, 110`; `gru_cell.py:107`; `structure.py:76-115`.
- Misuse of dense-only helpers in structured mode raises `RuntimeError`: `gru_cell.py:254`, `:284`.
- Optional `torch_structured` raises a descriptive `ImportError` with install hint only when a structured kind that needs it is requested: `structure.py:62-68`.
- Triton kernels assert `is_cuda` and `dtype == float32` at the wrapper entry: `scan.py:1661-1662`.
- Per-tensor symmetric / frozen requirement for in-kernel fake-quant is silently degraded (returns `None`) rather than raising — see `_extract_h_quant_params` (`gru_layer.py:28`).
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
