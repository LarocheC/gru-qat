# Technology Stack

**Analysis Date:** 2026-05-13

## Languages

**Primary:**
- Python `>=3.10` — entire library, tests, and benches. mypy targets `py310`; ruff `target-version = "py310"` (`pyproject.toml`).

**Embedded / DSL:**
- Triton kernel language (`triton.language as tl`) — persistent fwd/bwd kernels in `src/gru_qat/triton_kernels/scan*.py`. Not a separate compiled language, but is the hot path that all speed targets depend on. Kernels use `@triton.jit`, `triton.autotune`, `tl.dot`, `tl.atomic_add(sem="release"|"acquire")`.

**Secondary:**
- None. No C/C++/CUDA `.cu` files in this repo — the CUDA path is delegated to PyTorch + Triton.

## Runtime

**Environment:**
- CPython 3.10+. Lockfile `uv.lock` resolves wheels for cp310/cp311/cp312/cp313 on linux x86_64 / aarch64, macOS arm64, and win_amd64.
- CUDA runtime is required for the fast path and most tests. The PyTorch reference path runs on CPU but is "slow on purpose" (per `SCOPE.md`, `CLAUDE.md`).

**Package Manager:**
- `uv` (Astral). Documented in `CLAUDE.md` and `DEVELOPMENT.md`: `uv sync` to bootstrap; `uv pip install -e ".[dev]"` for dev extras.
- Lockfile: `uv.lock` (present, ~186 KB, committed to repo).

**Build Backend:**
- `hatchling` (`pyproject.toml [build-system]`). Wheel target: `src/gru_qat` (src-layout).

## Frameworks

**Core (runtime dependencies, `pyproject.toml [project]`):**
- `torch>=2.2` — required. Locked at `torch==2.11.0` in `uv.lock`. Provides `nn.Module`, autograd, fake-quant STE machinery, `torch.compile`, and the CUDA wheels (`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-cuda-runtime`, etc., pulled transitively).
- `numpy>=1.24` — locked at `numpy==2.4.4` / `2.2.6` (multi-version resolution).

**Optional extras (`pyproject.toml [project.optional-dependencies]`):**
- `triton = ["triton>=2.2"]` — fast-path runtime. Locked at `triton==3.6.0` in `uv.lock`. Tests gate with `pytest.importorskip("triton")` in `tests/test_triton_*.py`.
- `torch-structured` — NOT on PyPI and NOT declared in `pyproject.toml`. Installed from git: `uv pip install git+https://github.com/LarocheC/torch-structured`. Lazy-imported in `src/gru_qat/structure.py:_import_torch_structured`. See `INTEGRATIONS.md` for details.

**Testing (`[project.optional-dependencies].dev`):**
- `pytest>=7` — locked at `pytest==9.0.3`. Config in `pyproject.toml [tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-ra"`, custom marker `slow`.
- `pytest-xdist>=3` — locked at `pytest-xdist==3.8.0`. Parallel test runner; not invoked in default `pytest -q`.

**Lint / Type-check (`[project.optional-dependencies].dev`):**
- `mypy>=1.7` — locked at `mypy==2.0.0`. Strict mode, scoped to `src/gru_qat` only (`pyproject.toml [tool.mypy]`).
- `ruff>=0.1` — locked at `ruff==0.15.12`. `line-length = 100`, `target-version = "py310"` (`pyproject.toml [tool.ruff]`).

**Build/Dev:**
- `hatchling` — wheel builder. Configured at `pyproject.toml [tool.hatch.build.targets.wheel]`.

## Key Dependencies

**Critical (the fast path falls over without these):**
- `torch==2.11.0` — autograd graph, `nn.Module` ownership of quantizer state, `torch.compile` for the `compile_step=True` path (`gru_layer.py`), all GEMMs in the reference cell (`gru_cell.py`).
- `triton==3.6.0` — `@triton.jit`, `triton.autotune`, `tl.atomic_add` semaphore ops (release/acquire pattern documented in `DEVELOPMENT.md` "Cross-CTA barriers" and `scan.py:gru_scan_fwd_persistent_kernel`).
- `torch-structured` (external, git URL) — Monarch (`ts.monarch.blockdiag_linear.BlockdiagLinear`), Butterfly (`ts.Butterfly`), LDR (`torch_structured.structured.layers.LDRSubdiagonal`). Only required when `StructureConfig.kind` is one of `monarch|butterfly|ldr|circulant` (Circulant uses a thin local impl actually — see `src/gru_qat/structure.py:_CirculantLinear`).

**CUDA stack (pulled transitively by `torch==2.11.0`, see `uv.lock`):**
- `nvidia-cublas==13.1.0.3`
- `nvidia-cudnn-cu13==9.19.0.56`
- `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-cupti==13.0.85`, `nvidia-cuda-nvrtc==13.0.88`
- `nvidia-cufft==12.0.0.61`, `nvidia-cufile==1.15.1.6`, `nvidia-curand==10.4.0.35`
- `nvidia-cusolver==12.0.4.66`, `nvidia-cusparse==12.6.3.3`, `nvidia-cusparselt-cu13==0.8.0`
- `nvidia-nccl-cu13==2.28.9`, `nvidia-nvjitlink==13.0.88`, `nvidia-nvshmem-cu13==3.4.5`, `nvidia-nvtx==13.0.85`
- `cuda-bindings==13.2.0`, `cuda-pathfinder==1.5.4`, `cuda-toolkit==13.0.2`

**Standard-library only otherwise.** No web/server frameworks. No database client libraries. No HTTP client libraries.

## Configuration

**Environment:**
- No `.env` file present. No environment variables read by source code (`grep "os.environ"` in `src/` returns nothing).
- Hardware selection is via `torch.cuda.is_available()` in tests / benches; no env-var override.
- bench scripts read CLI flags only (`argparse` in `bench/bench_layer.py:main`, `bench/bench_triton_train.py:main`).

**Build:**
- `pyproject.toml` is the single source of truth. No `setup.py`, no `setup.cfg`, no `Makefile`, no `tox.ini`.
- `[tool.hatch.build.targets.wheel] packages = ["src/gru_qat"]` — src-layout wheel.

**Lint/Type-check config (all inline in `pyproject.toml`):**
- `[tool.mypy]`: `python_version = "3.10"`, `strict = true`, `files = ["src/gru_qat"]` — tests and benches are intentionally NOT mypy-checked.
- `[tool.ruff]`: `line-length = 100`, `target-version = "py310"`.

**Test config (`pyproject.toml [tool.pytest.ini_options]`):**
- `testpaths = ["tests"]`
- `addopts = "-ra"` (short summary of all non-pass outcomes)
- Custom marker: `slow` (`-m "not slow"` to skip).
- CUDA-only tests gate via the `cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), ...)` idiom defined per-file in `tests/test_triton_*.py`. Triton tests additionally use `pytest.importorskip("triton")`.

## Platform Requirements

**Development:**
- Python 3.10+ (lockfile resolves cp310–cp313).
- `uv` installed (Astral's package manager).
- For Triton path + most tests: NVIDIA GPU + matching CUDA runtime. Bench numbers in `README.md` / `DEVELOPMENT.md` are measured on **RTX 2000 Ada** (sm_89). `scan.py:_AUTOTUNE_CONFIGS_FWD` lists "Configs tuned for sm_89 (Ada). SMEM ~100KB/CTA".
- For structured matrix support: `torch-structured` installed from git URL (see `INTEGRATIONS.md`).

**Production:**
- No deployment target in this repo. `SCOPE.md` explicitly states "Not a deployment runtime."
- Library produces "quantized weights and a reference int kernel suitable for porting to TFLite/ONNX Runtime / a custom embedded runtime, but we don't target those backends directly" (`SCOPE.md`).

---

*Stack analysis: 2026-05-13*
