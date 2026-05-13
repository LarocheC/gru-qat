# External Integrations

**Analysis Date:** 2026-05-13

## APIs & External Services

**None.** This is a research/library project. No HTTP clients, no SDKs to remote services, no telemetry, no cloud APIs. `grep -r "requests\|httpx\|aiohttp\|boto3"` over `src/` returns nothing.

## Library / Framework Integrations (the real "integrations")

The only external surfaces this library couples to are Python/CUDA libraries. Each one has a specific contract the codebase relies on.

### PyTorch (`torch>=2.2`, locked at 2.11.0)

**Role:** Reference compute backend; parity baseline; autograd host.

**Coupling points (file:line-ish):**
- `torch.nn.GRUCell` — parity baseline. `tests/test_parity.py` asserts `GRUCellQuant` with Identity quantizers matches `nn.GRUCell` to `<1e-5` (the contract documented in `SCOPE.md` "Success criteria" and `CLAUDE.md` "Parity tolerance"). This pins the manual unroll's correctness.
- `torch.nn.GRU` — cuDNN ceiling benchmark in `bench/bench_layer.py:build_cudnn_gru`. Explicitly NOT used at runtime (closed kernel; can't insert fake-quant — see `SCOPE.md` "What this is not").
- `nn.Module` / `nn.Parameter` / `register_buffer` — `FakeQuantize` is an `nn.Module` so it can hold observer state, calibrated scales, learnable step sizes (`src/gru_qat/quantizers.py:FakeQuantize`).
- `torch.autograd.Function` — STE primitives in `src/gru_qat/ste.py` (`STERound`, `STEClamp`, `fake_quant_ste`).
- `torch.compile` — `GRULayer(compile_step=True)` fuses the elementwise per-step body (`src/gru_qat/gru_layer.py`).
- `torch.set_float32_matmul_precision("high")` — used in parity tests to ensure both PyTorch matmul and Triton kernels see TF32 inputs (documented in `README.md` "Numerical parity" section).
- PyTorch fake-quant API is *not* used directly; this library implements its own `FakeQuantize` because the official APIs don't support all the granularity combinations needed (see `SCOPE.md` "Quantizer design").

**Version sensitivity:** `torch>=2.2` is the declared floor. The CUDA wheels pulled at lock time (`nvidia-cublas==13.1.0.3`, `nvidia-cudnn-cu13==9.19.0.56`, etc.) follow torch 2.11.0's transitive deps.

### Triton (`triton>=2.2`, locked at 3.6.0)

**Role:** Fast-path persistent kernels. The whole reason this library exists in its current shape — cuDNN's GRU is closed; Triton lets us own the kernel.

**Coupling points:**
- `@triton.jit` decorator — every kernel in `src/gru_qat/triton_kernels/scan*.py`.
- `triton.autotune` over `BLOCK_B/OH/K` — `scan.py:_AUTOTUNE_CONFIGS_FWD` and `_AUTOTUNE_CONFIGS_BWD`. Configs are tuned for **sm_89 (Ada)** with "SMEM ~100KB/CTA" budget assumed.
- `tl.dot` — matmul intrinsic, drives all GEMM in the kernels.
- `tl.atomic_add(barrier_ptr + t, 1, sem="release")` / `tl.atomic_add(barrier_ptr + t, 0, sem="acquire")` — release/acquire semaphore pattern for cross-CTA visibility between timesteps in the persistent kernels. **Critical contract** documented in `DEVELOPMENT.md` "Cross-CTA barriers" and `CLAUDE.md`: do NOT swap this for `tl.load(cache_modifier='.cv')` — the cache modifier is not a fence substitute and caused ~0.2 absolute drift before the fix.
- `tl.extra.libdevice.rint` — bit-identical to `torch.round` per the parity note in `README.md`.
- Persistent kernel pattern with grid sized to `<= GPU SM count` — wrapper enforces this (otherwise scheduled-but-not-running CTAs deadlock on spin-wait, see comment in `scan.py:gru_scan_fwd_persistent_kernel`).

**Availability check:**
- `src/gru_qat/triton_kernels/__init__.py:is_available()` returns `True` iff `import triton` succeeds AND `torch.cuda.is_available()`.
- Tests use `triton = pytest.importorskip("triton")` and a per-file `cuda_only` marker (`tests/test_triton_scan.py:12,25`, `tests/test_triton_monarch.py:30`, etc.). On a CPU-only host the Triton suite skips cleanly.

**Version sensitivity:** `triton==3.6.0` is locked. The persistent-kernel pattern and `sem="release|acquire"` flags are Triton-3-era APIs.

### torch-structured (external, git URL, NOT on PyPI)

**Role:** Provides structured-matrix `nn.Linear` replacements (Monarch / Butterfly / LDR) used as the H×H hidden GEMM when `StructureConfig.kind` ≠ `"dense"`.

**Install:**
```bash
uv pip install git+https://github.com/LarocheC/torch-structured
```
(Documented in `CLAUDE.md`, `DEVELOPMENT.md` "Working agreement", and `README.md`.)

**Coupling points (lazy-imported):**
- `src/gru_qat/structure.py:_import_torch_structured` — soft-import wrapper. Raises a clear `ImportError` with install hint if missing.
- `ts.monarch.blockdiag_linear.BlockdiagLinear(in_features, out_features, bias=False, nblocks=...)` — Monarch factor (`src/gru_qat/structure.py:make_structured_linear`, kind `"monarch"`).
- `ts.Butterfly(in_features, out_features, bias=False, init=..., nblocks=...)` — Butterfly factor (`src/gru_qat/structure.py`, kind `"butterfly"`).
- `torch_structured.structured.layers.LDRSubdiagonal(layer_size, r, bias)` — LDR (low-displacement rank) factor (`src/gru_qat/structure.py`, kind `"ldr"`). Imported separately because `torch_structured.__init__` doesn't auto-import the `structured` submodule.
- `torch_structured.butterfly.multiply.butterfly_multiply` — used in `tests/test_butterfly_dispatch.py:270` for parity vs the Triton butterfly kernel.
- `src/gru_qat/triton_kernels/scan_butterfly.py:928` — comment notes "NBLOCKS is the stacked-butterfly count from torch_structured", binding kernel semantics to upstream's stacked-butterfly representation.

**Optional / kind-conditional:**
- `StructureConfig.kind="dense"` (default) — no `torch-structured` import. Library is fully usable for the dense + Triton path without it.
- `kind="diagonal"` — uses a local `_DiagonalLinear` (`src/gru_qat/structure.py:177`); no `torch-structured` needed.
- `kind="circulant"` — uses a local `_CirculantLinear` (mirrors upstream's private impl). Comment in `structure.py` notes this is deliberate to avoid depending on a private symbol.
- `kind in {"monarch", "butterfly", "ldr"}` — `torch-structured` required; raises clean `ImportError` otherwise. See `_NEEDS_TORCH_STRUCTURED = {"monarch", "circulant", "butterfly", "ldr"}` in `structure.py:57` (the set is slightly broader than the modules that strictly need the import, but the actual import gating is per-kind in `make_structured_linear`).

**Version sensitivity:** Not pinned in `pyproject.toml` (it's not even listed there — install is documented prose). Tests pull whatever HEAD of `LarocheC/torch-structured` is checked out.

### CUDA toolchain

**Pulled transitively by `torch==2.11.0`** (see `uv.lock`):
- `nvidia-cublas==13.1.0.3`
- `nvidia-cudnn-cu13==9.19.0.56`
- `nvidia-cuda-runtime==13.0.96`, `nvidia-cuda-cupti==13.0.85`, `nvidia-cuda-nvrtc==13.0.88`
- `nvidia-cufft==12.0.0.61`, `nvidia-cufile==1.15.1.6`, `nvidia-curand==10.4.0.35`
- `nvidia-cusolver==12.0.4.66`, `nvidia-cusparse==12.6.3.3`, `nvidia-cusparselt-cu13==0.8.0`
- `nvidia-nccl-cu13==2.28.9`, `nvidia-nvjitlink==13.0.88`, `nvidia-nvshmem-cu13==3.4.5`, `nvidia-nvtx==13.0.85`
- `cuda-bindings==13.2.0`, `cuda-pathfinder==1.5.4`, `cuda-toolkit==13.0.2`

**Hardware assumption (not enforced, but tuned for):**
- `scan.py:_AUTOTUNE_CONFIGS_FWD`: "Configs tuned for sm_89 (Ada). SMEM ~100KB/CTA". Tile sizes assume sm_89 SMEM budget.
- All published bench numbers in `README.md` and `DEVELOPMENT.md` are RTX 2000 Ada (sm_89). Other architectures will likely work but autotune configs may be suboptimal.

## Data Storage

**Databases:** None. No SQL, no NoSQL, no embedded DB at runtime.
**File Storage:** None at runtime. Library produces in-memory tensors. `SCOPE.md` explicitly says quantized weights are emitted "in the simulator's layout" for downstream consumers — no built-in file format here.
**Caching:**
- `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` — tool caches (gitignored).
- Triton's on-disk kernel cache (default location, not configured by this repo).

## Authentication & Identity

**None.** Library has no users, no auth, no sessions.

## Monitoring & Observability

**Error Tracking:** None.
**Logs:** None. No `logging` import in `src/gru_qat/`. Calibration reports stats via return values (`src/gru_qat/calibration.py:calibrate` returns a stats summary dict).
**Profiling:** Manual via `bench/bench_layer.py`, `bench/bench_triton_fwd.py`, `bench/bench_triton_train.py` — each runs `torch.cuda.synchronize()` around `time.perf_counter()` and reports `statistics.median(samples)` ms/iter.

## CI/CD & Deployment

**Hosting:** None.
**CI Pipeline:** No `.github/workflows/`, no `.gitlab-ci.yml`, no `Jenkinsfile`. `DEVELOPMENT.md` "Working agreement" says "Each phase has tests under `tests/`. CI green ⇒ phase landed." — but CI config is not in this repo (likely lives in the dev's local workflow or `bd` tracker).
**Pre-commit hooks:** None detected.

## Environment Configuration

**Required env vars:** None. `grep "os.environ"` and `grep "getenv"` in `src/gru_qat/` return nothing.

**Optional / runtime config knobs (all Python-level, not env-driven):**
- `torch.set_float32_matmul_precision("high")` — caller's responsibility; bench scripts and tests set it explicitly.
- `use_triton: bool | str = "auto"` on `GRULayer.__init__` — controls fast-path dispatch (`src/gru_qat/gru_layer.py:GRULayer`).

**Secrets location:** Not applicable (no auth, no APIs).

## Webhooks & Callbacks

**Incoming:** None.
**Outgoing:** None.

## Build / Source Dependencies via VCS

- `git+https://github.com/LarocheC/torch-structured` — only non-PyPI source dependency. User-owned (per `MEMORY.md` reference: "torch-structured repo — user's structured-matrix library"). No version pin; tracks HEAD at install time.

---

*Integration audit: 2026-05-13*
