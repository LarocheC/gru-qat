# Codebase Structure

**Analysis Date:** 2026-05-13

## Directory Layout

```
gru-triton/
├── src/
│   └── gru_qat/                    # Main package (importable as `gru_qat`)
│       ├── __init__.py             # Public API surface (re-exports)
│       ├── ste.py                  # Straight-Through Estimator autograd primitives
│       ├── quantizers.py           # FakeQuantize hierarchy, QuantRecipe, PRESETS
│       ├── calibration.py          # calibrate(), freeze_all()
│       ├── structure.py            # StructureConfig, make_structured_linear factory
│       ├── gru_cell.py             # GRUCellQuant — single GRU step (reference path)
│       ├── gru_layer.py            # GRULayer — multi-step driver, Triton dispatch
│       └── triton_kernels/         # Phase-5 persistent Triton kernels
│           ├── __init__.py         # Design notes, is_available() helper
│           ├── scan.py             # Dense fwd+bwd (autotune + persistent variants)
│           ├── scan_diagonal.py    # Diagonal hidden weights (elementwise recurrence)
│           ├── scan_monarch.py     # Monarch block-diagonal hidden weights
│           └── scan_butterfly.py   # Butterfly (log_H 2×2 mixing stages)
├── tests/                          # pytest suite (~100 tests)
│   ├── test_ste.py                 # STE primitives
│   ├── test_quantizers.py          # FakeQuantize variants
│   ├── test_parity.py              # Cell vs nn.GRUCell at fp32 (< 1e-5)
│   ├── test_qat_smoke.py           # End-to-end QAT trains on a toy task
│   ├── test_calibration.py         # calibrate() round-trip
│   ├── test_structure.py           # Structured cells (all 4 kinds)
│   ├── test_triton_scan.py         # Dense Triton fwd+bwd + persistent + QAT
│   ├── test_triton_diagonal.py     # Diagonal Triton fwd+bwd + QAT + dispatch
│   ├── test_triton_monarch.py      # Monarch Triton fwd+bwd + QAT + dispatch
│   └── test_butterfly_dispatch.py  # Butterfly Triton fwd+bwd + QAT + dispatch
├── bench/                          # Standalone latency harnesses (not collected by pytest)
│   ├── bench_layer.py              # Dense train-step bench vs cuDNN/compile/Triton
│   ├── bench_triton_fwd.py         # Forward-only bench across variants
│   └── bench_triton_train.py       # Train-step bench across variants
├── .planning/
│   └── codebase/                   # Generated codebase maps (this directory)
├── .beads/                         # bd (Beads) issue-tracker state — committed
├── .claude/                        # Claude Code agent state
├── pyproject.toml                  # Hatchling build, mypy strict, ruff, pytest config
├── uv.lock                         # Reproducible env lockfile (uv)
├── README.md                       # User-facing intro + numerical-parity table
├── SCOPE.md                        # Design rationale, non-goals, success criteria
├── DEVELOPMENT.md                  # File map, phase status, bench numbers, agent guardrails
├── CLAUDE.md                       # Project guidance for Claude Code agents
└── AGENTS.md                       # Agent-specific notes
```

## Directory Purposes

**`src/gru_qat/`:**
- Purpose: The single importable package. Layout is flat; each module has one clear responsibility.
- Contains: Public API (`__init__.py`), reference path (`gru_cell.py`, `gru_layer.py`), quantization primitives (`ste.py`, `quantizers.py`), structure factory (`structure.py`), calibration (`calibration.py`), Triton subpackage.
- Key files: `gru_cell.py` (reference path heart), `gru_layer.py` (dispatch), `quantizers.py` (pluggability surface).

**`src/gru_qat/triton_kernels/`:**
- Purpose: One Triton kernel pair (fwd + bwd) per structured-hidden variant. Each file is self-contained: kernel(s), Python wrappers, autograd `Function`, `extract_*_factors` helper.
- Contains: `scan.py` (dense), `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py`. `__init__.py` carries the original Phase-5 design notes and an `is_available()` helper.
- Key files: `scan.py` is the largest (~1700 lines) and the template that other variants follow.

**`tests/`:**
- Purpose: pytest suite. Mirrors the source layout one-to-one (one test file per source module that has externally observable behavior).
- Contains: Unit tests for primitives, parity tests vs `torch.nn.GRUCell`, QAT smoke tests, Triton variant tests gated on CUDA + `torch_structured`.
- Key files: `test_parity.py` (the < 1e-5 contract), `test_qat_smoke.py` (end-to-end), `test_triton_*.py` (kernel parity vs PyTorch reference).

**`bench/`:**
- Purpose: Standalone latency-measurement scripts. Run manually (`python bench/...`); not collected by pytest.
- Contains: `bench_layer.py` (dense), `bench_triton_fwd.py` (fwd-only across variants), `bench_triton_train.py` (train-step across variants).
- Key files: `bench_layer.py` is the canonical entry; the others follow its argparse pattern.

**`.planning/codebase/`:**
- Purpose: Codebase maps written by GSD mapping agents.
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md` (this file), and other focus-area docs.

**`.beads/`:**
- Purpose: Beads (`bd`) issue-tracker state. Use `bd ready`, `bd show <id>`, etc. — do not edit by hand.

## Key File Locations

**Entry Points:**
- `src/gru_qat/__init__.py`: Re-exports the user-visible classes/functions.
- `src/gru_qat/gru_layer.py`: `GRULayer` class — the primary user entry point.
- `src/gru_qat/gru_cell.py`: `GRUCellQuant` — single-step entry; also usable directly for streaming.
- `bench/bench_layer.py`: Canonical latency benchmark.

**Configuration:**
- `pyproject.toml`: Build (hatchling, `packages = ["src/gru_qat"]`), strict mypy on `src/gru_qat`, ruff (`line-length=100`, `target-version="py310"`), pytest (`testpaths=["tests"]`, `slow` marker).
- `uv.lock`: Reproducible environment for `uv sync`.
- `.gitignore`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`: Tooling state.

**Core Logic:**
- `src/gru_qat/gru_cell.py`: Reference path single-step (the executable spec).
- `src/gru_qat/gru_layer.py`: Multi-step driver and Triton dispatch.
- `src/gru_qat/quantizers.py`: `FakeQuantize` hierarchy.
- `src/gru_qat/ste.py`: STE autograd primitives.
- `src/gru_qat/structure.py`: Structured-matrix factory.
- `src/gru_qat/calibration.py`: Observer-mode forwards + freeze.
- `src/gru_qat/triton_kernels/scan*.py`: Persistent fwd/bwd kernels per variant.

**Testing:**
- `tests/test_parity.py`: Fp32 parity vs `nn.GRUCell` (< 1e-5).
- `tests/test_qat_smoke.py`: End-to-end QAT trains on a toy regression task.
- `tests/test_calibration.py`: Calibrate-then-freeze round trip.
- `tests/test_triton_*.py`: Each Triton variant has its own parity + QAT + dispatch test file.

**Reference docs:**
- `SCOPE.md`: Design decisions and non-goals — read first.
- `DEVELOPMENT.md`: File map, phase status, bench numbers, agent guardrails.
- `CLAUDE.md`: Fast-path summary for Claude Code agents.
- `README.md`: User-facing intro with the numerical-parity table.

## Naming Conventions

**Files:**
- Python modules: lowercase with underscores, one concept per file (`gru_cell.py`, `quantizers.py`, `scan_diagonal.py`).
- Triton kernel files: `scan_<kind>.py` per structured-hidden variant; the dense kernel keeps the unsuffixed name `scan.py`.
- Tests: `test_<module>.py` mirroring the source module name. Triton kernel tests use `test_triton_<kind>.py` (exception: `test_butterfly_dispatch.py`).
- Bench scripts: `bench_<scope>.py`.

**Directories:**
- Single package directory `gru_qat/` under `src/`. Subpackage `triton_kernels/` for the GPU-dependent kernels.

**Classes:**
- PascalCase. GRU-specific suffixes: `GRUCellQuant`, `GRULayer`, `CellWeights`.
- Quantizer subclasses: `FakeQuantize`, `FakeQuantizePerTensor`, `FakeQuantizePerChannel`, `FakeQuantizePerGroup`, `Identity`.
- STE wrappers: `STERound`, `STEClamp`.
- Config dataclasses end in `Config` or `Recipe`: `QuantizerConfig`, `QuantRecipe`, `StructureConfig`.
- Private structured-linear wrappers prefix `_`: `_DiagonalLinear`, `_CirculantLinear`, `_ButterflyLinear`, `_LDRLinear`.
- Triton autograd functions end in `Function`: `GRUScanFunction`, `GRUScanPersistentFunction`, `GRUScanDiagonalFunction`.

**Functions:**
- snake_case. Public API verbs: `make_quantizer`, `make_structured_linear`, `calibrate`, `freeze_all`.
- Quantizer factory inside cells: `factory(config)` returns a no-arg builder.
- Triton public wrappers: `gru_scan`, `gru_scan_persistent`, `gru_scan_diagonal`, `gru_scan_monarch`, `gru_scan_butterfly_triton`.
- Triton kernels: `gru_scan_<variant>_<fwd|bwd>_kernel`, decorated with `@triton.jit`.
- Helper `extract_<kind>_factors(cell)` returns `(weights, bh_cat)` for the corresponding fast path.
- Private helpers prefix `_`: `_compute_scale_zp`, `_extract_h_quant_params`, `_fake_quant`, `_validate_shapes`.

**Attributes:**
- Cell weight parameters: `W_ir`, `W_iz`, `W_in`, `W_hr`, `W_hz`, `W_hn` (gate ∈ {r, z, n}, side ∈ {i, h}).
- Cell biases: `b_ir`, `b_iz`, ..., `b_hn`.
- Cell quantizers: `quant_x`, `quant_h_in`, `quant_h_out`, `quant_W_*`, `quant_gate_*`, `quant_struct_W*_*`.
- Structured-mode cell submodules: `struct_Wi_r/z/n`, `struct_Wh_r/z/n`.

## Where to Add New Code

**New quantization scheme (e.g. log-quant, NF4):**
- Primary code: Subclass `FakeQuantize` in `src/gru_qat/quantizers.py`; override `_compute_scale_zp`.
- Factory: Add a branch to `make_quantizer` (`src/gru_qat/quantizers.py:236`).
- Recipe entry: Optionally add a preset to `PRESETS` (`quantizers.py:284`).
- Tests: New cases in `tests/test_quantizers.py`.
- No changes needed in `gru_cell.py` or `gru_layer.py`.

**New structured-matrix kind:**
- Type literal: Add the kind to `StructuredKind` in `src/gru_qat/structure.py:31`.
- Factory: Add a branch in `make_structured_linear` (`structure.py:118`) constructing the underlying `nn.Module`; add shape validation in `_validate_shapes` (`structure.py:76`).
- PyTorch path picks it up automatically through `cell.step_structured`.
- For Triton speed: add a new `src/gru_qat/triton_kernels/scan_<kind>.py` following the Monarch / Butterfly template (kernel + `torch.autograd.Function` + `gru_scan_<kind>` wrapper + `extract_<kind>_factors`). Add a dispatch branch in `GRULayer._forward_fast_dispatch` (`gru_layer.py:202`) and extend `_fast_dispatch_eligible` (`gru_layer.py:100`).
- Tests: New file `tests/test_<kind>_dispatch.py` (or `test_triton_<kind>.py`) and parametrize `tests/test_structure.py`.

**New Triton kernel variant (or different hardware backend):**
- Code: New file under `src/gru_qat/triton_kernels/`.
- Dispatch: New branch in `GRULayer._forward_fast_dispatch`; extend `_fast_dispatch_eligible`.
- Tests: New `tests/test_triton_<kind>.py` with `cuda_only` marker following the pattern in `test_triton_scan.py`.
- Bench: Register the variant in `bench/bench_triton_train.py` / `bench/bench_triton_fwd.py`.

**New gradient estimator (e.g. LSQ):**
- Edit `src/gru_qat/ste.py`. `STERound.apply` / `STEClamp.apply` / `fake_quant_ste` are the swap points.
- Plumb a learnable scale parameter via `QuantizerConfig.learnable_scale` (already declared, not implemented).

**New cell variant (e.g. LSTM later):**
- Copy `gru_cell.py` to `lstm_cell.py` with four gates + cell state quantizer.
- New `lstm_layer.py` mirroring `gru_layer.py`.
- Quantizer / structure infrastructure unchanged.
- Add a re-export in `src/gru_qat/__init__.py`.

**New benchmark:**
- Add `bench/bench_<scope>.py` mirroring the argparse pattern of `bench/bench_layer.py`.
- Bench scripts run manually; do not add them to `pyproject.toml` `[tool.pytest.ini_options].testpaths`.

**Utilities / shared helpers:**
- Keep them inside the module that owns them. The package layout is intentionally flat — no `utils.py` catchall.

## Special Directories

**`.beads/`:**
- Purpose: Beads issue-tracker state. The project uses `bd` for all task tracking (`bd ready`, `bd show <id>`, `bd update <id> --claim`, `bd close <id>`).
- Generated: No (state is committed via `bd dolt push`).
- Committed: Yes.

**`.claude/`:**
- Purpose: Claude Code agent state.
- Generated: Yes (by Claude Code).
- Committed: Yes.

**`.planning/`:**
- Purpose: GSD planning artifacts and codebase maps.
- Generated: Yes (by GSD mapping commands).
- Committed: Typically yes — these documents are consumed by other GSD commands.

**`.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`:**
- Purpose: Tooling caches and virtualenv.
- Generated: Yes.
- Committed: No (in `.gitignore`).

**`uv.lock`:**
- Purpose: Reproducible-env lockfile produced by `uv sync`.
- Generated: Yes.
- Committed: Yes.

---

*Structure analysis: 2026-05-13*
