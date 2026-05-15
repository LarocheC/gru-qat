---
phase: 07-audit-report-findings-handling
plan: 01
subsystem: triton-kernels-validation-and-lint
tags: [hardening, lint, mypy, ruff, validation, audit-close]
requires:
  - "Phase 6 EDG-04 GRULayer.forward ValueError guard convention"
provides:
  - "Hardened gru_scan* shape/dtype/is_cuda validation in the 10 assert-bearing callees (gru-triton-7rj)"
  - "mypy 0 / ruff 0 across src/gru_qat under strict mode (gru-triton-4m6)"
  - "tests/test_scan_wrapper_validation.py — end-to-end public-entry-point validation tests"
  - "pyproject.toml mypy third-party overrides + ruff per-file-ignores config"
affects:
  - "src/gru_qat/triton_kernels/scan*.py — the 4 Triton-kernel files"
  - "src/gru_qat/{ste,structure,gru_layer,gru_cell,calibration}.py — lint cleanup"
tech-stack:
  added: []
  patterns:
    - "if <bad condition>: raise ValueError(...) input-validation guard mirroring GRULayer.forward EDG-04"
    - "[[tool.mypy.overrides]] ignore_missing_imports for stub-less third-party deps (triton, torch_structured)"
    - "[tool.ruff.lint.per-file-ignores] for deliberate repo idioms (pytest.importorskip, kernel-local bindings)"
    - "# type: ignore[untyped-decorator] / [no-untyped-def] for @triton.jit kernels under mypy strict"
    - "cast() narrowing of torch nn.Module attribute-access Tensor | Module unions"
key-files:
  created:
    - "tests/test_scan_wrapper_validation.py"
  modified:
    - "pyproject.toml"
    - "src/gru_qat/triton_kernels/scan.py"
    - "src/gru_qat/triton_kernels/scan_diagonal.py"
    - "src/gru_qat/triton_kernels/scan_monarch.py"
    - "src/gru_qat/triton_kernels/scan_butterfly.py"
    - "src/gru_qat/ste.py"
    - "src/gru_qat/structure.py"
    - "src/gru_qat/gru_layer.py"
    - "src/gru_qat/gru_cell.py"
    - "src/gru_qat/calibration.py"
decisions:
  - "D-02: gru-triton-7rj fixed in-phase with the D-37/D-50 two-commit failing-test-before-fix discipline (no xfail)."
  - "D-06: mypy 0 / ruff 0 reached via systematic fixes + scoped mypy overrides + ruff per-file-ignores; strict = true preserved."
  - "ruff F401 added to the tests/* per-file-ignores block: 2 of the 4 baseline F401 errors live in test files (test_triton_scan.py, test_triton_monarch.py); per-file-ignores keeps those 5 test files untouched as the plan requires (deviation from the RESEARCH assumption that all 4 F401 were in src)."
metrics:
  duration: "~25 min"
  tasks_completed: 2
  files_created: 1
  files_modified: 10
  completed_date: "2026-05-15"
---

# Phase 7 Plan 01: gru_scan validation hardening + mypy/ruff cleanup Summary

JWT-style one-liner: Closed `gru-triton-7rj` by converting every bare shape/dtype/`is_cuda` `assert` in the 10 assert-bearing `gru_scan*` Triton callees to `python -O`-surviving `if ... raise ValueError`, and closed `gru-triton-4m6` by clearing the 145-error mypy / 23-error ruff debt to 0/0 under unchanged strict mode.

## What Was Built

### Task 1 — gru-triton-7rj: shape/dtype validation hardening (two-commit)

**Commit A (`b87d986`, `test(07-01)`):** New `tests/test_scan_wrapper_validation.py` — 13 tests exercising the public `gru_scan`, `gru_scan_persistent`, `gru_scan_diagonal`, `gru_scan_monarch`, `gru_scan_butterfly_triton` entry points end-to-end with a non-CUDA tensor, a non-float32 dtype, and malformed shapes. Each test asserts the raised type is `ValueError`/`RuntimeError`, **never** `AssertionError` — the precise `python -O` survival check (an `assert` is stripped under `-O` and raises `AssertionError`; an `if ... raise` is not). The non-CUDA-tensor cases are CPU-runnable; the shape/dtype cases are gated with the project `cuda_only` idiom. Committed failing (all 13 raised `AssertionError`).

**Commit B (`242a986`, `fix(07-01)`):** Converted every bare shape/dtype/`is_cuda` `assert` in the 10 callee functions enumerated in the plan — `gru_scan_forward`, `gru_scan_forward_persistent`, `gru_scan_backward_triton`, `gru_scan_backward_persistent` (scan.py); `gru_scan_diagonal_forward/backward_triton`; `gru_scan_monarch_forward/backward_triton`; `gru_scan_butterfly_forward/backward_triton` — to `if <bad condition>: raise ValueError(...)`, mirroring the Phase 6 `GRULayer.forward` EDG-04 guard. Each message names the offending field and the expected constraint. The assert-free public `.apply` wrappers were left untouched (the guards live in the callees they dispatch into). `@triton.jit` kernel-body asserts and the `_pytorch` reference-helper asserts (`scan_monarch.py:98-100`, `scan_diagonal.py:100`) were deliberately left as-is per the plan's scope boundary.

After Commit B all 13 wrapper-validation tests pass (CUDA host — `cuda_only` cases ran).

### Task 2 — gru-triton-4m6: mypy 0 / ruff 0 lint cleanup

**ruff 23 → 0:** Added `[tool.ruff.lint.per-file-ignores]` — `tests/*` ignores `E402`/`F841`/`F401` (the `pytest.importorskip` import idiom + test-scaffolding bindings), `scan_butterfly.py`/`scan.py` ignore `F841` (intentional kernel-local bindings). Deleted 2 genuine unused `src` imports (`factory` in `gru_cell.py`, `math` in `scan_butterfly.py`); converted the 2 `E731` grid lambdas to `def`. The 5 ruff-affected test files were not edited.

**mypy 145 → 0 (strict preserved):** `[[tool.mypy.overrides]] ignore_missing_imports` for the stub-less `triton` / `torch_structured` modules. `# type: ignore[untyped-decorator]` on the 8 `@triton.jit` + 2 `@triton.autotune` decorators and `# type: ignore[no-untyped-def]` on the kernel defs (kernel pointer/`constexpr` params can't be meaningfully annotated). Annotated the autograd `Function.forward`/`backward` `ctx` params as `Any` with explicit return types — this also retired 14 now-redundant `# type: ignore[override]` comments. Narrowed `torch_structured` `Module`-vs-`Tensor` unions in the `extract_*` helpers with `cast()`. Per-line fixes in `ste.py` / `structure.py` / `gru_layer.py` / `gru_cell.py` (`no-any-return` casts, `Iterable[object]` / `dict[str,int]` type args, `Callable[..., torch.Tensor]` for the variadic per-step `body`, removed one always-true `comparison-overlap` clause).

## Verification

- `uv run pytest tests/test_scan_wrapper_validation.py -q` — 13 passed.
- `uv run mypy` — `Success: no issues found in 12 source files`.
- `uv run ruff check src tests` — `All checks passed!`.
- Regression gate (touched-file subset): `test_triton_scan.py` + `test_parity.py` + `test_scan_wrapper_validation.py` — 37 passed; broader `test_structure.py` + `test_triton_diagonal.py` + `test_triton_monarch.py` + `test_butterfly_dispatch.py` + `test_ste.py` + `test_quantizers.py` — 85 passed, 1 skipped (pre-existing phase-2 simulator skip). No lint-pass behavior regression.
- `git log` confirms `test(07-01)` (`b87d986`) precedes `fix(07-01)` 7rj (`242a986`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] ruff F401 split between src and test files**
- **Found during:** Task 2
- **Issue:** The RESEARCH baseline counted 4 `F401` (unused-import) errors and the plan said "Delete the 4 genuine `F401` unused imports (in src files)". The live `ruff` run showed only 2 of the 4 `F401` are in `src` (`gru_cell.py`, `scan_butterfly.py`); the other 2 are in test files (`test_triton_scan.py:10`, `test_triton_monarch.py:20`). The plan's acceptance criterion firmly requires the 5 ruff-affected test files to be absent from `git diff` — but `F401` was not in the planned `tests/*` per-file-ignores set, so ruff could not reach 0 without either editing those test files or extending the ignore set.
- **Fix:** Extended the `tests/*` `per-file-ignores` block to include `F401` (documented as test-scaffolding / re-export imports). The 2 genuine `src` `F401` were deleted as planned. This honors both the "ruff 0" and "5 test files untouched" criteria.
- **Files modified:** `pyproject.toml`
- **Commit:** `cf0ef0f`

**2. [Rule 1 - Bug] Redundant always-true identity check (`comparison-overlap`)**
- **Found during:** Task 2
- **Issue:** `gru_cell.py:freeze_quantizers` had `if isinstance(module, FakeQuantize) and module is not self:` — a `FakeQuantize` is never the `GRUCellQuant` `self`, so `module is not self` is statically always `True` (mypy `comparison-overlap`).
- **Fix:** Removed the dead `and module is not self` clause (a true no-op removal — semantics unchanged); added a comment explaining the `isinstance` check already excludes `self`.
- **Files modified:** `src/gru_qat/gru_cell.py`
- **Commit:** `cf0ef0f`

No authentication gates occurred. No architectural (Rule 4) changes were needed.

## Known Stubs

None — no stub/placeholder code was introduced.

## Self-Check: PASSED

- `tests/test_scan_wrapper_validation.py` — FOUND.
- All 10 modified `src` / `pyproject.toml` files — FOUND.
- Commit `b87d986` (test 7rj) — FOUND in `git log`.
- Commit `242a986` (fix 7rj) — FOUND in `git log`, after `b87d986`.
- Commit `cf0ef0f` (mypy/ruff 4m6) — FOUND in `git log`.
- mypy 0 / ruff 0 — verified live.
