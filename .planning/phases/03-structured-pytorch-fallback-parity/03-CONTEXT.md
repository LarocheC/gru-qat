# Phase 3: Structured PyTorch fallback parity - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Pin the Circulant and LDR per-step PyTorch paths (`_CirculantLinear`, `_LDRLinear` in `src/gru_qat/structure.py`) against independent hand-rolled mathematical references at < 1e-5 abs (fwd + bwd via autograd-grad comparison). Plus confirm STR-03 graceful degradation: when `torch-structured` is missing, the optional-dep kinds (monarch, butterfly, LDR) raise `ImportError` with a clear install-hint message; local-impl kinds (circulant, diagonal, dense) continue to work without it.

In scope:
- `_CirculantLinear` fwd + autograd-bwd parity vs hand-rolled FFT and Toeplitz constructions (cross-checked against each other first).
- `_LDRLinear` fwd + autograd-bwd parity vs hand-rolled full-matrix construction.
- STR-03 missing-`torch-structured` simulation via `monkeypatch` on `_import_torch_structured`.
- Slow-tier marker convention for long-H grid cases (consistent with Phase 1/2).

Explicitly NOT in scope for Phase 3:
- Monarch, Butterfly, Diagonal — already covered by Phase 2 (Triton) and the existing `test_structure.py` integration tests.
- LDR / circulant shape-validator edge cases — existing `test_structure.py` handles "non-square circulant/ldr raises".
- Quant-on (non-Identity) parity — Phase 4 owns bit-identity for quant-on across all structured kinds.
- Performance / parameter-count claims for circulant or LDR (out of scope; correctness only).
- Triton-tier disposition (Option C from Phase 2) — no `tl.dot` here, so no TF32 issue.

</domain>

<decisions>
## Implementation Decisions

### Hand-rolled circulant reference (D-29..31)
- **D-29:** Two reference constructions, cross-checked against each other before comparison to production:
  - **Toeplitz form:** build `C ∈ R^{H×H}` explicitly as `C[i, j] = c[(j - i) mod H]` for kernel vector `c ∈ R^H`. Compute `y_toeplitz = x @ C.T`.
  - **FFT form:** `y_fft = real(ifft(fft(c, n=H) * fft(x, n=H)))` (broadcast `c` against `x`'s batch dim).
  - **Self-consistency test** (`test_handrolled_circulant_self_consistent`): assert FFT and Toeplitz forms agree at < 1e-5 abs. Catches reference math bugs BEFORE comparing to production. Tier 1.
  - **Parity test** (`test_circulant_matches_handrolled_toeplitz`): `_CirculantLinear(kernel=c).forward(x)` matches `y_toeplitz` at < 1e-5 abs across a shape grid. Tier 2.
- **D-30:** Backward parity uses autograd:
  - Build the Toeplitz `C` from `c` with `c.requires_grad_(True)`; compute `y_ref = x @ C.T`; backprop a fixed scalar loss; extract `c.grad`.
  - Compute `_CirculantLinear(kernel=c.clone().requires_grad_(True)).forward(x)`; backprop the same loss; extract gradient.
  - Assert both gradients agree at < 1e-5 abs. Named per-tensor failure messages ("kernel_c").
- **D-31:** Circulant tests live in `tests/test_structure_parity.py` (new file per D-35). Helpers `_build_toeplitz_from_kernel` and `_circulant_via_fft` are module-level `_underscore_prefixed` per existing convention. Fully typed.

### Hand-rolled LDR reference (D-32..33)
- **D-32:** Build LDR's effective H×H matrix from the displacement-rank formula and compare via plain matmul:
  - `_LDRLinear` parameterizes `A` via low-rank factors `G_k, H_k ∈ R^{H × r}` and uses some displacement operator (e.g., circulant shift `S`) such that `A = sum_k op_k(G_k, H_k)`.
  - **Hand-rolled reference:** Read `torch-structured`'s LDR source to identify the exact displacement formula. Independently construct `A ∈ R^{H×H}` by summing the rank-r outer products with the corresponding shift-matrix operations. Then `y_ref = x @ A.T`.
  - **Parity test** (`test_ldr_matches_handrolled_reference`): production `_LDRLinear.forward(x)` matches `y_ref` at < 1e-5 abs across a shape grid (with valid `H` and `rank ≤ H`).
  - Backward via the same autograd-gradient comparison pattern as D-30.
- **D-33:** LDR edge cases (rank > H, rank = 0, non-square) are NOT tested in Phase 3 — existing `test_structure.py` covers shape-validator behavior. Phase 6 owns edge-case sweeps. Phase 3 tests valid configs only.

### STR-03 graceful-degradation testing (D-34)
- **D-34:** Simulate missing `torch-structured` via `monkeypatch.setattr("gru_qat.structure._import_torch_structured", _raise_importerror_with_hint)`. In-process, fast.
  - **Test family** (`test_missing_torch_structured_raises_clear_error`): parametrize over `kind ∈ {"monarch", "butterfly", "ldr"}` — each must raise `ImportError` with a message containing the string `"torch-structured"` and the install URL/command. Use `pytest.raises(ImportError, match=r"torch-structured")`.
  - **Companion test** (`test_local_impls_work_without_torch_structured`): parametrize over `kind ∈ {"circulant", "diagonal", "dense"}` — `make_structured_linear(kind=k, ...)` returns a working layer that produces finite output. These kinds have local impls and must NOT depend on the missing dep.
  - Both tests use the same monkeypatch fixture.

### File location (D-35)
- **D-35:** New file `tests/test_structure_parity.py` (mirrors Phase 1's `test_layer_parity.py` and Phase 2's `test_triton_<kind>_strict.py` naming). Existing `tests/test_structure.py` stays unchanged as the smoke/integration tier (finite-output, gradient-flow, training-loop, int8-QAT). Two clear tiers, one file each.

### Shape grid (D-36)
- **D-36:** Phase 3 grid for `_CirculantLinear` and `_LDRLinear`:
  - `H ∈ {8, 32, 128, 512}` (no H=1, H=2 — those are non-trivial for circulant/LDR and live in Phase 6).
  - `B ∈ {1, 4, 32}`.
  - For LDR additionally: `rank ∈ {1, 4, 8}` (with `rank ≤ H`).
  - Long-H = 512 marked `@pytest.mark.slow` to keep `pytest -q` fast.
  - Total: circulant ~12 fast + 3 slow per test family (forward, backward) = ~30 cases; LDR ~36 fast + 9 slow per family = ~90 cases.

### Discipline (carried forward)
- **D-37:** Two-commit failing-test-before-fix from D-10..12 / D-27. No `@pytest.mark.xfail`. `bd create` per finding before commit A.
- **D-38:** D-28 locks extend to Phase 3: `tests/test_parity.py` AND `tests/test_layer_parity.py` MUST remain unchanged (`git diff` empty across Phase 3 commits). The Phase 2 strict-tier files are NOT locked (Phase 3 shouldn't touch them either, but they're not in the LOCKED contract).
- **D-39:** No `src/` modifications unless a parity test surfaces a real bug. Hand-rolled references stay in the test file (not promoted to `src/`).

### Phase 2 disposition does NOT carry forward
- **D-40:** The Option C / tight-TF32 / `< 5e-4` disposition from Phase 2 applies ONLY to Triton kernels. Phase 3 paths are pure PyTorch (no `tl.dot`), so `torch.set_float32_matmul_precision('highest')` is unnecessary AND the strict `< 1e-5` bound is achievable. Don't import the Phase 2 disposition mentally.

### Claude's Discretion
- Exact `pytest.parametrize` id strings.
- Whether to share a helper module across Phase 1/2/3 strict-tier tests — default: no, inline per file (consistent with D-18 from Phase 2).
- Exact form of the "install hint" string check in STR-03 — default: `match=r"torch-structured"` is enough.
- Plan structure: 2 plans (Plan 03-01: circulant + STR-03; Plan 03-02: LDR + phase-exit SUMMARY) OR 3 plans (split SUMMARY into its own). Planner decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project + phase
- `.planning/PROJECT.md` — tolerance contract, Key Decisions (Phase 1 + Phase 2 dispositions are logged).
- `.planning/REQUIREMENTS.md` §STR-01..03 — the three requirements this phase implements.
- `.planning/ROADMAP.md` §"Phase 3: Structured PyTorch fallback parity" — success criteria.
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-CONTEXT.md` D-10..12 — two-commit discipline carried forward.
- `.planning/phases/02-triton-fast-path-parity-vs-reference/02-CONTEXT.md` D-19..21, D-27, D-28 — file-naming convention, no-xfail, locked contracts.
- `.planning/phases/01-VERIFICATION.md`, `02-VERIFICATION.md` — prior phases closed; their test files are stable references.

### Codebase
- `src/gru_qat/structure.py` — `StructureConfig`, `_CirculantLinear`, `_LDRLinear`, `_import_torch_structured` (the monkeypatch target for D-34), `make_structured_linear` factory.
- `src/gru_qat/__init__.py` — public API surface (verify what's exported).
- `tests/test_structure.py` — existing smoke/integration tier; do NOT modify. Read to understand the existing `KINDS` parametrize style and the shape-validator coverage.
- `tests/test_parity.py` AND `tests/test_layer_parity.py` — LOCKED per D-28/D-38. Verifier asserts `git diff` empty across Phase 3.
- `tests/test_triton_diagonal.py` / `_monarch.py` / `_butterfly.py` (existing) AND `tests/test_triton_<kind>_strict.py` (Phase 2) — not LOCKED but not Phase 3's concern.
- `.planning/codebase/TESTING.md` — relative-error idiom, parametrize style, marker discipline.
- `.planning/codebase/STRUCTURE.md` — confirms `_CirculantLinear` is a local impl; `_LDRLinear` lives in (or wraps) `torch-structured`.

### External
- `torch-structured` library — for LDR reference investigation per D-32. URL in `DEVELOPMENT.md`: `https://github.com/LarocheC/torch-structured`. Install: `uv pip install git+https://github.com/LarocheC/torch-structured`.
- PyTorch FFT docs (for FFT-based circulant reference per D-29): https://docs.pytorch.org/docs/stable/fft.html

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Relative-error idiom** (from Phase 1/2 PATTERNS): `(a - b).abs().max() / max(b.abs().max(), 1e-6)`. Phase 3 uses **absolute** error at < 1e-5 (consistent with the Phase 2 strict-tier files and the fact that this is fp32-vs-fp32 with no TF32).
- **`torch.manual_seed(0)`** at the top of each test using randomness.
- **`KINDS` parametrize** in `tests/test_structure.py` — Phase 3 doesn't extend it; uses kind-specific named tests.
- **`pytest.importorskip("torch_structured")`** at the top of files that need it. Phase 3's test file needs this guarded for LDR but NOT for circulant (local impl). Solution: split — STR-03 uses monkeypatch (in-process), circulant parity tests do NOT need importorskip, LDR parity tests DO.

### Established Patterns
- **One test file per concept**, mirroring src/. `test_structure_parity.py` is the natural home.
- **Module-level underscore helpers** for math constructions (`_build_toeplitz_from_kernel`, `_circulant_via_fft`, `_build_ldr_matrix_from_factors`). No `conftest.py`.
- **`from __future__ import annotations` + PEP 604 union syntax** + full type annotations.
- **`# noqa: E402`** only when `pytest.importorskip` precedes the imports it guards.

### Integration Points
- `tests/test_structure_parity.py` is a new file. No `src/` modifications unless a parity test surfaces a bug → D-27 two-commit protocol.
- The monkeypatch target `gru_qat.structure._import_torch_structured` is the existing lazy-import helper at `src/gru_qat/structure.py:60` per ARCHITECTURE.md.
- LDR reference construction requires understanding `torch-structured`'s LDR parameterization; the planner / executor will need to read that library's source (it's installed as a dev dep per DEVELOPMENT.md).

</code_context>

<specifics>
## Specific Ideas

- **Three new test functions** for circulant (per D-29..31): `test_handrolled_circulant_self_consistent`, `test_circulant_matches_handrolled_toeplitz` (fwd parametrized), `test_circulant_backward_matches_autograd_reference` (bwd parametrized).
- **Two new test functions** for LDR (per D-32): `test_ldr_matches_handrolled_reference` (fwd parametrized), `test_ldr_backward_matches_autograd_reference` (bwd parametrized).
- **Two new test functions** for STR-03 (per D-34): `test_missing_torch_structured_raises_clear_error` (parametrized over monarch/butterfly/ldr), `test_local_impls_work_without_torch_structured` (parametrized over circulant/diagonal/dense).
- **Total Phase 3 net-new tests:** ~7 named functions + ~120 parametrized cases.
- **CUDA NOT required** — pure PyTorch, runs on CPU. No `@cuda_only` decorators in this phase.

</specifics>

<deferred>
## Deferred Ideas

- **LDR edge cases (rank > H, rank = 0, non-square)** — existing shape validators cover the error path. Phase 6 may add explicit edge-case tests if needed.
- **Performance comparison: hand-rolled Toeplitz vs `_CirculantLinear` vs FFT** — out of scope (correctness audit only; no perf claims in Phase 3).
- **Quant-on bit-identity for circulant / LDR** — Phase 4 owns this.
- **Pre-existing mypy/ruff debt in src/** — already filed as `gru-triton-4m6` from Phase 1.
- **Long-T accumulation drift testing on structured layers** — Phase 6 edge-case sweeps if needed.

</deferred>

---

*Phase: 3-structured-pytorch-fallback-parity*
*Context gathered: 2026-05-13*
