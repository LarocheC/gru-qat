# Phase 3: Structured PyTorch fallback parity - Pattern Map

**Mapped:** 2026-05-13
**Files analyzed:** 1 new test file (tests/test_structure_parity.py) + 5 in-repo analog files + 2 external library files (torch_structured LDR + krylov)
**Analogs found:** 1 file-shape exact (test_layer_parity.py) + 4 idiom-exact (test_structure.py, test_triton_diagonal.py, test_triton_monarch.py, test_parity.py); 3 novel patterns flagged (monkeypatch, torch.fft in tests, external-library-as-spec)

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `tests/test_structure_parity.py` (new) | test (pytest module) | request-response (build layer → forward/backward → compare to hand-rolled reference) | `tests/test_layer_parity.py` (Phase 1 output) | exact in shape — module-level underscore helpers, FAST/SLOW grid split, autograd-grad backward parity, relative-or-absolute error idiom; differs in target (per-step PyTorch path not reference loop) |

Secondary analogs cited for cross-cutting patterns:

- `tests/test_structure.py` — `KINDS` parametrize style (Phase 3 doesn't reuse `KINDS` directly per CONTEXT D-36, but inherits the structured-cell builder pattern and `pytest.importorskip("torch_structured")` idiom at module top).
- `tests/test_parity.py` — `pytest.raises(<Error>, match="...")` idiom (used for STR-03 in this phase per D-34) and Identity-quantizer / `recipe=PRESETS["fp32"]` pattern.
- `tests/test_triton_diagonal.py` + `tests/test_triton_monarch.py` — module-level `_make_<kind>_layer` underscore-prefix helper convention (CONTEXT "Established Patterns").
- `src/gru_qat/structure.py` — production paths under test (`_CirculantLinear`, `_LDRLinear`, `_import_torch_structured`).
- `torch_structured/structured/layers.py` + `torch_structured/structured/krylov.py` (external, dev-dep) — LDR spec source for D-32 hand-rolled reference.

---

## Pattern Assignments

### `tests/test_structure_parity.py` (new) — strict-tier hand-rolled parity

**Primary analog:** `tests/test_layer_parity.py` (Phase 1 output — same FAST/SLOW grid pattern, same `set_float32_matmul_precision('highest')` preamble, same autograd-grad backward idiom, but Phase 3 compares against hand-rolled math rather than against `nn.GRU`).
**Secondary analogs:** `tests/test_structure.py` (per-kind cell builder + `pytest.importorskip`), `tests/test_parity.py` (`pytest.raises` for error-path tests).

---

#### Imports + module preamble

**Model on** `tests/test_layer_parity.py:1-33` but adapted for Phase 3:
- No `nn.GRU` import (Phase 3 doesn't compare to `nn.GRU`; the layer-level audit is Phase 1's territory).
- No file-level `pytest.importorskip("torch_structured")` — circulant tests must run on CPU-only machines with no `torch-structured` (the local-impls test family needs to assert this works). LDR tests skip individually via `pytest.importorskip` at the LDR section, OR via a per-test guard. **Planner: prefer per-test guard** so that one file can host both circulant (no dep) and LDR (dep needed) families.
- `from __future__ import annotations` + PEP 604 union syntax (per CONTEXT "Established Patterns").
- No `# noqa: E402` unless a `pytest.importorskip` precedes the imports it guards.

```python
"""Hand-rolled parity tests for the Circulant and LDR PyTorch fallback paths — Phase 3 audit.

Pins ``_CirculantLinear`` and ``_LDRLinear`` in ``src/gru_qat/structure.py``
against independent hand-rolled mathematical references at < 1e-5 abs (fwd
+ bwd via autograd-grad comparison). Plus STR-03 graceful-degradation:
when ``torch-structured`` is missing, optional-dep kinds (monarch, butterfly,
ldr) must raise ImportError with a clear install hint, while local-impl kinds
(circulant, diagonal, dense) continue to work.

Pure PyTorch — no Triton, no CUDA. Pairs with ``tests/test_structure.py``
(smoke/integration tier, finite-output + gradient-flow + training-loop +
int8-QAT). Two clear tiers, one file each.

This module sets ``torch.set_float32_matmul_precision('highest')`` at import
time because Phase 3 audits the math (per CONTEXT D-40 — TF32 'high' is for
Triton kernel files only). The < 1e-5 abs bound is achievable on fp32-vs-fp32
without TF32 in play.
"""

from __future__ import annotations

import pytest
import torch

# IEEE-754 fp32 matmul. Per CONTEXT D-40, Phase 3 paths are pure PyTorch (no
# ``tl.dot``) so 'highest' is achievable and < 1e-5 abs is the strict bound.
torch.set_float32_matmul_precision("highest")

from gru_qat.structure import (  # imported here so monkeypatch can target it later
    _CirculantLinear,
    _LDRLinear,
    make_structured_linear,
    StructureConfig,
)
```

Notes:
- `_CirculantLinear` is exposed via `src/gru_qat/structure.py:207-225` and is a public-test-import (underscore is convention-only — Python doesn't enforce it across modules).
- `_LDRLinear` is exposed via `src/gru_qat/structure.py:239-247`.
- `make_structured_linear` is the production factory (`src/gru_qat/structure.py:118-174`) used by STR-03 graceful-degradation tests.
- **Do NOT** add `pytest.importorskip("torch_structured")` at module top — see "Companion test" pattern below.

---

#### Helper construction pattern (D-29: Toeplitz + FFT reference)

**Analog for `_make_<kind>_layer` style:**
- `tests/test_triton_diagonal.py:35-48` (`_make_diagonal_layer`).
- `tests/test_triton_monarch.py:35-52` (`_make_monarch_layer`).
- `tests/test_layer_parity.py:36-53` (`_make_dense_fp32_layer`).

All three define module-level, single-underscore-prefixed, fully-typed helpers. Phase 3 follows the same shape:

**`_build_toeplitz_from_kernel`** (new, no existing analog inside `src/` or `tests/` for explicit circulant-matrix construction — the production path at `src/gru_qat/structure.py:219-222` is FFT-only):

```python
def _build_toeplitz_from_kernel(c: torch.Tensor) -> torch.Tensor:
    """Build the H×H circulant matrix C from the length-H kernel vector c.

    Per D-29, the matrix construction is::

        C[i, j] = c[(j - i) mod H]

    First column equals c. Each subsequent column is a cyclic shift down.
    This is the canonical Toeplitz form (special case where the diagonals
    wrap), and is the matrix that ``_CirculantLinear(col=c).forward(x)``
    implements via FFT.

    Used as one of the two independent references in the
    self-consistency check (FFT-form vs Toeplitz-form) before either is
    compared to ``_CirculantLinear``.
    """
    H = c.shape[0]
    idx = torch.arange(H)
    # C[i, j] = c[(j - i) mod H] — vectorized via outer arithmetic.
    j_minus_i_mod_H = (idx[None, :] - idx[:, None]) % H
    return c[j_minus_i_mod_H]
```

**`_circulant_via_fft`** (new — uses `torch.fft.rfft` to MATCH the production path's rfft form per `src/gru_qat/structure.py:220-222`; or full `torch.fft.fft` to provide a TRULY independent reference. **Planner decision:** D-29 says "FFT form: `y_fft = real(ifft(fft(c, n=H) * fft(x, n=H)))`" — that's full complex FFT, deliberately divergent from the production's rfft. Use full FFT so the self-consistency check exercises a genuinely independent FFT path).

```python
def _circulant_via_fft(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply the circulant matrix defined by kernel ``c`` to ``x`` via full
    complex FFT (NOT rfft — the production path uses rfft so this is the
    independent reference).

    For 1D x of shape (..., H)::

        y = real(ifft(fft(c, n=H) * fft(x, n=H, dim=-1), dim=-1, n=H))

    Broadcasts c against x's leading dims by unsqueezing then expanding.
    The .real cast is safe because c and x are real-valued; any imaginary
    part is floating-point noise.
    """
    H = c.shape[0]
    c_f = torch.fft.fft(c, n=H)
    x_f = torch.fft.fft(x, n=H, dim=-1)
    y = torch.fft.ifft(c_f * x_f, n=H, dim=-1)
    return y.real
```

**`_build_ldr_matrix_from_factors`** (new — see D-32 / "External-library spec read" below; this is the largest hand-rolled helper):

```python
def _build_ldr_matrix_from_factors(
    subd_A: torch.Tensor,  # (n-1,)
    subd_B: torch.Tensor,  # (n-1,)
    G: torch.Tensor,       # (r, n)
    H: torch.Tensor,       # (r, n)
) -> torch.Tensor:
    """Construct the dense n×n matrix M that LDRSubdiagonal applies to x.

    Per torch_structured/structured/krylov.py:245 and layers.py:223::

        M @ x = sum_i Krylov(A, G[i]) @ Krylov(B, H[i]) @ x

    where Krylov(A, v) is the n×n matrix whose columns are
    [v, A@v, A^2@v, ..., A^{n-1}@v] and A is the shift-down-by-one operator
    weighted by ``subd_A`` (subdiagonal entries). Concretely, A's only
    non-zero entries are A[i+1, i] = subd_A[i] for i in 0..n-2.

    The slow ``Krylov(linear_map, v)`` reference is at
    torch_structured/structured/krylov.py:264-272 — use that exact loop
    here, NOT the fast FFT-based ``krylov_multiply``, so the hand-rolled
    matrix is provably independent from the production path.
    """
    r, n = G.shape

    # Build A as an explicit (n, n) subdiagonal matrix.
    A = torch.zeros(n, n, dtype=G.dtype, device=G.device)
    A[1:, :-1].fill_diagonal_(0)
    A[torch.arange(1, n), torch.arange(n - 1)] = subd_A
    B = torch.zeros(n, n, dtype=H.dtype, device=H.device)
    B[torch.arange(1, n), torch.arange(n - 1)] = subd_B

    # Krylov(A, v) = [v, A@v, A^2@v, ...] (column-stacked).
    def _krylov_explicit(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Mirrors krylov.py:264-272 (slow form). v shape (n,) -> (n, n).
        cols = [v]
        for _ in range(n - 1):
            v = M @ v
            cols.append(v)
        return torch.stack(cols, dim=-1)  # (n, n)

    # M = sum_i K_A(G[i]) @ K_B(H[i]).T   (per the LDR displacement formula;
    # see krylov.py:245 — out = K_A(G[i]) @ K_B(H[i]).T @ x summed over i).
    # Planner: VERIFY the transpose convention against subdiag_mult's call
    # to krylov_transpose_multiply (krylov.py:257) — the K^T may be on H,
    # not on K_B. Build both candidates, sanity-check against a small case
    # vs. _LDRLinear, pick the one that matches.
    M = torch.zeros(n, n, dtype=G.dtype, device=G.device)
    for i in range(r):
        K_A = _krylov_explicit(A, G[i])  # (n, n)
        K_B = _krylov_explicit(B, H[i])  # (n, n)
        M = M + K_A @ K_B.T
    return M
```

**NEW PATTERN FLAG (D-32 / external library spec):** This helper reads from `torch_structured/structured/layers.py:211-225` (LDRSubdiagonal class) and `torch_structured/structured/krylov.py:245-259` (the `subdiag_mult` definition the production path calls). **The reference impl is informed by an EXTERNAL library's source code**, not an in-repo analog. The planner / executor will need to:

1. Run `python -c "import torch_structured; print(torch_structured.__file__)"` to locate the install path (confirmed at `/home/claroche/miniconda3/lib/python3.13/site-packages/torch_structured/` on this system).
2. Read `torch_structured/structured/layers.py` (`LDRSubdiagonal.forward` at line 223 calls `kry.subdiag_mult(self.subd_A, self.subd_B, self.G, self.H, x)`).
3. Read `torch_structured/structured/krylov.py:245-259` (the `subdiag_mult` docstring is the LDR displacement formula spec: `sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x`).
4. Read `torch_structured/structured/krylov.py:264-272` (the slow `Krylov(linear_map, v, m=None)` explicit construction — this is the natural hand-rolled reference; the production path uses an FFT-based fast version which is what we're auditing).
5. **Verify on a single small case** (`H=8, r=2, batch=1`) that `_build_ldr_matrix_from_factors(...) @ x` agrees with `_LDRLinear(LDRSubdiagonal(8, r=2))(x)` to < 1e-5 BEFORE wiring it into the parametrized grid. If it doesn't agree, adjust the `K_A @ K_B.T` vs `K_A @ K_B` convention (see the TRANSPOSE warning in the helper docstring above).

Parameter shapes for `LDRSubdiagonal(layer_size=n, r=r)` (from `torch_structured/structured/layers.py:131-140, 215-221`):
- `subd_A`, `subd_B`: shape `(n-1,)`, initialized to ones (line 217, 221).
- `G`, `H`: shape `(r, n)`, initialized via `torch.nn.init.normal_(std=init_stddev)` (line 136-137).

---

#### Parametrize style for shape grids (D-36)

**Analog:** `tests/test_layer_parity.py:155-175` for FAST/SLOW split and `tests/test_structure.py:73-76` for `@pytest.mark.parametrize("kind", KINDS)` — but Phase 3 does NOT use the `KINDS` parametrize because circulant and LDR tests have different shape constraints (LDR adds `rank`). Use kind-specific named tests with their own grids per CONTEXT D-36:

```python
# Circulant: square; power-of-2 (per src/gru_qat/structure.py:95-98 validator).
FAST_CIRC_GRID = [
    (B, H)
    for B in (1, 4, 32)
    for H in (8, 32, 128)
]  # 9 cases
SLOW_CIRC_GRID = [
    (B, H)
    for B in (1, 4, 32)
    for H in (512,)
]  # 3 cases

# LDR: square; rank ≤ H. n must be power-of-2 for the production fast path
# (krylov.py:249-251 zero-pads otherwise; we test the non-padded case).
FAST_LDR_GRID = [
    (B, H, rank)
    for B in (1, 4, 32)
    for H in (8, 32, 128)
    for rank in (1, 4, 8)
    if rank <= H
]  # ~27 cases
SLOW_LDR_GRID = [
    (B, H, rank)
    for B in (1, 4, 32)
    for H in (512,)
    for rank in (1, 4, 8)
]  # 9 cases
```

Notes:
- CONTEXT D-36 spec: H ∈ {8, 32, 128, 512}, B ∈ {1, 4, 32}, LDR rank ∈ {1, 4, 8}. Total ~12 fast + 3 slow circulant; ~36 fast + 9 slow LDR per test family (forward + backward = ~30 + ~90 cases).
- **No T dimension** in Phase 3 grids — Phase 3 audits the linear layer in isolation (the GRU time loop is Phase 1's territory). `_CirculantLinear` and `_LDRLinear` are stateless single-step modules.
- Fast vs slow split via `@pytest.mark.slow` per `tests/test_qat_smoke.py:88` / `tests/test_layer_parity.py:560-561`.

---

#### Self-consistency test (D-29)

`test_handrolled_circulant_self_consistent` — runs BEFORE the parity test family to catch reference-math bugs.

**Analog:** No direct in-repo analog (Phase 1 and Phase 2 don't have a "two-reference cross-check" — they compare a single hand-rolled or library path to the production). Mechanically simple:

```python
@pytest.mark.parametrize("B,H", FAST_CIRC_GRID)
def test_handrolled_circulant_self_consistent(B: int, H: int) -> None:
    """Before comparing either hand-rolled form to ``_CirculantLinear``,
    assert that the two hand-rolled forms agree with each other to < 1e-5
    abs. Catches algebra mistakes in ``_build_toeplitz_from_kernel`` and
    ``_circulant_via_fft`` BEFORE they masquerade as a production-path bug.
    """
    torch.manual_seed(0)
    c = torch.randn(H) / (H ** 0.5)
    x = torch.randn(B, H)

    C = _build_toeplitz_from_kernel(c)
    # y_toeplitz: per CONTEXT D-29, y = x @ C.T (NOT x @ C — the matrix
    # _CirculantLinear applies is C as defined, and y = C @ x in column-
    # vector convention. With x shape (B, H), torch convention is y[b, :] =
    # x[b, :] @ C.T which is the same as (C @ x[b, :].T).T.)
    y_toep = x @ C.T

    y_fft = _circulant_via_fft(c, x)

    max_diff = (y_toep - y_fft).abs().max().item()
    assert max_diff < 1e-5, f"toeplitz vs fft self-consistency: {max_diff:.4e} (B={B},H={H})"
```

**Planner sanity check:** The `x @ C.T` vs `x @ C` convention depends on what `_CirculantLinear.forward` actually computes. The production code at `src/gru_qat/structure.py:219-225` is:

```python
def forward(self, x):
    col_f = torch.fft.rfft(self.col)
    x_f = torch.fft.rfft(x, dim=-1)
    y = torch.fft.irfft(col_f * x_f, n=self.n, dim=-1)
    ...
```

This computes `y[b, k] = sum_j col[(k-j) mod n] * x[b, j]` — i.e., circular convolution of `col` with `x`. The Toeplitz matrix that represents this operation is `C[k, j] = col[(k - j) mod n]` (first column = col). With that definition, `y = x @ C.T` is the correct PyTorch idiom (transpose because matrix multiplies row vectors on the right). **The helper above uses `c[(j-i) mod H]` indexing — that's the convention where C's first ROW is `col` (i.e., the transpose of what production computes).** Reconcile this before the self-consistency test goes red:

Two options for the helper:
- **(a)** `_build_toeplitz_from_kernel(c)` returns C with `C[i,j] = c[(i-j) mod H]` (first column = c), then `y_toep = x @ C.T`.
- **(b)** `_build_toeplitz_from_kernel(c)` returns C with `C[i,j] = c[(j-i) mod H]` (first row = c), then `y_toep = x @ C`.

These produce the same `y_toep` (b is the transpose of a). Pick whichever reads clearer. The CONTEXT D-29 spec is silent on this convention — Claude's discretion per CONTEXT.

---

#### Forward parity test (D-29 / D-31)

`test_circulant_matches_handrolled_toeplitz` — production vs hand-rolled. Mirrors `tests/test_layer_parity.py:185-205` (forward parity body) but compares against the hand-rolled matrix, not against `nn.GRU`:

```python
@pytest.mark.parametrize("B,H", FAST_CIRC_GRID)
def test_circulant_matches_handrolled_toeplitz(B: int, H: int) -> None:
    """``_CirculantLinear(col=c).forward(x)`` must match the explicit
    Toeplitz matrix construction ``x @ C.T`` (with C built from c via
    ``_build_toeplitz_from_kernel``) to < 1e-5 abs across the fast grid.

    Strict-tier bound (D-40): pure-PyTorch fp32 with 'highest' precision,
    no TF32, no STE, no nonlinearities — algebraic equality between two
    paths that compute the same circular convolution.
    """
    torch.manual_seed(0)
    layer = _CirculantLinear(H, bias=False)
    # The layer initializes col internally; read it out for the reference.
    c = layer.col.detach().clone()

    x = torch.randn(B, H)

    y_prod = layer(x)
    C = _build_toeplitz_from_kernel(c)
    y_ref = x @ C.T  # matches the indexing convention chosen above

    max_diff = (y_prod - y_ref).abs().max().item()
    # Absolute error (no TF32, so no need for relative-error floor).
    assert max_diff < 1e-5, f"circulant fwd max abs diff {max_diff:.4e} (B={B},H={H})"
```

---

#### Backward parity test (D-30 — autograd-grad comparison)

**Primary analog:** `tests/test_layer_parity.py:486-557` (`test_layer_backward_matches_nn_gru`). The detach-clone-requires_grad-twice pattern is at lines 516-519, the shared `g = torch.randn_like(out_ref)` pattern is at line 528, and the per-tensor named-failure loop is at lines 546-557.

```python
@pytest.mark.parametrize("B,H", FAST_CIRC_GRID)
def test_circulant_backward_matches_autograd_reference(B: int, H: int) -> None:
    """Backward parity: gradient w.r.t. kernel c. Build the Toeplitz matrix
    C from c with ``c.requires_grad_(True)``, compute ``y_ref = x @ C.T``,
    backprop a shared random ``g``, extract ``c.grad``. Compute the same
    on ``_CirculantLinear`` and assert the two gradients agree to < 1e-5
    abs.

    Per D-30 / D-37 — uses autograd-vs-autograd, no manual gradient math.
    """
    torch.manual_seed(0)
    c_init = torch.randn(H) / (H ** 0.5)

    # Two leaves, independent autograd graphs (detach-clone idiom from
    # tests/test_layer_parity.py:516-519).
    c_ref = c_init.detach().clone().requires_grad_(True)
    c_prod = c_init.detach().clone().requires_grad_(True)
    x = torch.randn(B, H)
    g = torch.randn(B, H)  # shared downstream gradient signal

    # Reference path: build C as a function of c_ref (so autograd flows back).
    C = _build_toeplitz_from_kernel(c_ref)
    y_ref = x @ C.T
    y_ref.backward(g)

    # Production path: install c_prod as the layer's parameter directly.
    layer = _CirculantLinear(H, bias=False)
    with torch.no_grad():
        layer.col.copy_(c_prod)
    # Re-attach autograd: replace the Parameter's tensor with c_prod so the
    # backward populates c_prod.grad rather than layer.col.grad. Simpler:
    # use layer.col as the leaf directly.
    layer.col = torch.nn.Parameter(c_prod)  # re-assign so the layer uses c_prod
    y_prod = layer(x)
    y_prod.backward(g)

    # Named per-tensor failure messages (D-30 spec: "kernel_c").
    for name, ref_t, prod_t in [("kernel_c", c_ref.grad, c_prod.grad)]:
        max_diff = (ref_t - prod_t).abs().max().item()
        assert max_diff < 1e-5, f"{name} max abs diff {max_diff:.4e} (B={B},H={H})"
```

Planner note: the `layer.col = nn.Parameter(c_prod)` re-assignment is correct PyTorch, but unusual. Alternative idiom: clone c_prod into `layer.col.data` first, then run backward and read `layer.col.grad` — this is what `tests/test_parity.py:18-44` does. Both work; pick whichever reads clearer.

---

#### LDR forward + backward parity tests (D-32)

Same shape as the circulant tests, but the reference comes from `_build_ldr_matrix_from_factors`:

```python
# Module-top: LDR needs torch_structured (the production path's LDRSubdiagonal
# is imported from there). Module-top skip per tests/test_structure.py:25.
torch_structured = pytest.importorskip("torch_structured")

from torch_structured.structured.layers import LDRSubdiagonal  # noqa: E402


@pytest.mark.parametrize("B,H,rank", FAST_LDR_GRID)
def test_ldr_matches_handrolled_reference(B: int, H: int, rank: int) -> None:
    """``_LDRLinear(LDRSubdiagonal(H, r=rank))(x)`` must match ``x @ M.T``
    where M is the dense matrix built from the layer's (subd_A, subd_B, G, H)
    factors via ``_build_ldr_matrix_from_factors``.

    The dense matrix is constructed from the explicit Krylov formula at
    torch_structured/structured/krylov.py:264-272 (slow form); the
    production path uses the FFT-based fast form at krylov.py:245-259.
    The two should agree algebraically — < 1e-5 abs under 'highest'.
    """
    torch.manual_seed(0)
    ldr = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    layer = _LDRLinear(ldr)

    # Read factors out of the layer for the hand-rolled reference.
    subd_A = ldr.subd_A.detach().clone()
    subd_B = ldr.subd_B.detach().clone()
    G = ldr.G.detach().clone()
    H_factor = ldr.H.detach().clone()

    x = torch.randn(B, H)

    y_prod = layer(x)
    M = _build_ldr_matrix_from_factors(subd_A, subd_B, G, H_factor)
    y_ref = x @ M.T  # planner: verify M vs M.T convention against production output

    max_diff = (y_prod - y_ref).abs().max().item()
    assert max_diff < 1e-5, f"ldr fwd max abs diff {max_diff:.4e} (B={B},H={H},rank={rank})"
```

Backward follows the same detach-clone-twice pattern as circulant (per D-32 last line: "Backward via the same autograd-gradient comparison pattern as D-30"), but the leaves to grad-check are all four of {`subd_A`, `subd_B`, `G`, `H`} — four named entries in the per-tensor loop instead of one.

---

#### STR-03 missing-dep test (D-34) — INTRODUCES `monkeypatch`

**NEW PATTERN FLAG:** Per `TESTING.md:201-203`: **"Mocking: None. No `unittest.mock`, no `pytest-mock`, no monkeypatching."** A grep across `tests/` and `src/` confirms zero existing uses of `monkeypatch`. **Phase 3 introduces this convention.** The planner should:
- Document this in PLAN.md as a NEW test convention.
- Update `.planning/codebase/TESTING.md` "Mocking" section after Phase 3 closes (Phase 3 SUMMARY task).
- Use `pytest`'s built-in `monkeypatch` fixture — no new dep needed (`pytest-mock` is NOT required).

**Pattern source (external — pytest docs):** `https://docs.pytest.org/en/stable/how-to/monkeypatch.html` for `monkeypatch.setattr`. Standard pytest API; no in-repo precedent.

**The error to assert** (from `src/gru_qat/structure.py:60-69`):
```python
def _import_torch_structured():
    """Soft-import torch_structured. Raises a clear error on missing dep."""
    try:
        import torch_structured as ts  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "torch-structured is required for structured GRU weights. "
            "Install with: pip install 'gru-qat[structured]'"
        ) from e
    return ts
```

The install hint string is `pip install 'gru-qat[structured]'` (NOT `pip install torch-structured`). D-34's CONTEXT says "match the string 'torch-structured'" — that matches the message (which starts with "torch-structured is required..."), so `match=r"torch-structured"` works on the existing error.

```python
def _raise_missing_torch_structured() -> None:
    """Stand-in for _import_torch_structured that always raises the
    ImportError the production helper would raise on a missing install.
    Matches the production message verbatim (src/gru_qat/structure.py:65-68)
    so the test asserts the user-facing string, not just any ImportError.
    """
    raise ImportError(
        "torch-structured is required for structured GRU weights. "
        "Install with: pip install 'gru-qat[structured]'"
    )


@pytest.mark.parametrize("kind", ["monarch", "butterfly", "ldr"])
def test_missing_torch_structured_raises_clear_error(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per STR-03: when ``torch-structured`` is missing, each optional-dep
    structured kind must raise ImportError with a message that mentions
    ``torch-structured`` (so the user knows what to install).

    Simulated via ``monkeypatch.setattr`` on the lazy-import helper at
    ``src/gru_qat/structure.py:60`` — in-process, fast, no need to uninstall
    the actual library. Resets cleanly between tests via pytest's fixture.
    """
    monkeypatch.setattr(
        "gru_qat.structure._import_torch_structured",
        _raise_missing_torch_structured,
    )
    # Special case: 'ldr' has a DIFFERENT import path — it imports
    # `from torch_structured.structured.layers import LDRSubdiagonal` directly
    # at src/gru_qat/structure.py:164, NOT through `_import_torch_structured`.
    # So the monkeypatch on _import_torch_structured will NOT affect 'ldr'.
    # Planner: either (a) also monkeypatch the LDR import path, or (b) document
    # the LDR exception in the test and assert against the LDR-specific
    # ImportError at src/gru_qat/structure.py:166-169 ("torch-structured is
    # required for kind='ldr'").
    # See `src/gru_qat/structure.py:160-172` for the LDR import branch.

    cfg = StructureConfig(kind=kind, nblocks=4, butterfly_nblocks=1, ldr_rank=2)
    with pytest.raises(ImportError, match=r"torch-structured"):
        make_structured_linear(cfg, 32, 32)
```

**Planner: resolve the LDR import-path divergence.** The CONTEXT D-34 spec parametrizes over `{"monarch", "butterfly", "ldr"}` and says "each must raise ImportError with a message containing 'torch-structured'". The LDR branch at `src/gru_qat/structure.py:160-172` does:

```python
if cfg.kind == "ldr":
    try:
        from torch_structured.structured.layers import LDRSubdiagonal
    except ImportError as e:
        raise ImportError(
            "torch-structured is required for kind='ldr'. "
            "Install with: pip install 'gru-qat[structured]'"
        ) from e
```

So LDR's missing-dep path is **independent** of `_import_torch_structured`. Two options:
- **(a)** Test LDR via a separate `monkeypatch` that patches `torch_structured.structured.layers` import (more invasive — needs `monkeypatch.setitem(sys.modules, ...)`).
- **(b)** Skip LDR in this test family, document the asymmetry, and add a separate non-monkeypatched test that asserts the LDR error message format.

Either is reasonable. Recommend (a) for STR-03 completeness; the `sys.modules` patch is the canonical way to simulate a missing import (see pytest docs).

---

#### STR-03 companion test (D-34)

```python
@pytest.mark.parametrize("kind", ["circulant", "diagonal", "dense"])
def test_local_impls_work_without_torch_structured(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per STR-03: local-impl kinds (circulant, diagonal, dense) must NOT
    depend on torch-structured. Simulate the missing dep via the same
    monkeypatch as the failing-kinds test, then assert the layer builds
    and produces finite output.
    """
    monkeypatch.setattr(
        "gru_qat.structure._import_torch_structured",
        _raise_missing_torch_structured,
    )
    cfg = StructureConfig(kind=kind)
    layer = make_structured_linear(cfg, 32, 32)
    x = torch.randn(4, 32)
    y = layer(x)
    assert torch.isfinite(y).all(), f"{kind} produced non-finite output without torch_structured"
    assert y.shape == (4, 32)
```

---

## Shared Patterns

### `'highest'` module-level precision preamble

**Source:** `tests/test_layer_parity.py:28-33` (Phase 1 output).
**Apply to:** Top of `tests/test_structure_parity.py`, immediately after the `import torch` statement (before any `pytest.importorskip` blocks).

```python
# Per CONTEXT D-40: Phase 3 paths are pure PyTorch (no tl.dot), so 'highest'
# is achievable and < 1e-5 abs is the strict bound.
torch.set_float32_matmul_precision("highest")
```

**Why module-level not per-test:** `set_float32_matmul_precision` is global state. Setting it once at import is the cleanest signal and matches `tests/test_layer_parity.py:33` exactly.

### Absolute-error assertion (strict tier)

**Source:** Phase 2 strict-tier files; CONTEXT D-40 ("the strict `< 1e-5` bound is achievable").

```python
max_diff = (ref - prod).abs().max().item()
assert max_diff < 1e-5, f"<name> max abs diff {max_diff:.4e} (<shape-vars>)"
```

**Apply to:** All forward and backward parity assertions in the new file. No `1e-6` relative-error floor needed — fp32-vs-fp32 with `'highest'` and no TF32 should produce values larger than the floor anyway, and the strict tier is auditing math.

### Module-level underscore helpers (no `conftest.py`)

**Source:** `TESTING.md:198-200`, plus universal convention across `tests/test_triton_*.py:35,51`.
**Apply to:** `_build_toeplitz_from_kernel`, `_circulant_via_fft`, `_build_ldr_matrix_from_factors`, `_raise_missing_torch_structured`. Single underscore, fully typed, no class wrapper.

### `torch.manual_seed(0)` per test

**Source:** `tests/test_layer_parity.py:191, 308, ...` and TESTING.md "Bench-Style Smoke Tests vs. Correctness Tests" (line 136).
**Apply to:** Top of every test function in the new file that uses randomness. Reset at the top of the body, not at module scope.

### Detach-clone-twice for two independent autograd graphs

**Source:** `tests/test_layer_parity.py:480-484, 516-519`.

```python
x_ref = torch.randn(B, H, requires_grad=True)
x_ours = x_ref.detach().clone().requires_grad_(True)
```

**Apply to:** Both circulant and LDR backward tests. **Critical:** sharing a single requires_grad leaf across both forward paths builds one autograd tape over both and the second `.backward(g)` raises `RuntimeError: Trying to backward through the graph a second time`. The detach-clone idiom is non-negotiable.

### Shared downstream gradient `g`

**Source:** `tests/test_layer_parity.py:524-528`.

```python
g = torch.randn_like(out_ref)
out_ref.backward(g)
out_ours.backward(g)
```

Per the `test_layer_parity.py:498-501` docstring: *"Uses a shared random g so both autograd graphs see the same upstream gradient signal — every output element contributes independently, which is more discriminating than `out.sum().backward()`."*

**Apply to:** Both backward parity tests. Don't use `loss = out.sum(); loss.backward()` — less diagnostic.

### Per-tensor named-failure loop

**Source:** `tests/test_layer_parity.py:546-557`.

```python
for name, ref_t, prod_t in [
    ("kernel_c", c_ref.grad, c_prod.grad),
    # ...
]:
    max_diff = (ref_t - prod_t).abs().max().item()
    assert max_diff < 1e-5, f"{name} max abs diff {max_diff:.4e} (B={B},H={H})"
```

**Apply to:** Backward tests with multiple gradient tensors (LDR has 4: `subd_A`, `subd_B`, `G`, `H`). The `{name}` in the failure message is the single most-diagnostic piece of information when a bd issue is filed.

### `pytest.raises(<Error>, match="...")` for error-path tests

**Source:** `tests/test_parity.py:186, 204` and `tests/test_structure.py:195, 208, 243, 251, 257`.

```python
with pytest.raises(ImportError, match=r"torch-structured"):
    make_structured_linear(cfg, 32, 32)
```

**Apply to:** STR-03 missing-dep tests. Use `match=` with a regex, not a literal substring (the production message has variable whitespace). The `r"torch-structured"` regex is intentionally loose per CONTEXT D-34 (match string is sufficient).

### `@pytest.mark.slow` for long-H grid cases

**Source:** `tests/test_qat_smoke.py:88` (registered in `pyproject.toml:44-45`).
**Apply to:** Each fast test family gets a `_slow` sibling parametrized over `SLOW_CIRC_GRID` / `SLOW_LDR_GRID` (H=512). Same body, slow decorator.

### Two-commit failing-test-before-fix discipline (carried from Phase 1 / Phase 2)

**Source:** Phase 1 PATTERNS lines 393-399; CONTEXT D-37.
**Apply to:** Any parity failure surfaced by Phase 3.
1. Commit A: failing test only, no `src/` changes. Capture `pytest --tb=short` tail.
2. `bd create` per finding; `bd update <id> --notes <pytest-tail>`.
3. Commit B: fix in `src/gru_qat/structure.py` (no test changes); test passes; `bd close <id>` after CI green.
4. **Never** `@pytest.mark.xfail` — silent in `pytest -q`, defeats audit signal.

### Locked files per D-28 / D-38

**Source:** CONTEXT D-38.
**Apply to:** Verifier asserts `git diff <phase-3-base>..HEAD -- tests/test_parity.py tests/test_layer_parity.py` is empty across Phase 3 commits. Pattern: do NOT import helpers FROM these locked files into the new file (duplicate per D-18 from Phase 2 — small helpers re-inlined).

---

## No Analog Found

| Test / Pattern | Role | Data Flow | Why No Analog |
|----------------|------|-----------|---------------|
| `monkeypatch` usage in `tests/` | test infrastructure | request-response (monkeypatched _import call) | `TESTING.md:201-203` explicitly documents "No `unittest.mock`, no `pytest-mock`, no monkeypatching" — Phase 3 INTRODUCES this convention. Use pytest's built-in `monkeypatch` fixture (no new dep). After Phase 3 closes, update TESTING.md to lift the "no monkeypatching" restriction or document the exception. **Planner: add this as a phase SUMMARY task.** |
| `torch.fft.fft` / `torch.fft.ifft` (full complex) in test code | math reference | request-response (FFT-based circulant reference) | The only existing `torch.fft.*` usage in the repo is the production `_CirculantLinear.forward` at `src/gru_qat/structure.py:220-222` (which uses `rfft`/`irfft`). Phase 3 introduces `fft`/`ifft` (full complex) in test code as the genuinely independent FFT path for the self-consistency check. Pattern is mechanically standard — no analog needed, but flag as a new test-side import. |
| `_build_ldr_matrix_from_factors` reading torch_structured source | math reference | external-library-spec-read | No in-repo precedent for "build a reference impl by reading an external library's source." The pattern is:<br>1. Locate install: `python -c "import torch_structured; print(torch_structured.__file__)"` → `/home/claroche/miniconda3/lib/python3.13/site-packages/torch_structured/`.<br>2. Read `torch_structured/structured/layers.py:211-225` (the `LDRSubdiagonal` class — see params at lines 215-221, forward at 223).<br>3. Read `torch_structured/structured/krylov.py:245-259` (`subdiag_mult` — the displacement-rank formula spec).<br>4. Read `torch_structured/structured/krylov.py:264-272` (slow `Krylov` construction — natural hand-rolled reference).<br>5. Verify on a single H=8, r=2 case BEFORE wiring into the grid.<br>**This is a non-codebase analog and must be called out in PLAN.md so the executor knows to read external source.** |
| `pytest.MonkeyPatch` type annotation | test infrastructure | — | No existing `pytest.MonkeyPatch` annotation anywhere. Standard pytest API; safe to introduce. |
| Two-reference cross-check (FFT + Toeplitz before either compared to production) | test methodology | request-response | Phase 1 and Phase 2 compare a SINGLE hand-rolled or library path to production. The "FFT vs Toeplitz self-consistency first, THEN compare the verified reference to production" pattern is novel. Mechanically just an extra parametrized test; rationale is documented inline in the test docstring (catches reference-math bugs before they're misread as production-path bugs). |

No `src/` files in the no-analog table — per CONTEXT D-39, no `src/` modifications are expected unless a parity test surfaces a real bug.

---

## Metadata

**Analog search scope:**
- `tests/` — read `test_structure.py` (full), `test_layer_parity.py` (imports + backward test body lines 475-557), `test_triton_diagonal.py` (helper conventions lines 1-90), `test_triton_monarch.py` (helper conventions lines 1-90), `test_parity.py` (referenced via Phase 1 PATTERNS for `pytest.raises` idiom).
- `src/gru_qat/structure.py` — full file, focus on `_CirculantLinear` (lines 207-225), `_LDRLinear` (lines 239-247), `_import_torch_structured` (lines 60-69), LDR import branch (lines 160-172), `make_structured_linear` factory (lines 118-174).
- External: `torch_structured/structured/layers.py:180-269` (LDR class hierarchy), `torch_structured/structured/krylov.py:200-290` (`subdiag_mult` and explicit `Krylov` construction).
- `.planning/codebase/TESTING.md` (full — read for "Mocking: None" confirmation at line 201, parametrize style, `pytest.importorskip` idiom, no-conftest convention).
- `.planning/phases/01-reference-path-parity-vs-nn-gru/01-PATTERNS.md` (full — read for shared patterns continuity).
- `.planning/phases/02-triton-fast-path-parity-vs-reference/02-PATTERNS.md` (full — read for strict-tier conventions; Phase 3 inherits the absolute-error and `'highest'` patterns but NOT the TF32 disposition per D-40).

**Grep audit:**
- `grep "torch.fft" tests/ src/` → only `src/gru_qat/structure.py:220-222`. New pattern for tests.
- `grep "monkeypatch" tests/ src/` → zero hits. New pattern for the repo.
- `grep "pytest.raises" tests/` → 10 hits, idiom well-established.

**Files scanned:** 7 in-repo (5 test files, 1 src module, 1 conventions doc) + 2 external library files + 2 prior phase pattern maps.

**Pattern extraction date:** 2026-05-13.
