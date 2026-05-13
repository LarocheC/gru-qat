# Phase 3: Structured PyTorch fallback parity - Discussion Log

> **Audit trail only.** Decisions are captured in CONTEXT.md.

**Date:** 2026-05-13
**Phase:** 3-structured-pytorch-fallback-parity
**Areas discussed:** Hand-rolled circulant reference, Hand-rolled LDR reference, STR-03 graceful-degradation testing, File location

---

## Hand-rolled circulant reference

### Formulation

| Option | Description | Selected |
|--------|-------------|----------|
| Both (FFT ↔ Toeplitz cross-check, then compare to production) | Tier 1: self-consistency test. Tier 2: production vs Toeplitz. | ✓ |
| Toeplitz only | Simplest; mathematically unambiguous. | |
| FFT only | Closer to production impl; risks self-testing. | |

**User's choice:** Both. Captured as D-29.

### Backward parity rigor

| Option | Description | Selected |
|--------|-------------|----------|
| Compare gradients via autograd vs hand-rolled | autograd grad on hand-rolled C vs production .grad. | ✓ |
| Forward parity only | Skip backward. | |

**User's choice:** Autograd-grad comparison. Captured as D-30.

---

## Hand-rolled LDR reference

### Formulation

| Option | Description | Selected |
|--------|-------------|----------|
| Build full H×H matrix from G_k, S_k, then matmul | Simple, slow, obviously correct. | ✓ |
| Displacement-operator form directly | Faster, mirrors production. Risks shared bugs. | |
| Read torch-structured source + reimplement independently | Strongest independence; most work. | |

**User's choice:** Full matrix build. Captured as D-32. (Implicitly per the chosen approach we DO need to read torch-structured to identify the displacement formula — noted in D-32 itself.)

### Edge cases

| Option | Description | Selected |
|--------|-------------|----------|
| Existing shape-validator tests cover this | Trust test_structure.py's existing coverage. | ✓ |
| Add edge tests in this phase | Would be scope creep into Phase 6. | |

**User's choice:** Trust existing. Captured as D-33.

---

## STR-03 graceful-degradation testing

### Simulation approach

| Option | Description | Selected |
|--------|-------------|----------|
| monkeypatch on _import_torch_structured | In-process, fast. | ✓ |
| subprocess with sanitized PYTHONPATH | Most realistic; CI-fragile. | |
| pytest-import-error plugin | Overkill. | |

**User's choice:** monkeypatch. Captured as D-34.

### Coverage scope

| Option | Description | Selected |
|--------|-------------|----------|
| All optional-dep kinds: monarch + butterfly + ldr | Parametrize the missing-dep test. | ✓ |
| One representative test (just monarch) | Faster; trusts shared code path. | |

**User's choice:** All three. Captured as D-34 (the "test family" bullet).

---

## File location

### Where Phase 3 tests live

| Option | Description | Selected |
|--------|-------------|----------|
| New tests/test_structure_parity.py | Mirrors Phase 1/2 naming. Clear separation. | ✓ |
| Extend tests/test_structure.py | One place for all structured tests; mixes tiers. | |

**User's choice:** New file. Captured as D-35.

### Plan structure preview

| Option | Description | Selected |
|--------|-------------|----------|
| 2-3 plans | Plan 03-01 circulant + STR-03; Plan 03-02 LDR; Plan 03-03 audit-kickoff (optional fold-in). | ✓ |
| Let the planner decide | — | |

**User's choice:** 2-3 plans guidance. Captured as D's "Claude's Discretion" — planner will choose 2 vs 3.

---

## Claude's Discretion

- Exact pytest.parametrize id strings.
- Shared helpers across Phase 1/2/3 strict-tier tests — default inline per file.
- "install hint" string match form (`match=r"torch-structured"` is enough).
- Whether to fold the audit-kickoff into Plan 03-02 or split into a separate Plan 03-03.

## Deferred Ideas

- LDR edge cases (rank > H, rank = 0, non-square) — Phase 6.
- Performance comparison (Toeplitz vs _CirculantLinear vs FFT) — out of scope.
- Quant-on for circulant / LDR — Phase 4.
- Long-T accumulation drift on structured layers — Phase 6.
- Pre-existing mypy/ruff debt — already bd:gru-triton-4m6.
