# Phase 7: Audit report + findings handling - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the parity-audit milestone. Two deliverables:

1. **Findings handling** — triage all 14 open `bd` issues from Phases 1–6 into a disposition framework; fix the genuine bugs in-phase, document the irreducible divergences, and leave `bd ready` empty (every issue closed or deferred-to-v2 with a reference).
2. **`AUDIT-REPORT.md`** — a repo-root report summarizing what was checked across all 28 v1 requirements, what passed, what was fixed, and the residual known-but-accepted divergences with rationale.

Phase 7 is primarily a **closure** phase, but the user has scoped it to also **fix every genuine bug** (not just document) — see D-02. The one thing Phase 7 explicitly does NOT do is the `input_precision="ieee"` TF32-elimination kernel rewrite (out of scope — see D-03 and Deferred Ideas).

**Requirements:** RPT-01, RPT-02, RPT-03.
</domain>

<decisions>
## Implementation Decisions

### A. Finding disposition framework (the 14 open bd issues)

- **D-01:** Triage all 14 open `bd` issues into three buckets with explicit criteria:
  - **FIX** — a genuine code bug with a tractable, non-TF32-rooted fix. Fixed in-phase.
  - **ACCEPTED-DIVERGENCE** — an *irreducible* divergence whose root cause is the TF32 `tl.dot` reduction-order non-associativity. Documented, not code-fixed (the fix would be the out-of-scope `ieee` rewrite). bd issue closed with a resolution note pointing at the AUDIT-REPORT section.
  - **INDIVIDUAL** — issues that fit neither: hardware limits and process findings, dispositioned case-by-case (see D-04).

- **D-02 (FIX bucket — fix in-phase):** Phase 7 fixes every genuine tractable bug. Known FIX-bucket members:
  - `gru-triton-n20` — shared `QuantizerConfig` instance silent-correctness bug (see D-07).
  - `gru-triton-7rj` — `scan*.py` `gru_scan*` wrappers use `assert` for shape validation (stripped under `python -O`); convert to `if … raise ValueError` mirroring the `GRULayer.forward` guard convention.
  - Any dense/diagonal backward-drift issue (`gru-triton-mjy`, `lht`, `e7t`, `fpl`) where the planner/researcher determines the failure has a tractable non-TF32 root cause — those get FIXed. Where the failure is purely TF32-rooted, it moves to ACCEPTED-DIVERGENCE instead. The FIX-vs-DIVERGENCE call per issue is delegated to research/planning, which must inspect each issue's actual root cause.
  - All FIX-bucket work follows the D-37/D-50 two-commit discipline: failing test (Commit A) committed BEFORE the fix (Commit B), verifiable in `git log`. **No `@pytest.mark.xfail`.**

- **D-03 (ACCEPTED-DIVERGENCE bucket):** The irreducible TF32 `tl.dot` reduction-order family — candidate members `gru-triton-in0`, `q3k`, `lqk`, `5rk`, and any of `mjy`/`lht`/`e7t`/`fpl` that research confirms are purely TF32-rooted. These are NOT code-fixed. The `input_precision="ieee"` kernel rewrite that would eliminate them is **explicitly out of scope** — PROJECT.md's locked "Option C / tiered tolerance" Key Decision already accepted this root cause. Each ACCEPTED-DIVERGENCE issue is closed with a resolution note referencing the AUDIT-REPORT's residual-divergences section.

- **D-04 (INDIVIDUAL bucket):**
  - `gru-triton-e0l` — Monarch bwd SMEM OOM / `tl.dot` K<16 on consumer GPUs (RTX 2000 Ada). A genuine **hardware limit**, not code-fixable in-milestone without a kernel-tiling redesign. Disposition: documented HW-limit, covered by the existing `_skip_if_monarch_bwd_hw_limit` skip; AUDIT-REPORT records it; the kernel-tiling redesign is deferred to v2.
  - `gru-triton-u00` — F-04-05-D parallel-execution race. A **process** finding, not a code bug — already mitigated by the single-plan discipline adopted in Phases 5–6. Disposition: AUDIT-REPORT process-note; close (no code change).

### B. ROADMAP success criterion #3 — honest green gate

- **D-05:** Introduce a `divergence` pytest marker, registered in `pyproject.toml [tool.pytest.ini_options] markers`. Mark every strict-tier test case whose failure is an irreducible TF32 ACCEPTED-DIVERGENCE (D-03). Criterion #3's green gate is operationalized as:
  - CPU / CUDA fast tier: `pytest -q -m "not divergence"` → passes green.
  - Slow tier: `pytest -m "slow and not divergence" -q` → passes green.
  - The `divergence`-marked tests stay **live** (collected and run, NOT skipped, NOT `xfail`) — running `pytest -m divergence` reproduces the documented divergence on demand. They are executable documentation, not hidden failures.
  - **This is an explicit operationalization of criterion #3**, recorded in `AUDIT-REPORT.md` (and noted in ROADMAP). Criterion #3 as literally written ("`pytest -q` passes") is met by the `-m "not divergence"` gate; the reinterpretation is intentional and documented, not a silent loosening.

### C. mypy / ruff (criterion #3, second half — gru-triton-4m6)

- **D-06:** Fix the lint/type debt fully — both `src` and `tests` green:
  - `mypy` (already `strict` and config-scoped to `src/gru_qat` per `pyproject.toml`) → **0 errors** (clears the ~145-error baseline in `src/gru_qat/*`).
  - `ruff check src tests` → **0 errors** (clears src + the ~23 pre-existing errors in test files).
  - `gru-triton-4m6` closed on completion. This is the milestone's own audit-close hygiene — the audit should not report itself as "not green."
  - Note: this does NOT change the project convention that test files are not mypy-*strict* — `mypy` is config-scoped to `src/gru_qat`, so "tests green" here means **ruff**-clean tests, not mypy-strict tests.

### D. The gru-triton-n20 genuine bug

- **D-07:** Fix `gru-triton-n20` in Phase 7. The bug: a shared `QuantizerConfig` instance between `quant_h_in`/`quant_h_out` (and the six `quant_W_*` quantizers) makes `freeze_all` silently no-op the second quantizer. Fix: `deepcopy` (or equivalent per-quantizer config isolation) in `make_quantizer`.
  - **Entanglement (flag for planner):** the deepcopy fix changes quantizer scales, which **breaks ~18+ Phase 4 strict tests** whose bit-identity contract depended on both paths sharing the buggy `scale=1.0`. Phase 7 **absorbs this re-baseline**: each affected strict test is re-baselined to its genuinely-correct post-fix bound, OR — if the post-fix residual is itself TF32-rooted — moved into the `divergence` marker (D-05). The FIX-vs-rebaseline-vs-mark call is per-test and delegated to research/planning.
  - Failing-test-before-fix discipline (D-02) applies to the n20 fix itself.

### E. AUDIT-REPORT.md structure

- **D-08:** `AUDIT-REPORT.md` lives at **repo root**. Structure per ROADMAP criterion #2:
  - (a) A table of all **28 v1 requirements** (REF/TRI/STR/QNT/CAL/EDG/RPT) with status ∈ {PASS, FIX, ACCEPTED-DIVERGENCE}.
  - (b) A per-phase summary of what was checked and how — sourced from each `NN-SUMMARY.md` and `NN-VERIFICATION.md`; do not re-derive.
  - (c) A "residual known-but-accepted divergences" section. The TF32 `tl.dot` family gets **one consolidated entry** (shared root cause) with per-issue sub-bullets — not 8 separate top-level entries — for readability. Each entry states the rationale and why the fix is out of scope.
  - (d) A pointer from each finding to the `bd` issue that resolved it.

### F. Criterion #1 verification + criterion #4 end state

- **D-09 (criterion #1):** Phase 7 audits `git log` to confirm, for every Phase 1–6 finding, that the failing-test commit precedes the fix commit. Phases 4–6 followed D-37/D-50; earlier phases may not have. Where the test-before-fix ordering is genuinely absent in history, **document the gap in AUDIT-REPORT** — do NOT rewrite git history. Phase 7's own fixes (n20, 7rj, …) strictly follow the two-commit discipline.
- **D-10 (criterion #4):** End state — `bd ready` is empty. Every one of the 14 issues ends either CLOSED (FIXed, or ACCEPTED-DIVERGENCE closed-with-rationale, or INDIVIDUAL dispositioned) or DEFERRED with a v2 `bd` reference recorded in `REQUIREMENTS.md`. No issue is left plain-open. ACCEPTED-DIVERGENCE issues are **closed** (with a resolution note → AUDIT-REPORT), not left open.

### Claude's Discretion (delegated to research / planning)

- The per-issue FIX-vs-ACCEPTED-DIVERGENCE call for the ambiguous backward-drift issues (`mjy`, `lht`, `e7t`, `fpl`) — requires inspecting each issue's actual measured root cause.
- For the n20 re-baseline: per affected strict test, whether to re-baseline the bound or `divergence`-mark it.
- Plan/wave structure. NOTE: n20's fix touches `quantizers.py` + many strict test files; the `divergence` marker also touches strict test files + `pyproject.toml`; the lint cleanup touches `src` broadly. These overlap — sequence so no two plans write the same file concurrently (the F-04-05-D / `gru-triton-u00` race lesson). AUDIT-REPORT is written LAST (it reports final post-fix status).
- Whether the `divergence` marker is applied test-function-wide or per-parametrize-case.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope + requirements
- `.planning/ROADMAP.md` § Phase 7 — goal + 4 success criteria
- `.planning/REQUIREMENTS.md` — RPT-01/02/03 + the full 28-requirement v1 list the AUDIT-REPORT table must cover; v2 section is where deferred-issue bd refs are recorded
- `.planning/PROJECT.md` — core value, tolerance tiers, the locked "Option C / tiered tolerance" Key Decision (the basis for D-03 accepting the TF32 root cause)

### Findings inputs (the 14 bd issues + per-phase artifacts)
- `bd list --status=open` / `bd show <id>` — the 14 open issues; authoritative root-cause/fix-ref source
- `.planning/phases/02-*/02-*-SUMMARY.md`, `02-DISPOSITION.md` (Option C) — Phase 2 TF32 disposition
- `.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md` — per-cluster `h_scale_mult` table; the strict-test contract n20's re-baseline (D-07) must reconcile against
- `.planning/phases/04-*/04-VERIFICATION.md`, `04-SUMMARY.md` — Phase 4 verifier findings (the F-04-VERIFIER-* family)
- `.planning/phases/05-*/05-SUMMARY.md`, `.planning/phases/06-*/06-SUMMARY.md`, `06-VERIFICATION.md` — Phase 5/6 closure + the n20 deferral note
- `.planning/phases/06-edge-case-sweeps/deferred-items.md` — the 145-error mypy / 23-error ruff baseline (gru-triton-4m6) and the c2a/7rj handoffs
- `.planning/debug/butterfly-batch-invariance-c2a.md`, `.planning/debug/monarch-rounding-mismatch.md` — resolved/active debug sessions feeding the AUDIT-REPORT
- All `NN-SUMMARY.md` + `NN-VERIFICATION.md` across phases 1–6 — the per-phase summary (D-08 part b) is sourced from these

### Implementation surfaces (FIX-bucket targets)
- `src/gru_qat/quantizers.py` — `make_quantizer` (the n20 deepcopy fix, D-07)
- `src/gru_qat/triton_kernels/scan.py`, `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py` — `gru_scan*` wrappers (the 7rj `assert`→`ValueError` fix); also the bwd-drift issue sites
- `pyproject.toml` — `[tool.pytest.ini_options] markers` (register the `divergence` marker, D-05); `[tool.mypy]` / `[tool.ruff]` config (D-06 scope)
- `src/gru_qat/*` broadly — the ~145-error mypy cleanup surface (D-06)
- `tests/test_triton_*_strict.py` — the strict tests that get `divergence`-marked and/or n20-re-baselined (these were "locked for new test additions" in Phases 5–6 but Phase 7 OWNS them — re-baselining and marking are exactly Phase 7's job; D-51 no longer applies in Phase 7)

### Output
- `AUDIT-REPORT.md` — repo root (NEW; D-08)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_skip_if_monarch_bwd_hw_limit` — already the disposition mechanism for `gru-triton-e0l` (D-04); no new skip infrastructure needed.
- The `GRULayer.forward` `if … raise ValueError` guard added in Phase 6 — the exact convention the `gru-triton-7rj` `assert`→`ValueError` wrapper fix should mirror.
- Phase 4 `04-DISPOSITION.md` per-cluster `h_scale_mult` table — the contract the n20 re-baseline must reconcile against.
- Every phase's `NN-SUMMARY.md` / `NN-VERIFICATION.md` — pre-written per-phase narrative; the AUDIT-REPORT per-phase section quotes/condenses these rather than re-deriving.

### Established Patterns
- D-37/D-50 two-commit failing-test-before-fix discipline; **no `@pytest.mark.xfail`** anywhere.
- bd-issue discipline — every finding has a `bd` issue; Phase 7 closes them, recording resolution notes.
- Single-plan-per-shared-file to avoid the F-04-05-D parallel-write race.
- `pytest` markers (`slow`, `cuda_only`) already exist — `divergence` is a new sibling marker registered the same way.

### Integration Points
- `pyproject.toml` — one new marker registration + (D-06) whatever config the lint cleanup needs.
- `AUDIT-REPORT.md` at repo root — the only wholly-new file.
- The FIX bucket touches `quantizers.py`, the `scan*.py` wrappers, and broad `src/gru_qat` lint surface; the `divergence` marker + n20 re-baseline touch `tests/test_triton_*_strict.py`. Sequence to avoid concurrent same-file writes.

</code_context>

<specifics>
## Specific Ideas

- The TF32 `tl.dot` reduction-order non-associativity is the milestone's single most-cited root cause (Phase 2 `gru-triton-rwm`, the F-04-VERIFIER-* family). The AUDIT-REPORT's residual-divergences section should treat it as ONE phenomenon with one rationale, not eight unrelated bugs.
- "Fix everything" was the user's intent — but bounded: the user explicitly accepted that the irreducible TF32 cases and the `e0l` hardware limit are NOT code-fixed in Phase 7. The boundary is "every *genuine tractable* bug," not literally every open issue.
- `gru-triton-n20` is a real silent-correctness bug — the audit must not ship it unfixed. Its fix re-bases the Phase 4 strict-test contract; that re-baseline is in-scope Phase 7 work, not a regression.
- Criterion #3's green gate is `pytest -q -m "not divergence"` — this operationalization is itself an audit finding and must be stated plainly in `AUDIT-REPORT.md`.

</specifics>

<deferred>
## Deferred Ideas

| Idea | Why deferred | Suggested target |
|---|---|---|
| `input_precision="ieee"` TF32-elimination kernel rewrites | Would eliminate the entire ACCEPTED-DIVERGENCE family, but PROJECT.md's locked "Option C / tiered tolerance" decision and SCOPE explicitly defer it; milestone-ballooning | v2 |
| `gru-triton-e0l` Monarch-bwd kernel-tiling redesign for consumer-GPU SMEM | A genuine hardware constraint; needs a kernel redesign, not a bugfix | v2 |
| ACT-01 (per-channel `min_max` observer done right), ACT-02 (LSQ/PACT) | Already in REQUIREMENTS.md v2 | v2 |
| PERF-01 / PERF-02 (cuDNN comparison + QAT-overhead bench re-validation) | Correctness-only milestone; perf is out of scope | v2 |

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 7-Audit report + findings handling*
*Context gathered: 2026-05-15*
