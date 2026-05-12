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
