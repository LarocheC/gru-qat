# Phase 6 — deferred / out-of-scope items

Items discovered during plan 06-01 execution that are NOT caused by this
plan's changes. Logged per the SCOPE BOUNDARY rule — not fixed here.

## Pre-existing lint / type baseline (out of scope)

The repo has a pre-existing red `ruff` / `mypy` baseline, confirmed
identical on the plan's base commit `3bdcf7b`:

- `mypy`: 145 errors across 10 source files (e.g. `gru_layer.py:130/132/
  208/212/287`, `gru_cell.py:462/504`, `calibration.py:33`). None are in
  the lines added by plan 06-01 (the `gru_layer.py` T=0/B=0 guard and the
  `structure.py` butterfly H<2 guard add zero mypy errors).
- `ruff`: 23 errors, all in pre-existing test files (not in the new
  `tests/test_edge_cases.py`, which is ruff-clean).

Plan 06-01's own changes (`gru_layer.py`, `structure.py`,
`tests/test_edge_cases.py`) are `ruff`-clean and introduce no new `mypy`
errors. Cleaning the pre-existing baseline is a separate hygiene task,
out of Phase 6's edge-sweep scope.

## Open findings handed off (tracked in beads)

- `gru-triton-c2a` — butterfly Triton kernel violates batch-invariance at
  H=512 (periodic stride-8 batch-tiling correctness bug). Commit-A failing
  test landed (`test_butterfly_partial_batch_tile`); deep `scan_butterfly.py`
  kernel fix handed off per the Task-3 CONTEXT-BUDGET handoff.
- `gru-triton-7rj` — `scan*.py` `gru_scan*` wrappers use `assert` for shape
  validation (stripped under `python -O`). Out of EDG-04 scope; filed, not
  fixed, per the plan.
