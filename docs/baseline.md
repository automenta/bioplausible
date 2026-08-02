# Gated Baseline Snapshot

Captured for Sprint −1.3 (2026-08-02) at `v0.1-pre-sprint0`.
Every number here maps 1:1 to a gate in Sprint 5.5. Style-only metrics are
deliberately excluded (deferred); only correctness gates are snapshot.

## Fast CPU Gate (`tests/unit/ tests/property/`)

Command: `uv run pytest tests/unit/ tests/property/ -q --no-cov`

| Metric | Baseline | Gate in 5.5 |
|--------|----------|-------------|
| Collected | 1203 | — |
| Passed | 1200 | must not decrease |
| Failed | 0 | 0 |
| Xfailed | 1 (biology AdaptiveFA) | 0 after Sprint 1.5.2 / documented |
| Skipped | 1 | — |

## Full Suite (`--collect-only`)

| Metric | Baseline |
|--------|----------|
| Collected | 1626 |

## Pyright (strict)

Command: `uv run pyright .`

| Metric | Baseline | Gate in 5.5 |
|--------|----------|-------------|
| Errors | 0 | == 0 |
| Warnings | 2440 | ≤ baseline (must not grow) |

## Ruff Correctness Set (`--select E,F,W,C90`)

Command: `uv run ruff check --select E,F,W,C90 .`

| Metric | Baseline | Gate in 5.5 |
|--------|----------|-------------|
| Correctness errors | 635 | ≤ baseline |

Top offenders (correctness set only): E501 line-too-long 362, C901 complex 132→(106 minus refactors),
E402 import-not-at-top 95, F821 undefined 31, F401 unused-import 9, F841 9,
E741 8, E722 bare-except 7, F822 4, F541 3, W293 3, W291 1. (C901 dropped after
Sprint 0.3 refactors of engine/trainer.)

## Coverage (Sprint 5.5 gate uses `--cov`)

Command: `uv run pytest --cov`

| Metric | Baseline | Gate in 5.5 |
|--------|----------|-------------|
| Coverage | 20.84% | floor 50% → 85% over time |

NOTE: coverage is far below the 50% floor. Sprint 5.5 must expand test coverage;
this is the largest gap between baseline and CI-green.

## Sprint −1.3 Exit

- `git tag v0.1-pre-sprint0` created (2026-08-02).
- Numbers above are the frozen reference for CI baseline assertions.
