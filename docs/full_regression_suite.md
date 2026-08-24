# Full Regression Suite

## Overview

This document describes the single-command regression test suite that validates all kernel backends, integration tests, export pipelines, and strategy permutations.

## Quick Start

Run the full regression suite with:

```bash
uv run pytest tests/unit/acceleration/ tests/unit/validation/ tests/integration/test_kernel_*.py -x --tb=short
```

This executes:
- **Kernel backend tests** (`tests/unit/acceleration/`) — protocol compliance, dtype support, mixed precision parity
- **Validation tests** (`tests/unit/validation/`) — kernel parity, family parity, registry audit, reproducibility
- **Integration tests** (`tests/integration/test_kernel_*.py`) — end-to-end kernel dispatch, contrastive learning, export CLI

## Targeted Test Groups

### 1. Kernel Backend Unit Tests

```bash
# All acceleration tests (protocol, dtypes, mixed precision, export)
uv run pytest tests/unit/acceleration/ -x --tb=short

# Specific families
uv run pytest tests/unit/acceleration/test_fa_kernel_init.py -x
uv run pytest tests/unit/acceleration/test_eqprop_kernel_backend.py -x
uv run pytest tests/unit/acceleration/test_export_torch_export.py -x
uv run pytest tests/unit/acceleration/test_mixed_precision.py -x
```

### 2. Kernel Validation Tests

```bash
# Family parity (cross-hardware, cross-dtype)
uv run pytest tests/unit/validation/test_family_kernel_parity.py -x

# Registry audit
uv run pytest tests/unit/validation/test_registry_audit.py -x

# Reproducibility
uv run pytest tests/unit/validation/test_reproducibility.py -x
```

### 3. Integration Tests

```bash
# Kernel dispatch + training
uv run pytest tests/integration/test_kernel_*.py -x

# Strategy optimizer wiring
uv run pytest tests/unit/test_strategy_optimizer_wiring.py -x
```

### 4. Strategy Permutation Benchmarks

```bash
# Quick smoke test (1 model, 1 dataset, 1 precision, 2 epochs)
uv run python tools/benchmark_strategy_permutations.py \
    --models backprop_mlp \
    --datasets digits \
    --precisions fp32 \
    --epochs 2

# Full sweep (all models, all datasets, all precisions, 20 epochs)
uv run python tools/benchmark_strategy_permutations.py \
    --models backprop_mlp standard_fa pepita diff_target_prop predictive_coding_hybrid eqprop \
    --datasets digits mnist fashion_mnist \
    --precisions fp32 fp16 bf16 \
    --epochs 20
```

### 5. Export CLI Tests

```bash
# Export trained kernel (backprop on CPU)
uv run biopl-export-trained-kernel --algorithm backprop --target cpu --epochs 10 --output ./exports/backprop_cpu

# Export trained kernel (FA on CPU)
uv run biopl-export-trained-kernel --algorithm fa --target cpu --epochs 10 --output ./exports/fa_cpu

# Export trained kernel (EQPROP on CPU)
uv run biopl-export-trained-kernel --algorithm eqprop --target cpu --epochs 10 --output ./exports/eqprop_cpu

# Export untrained kernel (manifest only)
uv run biopl-export-kernel --algorithm backprop --target cpu --output ./exports/backprop_manifest
```

## Continuous Integration Gate

The CI pipeline runs in this order:

```bash
# 1. Format check
uv run ruff format --check .

# 2. Lint
uv run ruff check --fix .

# 3. Type check (strict)
uv run pyright .

# 4. Full test suite with coverage floor
uv run pytest --cov=computronium --cov-report=term-missing --cov-fail-under=55

# 5. Dependency audit
uv run pip-audit
```

## Test Selection by Markers

```bash
# Skip slow tests
uv run pytest -m "not slow" ...

# GPU-only tests (require CUDA)
uv run pytest -m gpu ...

# Benchmark tests (produce JSONL)
uv run pytest -m benchmark ...
```

## Expected Test Counts

| Test Group | Approximate Tests | Duration |
|------------|-------------------|----------|
| `tests/unit/acceleration/` | 100+ | ~30s |
| `tests/unit/validation/` | 60+ | ~60s |
| `tests/integration/test_kernel_*.py` | 20+ | ~30s |
| Strategy benchmark (smoke) | 8 permutations | ~10s |

## Troubleshooting

### Coverage Failure

If coverage fails (`--cov-fail-under=55`), the issue is typically in untested kernel backend code paths. Focus on:

```bash
uv run pytest tests/unit/acceleration/ --cov=computronium.acceleration --cov-report=term-missing
```

### CUDA Tests on CPU-Only Machines

Tests marked `@pytest.mark.skipif(not torch.cuda.is_available())` will be skipped automatically. To force CPU fallbacks:

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest tests/unit/acceleration/test_mixed_precision.py -x
```

### ONNX Export Failures

Spectral norm parametrization blocks ONNX export for EQPROP. This is expected:

```
ONNX export skipped for eqprop: Exporting the operator 'aten::vdot' to ONNX opset version 11 is not supported
```

The state dict and manifest are still produced correctly.

## Artifacts

All test runs produce artifacts in `artifacts/`:
- `strategy_benchmark_report.json` — permutation benchmark results
- `exports/*/manifest.json` — hardware manifests
- `exports/*/state_dict.pt` — trained weights
- `exports/*/export_summary.json` — export summaries