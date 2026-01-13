# TorEqProp Verification Results

**Generated**: 2026-01-13 22:51:10


## Executive Summary

**Verification completed in 76.9 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 2 |
| Passed | 2 ✅ |
| Partial | 0 ⚠️ |
| Failed | 0 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 100.0/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 14 | Transformer EqProp | ✅ | 100 | 75.0s |
| 22 | Golden Reference Harness | ✅ | 100 | 1.8s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 14: Transformer EqProp


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 75.0s

🧪 **Evidence Level**: Smoke Test


**Claim**: Equilibrium Transformer can solve sequence manipulation tasks (Reversal).

**Experiment**: Learn to reverse a sequence of length 8. N=3 seeds.

| Metric | Mean | StdDev |
|--------|------|--------|
| Accuracy | 100.0% | 0.0% |

**Key Finding**: Iterative equilibrium attention successfully routes information
from pos $i$ to $L-i-1$.




## Track 22: Golden Reference Harness


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 1.8s

🧪 **Evidence Level**: Smoke Test


**Claim**: NumPy kernel matches PyTorch autograd to within numerical tolerance.

**Experiment**: Compare hidden states at each relaxation step.

| Metric | Value | Threshold |
|--------|-------|-----------|
| Max Hidden Diff | 3.28e-07 | < 1.00e-05 |
| Output Diff | 2.24e-07 | < 1.00e-05 |
| Steps Compared | 30 | - |

**Step-by-Step Comparison** (first/last steps):

| Step | Max Difference |
|------|----------------|
| 0 | 1.79e-07 |
| 1 | 2.38e-07 |
| 2 | 2.38e-07 |
| 3 | 3.28e-07 |
| 4 | 1.79e-07 |
| 28 | 2.38e-07 |
| 29 | 2.38e-07 |

**Purpose**: This harness enables safe optimization of the engine. Any new kernel
implementation must pass this test before deployment.

**Status**: ✅ VALIDATED - Safe to optimize


