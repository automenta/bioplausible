# TorEqProp Verification Results

**Generated**: 2026-01-14 19:03:17


## Executive Summary

**Verification completed in 31.8 seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | 3 |
| Passed | 3 ✅ |
| Partial | 0 ⚠️ |
| Failed | 0 ❌ |
| Stubs (TODO) | 0 🔧 |
| Average Score | 100.0/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
| 42 | Holomorphic EP | ✅ | 100 | 4.2s |
| 43 | Directed EP | ✅ | 100 | 4.0s |
| 44 | Finite-Nudge EP | ✅ | 100 | 23.5s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 42: Holomorphic EP


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 4.2s

🧪 **Evidence Level**: Smoke Test


**Claim**: Holomorphic EP learns using complex-valued states and weights.

**Results**:
- Initial Loss: 2.4805
- Final Loss: 2.1271
- Complex Weights: ✅ Yes
- Learning: ✅ Yes




## Track 43: Directed EP


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 4.0s

🧪 **Evidence Level**: Smoke Test


**Claim**: Directed EP learns with asymmetric forward/feedback weights.

**Results**:
- Asymmetric: ✅ Yes
- Initial Loss: 2.2926
- Final Loss: 2.1053




## Track 44: Finite-Nudge EP


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 23.5s

🧪 **Evidence Level**: Smoke Test


**Claim**: Finite-Nudge EP learns stably with large beta (1.0).

**Results**:
- Initial Loss: 2.2813
- Final Loss: 2.0970
- Stability: ✅ Stable

