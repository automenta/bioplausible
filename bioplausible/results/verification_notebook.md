# TorEqProp Verification Results

**Generated**: 2026-01-15 11:11:01


## Executive Summary

**Verification completed in 27.2 seconds.**

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
| 42 | Holomorphic EP | ✅ | 100 | 3.6s |
| 43 | Directed EP | ✅ | 100 | 2.1s |
| 44 | Finite-Nudge EP | ✅ | 100 | 21.4s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 42: Holomorphic EP


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 3.6s

🧪 **Evidence Level**: Smoke Test


**Claim**: Holomorphic EP learns using complex-valued states and weights.

**Results**:
- Initial Loss: 2.4805
- Final Loss: 2.1271
- Complex Weights: ✅ Yes
- Learning: ✅ Yes




## Track 43: Directed EP


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 2.1s

🧪 **Evidence Level**: Smoke Test


**Claim**: Directed EP learns with asymmetric forward/feedback weights.

**Results**:
- Asymmetric: ✅ Yes
- Initial Loss: 2.2926
- Final Loss: 2.1053




## Track 44: Finite-Nudge EP


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 21.4s

🧪 **Evidence Level**: Smoke Test


**Claim**: Finite-Nudge EP learns stably with large beta (1.0).

**Results**:
- Initial Loss: 2.2813
- Final Loss: 2.0970
- Stability: ✅ Stable

