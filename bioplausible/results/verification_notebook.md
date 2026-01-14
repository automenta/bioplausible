# TorEqProp Verification Results

**Generated**: 2026-01-14 14:24:03


## Executive Summary

**Verification completed in 1.2 seconds.**

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
| 0 | Framework Validation | ✅ | 100 | 1.0s |
| 53 | NEBC Contrastive Hebbian | ✅ | 100 | 0.1s |
| 54 | NEBC Deep Hebbian Chain | ✅ | 100 | 0.1s |


**Seed**: 42 (deterministic)

**Reproducibility**: All experiments use fixed seeds for exact reproduction.

---


## Track 0: Framework Validation


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 1.0s

🧪 **Evidence Level**: Smoke Test


**Framework Self-Test Results**

| Test | Status |
|------|--------|
| Cohen's d calculation | ✅ |
| Statistical significance (t-tests) | ✅ |
| Evidence classification | ✅ |
| Human-readable interpretations | ✅ |
| Statistical comparison formatting | ✅ |
| Reproducibility hashing | ✅ |

**Tests Passed**: 6/6

**Purpose**: This track validates the validation framework itself, ensuring all statistical
functions work correctly before running model validation tracks.


**Limitations**:
- Framework-level test only, does not validate EqProp models



## Track 53: NEBC Contrastive Hebbian


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 0.1s

🧪 **Evidence Level**: Smoke Test

ContrastiveHebbianLearning runs train_step without error.



## Track 54: NEBC Deep Hebbian Chain


✅ **Status**: PASS | **Score**: 100.0/100 | **Time**: 0.1s

🧪 **Evidence Level**: Smoke Test

DeepHebbianChain maintains signal through 50 layers.

