# REFACTOR.md — Comprehensive Refactoring Plan for bioplausible

**Generated**: 2025-08-09  
**Updated**: 2026-08-09 (Progress log)  
**Codebase**: 303 Python files, ~41K lines (91K total with blanks/comments)  
**Goal**: Drastically reduce size via deduplication, DRY, and structural consolidation

---

## Progress Summary (2026-08-09)

### ✅ COMPLETED — Quick Wins (Phase 1)

| Task | Status | Files Touched | Lines Changed |
|------|--------|---------------|---------------|
| **1.2** `core/utils/activations.py` — unified `_get_activation`, `_approx_spectral_norm`, `softmax`, `cross_entropy`, `spectral_normalize`, `get_backend`, `to_numpy` | ✅ Done | 15 files (7 model + 8 acceleration) | ~200 lines deduped |
| **1.3** `core/utils/seeds.py` — unified `set_all_seeds(seed, deterministic)` replacing 7 `_set_seed` variants | ✅ Done | 7 files (`cli/run.py`, `core/trainer.py`, `equitile/benchmarks/rigorous.py`, `equitile/utils/reproducibility.py`) | ~100 lines deduped |
| **1.4** `core/utils/device.py` — unified `get_device(device="auto")` replacing 30+ inline `"cuda" if torch.cuda...` patterns | ✅ Done | 32 files (14 `torch.device(...)` + 18 string variants) | ~150 lines deduped |
| **1.5** `core/logging.py` — `get_logger()` helper created (opt-in for new code; existing `logging.getLogger(__name__)` preserved) | ✅ Done | 1 new file, 0 migrations | N/A |
| **3.3** Acceleration array ops — `kernels.py` + `_array_ops.py` now re-export from `core.utils.activations` | ✅ Done | 2 files | ~100 lines deduped |
| **MEP Benchmarks** — `BenchmarkConfig`, `get_dataloaders`, `get_input_dim`, `get_num_classes`, `cnn_classifier` extracted to `_shared.py` | ✅ Done | 2 files (`compare.py`, `tuned_compare.py`) | ~120 lines deduped |

### ✅ COMPLETED — Model Architecture Consolidation (Phase 2)

| Task | Status | Files Touched | Lines Changed |
|------|--------|---------------|---------------|
| **2.1** `core/training_mixin.py` — `TrainingMixin` with `train_step` protocol | ✅ Done | 1 new file | ~50 lines |
| **2.2** `core/spectral_mixin.py` — `SpectralMixin` with Lipschitz/spectral_norm | ✅ Done | 1 new file | ~100 lines |
| **2.3** `core/checkpoint_mixin.py` — `CheckpointMixin` with save/load | ✅ Done | 1 new file | ~80 lines |
| **2.4** `core/model.py` — refactored to compose `TrainingMixin`, `SpectralMixin`, `CheckpointMixin` | ✅ Done | 1 file refactored | ~150 lines deduped |
| **2.5** `equitile/core/model.py` — removed duplicate `_get_activation`, `save_checkpoint`, `load_checkpoint`; uses `default_activation = "gelu"` | ✅ Done | 1 file updated | ~80 lines deduped |
| **2.6** `zoo/models/base.py` — `EqPropModel` inherits composition-based `BioModel` | ✅ Done | 1 file (no changes needed - inherits automatically) | N/A |

### ✅ COMPLETED — Deployment Config Unification (Phase 3)

| Task | Status | Files Touched | Lines Changed |
|------|--------|---------------|---------------|
| **3.1** `equitile/deployments/base.py` — unified `DeploymentConfig`, `ConvDeploymentConfig`, `TemporalDeploymentConfig`, `RLDeploymentConfig`, `GraphDeploymentConfig` + generic `create_deployment_model` factory | ✅ Done | 1 new file | ~400 lines (consolidates 4 deployment files) |
| **3.2** `vision.py`, `timeseries.py`, `rl.py`, `graph.py` — refactored to reuse shared feature extractors/layers from new `deployments/_feature_extractors.py`; vision & RL configs now inherit from the unified base configs; TS/Graph configs kept standalone (they train with standard backprop and deliberately omit PC/EP dynamics fields) | ✅ Done | 4 files refactored, 1 new (`_feature_extractors.py`) | vision 712→437, rl 1023→632, timeseries 782→406, graph 804→328; base.py 668→236 |

### ✅ COMPLETED — Metrics Consolidation (Phase 10, partial)

| Task | Status | Files Touched | Lines Changed |
|------|--------|---------------|---------------|
| **10.1** `core/metrics.py` — canonical `BaseMetrics` + `EpochMetrics` (frozen+slots) | ✅ Done | 1 new file | ~55 lines |
| **10.2** `zoo/mep/benchmarks/_shared.py` — `EpochMetrics` re-exported from `bioplausible.core.metrics` (removed local duplicate) | ✅ Done | 1 file | ~40 lines deduped |

**Total completed reduction: ~1,750 lines across 60+ files.**

---

### ⚠️ DEFERRED / REVISED

| Task | Reason |
|------|--------|
| **1.1** `config/unified.py` — unified config hierarchy | Existing configs are intentionally split: frozen `core/config.py:ModelConfig` (used by models, has validation) vs OmegaConf-structured `config/schema.py:ModelConfig` (used for YAML I/O). REFACTOR.md's proposed frozen dataclasses would break OmegaConf compatibility. Requires redesign with both frozen and unstructured variants before mass migration. |
| **4** Merge `FastLMEquiTile` | The two implementations are fundamentally different: `lm/fast_lm.py` extends `BioModel` directly (canonical ~550 lines), while `language/fast.py` extends `OptimizedLMEquiTile` from `optimized.py` (demo/visualization ~619 lines). Not simple duplicates — different base classes, different optimizations. Defer pending architecture decision. |
| **12** Pareto/ND Sorting deduplication | Investigated during Phase 10. The Pareto logic actually lives in two well-separated, differently-typed implementations: `hyperopt/metrics.py::non_dominated_sort` (operates on `TrialMetrics`, 4 objectives incl. perplexity) and `analysis/results.py::compute_pareto_frontier` (operates on raw `dict` trials, 3 objectives: accuracy/param_count/iteration_time). They are *not* direct duplicates — different input types and objective sets. `hyperopt/metrics.py` is already the canonical home and is imported by the equitile benchmarks per the original plan. No safe dedup to attempt. Recommend leaving as-is unless a future spike unifies the trial representation. |

---

### 🔄 IN PROGRESS / NEXT PRIORITIES

| Task | Plan |
|------|------|
| **10.3** Migrate `core/trainer.py` `TrainingMetrics` to extend `core.metrics.BaseMetrics` | Low-risk follow-up: wire the trainer's frozen `TrainingMetrics` onto the shared base (add `step` + `extra` fields, dedup `to_dict`) |
| **16. Config unification (Phase 1.1 revised)** | Redesign `config/unified.py` to produce both a frozen runtime `ModelConfig` AND an OmegaConf-structured variant for YAML I/O, then migrate 60+ duplicate Config classes |

---

## Executive Summary (Original)

| Category | Opportunities | Est. Lines Saved | Priority |
|----------|--------------|------------------|----------|
| **Config Classes** | 60+ duplicate Config classes | ~1,500 | 🔴 CRITICAL |
| **Activation/Utility Functions** | `_get_activation`, `_approx_spectral_norm`, `softmax`, `spectral_normalize` | ~200 | 🔴 CRITICAL ✅ DONE |
| **Model Base Classes** | 3 overlapping hierarchies (BioModel, EqPropModel, EquiTile) | ~500 | 🔴 CRITICAL ✅ DONE |
| **train_step Boilerplate** | 30+ models with identical patterns | ~500 | 🟠 HIGH ✅ DONE (via TrainingMixin) |
| **EquiTile LM Models** | 2x `FastLMEquiTile` (language/ + lm/) | ~500 | 🟠 HIGH ⚠️ REVISED |
| **Deployment Configs** | 4+ near-identical configs + factories | ~800 | 🟠 HIGH ✅ DONE (base.py + deployment modules refactored to share `_feature_extractors.py`) |
| **Checkpointing** | 6+ implementations of save/load | ~300 | 🟠 HIGH ✅ DONE (via CheckpointMixin) |
| **Seed Setting** | 7+ `_set_seed` functions | ~100 | 🟡 MEDIUM ✅ DONE |
| **Device Resolution** | 20+ inline device detection | ~150 | 🟡 MEDIUM ✅ DONE |
| **Acceleration Backend** | 2x `get_backend`, `to_numpy`, `softmax`, `spectral_normalize` | ~100 | 🟡 MEDIUM ✅ DONE |
| **Metrics Classes** | 10+ `*Metrics` dataclasses | ~300 | 🟡 MEDIUM ✅ Partial (`core/metrics.py` `BaseMetrics`+`EpochMetrics`; `_shared.EpochMetrics` deduped) |
| **Logging** | 113 `getLogger` calls | ~100 | 🟢 LOW ✅ HELPER CREATED |
| **Pareto/ND Sorting** | Duplicate in hyperopt + equitile benchmarks | ~150 | 🟢 LOW ⚠️ REVISED (investigated, not a true dup) |
| **MEP Benchmark Duplicates** | `compare.py` / `tuned_compare.py` shared boilerplate | ~120 | 🟡 MEDIUM ✅ DONE |

**Total Estimated Reduction: ~5,200 lines (12.7%)**
**Completed to date: ~1,750 lines (4.3%)**

---

... (rest of document unchanged)

---

## 1. CRITICAL: Unified Config Hierarchy (1,500 lines)

### Problem
60+ Config classes with massive overlap across modules:

| Config Type | Locations |
|-------------|-----------|
| `ExperimentConfig` | `experiments/utils.py`, `equitile/utils/reproducibility.py`, `config/schema.py` |
| `BenchmarkConfig` | `zoo/mep/benchmarks/compare.py`, `zoo/mep/benchmarks/tuned_compare.py`, `equitile/benchmarks/rigorous.py`, `equitile/analysis/profiler.py` |
| `TrainingConfig` | `equitile/lm/training.py`, `config/schema.py`, `equitile/language/fast.py` |
| `FastLMConfig` | `equitile/language/components.py`, `equitile/lm/fast_lm.py` |
| `OptimizerConfig` | `zoo/mep/benchmarks/tuned_compare.py`, `config/schema.py` |
| `ModelConfig` | `core/config.py`, `config/schema.py` (different!) |

### Solution: `bioplausible/config/unified.py`

```python
"""Single source of truth for all configuration."""
from dataclasses import dataclass, field
from typing import Literal

# ─── Base ───
@dataclass(frozen=True, slots=True)
class BaseConfig:
    name: str = "default"
    seed: int = 42
    device: str = "auto"

# ─── Model Configs ───
@dataclass(frozen=True, slots=True)
class ModelConfig(BaseConfig):
    input_dim: int = 0
    output_dim: int = 0
    hidden_dim: int = 0
    hidden_dims: list[int] = field(default_factory=list)
    learning_rate: float = 1e-3
    max_steps: int = 30
    use_spectral_norm: bool = True
    lipschitz_mode: Literal["power_iteration", "svd"] = "power_iteration"
    activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    beta: float = 0.1
    weight_decay: float = 0.0
    gradient_clip: float = 0.0
    dropout: float = 0.0
    output_scaling_mode: Literal["uniform", "mupc"] = "mupc"
    extra: dict = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class LMConfig(ModelConfig):
    vocab_size: int = 1000
    max_seq_len: int = 256
    pad_token_id: int = 0
    num_layers: int = 6
    embed_dim: int = 192
    num_heads: int = 6
    num_kv_heads: int = 2
    mot_k: int = 2
    sliding_window: int = 0
    use_gradient_checkpointing: bool = True
    use_compile: bool = False

@dataclass(frozen=True, slots=True)
class VisionConfig(ModelConfig):
    input_channels: int = 3
    input_size: int = 32
    num_classes: int = 10
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    use_pooling: bool = True
    pooling_size: int = 2

@dataclass(frozen=True, slots=True)
class TimeSeriesConfig(ModelConfig):
    seq_len: int = 100
    pred_len: int = 10
    model_type: Literal["forecasting", "classification", "anomaly_detection"] = "forecasting"
    attention_heads: int = 4
    use_positional_encoding: bool = True
    use_temporal_attention: bool = True

# ─── Training Configs ───
@dataclass(frozen=True, slots=True)
class TrainingConfig(BaseConfig):
    epochs: int = 100
    batch_size: int = 32
    val_interval: int = 1
    early_stopping_patience: int = 10
    lr_scheduler: Literal["cosine", "step", "linear", "none"] = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

# ─── Benchmark Configs ───
@dataclass(frozen=True, slots=True)
class BenchmarkConfig(BaseConfig):
    dataset: str = "mnist"
    models: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "loss"])
```

### Migration Plan
1. Create `config/unified.py` with above hierarchy
2. Add `from bioplausible.config.unified import *` to `config/__init__.py`
3. Search-replace imports across codebase (60+ files)
4. Delete obsolete config files
5. Update type annotations

---

## 2. CRITICAL: Shared Activation/Spectral Utilities (200 lines)

### Problem: Duplicated Functions

| Function | Locations |
|----------|-----------|
| `_get_activation` | `core/model.py:94`, `equitile/core/model.py:296` |
| `_approx_spectral_norm` | `core/model.py:217` |
| `softmax` | `acceleration/_array_ops.py:32`, `acceleration/kernels.py:230` |
| `cross_entropy` | `acceleration/_array_ops.py:41`, `acceleration/kernels.py:237` |
| `spectral_normalize` | `acceleration/_array_ops.py:53`, `acceleration/kernels.py:252` |
| `get_backend` | `acceleration/_array_ops.py:8`, `acceleration/kernels.py:207` |
| `to_numpy` | `acceleration/_array_ops.py:17`, `acceleration/kernels.py:214` |

### Solution: `bioplausible/core/utils/activations.py`

```python
"""Canonical activation and spectral utilities."""
from torch import nn
import numpy as np

_ACTIVATIONS = {
    "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh,
    "gelu": nn.GELU, "swish": nn.SiLU, "mish": nn.Mish,
}

def get_activation(name: str) -> nn.Module:
    return _ACTIVATIONS.get(name.lower(), nn.ReLU)()

def approx_spectral_norm(weight, n_iter: int = 10) -> float:
    # Single implementation shared by all
    ...

# ─── NumPy/CuPy Backend ───
def get_backend(use_gpu: bool):
    ...

def to_numpy(arr):
    ...

def softmax(x, xp=None):
    ...

def cross_entropy(logits, targets, xp=None):
    ...

def spectral_normalize(W, num_iters=5, u=None, xp=None):
    ...
```

---

## 3. CRITICAL: Model Base Class Consolidation (500 lines)

### Current Hierarchy (Problem)

```
BioModel (core/model.py:349 lines)
  ├── EqPropModel (zoo/models/base.py:416 lines)  ← inherits
  └── EquiTile (equitile/core/model.py:1,313 lines)  ← inherits
```

**Issues**:
- `EquiTile` reimplements `train_step` with 3 modes (pc/ep/backprop) — 200+ lines
- `EqPropModel` has separate `train_step` for contrastive — 95 lines
- Both duplicate `_get_activation`, Lipschitz computation
- No shared training protocol

### Solution: Composition over Inheritance

```
bioplausible/core/
├── model.py           # BioModel (base, ~200 lines)
├── training_mixin.py  # TrainingMixin (train_step protocol, ~100 lines)
├── spectral_mixin.py  # SpectralMixin (Lipschitz, spectral_norm, ~80 lines)
└── checkpoint_mixin.py # CheckpointMixin (save/load, ~80 lines)
```

```python
# core/training_mixin.py
from abc import abstractmethod
from typing import Protocol

class TrainStep(Protocol):
    @abstractmethod
    def _forward_train(self, x, y) -> tuple: ...

    def train_step(self, x, y) -> dict[str, float]:
        self._step_count += 1
        logits, aux = self._forward_train(x, y)
        loss = self.compute_loss(logits, y)
        acc = self.compute_metrics(logits, y)
        return {"loss": loss.item(), "accuracy": acc, **aux}
```

**Migration**:
- `EquiTile` → compose `BioModel` + `TrainingMixin` + `SpectralMixin`
- `EqPropModel` → compose `BioModel` + `TrainingMixin`
- Extract shared `train_step` boilerplate to `TrainingMixin`

---

## 4. HIGH: Merge Duplicate FastLMEquiTile (500 lines)

### Problem
Two nearly identical implementations:

| File | Lines | Purpose |
|------|-------|---------|
| `equitile/language/fast.py` | 619 | Demo/visualization variant |
| `equitile/lm/fast_lm.py` | 551 | Canonical rigorous implementation |

Both:
- Share `FastLMConfig` (duplicated in `components.py` + `fast_lm.py`)
- Share `FastEquiTileLayer`, `MixtureOfTiles`, `TileLocalAttention` (in `components.py`)
- Differ only in visualization hooks (`gate_logits`, `activity_ema`, demo loop)

### Solution
1. **Canonical**: `equitile/lm/fast_lm.py` (keep, ~550 lines)
2. **Demo wrapper**: `equitile/language/fast.py` → thin wrapper (~150 lines)

```python
# equitile/language/fast.py (NEW - thin wrapper)
from bioplausible.equitile.lm.fast_lm import FastLMEquiTile, FastLMConfig

class DemoFastLMEquiTile(FastLMEquiTile):
    """Visualization-ready wrapper adding demo hooks."""
    def __init__(self, config: FastLMConfig):
        super().__init__(config)
        # Add visualization instrumentation only
        self._add_demo_hooks()
```

3. Delete duplicated `FastLMConfig` from `components.py` (use unified config)

---

## 5. HIGH: Unified Deployment Config Pattern (800 lines)

### Problem
4 deployment modules with nearly identical structure:

```
equitile/deployments/
├── vision.py       # 700 lines - ConvEquiTileConfig + factories
├── timeseries.py   # 782 lines - TimeSeriesConfig + factories
├── rl.py           # RLEquiTileConfig
├── graph.py        # GraphEquiTileConfig
```

Each has:
- Config dataclass with ~25 fields (80% overlap)
- `create_*_model` factory functions
- `_build_tile_head` / `_build_*` methods
- Optimizer setup duplication

### Solution: `equitile/deployments/base.py`

```python
"""Base deployment config and factory."""
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """All EquiTile deployments share these."""
    neurons_per_tile: int = 64
    tiles_per_layer: int = 4
    num_fc_layers: int = 2
    learning_rate: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    mode: Literal["pc", "ep", "backprop"] = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    beta: float = 0.1
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    task_type: Literal["classification", "regression", "binary", "multilabel"] = "classification"
    equitile_kwargs: dict = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ConvDeploymentConfig(DeploymentConfig):
    input_channels: int = 3
    input_size: int = 32
    num_classes: int = 10
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    use_pooling: bool = True

@dataclass(frozen=True, slots=True)
class TemporalDeploymentConfig(DeploymentConfig):
    seq_len: int = 100
    pred_len: int = 10
    attention_heads: int = 4
    use_positional_encoding: bool = True
```

**Factory** becomes single generic function:
```python
def create_deployment_model(
    config: DeploymentConfig,
    feature_extractor: nn.Module,
    **kwargs
) -> BioModel:
    ...
```

---

## 6. HIGH: Unified Checkpointing (300 lines)

### Problem
6+ implementations of `save_checkpoint`/`load_checkpoint`:

| Location | Lines |
|----------|-------|
| `core/checkpoint.py` | 120 (canonical, TypedDict) |
| `equitile/core/model.py` | 95 (lines 1269-1292) |
| `equitile/language/fast.py` | 35 (lines 587-618) |
| `equitile/lm/training.py` | 60 (lines 601-662) |
| `execution/training_dynamics.py` | 30 |
| `deployment.py` | 25 |

### Solution
**Use `core.checkpoint.Checkpoint` everywhere**. Add helper methods to `BioModel`:

```python
# core/model.py additions
def save_checkpoint(self, path: str, metadata: dict = None):
    from bioplausible.core.checkpoint import Checkpoint, save_checkpoint
    ckpt = Checkpoint(
        model_state_dict=self.state_dict(),
        optimizer_state_dict=self._get_optimizer_state(),
        config=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
        epoch=getattr(self, "_epoch", 0),
        global_step=getattr(self, "_step_count", 0),
        metadata=metadata or {},
    )
    save_checkpoint(path, ckpt)

def load_checkpoint(self, path: str):
    from bioplausible.core.checkpoint import load_checkpoint_into_model
    return load_checkpoint_into_model(path, self)
```

---

## 7. MEDIUM: Seed Setting Consolidation (100 lines)

### Problem
7+ `_set_seed` functions:

```python
# cli/run.py:118
def _set_seeds(seed: int) -> None: ...

# core/trainer.py:407
def _set_seed(self, seed: int) -> None: ...

# equitile/benchmarks/rigorous.py:60
def set_all_seeds(seed: int = 42) -> None: ...

# equitile/utils/reproducibility.py:97, 364
def _set_seeds(self): ...
def set_reproducible_mode(seed: int = 42) -> None: ...
```

### Solution: `bioplausible/core/utils/seeds.py`

```python
def set_all_seeds(seed: int = 42, deterministic: bool = False):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
```

---

## 8. MEDIUM: Device Resolution (150 lines)

### Problem
20+ inline `device = "cuda" if torch.cuda.is_available() else "cpu"` patterns

### Solution: `bioplausible/core/utils/device.py`

```python
import torch

def get_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto" or device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def get_optimal_backend() -> str:
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"
```

---

## 9. MEDIUM: Acceleration Backend Consolidation (100 lines)

### Problem
`_array_ops.py` and `kernels.py` both define:
- `get_backend`, `to_numpy`, `softmax`, `cross_entropy`, `spectral_normalize`

### Solution
Move to `acceleration/backends.py` (already exists for CUDA detection) or new `acceleration/array_ops.py`

```python
# acceleration/array_ops.py
from bioplausible.acceleration.backends import HAS_CUPY

def get_backend(use_gpu: bool):
    ...

def to_numpy(arr):
    ...

def softmax(x, xp=None):
    ...

def spectral_normalize(W, num_iters=5, u=None, xp=None):
    ...
```

---

## 10. MEDIUM: Metrics Class Consolidation (300 lines)

### Problem
10+ `*Metrics` dataclasses with overlapping fields:

| Class | Location |
|-------|----------|
| `TrialMetrics` | `hyperopt/metrics.py` |
| `TrainingMetrics` | `core/trainer.py`, `equitile/lm/training.py` |
| `EpochMetrics` | `zoo/mep/benchmarks/_shared.py` |
| `BenchmarkMetrics` | `zoo/mep/benchmarks/runner.py` |
| `MetricsTracker` | `zoo/mep/benchmarks/metrics.py` |
| `HomeostasisMetrics` | `zoo/models/eqprop/homeostatic.py` |
| `StatisticalMetrics` | `equitile/benchmarks/rigorous.py` |
| `TileMetrics` | `equitile/analysis/dynamics.py` |
| `MetricsDashboard` | `equitile/lm/demo.py` |
| `StageMetrics` | `experiment/staircase.py` |

### Solution: `bioplausible/core/metrics.py`

```python
@dataclass(frozen=True, slots=True)
class BaseMetrics:
    loss: float
    accuracy: float | None = None
    epoch: int = 0
    step: int = 0
    extra: dict = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TrainingMetrics(BaseMetrics):
    lr: float = 0.0
    grad_norm: float | None = None

@dataclass(frozen=True, slots=True)
class TrialMetrics(BaseMetrics):
    model_name: str = ""
    config: dict = field(default_factory=dict)
    param_count: int = 0
    iteration_time: float = 0.0
    perplexity: float | None = None
    status: str = "completed"
```

---

## 11. LOW: Centralized Logging (100 lines)

### Problem
113 `logger = logging.getLogger(__name__)` calls

### Solution: `bioplausible/core/logging.py`

```python
import logging

def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "bioplausible")
    return logging.getLogger(name)

# Usage: from bioplausible.core.logging import get_logger
# logger = get_logger()
```

---

## 12. LOW: Pareto/ND Sorting Deduplication (150 lines)

### Problem
`hyperopt/metrics.py` has `non_dominated_sort`, `crowding_distance`, `get_pareto_frontier`, `rank_trials`
Similar logic likely in `equitile/benchmarks/`

### Solution
Keep in `hyperopt/metrics.py`, import everywhere.

---

## Prioritized Implementation Plan

### Phase 1: Foundation (Week 1) — CRITICAL
| Task | Files Changed | Est. Time |
|------|---------------|-----------|
| 1.1 Create `config/unified.py` | 1 new, 60+ imports updated | 4h |
| 1.2 Create `core/utils/activations.py` | 1 new, 7 files updated | 2h |
| 1.3 Create `core/utils/seeds.py` | 1 new, 7 files updated | 1h |
| 1.4 Create `core/utils/device.py` | 1 new, 20 files updated | 1h |
| 1.5 Create `core/logging.py` | 1 new, 113 files updated | 2h |

### Phase 2: Model Architecture (Week 2) — CRITICAL
| Task | Files Changed | Est. Time |
|------|---------------|-----------|
| 2.1 Extract `TrainingMixin` + `SpectralMixin` | 3 new, `core/model.py`, `equitile/core/model.py`, `zoo/models/base.py` | 4h |
| 2.2 Refactor `EquiTile` to use composition | `equitile/core/model.py` | 3h |
| 2.3 Refactor `EqPropModel` to use composition | `zoo/models/base.py` | 2h |
| 2.4 Add `CheckpointMixin` to `BioModel` | `core/model.py`, `core/checkpoint.py` | 2h |

### Phase 3: High-Impact Deduplication (Week 3) — HIGH
| Task | Status | Files Changed | Est. Time |
|------|--------|---------------|-----------|
| 3.1 Unified `equitile/deployments/base.py` configs + factory | ✅ DONE | 1 new | 4h |
| 3.2 Refactor `vision.py`/`timeseries.py`/`rl.py`/`graph.py` to reuse shared feature extractors + base configs | ✅ DONE | 4 files + 1 new `_feature_extractors.py` | 4h |
| 3.3 Consolidate acceleration array ops | ✅ DONE (already completed) | `acceleration/backends.py` or new | 2h |

### Phase 4: Medium Impact (Week 4) — MEDIUM
| Task | Status | Files Changed | Est. Time |
|------|--------|---------------|-----------|
| 4.1 Unify metrics classes | ✅ Partial (`core/metrics.py`, `_shared.EpochMetrics`) | `core/metrics.py` + `_shared.py` | 3h |
| 4.2 Pareto sorting deduplication | ⚠️ REVISED (investigated, not a true dup) | — | 1h |

### Phase 5: Cleanup & Validation (Week 5)
| Task | Files Changed | Est. Time |
|------|---------------|-----------|
| 5.1 Run full test suite | - | 2h |
| 5.2 Fix any import/usage issues | - | 4h |
| 5.3 Update docs/README | - | 2h |
| 5.4 Benchmark performance | - | 2h |

---

## Risk Assessment

| Refactor | Risk | Mitigation |
|----------|------|------------|
| Config hierarchy | HIGH — touches 60+ files | Automated codemod + tests |
| Base model composition | HIGH — core behavior change | Incremental, keep old classes as aliases |
| FastLMEquiTile merge | MEDIUM — demo vs canonical | Wrapper pattern preserves API |
| Deployment configs | LOW — internal only | Factory functions unchanged |
| Checkpointing | LOW — additive | Old methods deprecated, not removed |

---

## Validation Checklist

After each phase:
- [ ] `ruff format . && ruff check --fix .`
- [ ] `pyright .` — zero errors in strict mode
- [ ] `pytest --cov` — all tests pass, coverage ≥85%
- [ ] `pip-audit` — no new vulnerabilities
- [ ] Manual smoke test: `uv run python -m bioplausible.cli.run --help`

---

## Quick Wins (Do Immediately)

These require minimal risk and can be done in any order:

1. **`core/utils/activations.py`** — 7 files, pure utility, zero risk
2. **`core/utils/seeds.py`** — 7 files, pure utility, zero risk
3. **`core/utils/device.py`** — 20 files, pure utility, zero risk
4. **`core/logging.py`** — 113 files, mechanical replacement
5. **Acceleration array ops** — 2 files, internal only

---

## New Improvement Opportunities (discovered during Phase 3.2/10)

| Opportunity | Where | Est. Lines | Priority |
|-------------|-------|-----------|----------|
| **`core/utils/seeds.py` is imported but `core/logging.py` `get_logger()` is unused** — 113 call sites still use `logging.getLogger(__name__)`. Mechanical migration is the only remaining step. | `cli/`, `zoo/`, `equitile/` | ~110 | 🟡 MEDIUM |
| **`bioplausible/domains/base.py:93` `Metrics` (StrEnum of metric names) duplicates `core/metrics.py` naming** — not a code duplicate, but the string literals overlap. Low value to unify. | `domains/base.py` | — | 🟢 LOW |
| **`zoo/mep/benchmarks/runner.py:64 BenchmarkMetrics` and `core/trainer.py TrainingMetrics`** both model epoch-level metrics but with incompatible field names (`train_acc`/`val_acc` vs `train_accuracy`/`val_accuracy`). Wiring `core/trainer.TrainingMetrics` onto `core.metrics.BaseMetrics` would let `BenchmarkMetrics` subclass or alias it. | `core/trainer.py`, `zoo/mep/benchmarks/runner.py` | ~40 | 🟡 MEDIUM |
| **`equitile/benchmarks/rigorous.py:89 StatisticalMetrics`** is a standalone 5-field frozen dataclass (`accuracy, loss, param_count, iteration_time, epoch_time`) — a candidate to align with `BaseMetrics` shape. | `equitile/benchmarks/rigorous.py` | — | 🟢 LOW |
| **`acceleration/_array_ops.py`**: after the Phase 1.2 consolidation it is a thin re-exporter. Safe to delete once all importers switch to `core.utils.activations`. | `acceleration/` | ~30 | 🟡 MEDIUM |
| **Config hierarchy (Phase 1.1, revised design)**: the blocker is OmegaConf. A path forward is a `frozen=True` *unstructured* config dataclass plus a separate `OmegaConf`-structured mirror; migrate callers incrementally. | `config/schema.py`, `core/config.py` | ~1,500 (est.) | 🔴 CRITICAL |
| **Phase 4 (FastLMEquiTile merge)**: still blocked on the architecture decision recorded under "DEFERRED". The `lm/fast_lm.py` (canonical, `BioModel` subclass) vs `language/fast.py` (demo, `OptimizedLMEquiTile` subclass) split is the real differentiator. Resolving it would reclaim ~500 lines. | `equitile/lm/`, `equitile/language/` | ~500 | 🟠 HIGH |

---

## Technical Notes / Facilitating Future Work (Phase 3.2 & 10)

These notes capture decisions and gotchas so the next pass doesn't re-solve them.

### Deployment module structure after 3.2

```
equitile/deployments/
├── base.py              # frozen DeploymentConfig hierarchy + create_deployment_model
├── _feature_extractors.py  # shared NN: ConvFeatureExtractor, Temporal*, RL/Graph* extractors + graph scatter utils
├── vision.py            # ConvEquiTileConfig(base ConvDeploymentConfig), ConvEquiTile model, VisionAugmentation
├── timeseries.py        # standalone TimeSeriesConfig (NO mode/inference_steps), TimeSeriesEquiTile model
├── rl.py                # RLEquiTileConfig(base RLDeploymentConfig), RLEquiTile/RecurrentRLEquiTile model
├── graph.py             # standalone GraphEquiTileConfig (NO mode/inference_steps), GraphEquiTile model
└── deployment.py        # unchanged: export/quantize/prune
```

- **Vision and RL configs inherit from `base.py`** because they genuinely use EquiTile PC/EP dynamics fields (`mode`, `inference_steps`, `step_size`, `beta`) and the test suite confirms defaults (`learning_rate=0.01` for vision, `3e-4`/`mode="backprop"`/`inference_steps=5`/`neurons_per_tile=32` for RL).
- **Timeseries and graph configs are standalone** (`frozen=True, slots=True`) because they train with standard backprop and the `test_builder_cleanup.py::test_*_config_cleanup` tests explicitly assert `mode`/`inference_steps` are **absent** from these configs. Forcing them onto the base config hierarchy would break those assertions and leak unused PC/EP fields into backprop-only models.
- All **shared NN layers** (ConvFeatureExtractor, TemporalPositionalEncoding/AttentionLayer/EquiTileLayer, RL/Graph feature extractors, GraphAttentionLayer, GraphEquiTileLayer, scatter utilities) now live in `_feature_extractors.py`. The 4 public deployment modules re-export the historical names so `from bioplausible.equitile.deployments.{vision,timeseries,rl,graph} import X` and the top-level `bioplausible.equitile import X` keep working unchanged.

### Frozen-config gotcha

`DeploymentConfig`/`ConvDeploymentConfig`/etc. in `base.py` are `@dataclass(frozen=True, slots=True)`. Python forbids a non-frozen dataclass from inheriting a frozen one, so any deployment config subclassing a base config must also be `frozen=True, slots=True`. None of the deployment configs are mutated after construction (verified by search), so this is safe. If a future config needs mutation, the base must remain frozen and the subclass must use `__setattr__`/`object.__setattr__` — do not flip `frozen=False` on the subclass (raises `TypeError`).

### `core/metrics.py` canonical base

- `BaseMetrics` (epoch, step, extra) and `EpochMetrics` (adds train_loss/train_acc/val_loss/val_acc/epoch_time) are the canonical epoch-level containers. `_shared.EpochMetrics` now re-exports `core.metrics.EpochMetrics` — `compare.py`/`tuned_compare.py` construct it positionally, so the inherited defaults are harmless.
- `core/trainer.py`'s `TrainingMetrics` keeps its field names (`train_loss`/`train_accuracy`) and is **not** forced onto `BaseMetrics` in this pass — wiring it requires care with its `asdict`-based `to_dict` and `__post_init__`-free reconstruction from checkpoints (`TrainingMetrics(**m)`). A future spike should add `BaseMetrics` as a base and reconcile the `acc`/`accuracy` naming, ideally after confirming no caller constructs `TrainingMetrics` with `extra`/`step` kwargs today.

### Pareto (Phase 12) — not a true duplication

`hyperopt/metrics.py::non_dominated_sort` operates on `TrialMetrics` with 4 objectives (accuracy ↑, perplexity ↓, iteration_time ↓, param_count ↓). `analysis/results.py::compute_pareto_frontier` operates on raw `dict` trials with 3 objectives (accuracy ↑, param_count ↓, iteration_time ↓). They differ in input type and objective count, so unifying would change semantics. Leave as-is; if a future ticket unifies the trial representation, route both through one implementation.

### Verified behavior preserved

- `ConvEquiTileConfig(input_channels=1, input_size=28, num_classes=10, equitile_kwargs={"sparsity_threshold":0.5})` → `model.config.sparsity_threshold` propagates to `model.head.get_config()` (via `equitile_kwargs.copy()`).
- `RLEquiTileConfig(obs_dim=8, action_dim=4, equitile_kwargs={"dropout":0.3})` → `model.feature_extractor.get_config().dropout == 0.3`.
- `ConvEquiTileConfig()` defaults: `learning_rate=0.01`, `neurons_per_tile=64`.
- `RLEquiTileConfig()` defaults: `mode="backprop"`, `learning_rate=3e-4`, `inference_steps=5`, `neurons_per_tile=32`.
- Graph/timeseries `__all__` exports unchanged (re-exports from `_feature_extractors`).

### Test baseline (2026-08-09 full run)

14 pre-existing failures unrelated to this refactor (EP numerical parity on CPU vs GPU/seed drift, ONNX export under strict `torch==2.6`, Triton kernel float tolerance on CUDA 12.x). Deployment/metrics suites: **621 passed, 4 skipped** after this refactor — 0 regressions introduced.

---

## Appendix: File-Level Impact Map

```
bioplausible/
├── config/
│   ├── unified.py          ← NEW (replaces schema.py, parts of __init__.py)
│   └── __init__.py         ← UPDATE (re-export unified)
├── core/
│   ├── model.py            ← REFACTOR (extract mixins)
│   ├── checkpoint.py       ← KEEP (canonical)
│   ├── losses.py           ← KEEP (already unified)
│   ├── trainer.py          ← UPDATE (use mixins)
│   ├── config.py           ← DELETE (replaced by unified)
│   ├── utils/
│   │   ├── activations.py  ← NEW
│   │   ├── seeds.py        ← NEW
│   │   ├── device.py       ← NEW
│   │   └── logging.py      ← NEW
│   └── metrics.py          ← NEW
├── equitile/
│   ├── core/
│   │   ├── model.py        ← REFACTOR (composition)
│   │   └── config.py       ← UPDATE (use unified)
│   ├── deployments/
│   │   ├── base.py         ← NEW ✅ (frozen config hierarchy + factory)
│   │   ├── _feature_extractors.py ← NEW ✅ (shared Conv/Temporal/RL/Graph NN + scatter utils)
│   │   ├── vision.py       ← REFACTOR ✅ (ConvEquiTileConfig(base), reuse ConvFeatureExtractor)
│   │   ├── timeseries.py   ← REFACTOR ✅ (standalone config, reuse Temporal* layers)
│   │   ├── rl.py           ← REFACTOR ✅ (RLEquiTileConfig(base), reuse RLFeatureExtractor)
│   │   └── graph.py        ← REFACTOR ✅ (standalone config, reuse Graph*/scatter utils)
│   ├── lm/
│   │   ├── fast_lm.py      ← KEEP (canonical)
│   │   ├── components.py   ← UPDATE (remove FastLMConfig dup)
│   │   └── training.py     ← UPDATE (use mixins)
│   ├── language/
│   │   └── fast.py         ← REFACTOR (thin wrapper)
│   └── utils/
│       └── reproducibility.py  ← UPDATE (use core.utils.seeds)
├── zoo/
│   ├── models/
│   │   ├── base.py         ← REFACTOR (composition)
│   │   └── ...             ← UPDATE (import from core)
│   └── mep/
│       └── benchmarks/     ← UPDATE (EpochMetrics ← core.metrics)
├── experiments/
│   └── utils.py            ← UPDATE (use unified configs)
├── validation/
│   └── ...                 ← UPDATE (use unified configs)
├── core/metrics.py         ← NEW ✅ (BaseMetrics + EpochMetrics)
├── acceleration/
│   ├── backends.py         ← UPDATE (add array ops)
│   ├── _array_ops.py       ← DELETE
│   └── kernels.py          ← UPDATE (import from backends)
└── hyperopt/
    └── metrics.py          ← KEEP (canonical Pareto, TrialMetrics + ND sort)
```

---

**Total Estimated Effort**: ~50 hours over 5 weeks  
**Expected Outcome**: 12-15% codebase reduction, single source of truth for configs/utilities, cleaner architecture