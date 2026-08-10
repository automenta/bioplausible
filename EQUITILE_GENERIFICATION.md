# EquiTile Generification Plan — Lifting Tile-Based Local Learning to General Infrastructure

**Date**: 2026-08-10  
**Goal**: Extract EquiTile's tile-based local learning primitives into general-purpose infrastructure usable by any bio-plausible algorithm (PC, EP, Hebbian, FA, Target Prop, etc.)

---

## 1. Current Architecture Analysis

### EquiTile-Specific Components (to be generified)

| Component | Location | Current Scope | General Applicability |
|-----------|----------|---------------|----------------------|
| **TileGraph / TileState** | `equitile/core/topology.py` | EquiTile only | **Universal** — graph of compute units with state |
| **Kernels** | `equitile/core/kernels.py` | EquiTile only | **Universal** — activity update, Hebbian updates, contrastive updates |
| **Optimizer Mixin** | `equitile/training/optimizer_mixin.py` | EquiTile only | **Universal** — multi-optimizer management (weights, importance, full) |
| **Task Handler** | `equitile/training/task_handler.py` | EquiTile only | **Universal** — task-type-aware loss/grad/metrics |
| **Config Hierarchy** | `equitile/core/config.py` | EquiTile + deployments | **Universal** — local learning dynamics params |
| **Deployment Feature Extractors** | `equitile/deployments/_feature_extractors.py` | Vision/TS/RL/Graph | **Universal** — any model needing feature extraction + tile head |

### Algorithm-Specific Logic (stays in EquiTile)

| Component | Why Algorithm-Specific |
|-----------|------------------------|
| PC/EP/Backprop mode switching | Different energy functions & nudge protocols |
| Free/Nudged phase management | EqProp-specific contrastive learning |
| Importance-based sparsity gating | EquiTile's specific adaptive computation |
| Dynamic tile growth/pruning | EquiTile-specific architecture search |
| Tile-specific LR adaptation | EquiTile-specific meta-learning |

---

## 2. Generification Strategy: Layered Architecture

```
bioplausible/
├── core/
│   ├── tile/
│   │   ├── topology.py          # TileGraph, TileState (GENERIC)
│   │   ├── kernels.py           # compute_activity_update, compute_hebbian_update, etc. (GENERIC)
│   │   ├── state.py             # TileStateDict, checkpointing helpers (GENERIC)
│   │   └── __init__.py
│   ├── local_learning/
│   │   ├── mixins.py            # LocalLearningMixin, MultiOptimizerMixin (GENERIC)
│   │   ├── task.py              # TaskHandler (GENERIC)
│   │   ├── config.py            # LocalLearningConfig base (GENERIC)
│   │   └── __init__.py
│   ├── model.py                 # BioModel (uses LocalLearningMixin optionally)
│   └── ...
├── equitile/
│   ├── core/
│   │   ├── config.py            # EquiTileConfig extends LocalLearningConfig
│   │   ├── model.py             # EquiTile uses core.tile + core.local_learning
│   │   └── ...
│   └── deployments/             # Uses core.tile kernels + feature extractors
```

---

## 3. Detailed Component Migration

### 3.1 `core/tile/topology.py` — **TileGraph & TileState** (NEW LOCATION)

**Current**: `equitile/core/topology.py`  
**New**: `bioplausible/core/tile/topology.py`

```python
# GENERIC — no EquiTile-specific logic
@dataclass
class TileState:
    id: int
    neurons: int
    layer_id: int
    # Dynamic state — algorithm-agnostic
    activity: Tensor | None = None
    prediction: Tensor | None = None  # For predictive coding
    error: Tensor | None = None      # For PC/EP/error-driven learning
    value: Tensor | None = None      # For value-based methods (RL, FA)
    # Metadata
    is_input: bool = False
    is_output: bool = False
    pos_x: float = 0.0
    pos_y: float = 0.0
    # Connectivity
    fwd_neighbors: list[int] = field(default_factory=list)
    bwd_neighbors: list[int] = field(default_factory=list)

class TileGraph:
    """Generic tile connectivity manager."""
    def build_layered(...) -> None: ...      # Universal layered topology
    def build_custom(...) -> None: ...       # Universal custom topology
    def get_boundary_tiles(...) -> dict: ... # Universal distributed support
    @property
    def all_tiles(self) -> list[TileState]: ...
```

**Consumers**: EquiTile, future GraphEquiTile, SpikingTileModel, HierarchicalPC, etc.

---

### 3.2 `core/tile/kernels.py` — **Mathematical Kernels** (NEW LOCATION)

**Current**: `equitile/core/kernels.py`  
**New**: `bioplausible/core/tile/kernels.py`

```python
# GENERIC — pure math, no algorithm assumptions
def compute_tile_prediction(inputs: list[Tensor], bias: Tensor | None, ...) -> Tensor: ...

def compute_activity_update(
    activity: Tensor,
    error: Tensor,
    fwd_feedback: list[Tensor],
    importance: float,
    step_size: float,
    lambda_error: float,
    clamp_min: float,
    clamp_max: float,
    clamp: bool,
) -> Tensor: ...

def compute_hebbian_update(
    src_act: Tensor, dst_err: Tensor, importance: float, batch_size: int
) -> tuple[Tensor, Tensor]: ...

def compute_contrastive_hebbian_update(
    src_free: Tensor, dst_free: Tensor,
    src_nudged: Tensor, dst_nudged: Tensor,
    learning_rate: float, beta: float, batch_size: int
) -> tuple[Tensor, Tensor]: ...

# NEW: Generic kernels for other algorithms
def compute_fa_update(...) -> Tensor: ...           # Feedback Alignment
def compute_target_prop_update(...) -> Tensor: ...  # Target Propagation
def compute_hebbian_anti_hebbian(...) -> Tensor: ... # Hebbian/Anti-Hebbian
```

**Consumers**: Any tile-based local learning algorithm.

---

### 3.3 `core/tile/state.py` — **Checkpointing & Serialization** (NEW LOCATION)

**Current**: `equitile/_internal/state_types.py` + checkpoint methods in EquiTile  
**New**: `bioplausible/core/tile/state.py`

```python
@dataclass(frozen=True, slots=True)
class TileStateDict:
    """Generic tile model state for checkpointing."""
    model_state_dict: dict[str, Tensor]
    tile_graph: TileGraph  # Serialized topology
    config: LocalLearningConfig
    training: dict[str, object]  # step_count, error_ema, etc.
    optimizers: dict[str, dict]  # weight_opt, importance_opt, full_opt

def save_tile_checkpoint(path: str, state: TileStateDict) -> None: ...
def load_tile_checkpoint(path: str, map_location) -> TileStateDict: ...
```

---

### 3.4 `core/local_learning/mixins.py` — **Mixin Infrastructure** (NEW LOCATION)

**Current**: `equitile/training/optimizer_mixin.py`  
**New**: `bioplausible/core/local_learning/mixins.py`

```python
class MultiOptimizerMixin:
    """Manages multiple optimizers for local learning (weights, importance, full)."""
    def _setup_optimizers(self) -> None:
        # Generic: subclasses define optimizer_groups()
        for name, params, lr in self.optimizer_groups():
            setattr(self, f"_optim_{name}", torch.optim.Adam(params, lr=lr))

    def optimizer_groups(self) -> list[tuple[str, list[Tensor], float]]:
        """Override: return [(name, params, lr), ...]"""
        raise NotImplementedError

    def reset_optimizers(self) -> None:
        self._setup_optimizers()
        # Restore schedulers...

class LocalLearningMixin:
    """Core local learning protocol — algorithm-agnostic."""
    def _run_inference(self, input_proj, steps, batch, device) -> None: ...
    def _compute_predictions(self, ...) -> None: ...
    def _compute_errors(self) -> None: ...
    def _relax(self, ...) -> None: ...
    
    # Algorithm-specific hooks (override in subclasses)
    def _train_step_local(self, x, y) -> dict: raise NotImplementedError
    def _compute_loss_and_delta(self, logits, y) -> tuple[Tensor, Tensor]: ...
```

---

### 3.5 `core/local_learning/task.py` — **TaskHandler** (NEW LOCATION)

**Current**: `equitile/training/task_handler.py`  
**New**: `bioplausible/core/local_learning/task.py`

```python
# GENERIC — used by any model
class TaskHandler:
    def __init__(self, task_type: Literal["classification", "regression", "binary", "multilabel"], output_dim: int): ...
    def compute_loss(self, logits: Tensor, y: Tensor) -> Tensor: ...
    def compute_loss_and_grad(self, logits: Tensor, y: Tensor) -> tuple[Tensor, Tensor]: ...
    def compute_metrics(self, logits: Tensor, y: Tensor) -> float: ...
```

**Consumers**: EquiTile, deployments, domains, zoo models, validation tracks.

---

### 3.6 `core/local_learning/config.py` — **Base Config** (NEW LOCATION)

**Current**: `equitile/core/config.py` (EquiTileConfig)  
**New**: `bioplausible/core/local_learning/config.py`

```python
@dataclass(frozen=True, slots=True)
class LocalLearningConfig:
    """Base config for any tile-based local learning algorithm."""
    # Architecture
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4
    
    # Learning
    learning_rate: float = 0.01
    importance_lr: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    dropout: float = 0.1
    
    # Dynamics (algorithm-agnostic params)
    inference_steps: int = 10
    step_size: float = 0.1
    clamp_activities: bool = True
    activity_clamp_min: float = -5.0
    activity_clamp_max: float = 5.0
    relaxation_tolerance: float = 1e-4
    
    # Task & Activation
    task_type: Literal["classification", "regression", "binary", "multilabel"] = "classification"
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    
    # Extensibility
    equitile_kwargs: dict[str, object] = field(default_factory=dict)  # Algorithm-specific

# EquiTile-specific extension
@dataclass(frozen=True, slots=True)
class EquiTileConfig(LocalLearningConfig):
    mode: Literal["pc", "ep", "backprop"] = "pc"
    lambda_error: float = 0.1
    beta: float = 0.1
    beta_anneal: float = 1.0
    inference_steps_free: int | None = None
    inference_steps_nudged: int | None = None
    use_symmetric_weights: bool = False
    ep_init_scale: float = 0.1
    importance_decay: float = 0.95
    importance_reg_coef: float = 0.01
    sparsity_penalty_coef: float = 0.05
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1
```

---

### 3.7 `core/tile/feature_extractors.py` — **Shared Feature Extractors** (MOVE)

**Current**: `equitile/deployments/_feature_extractors.py`  
**New**: `bioplausible/core/tile/feature_extractors.py`

```python
# GENERIC — any model can use these
class ConvFeatureExtractor(nn.Module): ...      # Vision
class TemporalFeatureExtractor(nn.Module): ...  # Time series
class RLFeatureExtractor(nn.Module): ...        # RL
class GraphFeatureExtractor(nn.Module): ...     # Graph
class TemporalPositionalEncoding(nn.Module): ...
class TemporalAttentionLayer(nn.Module): ...
class GraphAttentionLayer(nn.Module): ...
class GraphEquiTileLayer(nn.Module): ...
def scatter_mean/max/sum(...): ...
def create_graph_from_edges(...): ...
```

**Consumers**: Deployments, zoo models, custom architectures.

---

## 4. EquiTile Refactoring (Post-Generification)

### `equitile/core/model.py` — **Thin Algorithm Wrapper**

```python
from bioplausible.core.tile import TileGraph, TileState
from bioplausible.core.tile.kernels import (
    compute_activity_update, compute_hebbian_update, 
    compute_contrastive_hebbian_update, compute_tile_prediction
)
from bioplausible.core.local_learning import (
    LocalLearningMixin, MultiOptimizerMixin, TaskHandler
)
from bioplausible.core.local_learning.config import EquiTileConfig

class EquiTile(BioModel, LocalLearningMixin, MultiOptimizerMixin):
    """EquiTile: PC/EP/Backprop on tile substrate."""
    
    def __init__(self, config: EquiTileConfig, ...):
        # Use generic tile graph
        self.graph = TileGraph()
        self.graph.build_layered(...)
        
        # Generic init
        super().__init__(config)
        self.task_handler = TaskHandler(config.task_type, output_dim)
        
        # EquiTile-specific: importance parameters
        self.tile_importance = nn.Parameter(torch.ones(len(self.graph.tiles)))
        self.edge_importance = nn.Parameter(torch.ones(len(self.graph.edges)))
    
    def optimizer_groups(self) -> list[tuple[str, list[Tensor], float]]:
        """Define optimizer groups for MultiOptimizerMixin."""
        return [
            ("io", list(self.W_in.parameters()) + list(self.W_out.parameters()), self.config.learning_rate),
            ("importance", [self.tile_importance, self.edge_importance], self.config.importance_lr),
            ("full", self.parameters(), self.config.learning_rate) if self.config.mode in ("backprop", "ep") else None,
        ]
    
    # Algorithm-specific implementations only:
    def _train_step_pc(self, x, y): ...
    def _train_step_ep(self, x, y): ...
    def _train_step_backprop(self, x, y): ...
    def _ep_free_phase(self, ...): ...
    def _ep_nudged_phase(self, ...): ...
    def _ep_update(self, ...): ...
    def _update_importance(self): ...  # EquiTile-specific sparsity gating
```

---

## 5. New Algorithms Enabled by Generification

### 5.1 Hierarchical Predictive Coding (HPC)
```python
# bioplausible/zoo/models/predictive_coding/hierarchical.py
from bioplausible.core.tile import TileGraph, TileState
from bioplausible.core.tile.kernels import compute_activity_update
from bioplausible.core.local_learning import LocalLearningMixin, TaskHandler

class HierarchicalPC(BioModel, LocalLearningMixin):
    """Multi-scale PC with tile-based hierarchy."""
    def _train_step_local(self, x, y): ...
```

### 5.2 Feedback Alignment on Tile Substrate
```python
# bioplausible/zoo/models/fa/tile_fa.py
from bioplausible.core.tile import TileGraph
from bioplausible.core.tile.kernels import compute_fa_update
from bioplausible.core.local_learning import MultiOptimizerMixin

class TileFA(BioModel, MultiOptimizerMixin):
    """Feedback Alignment with tile-based local updates."""
    def optimizer_groups(self): ...
```

### 5.3 Target Propagation with Tiles
```python
# bioplausible/zoo/models/target_prop/tile_tp.py
from bioplausible.core.tile import TileGraph
from bioplausible.core.tile.kernels import compute_target_prop_update

class TileTargetProp(BioModel, LocalLearningMixin):
    """Target Prop using tile graph for inverse targets."""
    def _compute_targets(self, ...): ...
```

### 5.4 Spiking Tile Models
```python
# bioplausible/zoo/models/spiking/tile_snn.py
from bioplausible.core.tile import TileGraph, TileState
# TileState gets `spike_count`, `membrane_potential` fields
```

### 5.5 Graph Neural Networks with Local Learning
```python
# bioplausible/zoo/models/graph/tile_gnn.py
from bioplausible.core.tile.feature_extractors import GraphFeatureExtractor, GraphEquiTileLayer
# Reuse graph scatter kernels from core/tile/feature_extractors.py
```

---

## 6. Migration Plan (Phased)

### Phase 1: Core Infrastructure (Week 1-2)
| Step | Task | Files |
|------|------|-------|
| 1.1 | Create `core/tile/` directory structure | 4 new files |
| 1.2 | Move `TileGraph`, `TileState` → `core/tile/topology.py` | 1 move |
| 1.3 | Move kernels → `core/tile/kernels.py` + add FA/TP kernels | 1 move + extend |
| 1.4 | Create `core/tile/state.py` for checkpointing | 1 new |
| 1.5 | Move `TaskHandler` → `core/local_learning/task.py` | 1 move |
| 1.6 | Create `LocalLearningConfig` base in `core/local_learning/config.py` | 1 new |
| 1.7 | Create `MultiOptimizerMixin` / `LocalLearningMixin` in `core/local_learning/mixins.py` | 1 new |

### Phase 2: EquiTile Refactor (Week 2-3)
| Step | Task | Files |
|------|------|-------|
| 2.1 | Update `EquiTileConfig` to extend `LocalLearningConfig` | 1 edit |
| 2.2 | Refactor `EquiTile` to use `core.tile` + `core.local_learning` mixins | 1 major edit |
| 2.3 | Update imports across `equitile/` modules | ~10 files |
| 2.4 | Verify all 3 modes (PC/EP/Backprop) work | Tests |

### Phase 3: Deployments & Zoo Models (Week 3-4)
| Step | Task | Files |
|------|------|-------|
| 3.1 | Move `_feature_extractors.py` → `core/tile/feature_extractors.py` | 1 move |
| 3.2 | Update 4 deployment modules to import from core | 4 edits |
| 3.3 | Enable `zoo/models/` to use tile kernels (FA, TP, PC) | ~5 files |
| 3.4 | Add example: `TileFA`, `TileTargetProp` in zoo | 2 new |

### Phase 4: Validation & Documentation (Week 4)
| Step | Task | Files |
|------|------|-------|
| 4.1 | Full test suite + coverage | — |
| 4.2 | Add generification docs to `docs/architecture/tile_substrate.md` | 1 new |
| 4.3 | Update `REFACTOR.md` with completed generification | 1 edit |

---

## 7. Benefits Summary

| Metric | Before | After |
|--------|--------|-------|
| **Tile infrastructure reuse** | EquiTile only | All local learning algorithms |
| **Kernel duplication** | High (copy-paste risk) | Zero — single source in `core/tile/kernels.py` |
| **New algorithm effort** | ~500 lines boilerplate | ~100 lines (inherit mixins, override hooks) |
| **Deployment sharing** | Via `_feature_extractors.py` | Via `core/tile/feature_extractors.py` |
| **Checkpointing** | EquiTile-specific | Generic `TileStateDict` for any tile model |
| **Distributed support** | `equitile/training/distributed.py` | Generic `TileGraph.get_boundary_tiles()` in core |

---

## 8. Backward Compatibility

- **EquiTile API unchanged**: `EquiTile(config)`, `train_step()`, `forward()`, `save_checkpoint()` all work identically
- **Deployment APIs unchanged**: `ConvEquiTile`, `RLEquiTile`, etc. import from new locations via re-exports
- **Config classes**: `EquiTileConfig` extends `LocalLearningConfig` — all existing fields preserved
- **Registry entries**: Model registration unchanged

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Import breakage during moves | Use `sys.path` shims temporarily; run tests after each move |
| Subtle behavior changes in kernels | Freeze kernel implementations; add unit tests for each kernel |
| Mixin diamond inheritance | Use `Protocol` for mixin interfaces; explicit `__init__` chaining |
| Config validation gaps | `LocalLearningConfig.validate()` calls `super().validate()` in subclasses |

---

**Next Step**: Begin Phase 1.1 — create `core/tile/` directory and move topology/kernels.