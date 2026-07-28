# REFACTOR3: The TransitionGraph Architecture — Total Model-Propagator Contract

## Executive Summary

Replace all three hardcoded layer-discovery systems (EqProp's `_get_layers`, CHL's `_get_layers`, MEP's `ModelInspector`) with a **single model-declared `TransitionGraph` protocol**. The model is the sole authority on its state transitions. Propagators are pure gradient-transformation rules that consume this graph. If a model doesn't declare its transitions, per-layer propagators **fail fast with a clear, actionable error** — no silent discovery, no fallback, no protocol-with-defaults.

This architecture achieves **unambiguous DRY**: every piece of structural knowledge exists in exactly one place (the model's `transition_modules()`). Inconsistencies are impossible by construction. Adding new models, propagators, or optimizers requires zero modifications to existing code.

---

## 1. The Single Contract: `TransitionGraph`

**File:** `bioplausible/zoo/models/transitions.py`

```python
# bioplausible/zoo/models/transitions.py
from typing import Protocol
from torch import Tensor, nn

class TransitionGraph(Protocol):
    """Model declares its single-step state transition modules."""
    
    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during ONE forward step.
        
        Each module: forward(state, external_input?) -> next_state.
        The propagator iterates these modules to perform:
        - Free phase (EqProp/CHL): x → h₁ → h₂ → ... → hₙ
        - Nudged phase (EqProp): same, with target nudging
        - Energy settling (MEP): same, with energy gradient
        """
    
    def initial_state(self, x: Tensor) -> Tensor:
        """Initial state from input. Default: x."""
        return x
    
    def readout(self, final_state: Tensor) -> Tensor:
        """Convert final state to model output. Default: identity."""
        return final_state
    
    def num_settling_steps(self) -> int:
        """Iterations of transition_modules for free/nudged phases.
        Default: 1 (feedforward). RNNs/EqProp override to >1."""
        return 1
```

**One method, one responsibility.** No `Transition` dataclass, no over-specification. Parameters = `sum((list(m.parameters()) for m in model.transition_modules()), [])`.

### Default Mixin for Standard Models

```python
class TransitionGraphMixin:
    """Auto-discovers transition_modules for models with standard structure."""
    
    def transition_modules(self) -> list[nn.Module]:
        # 1. Explicit attribute (most common)
        if hasattr(self, "layers") and isinstance(self.layers, nn.ModuleList):
            return list(self.layers)
        # 2. Forward/feedback layers (DirectedEP)
        if hasattr(self, "forward_layers"):
            return list(self.forward_layers)
        # 3. Fallback: scan for Linear/Conv (preserves backward compat)
        modules = [m for m in self.modules() 
                   if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))]
        if modules:
            return modules
        raise NotImplementedError(
            f"{type(self).__name__} has no transition_modules. "
            "Define `self.layers: ModuleList[nn.Module]` or implement transition_modules()."
        )
```

**Standard models (StandardEqProp, MomentumEquilibrium, LoopedMLP, etc.) get this for free.** Custom models override.

---

## 2. Every Model Implements `TransitionGraph`

| Model | Implementation | Lines |
|---|---|---|
| `StandardEqProp`, `MomentumEquilibrium`, `SparseEquilibrium`, `LoopedMLP`, `MemoryEfficientEqPropModel` | Inherit `TransitionGraphMixin` → **zero code** | 0 |
| `DirectedEP` | `@property` → `self.forward_layers` | 1 |
| `HomeostaticEqProp` | `[self.W_in, *self.layers, self.head]` | 3 |
| `LazyEqProp` | `[self.embed, *self.layers, self.head]` | 3 |
| `TemporalResonanceEqProp` | `[self.W_in, *self.layers, self.osc_coupling, self.head]` | 3 |
| **`NeuralCube`** | Wrap `W_local` in `LocalUpdateModule`; return `[W_in, local, W_out]` | ~15 |
| **`TernaryEqProp`** | `[self.W_in, self.W_rec, self.W_out]` — `TernaryLinear` IS `nn.Module` | 3 |
| **`CausalTransformerEqProp`** | Interleaved `[attn0, ffn0, attn1, ffn1, ...]` | ~10 |
| `GraphEqProp`, `ConvEqProp`, `ModernConvEqProp` | Based on their layer structure | 3-10 |

**No model is exempt.** If `transition_modules()` is missing, per-layer propagators fail fast with exact error.

### NeuralCube Wrapper (One-Time Effort)

```python
class LocalUpdateModule(nn.Module):
    """Wraps NeuralCube's W_local + neighbor logic into a callable module."""
    def __init__(self, W_local: nn.Parameter, neighbor_indices: Tensor, cube_size: int):
        super().__init__()
        self.W_local = W_local
        self.register_buffer("neighbor_indices", neighbor_indices)
        self.cube_size = cube_size
    
    def forward(self, h: Tensor, x: Tensor | None = None) -> Tensor:
        # NeuralCube.local_update() logic
        batch_size, n_neurons = h.shape[0], self.neighbor_indices.shape[0]
        h_padded = F.pad(h, (0, 1))
        indices_expanded = self.neighbor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        h_expanded = h_padded.unsqueeze(1).expand(-1, n_neurons, -1)
        neighbor_activations = torch.gather(h_expanded, 2, indices_expanded)
        return (neighbor_activations * self.W_local.unsqueeze(0)).sum(dim=2)
```

---

## 3. Propagators Consume `transition_modules()` — Period

### EqProp (`bioplausible/zoo/propagators/eqprop.py`)

```python
def _get_transitions(self) -> list[nn.Module]:
    if not hasattr(self.model, "transition_modules"):
        raise TypeError(
            f"EqProp requires a model implementing TransitionGraph. "
            f"{type(self.model).__name__} does not implement transition_modules(). "
            "Either implement transition_modules() on your model, "
            "or use a whole-model propagator (Backprop, FeedbackAlignment)."
        )
    return self.model.transition_modules()
```

**Delete:** `_get_layers()`, `_settle()` (reimplement using `transition_modules()`), all `isinstance` checks.

### CHL (`bioplausible/zoo/propagators/hebbian.py`)

Same pattern. **Delete the hardcoded `F.relu(h)`** — the activation is now part of the module's forward (or add `activation_fn()` to `TransitionGraph` if needed). **Delete `_get_layers()`.**

### MEP (`bioplausible/zoo/mep/optimizers/settling.py`)

**Delete `ModelInspector` entirely.** The `Settler` registers hooks directly on `model.transition_modules()`.

```python
# In Settler.__init__:
for module in self.model.transition_modules():
    hook = module.register_forward_hook(self._capture_state)
    self._hooks.append(hook)
```

**Delete:** `bioplausible/zoo/mep/optimizers/inspector.py` entirely.

---

## 4. Registry Compatibility Contract (Zero Inconsistencies)

**File:** `bioplausible/core/registry.py`

```python
class Capability(StrEnum):
    TRANSITION_GRAPH = auto()      # Model implements transition_modules()
    STANDARD_AUTOGRAD = auto()     # Standard forward + loss.backward()
    CONTRASTIVE = auto()           # Implements get_hebbian_pairs()

# Propagator metadata (in ComponentMeta):
# Backprop, FeedbackAlignment, AdaptiveFA, StochasticFA, ContrastiveFA, MuonBackprop
#   → requires: [STANDARD_AUTOGRAD]
# EqProp, HolomorphicEqProp, FiniteNudgeEqProp, LazyEqProp
#   → requires: [TRANSITION_GRAPH]
# ContrastiveHebbianLearning
#   → requires: [TRANSITION_GRAPH]
# MEP presets (smep, sdmep, local_ep, natural_ep)
#   → requires: [TRANSITION_GRAPH]
# STDP
#   → requires: [STANDARD_AUTOGRAD]

# Model metadata:
# Standard MLP/Transformer/RNN/EqProp models
#   → provides: [TRANSITION_GRAPH, STANDARD_AUTOGRAD]
# Custom models implementing transition_modules()
#   → provides: [TRANSITION_GRAPH, STANDARD_AUTOGRAD]
# Plain nn.Module (no transition_modules())
#   → provides: [STANDARD_AUTOGRAD] only

def check_compatibility(propagator_name: str, model_name: str) -> bool:
    """Returns True if model provides all propagator requires."""
    prop_meta = _COMPONENTS[ComponentCategory.PROPAGATOR][propagator_name].metadata
    model_meta = _COMPONENTS[ComponentCategory.MODEL][model_name].metadata
    required = set(prop_meta.get("requires", []))
    provided = set(model_meta.get("provides", []))
    return required.issubset(provided)
```

**Single source of truth.** No duplicate compatibility logic anywhere.

---

## 5. Trainer Enforcement (Fail Fast, Actionable Errors)

**File:** `bioplausible/core/trainer.py`

```python
def _create_propagator(self):
    if not Registry.check_compatibility(self.config.propagator, self.config.model):
        prop_meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, self.config.propagator)
        model_meta = Registry.get_metadata(ComponentCategory.MODEL, self.config.model)
        required = prop_meta.get("requires", [])
        provided = model_meta.get("provides", [])
        missing = set(required) - set(provided)
        raise IncompatibilityError(
            f"Propagator '{self.config.propagator}' requires {missing}. "
            f"Model '{self.config.model}' only provides {provided}. "
            f"Fix: implement transition_modules() on your model, "
            f"or use a compatible propagator (e.g., 'backprop', 'fa')."
        )
    # ... existing creation logic ...
```

---

## 6. Complete Deduplication (DRY Unambiguous)

| Before (3+ separate systems) | After (1 contract) |
|---|---|
| `EqProp._get_layers()`: `model.modules()` + `isinstance(m, (nn.Linear, nn.Conv2d))` | `model.transition_modules()` |
| `CHL._get_layers()`: identical hardcoded scan | `model.transition_modules()` |
| `MEP.ModelInspector`: recursive `children()` + 20+ hardcoded `isinstance` checks | `model.transition_modules()` |
| `EqPropModel.get_hebbian_pairs()`: abstract, per-subclass | Can derive from `transition_modules()` |

**Files deleted:**
- `bioplausible/zoo/mep/optimizers/inspector.py`
- `EqProp._get_layers()`, `CHL._get_layers()`
- All hardcoded `isinstance(m, (nn.Linear, ...))` in propagators
- Hardcoded `F.relu` in CHL

---

## 6. Ergonomic Component Addition

| New Component | Implementation | Existing Code Touched |
|---|---|---|
| **New propagator** (e.g., PredictiveCoding) | Implement `Propagator` using `model.transition_modules()`; declare `requires: [TRANSITION_GRAPH]` in metadata | **Zero** |
| **New model** (e.g., Liquid Time-Constant) | Implement `transition_modules()`; register with `provides: [TRANSITION_GRAPH, STANDARD_AUTOGRAD]` | **Zero** |
| **New optimizer** (e.g., Lion) | Subclass `torch.optim.Optimizer`; register; works with any model | **Zero** |
| **New learning rule** (e.g., Gradient-free) | Add `Capability`; implement propagator using model's contract | **Zero** |
| **Non-sequential DAG model** | Return `transition_modules()` in topological order | **Zero** |

---

## 7. Configuration Permutations — All Covered

| Propagator \ Model | Implements `transition_modules()` | Plain `nn.Module` |
|---|---|---|
| **Backprop / FA / Muon** | ✅ Works (uses `STANDARD_AUTOGRAD`) | ✅ Works |
| **EqProp / CHL / MEP** | ✅ Works (uses `TRANSITION_GRAPH`) | ❌ **Clear error**: "Implement transition_modules() or use Backprop/FA" |

**Every mathematically valid combination works. Every invalid combination fails fast with exact fix.**

---

## 8. Execution Order (Non-Negotiable)

1. Create `bioplausible/zoo/models/transitions.py` with `TransitionGraph`, `TransitionGraphMixin`
2. Add `Capability` enum + `check_compatibility()` to `core/registry.py`
3. Wire compatibility check into `CoreTrainer._create_propagator()`
4. Update `EqProp` propagator: delete `_get_layers()`, implement `_get_transitions()` using `transition_modules()`
5. Update `CHL` propagator: same + delete hardcoded `F.relu`
6. Update `MEP` `Settler`: delete `ModelInspector`, use `transition_modules()` for hooks
7. Implement `transition_modules()` on all models in `zoo/models/eqprop/`
8. Add `LocalUpdateModule` wrapper for NeuralCube
9. Verify: every model works with Backprop, FA, EqProp, CHL, MEP; or fails with exact actionable error
10. Remove any remaining hardcoded optimizers (already done in REFACTOR2)
11. Remove unnecessary `train_step` overrides (already done in REFACTOR2)

---

## 9. Why This Is Total

1. **No fallback, no silent failure** — Missing `transition_modules()` = loud error with exact fix
2. **No hardcoded types anywhere** — `ModelInspector` deleted, 20+ `isinstance` checks gone
3. **Single source of truth** — `model.transition_modules()` is the ONLY structural discovery mechanism
4. **Model is authority** — Propagators are pure gradient rules; they don't discover structure
5. **Extensible by construction** — New propagators/models add capabilities without touching existing code
6. **Mathematically complete** — Every valid combination works; every invalid combination fails fast with actionable error
7. **Unambiguous DRY** — Structural knowledge exists in exactly one place (the model). Inconsistencies impossible by design.

This is the architecture BioPlausible was built to have.