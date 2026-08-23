# Sprint Backlog — Usability, Capability & Rigor (TODO2)

**Created**: 2026-08-23 | **Based on**: Actual testing of J0–J6 "completed" sprints | **Focus**: Fix gaps blocking AI/newcomer usage, demonstrations, and scientific rigor

**Guiding Principle**: Backwards compatibility: NONE. Professional, not explanatory. Self-documenting code. Working functionality > coverage.

---

## 🚨 CRITICAL: Blocking Issues (Must Fix First)

### 1. CLI Entry Points Broken
| Command | Error | Root Cause | Fix |
|---------|-------|------------|-----|
| `biopl parity` | `ImportError: cannot import name 'CoreTrainer'` | `cli/parity.py` imports from deprecated `core.trainer` | Update to `SystemTrainer` or remove |
| `biopl lab inspect` | `ValueError: Unknown category: ComponentCategory.MODEL` | `cli/lab.py` uses old registry API | Update to `SystemConfig.from_experiment()` |
| `biopl run` | Not tested | May have similar issues | Audit all CLI commands |

**Files to fix**: `bioplausible/cli/parity.py`, `bioplausible/cli/lab.py`, `bioplausible/cli/run.py`, `bioplausible/cli/validate.py`

---

### 2. Missing `compose_joint_system` Factory
README documents `compose_joint_system()` with 6 arguments (including `plasticity`) but **function doesn't exist**.

**Fix**: Implement `compose_joint_system()` in `core/system_trainer.py` matching README signature.

---

### 3. Geometry.forward() Requires Substrate (Undocumented)
```python
# Current (works)
output = geometry.forward(data, substrate)

# README shows (broken)
output = geometry.forward(data)
```
**Fix**: Update README examples + add `substrate` parameter with default `DigitalSubstrate()`.

---

### 4. Inconsistent Naming (Code vs README)
| Concept | Code | README | Action |
|---------|------|--------|--------|
| Thermodynamic credit | `ThermodynamicContrast` | `ThermodynamicContrastCredit` | Add alias `ThermodynamicContrastCredit = ThermodynamicContrast` |
| Joint plasticity config | `PlasticityConfig.routing()` | `PlasticityConfig.routing(gate_init_scale=0.1)` | Fix README |
| System composition | `compose_system` (5 args) | `compose_joint_system` (6 args) | Implement per #2 |

---

## 🔧 USABILITY IMPROVEMENTS (Reduce Boilerplate, DRY)

### 5. Quick-Start Script That Works Out of the Box
**Current**: `demo/main.py` launches web UI (heavy, requires NiceGUI)
**Need**: `scripts/quickstart.py` — trains EqProp vs Backprop on MNIST in <2 min, prints results

```python
# Target: uv run scripts/quickstart.py
# Output:
# Backprop:  57% accuracy (3 epochs)
# EqProp:    55% accuracy (3 epochs)  
# Both biologically plausible and standard learning work!
```

### 6. One-Line System Construction Helpers
Add to `core/presets.py` (new module, `__init__.py` re-exports):
```python
# Instead of 20 lines of config
system = create_backprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
system = create_eqprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10, beta=0.5, n_iters=20)
system = create_fa_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
system = create_routing_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
system = create_fast_weight_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
```

### 7. Unified Public API in `__init__.py`
```python
# bioplausible/__init__.py
from bioplausible.core.ontology import (
    System, DigitalSubstrate, FeedforwardGeometry, GeometryConfig,
    InstantaneousDynamics, EnergyMinimizationDynamics,
    BackpropCredit, ThermodynamicContrast, RandomProjectionsCredit,
    EuclideanUpdate, ParameterUpdateConfig,
    StateDynamicsConfig, CreditAssignmentConfig,
)
from bioplausible.core.joint import (
    CompositeState, StateRegistry, SystemContext,
    CoupledTransition, NullPlasticity, PlasticityConfig,
    RoutingPlasticity, FastWeightPlasticity,
)
from bioplausible.core.system_trainer import (
    compose_system, compose_joint_system,
    create_backprop_mlp, create_eqprop_mlp, create_fa_mlp,
    create_routing_mlp, create_fast_weight_mlp,
    SystemTrainer, SystemTrainerConfig,
)
from bioplausible.core.presets import *  # factories above

__all__ = [
    # Core ontology
    "System", "DigitalSubstrate", "FeedforwardGeometry", "GeometryConfig",
    "InstantaneousDynamics", "EnergyMinimizationDynamics",
    "BackpropCredit", "ThermodynamicContrast", "RandomProjectionsCredit",
    "EuclideanUpdate", "ParameterUpdateConfig",
    "StateDynamicsConfig", "CreditAssignmentConfig",
    # Joint architecture
    "CompositeState", "StateRegistry", "SystemContext",
    "CoupledTransition", "NullPlasticity", "PlasticityConfig",
    "RoutingPlasticity", "FastWeightPlasticity",
    # Factories
    "compose_system", "compose_joint_system",
    "create_backprop_mlp", "create_eqprop_mlp", "create_fa_mlp",
    "create_routing_mlp", "create_fast_weight_mlp",
    # Trainers
    "SystemTrainer", "SystemTrainerConfig",
]
```

### 8. Preset Configurations (YAML, No Code)
`configs/presets/` — config-driven experimentation:
```yaml
# configs/presets/eqprop_mnist.yaml
substrate: { type: digital }
geometry: { type: feedforward, input_dim: 784, output_dim: 10, hidden_dims: [256, 128] }
dynamics: { type: energy_minimization, max_steps: 20, step_size: 0.1, beta: 0.5 }
plasticity: { type: null }
credit: { type: thermodynamic_contrast, beta: 0.5 }
update: { type: euclidean, step_size: 0.01 }
training: { max_epochs: 10, batch_size: 64, device: auto }
```

CLI: `biopl run --config configs/presets/eqprop_mnist.yaml`

---

## 🧪 DEMONSTRATION & VALIDATION GAPS

### 9. Fix Failing Energy Invariant Tests
11/15 tests in `tests/integration/test_energy_invariants.py` fail. Core mathematical guarantees.

**Fix**: Restore Lyapunov/Control-Lyapunov proofs or mark `@pytest.mark.xfail` with precise reason.

### 10. Add Integration Test for Quick-Start
```python
# tests/integration/test_quickstart.py
def test_backprop_vs_eqprop_mnist():
    """Both algorithms train on same architecture, achieve >50% on MNIST in 3 epochs."""
```

### 11. Benchmark CLI Works
Verify `biopl benchmark run --suite adaptation_efficiency` produces output without errors.

---

## 📚 DOCUMENTATION SYNC (Self-Documenting Code)

### 12. README Quickstart Section
Replace broken `biopl lab` example:
```bash
# Quickstart (works in <2 min)
uv run scripts/quickstart.py

# Config-driven training
biopl run --config configs/presets/eqprop_mnist.yaml
```

### 13. API Docstrings (Google-Style, Behavior-Focused)
All public classes/functions: purpose, args with types, returns, invariants, side effects. No explanatory comments.

### 14. Architecture Diagram (Mermaid in README)
```mermaid
graph LR
    S[Substrate] --> G[Geometry]
    G --> D[StateDynamics]
    D --> M[Plasticity]
    M --> C[CreditAssignment]
    C --> U[ParameterUpdate]
    U --> S
```

---

## 🏗️ INFRASTRUCTURE

### 15. Pre-commit Hooks (Per AGENTS.md)
`.pre-commit-config.yaml` references removed files. Update to:
```yaml
- repo: local
  hooks:
    - id: ruff-format
      name: ruff format
      entry: uv run ruff format
      language: system
    - id: ruff-check
      name: ruff check
      entry: uv run ruff check --fix
      language: system
    - id: pyright
      name: pyright
      entry: uv run pyright
      language: system
    - id: pytest-fast
      name: pytest (property tests)
      entry: uv run pytest tests/property/ -q
      language: system
```

### 16. CI Pipeline (Per AGENTS.md Order)
GitHub Actions: `ruff format --check` → `ruff check` → `pyright` → `pytest tests/property/` → `pip-audit`

---

## 🤖 AI/AGENT-FRIENDLY IMPROVEMENTS

### 17. Machine-Readable API Schema
Generate `api_schema.json` from type hints for programmatic discovery.

### 18. Typed Exception Hierarchy
```python
# bioplausible/core/exceptions.py
class BioplausibleError(Exception): ...
class ConfigurationError(BioplausibleError): ...
class CompositionError(BioplausibleError): ...
class TrainingError(BioplausibleError): ...
```

### 19. Auto-Discovery of Valid Compositions
```python
# In SystemConfig
@classmethod
def valid_combinations(cls) -> list[dict]:
    """Return all valid 6-D coordinate combinations for AutoScientist."""
```

---

## 🔬 CORRECTNESS & RIGOR (Scientific Guarantees)

### 20. Gradient Equivalence Verification (CI Gate)
Add to property tests:
```python
# tests/property/test_gradient_equivalence.py
def test_eqprop_gradient_matches_bptt():
    """EqProp pseudo-gradient cosine ≥ 0.5 vs BPTT on same architecture."""

def test_fa_gradient_alignment_improves():
    """Feedback alignment matrices align with forward weights over training. cos(B, W^T) improvement > 0.05"""
```

### 21. Determinism Lock (L5) for All 6-D Coordinates
```python
# tests/property/test_determinism_extended.py
@pytest.mark.parametrize("coordinate", random_6d_coordinates(n=20))
def test_determinism_joint(coordinate):
    """Same seed + same device = bitwise equal params & metrics for any 6-D coordinate."""
```

### 22. Lyapunov/Control-Lyapunov Formal Verification
Restore failing energy invariant tests:

| System | Formal Guarantee | Test |
|--------|------------------|------|
| Symmetric + EnergyMinimization | LaSalle's invariance → fixed point | `test_symmetric_recurrent_converges` |
| PredictiveSettling | Control-Lyapunov → free energy non-increasing | `test_control_lyapunov_free_energy_decreases` |
| Neuromorphic | Passivity (‖n(a)-n(b)‖ ≤ ‖a-b‖) | `test_neuromorphic_passivity` |
| Quantum | Parameter-shift ≈ finite-diff (cos ≥ 0.999) | `test_quantum_parameter_shift` |

### 23. Locality Axiom Tests (L3) for All Credit Assignments
```python
# tests/property/test_locality_axioms.py
def test_thermodynamic_contrast_invariant_to_nonlocal_perturb():
    """Layer-0 pseudo-gradient unchanged by non-local weight perturbation."""

def test_fa_feedback_fixed_at_init():
    """FA backward weights ≠ forward transpose; separate memory; seed-independent."""
```

### 24. Zero-Extension Theorem Numerical Verification (J1)
Verify in CI with documented tolerance:
```python
# tests/property/joint/test_null_equivalence.py
def test_null_plasticity_preserves_5d_dynamics():
    """F_θ^Null = D_θ within numerical tolerance (1e-5)."""
```

---

## 🎭 VISUALIZATION & DEBUGGING (Demonstrability)

### 25. Joint State Inspector CLI
```bash
biopl lab inspect-state --coordinate digital/recurrent/energy_min/routing/thermo/euclidean \
    --task mnist --steps 50 --output state_evolution.html

# Output: Interactive Plotly showing:
# - Activity trajectories (x_t) per layer
# - Plastic state evolution (ψ_t) — gate logits, fast weights
# - Substrate state (σ_t) — conductance, noise
# - Energy per iteration
# - Spectral radius ρ(J_F) per step
```

### 26. 6-D Ontology Explorer (Interactive)
```bash
uv run scripts/ontology_explorer.py
# → NiceGUI at localhost:8080
# - Click axes to select primitives
# - Shows valid/invalid combinations in real-time
# - Generates config YAML or Python code
# - Links to relevant papers/benchmarks
```

### 27. Training Dynamics Visualizer
```python
from bioplausible.analysis.dynamics import plot_training_dynamics

plot_training_dynamics(
    trajectory=joint_trajectory,
    save_html="training_dynamics.html"
)
# Shows: energy, loss, accuracy, ρ(J_F), gate entropy, settling time
```

### 28. Plasticity Effect Comparison Benchmark
```bash
biopl benchmark compare --suite adaptation_efficiency \
    --plast null routing fast_weights \
    --output plasticity_comparison.html

# Output: Multi-panel figure:
# - Adaptation curves (loss vs episodes)
# - Gate entropy evolution (routing)
# - Fast weight matrix heatmaps
# - Resource usage (compute, memory, plastic state)
# - Stability proxies (ρ(J_F), Lyapunov)
```

---

## 🧠 AUTOSCIENTIST ACCESSIBILITY

### 29. Minimal Campaign Runner
```bash
biopl scientist explore --space joint_smoke \
    --objective adaptation_efficiency \
    --budget 10 \
    --output campaign_results/

# joint_smoke = {
#   substrate: [digital],
#   geometry: [feedforward, recurrent],
#   dynamics: [instantaneous, energy_minimization],
#   plasticity: [null, routing, fast_weights],
#   credit: [backprop, thermodynamic_contrast, random_projections],
#   update: [euclidean]
# }
```

### 30. Campaign Result Browser
```bash
biopl campaign list --format table
biopl campaign show <campaign_id> --include frontier,resources,stability
biopl campaign pareto <campaign_id> --objectives accuracy,adaptation_time,rho_jacobian
```

### 31. Hypothesis Template Library
Pre-built chain-of-thought templates for AutoScientist:
```python
# bioplausible/autoscientist/templates/
# - substrate_ablation.md      # "What if we change substrate?"
# - credit_swap.md             # "Does FA work better on memristive?"
# - plasticity_search.md       # "Does routing help adaptation?"
# - stability_frontier.md      # "Maximize adaptation s.t. ρ(J_F) < 0.99"
```

---

## ⚡ PERFORMANCE & PROFILING

### 32. Joint Kernel Profiler
```bash
biopl benchmark profile --coordinate digital/recurrent/energy_min/routing/thermo/euclidean \
    --batch-sizes 32,64,128 --device cuda \
    --output kernel_profile.json

# Output: FLOPs, memory, latency per kernel type
# - CoupledTransition.step
# - PlasticityPrimitive.step  
# - Stability estimators
# - Adapter projections
```

### 33. Empirical Resource Analysis
```python
# In analysis/profiling.py
def analyze_joint_system(coordinate: SystemCoordinate) -> ResourceUsage:
    """Empirical measurement of compute, memory, energy, plastic state capacity."""
    # Uses torch.profiler + nvml for GPU memory
    # Returns ResourceUsage for FrontierRecord
```

---

## 📋 PRIORITIZED EXECUTION PLAN

| Phase | Task | Effort | Impact | Category |
|-------|------|--------|--------|----------|
| **P0** | Fix `biopl parity`, `biopl lab` CLI imports | 1 hr | Unblocks all CLI usage | Usability |
| **P0** | Implement `compose_joint_system()` | 2 hrs | Matches README, enables 6-D API | Usability |
| **P0** | Fix `geometry.forward(substrate)` in README/examples | 30 min | Prevents immediate confusion | Usability |
| **P0** | Fix energy invariant tests or xfail with reason | 4 hrs | **Restores mathematical guarantees** | Rigor |
| **P1** | Create `scripts/quickstart.py` | 2 hrs | **Primary demonstration artifact** | Demo |
| **P1** | Add preset factory functions (`create_eqprop_mlp`, etc.) | 3 hrs | Reduces boilerplate 10x | Usability |
| **P1** | Fix naming inconsistencies (alias `ThermodynamicContrastCredit`) | 1 hr | Eliminates "works in code not docs" | Usability |
| **P1** | Gradient equivalence in CI gate (property tests) | 3 hrs | **Core scientific claim** | Rigor |
| **P2** | Unified `__init__.py` exports | 1 hr | `from bioplausible import *` works | Usability |
| **P2** | Preset YAML configs + `biopl run --config` | 2 hrs | Config-driven experimentation | Usability |
| **P2** | Joint state inspector CLI (`biopl lab inspect-state`) | 4 hrs | **Visual debugging of joint dynamics** | Demo |
| **P2** | Determinism lock for all 6-D coordinates | 2 hrs | Reproducibility guarantee | Rigor |
| **P3** | 6-D Ontology Explorer (interactive) | 6 hrs | **Exploration & discovery tool** | Demo/Capability |
| **P3** | Training dynamics visualizer | 3 hrs | Understand what's happening | Demo |
| **P3** | Plasticity comparison benchmark | 3 hrs | **Tangible evidence of value** | Demo |
| **P3** | Minimal campaign runner (`biopl scientist explore`) | 4 hrs | AutoScientist accessibility | Capability |
| **P3** | Joint kernel profiler | 3 hrs | Performance optimization | Capability |
| **P4** | Pre-commit hooks cleanup | 30 min | Developer experience | Infra |
| **P4** | Mermaid architecture diagram in README | 30 min | Visual understanding | Docs |
| **P4** | Machine-readable API schema | 2 hrs | AI agent usability | AI-friendly |
| **P4** | Typed exception hierarchy | 1 hr | Structured error handling | Usability |

**Total**: ~51 hours for usability + demonstrability + rigor + AI-accessibility

---

## ✅ ACCEPTANCE CRITERIA FOR TODO2

```bash
# 1. Newcomer/AI can run this and it works in <2 minutes
uv run scripts/quickstart.py
# → Backprop: 57% | EqProp: 55% | FA: 56%

# 2. All CLI commands in README work
biopl joint-validate --coordinate digital/feedforward/instantaneous/null/backprop/euclidean
biopl run --config configs/presets/eqprop_mnist.yaml
biopl benchmark run --suite adaptation_efficiency

# 3. One-line system creation works
from bioplausible import create_eqprop_mlp, SystemTrainer
system = create_eqprop_mlp(784, (256, 128), 10)
trainer = SystemTrainer(system, train_data=...)

# 4. Property tests pass (CI gate) - INCLUDING gradient equivalence
uv run pytest tests/property/ -q  # 351+ passing + new gradient tests

# 5. Type checking clean (strict mode)
uv run pyright .  # 0 errors

# 6. Pre-commit passes
uv run pre-commit run --all-files

# 7. Visualization works
uv run scripts/ontology_explorer.py  # Opens localhost:8080
biopl lab inspect-state --coordinate digital/recurrent/energy_min/routing/thermo/euclidean

# 8. AutoScientist accessible
biopl scientist explore --space joint_smoke --budget 5

# 9. Benchmark comparison produces publication-ready figures
biopl benchmark compare --suite adaptation_efficiency --plast null routing fast_weights
```

---

## 🔄 RELATION TO TODO.md

| TODO.md Sprint | Status | TODO2 Action |
|----------------|--------|--------------|
| J0–J6 Core | "Complete" | Fix CLI, API gaps, docs sync, **restore rigor** |
| J6 Hardening | In progress | Pre-commit, dead code, types |
| — | — | **Add**: Quick-start, presets, AI-friendly APIs, **visualization, AutoScientist access, rigor** |

**Philosophy**: 
- TODO.md = architectural completeness (done)
- TODO2.md = **usability + demonstrability + scientific rigor completeness**
- Backwards compatibility: NONE (per AGENTS.md)

---

## 🎯 DEFINITION OF DONE (Expanded)

A developer (or AI agent) can:
1. `git clone && cd bioplausible && uv sync --dev`
2. `uv run scripts/quickstart.py` → sees working results in <2 min
3. `uv run biopl run --config configs/presets/eqprop_mnist.yaml` → trains model
4. `from bioplausible import create_eqprop_mlp` → builds system in 1 line
5. Read README → all code examples copy-paste and run
6. Run tests → all property tests pass, type checking clean (strict mode)
7. `uv run scripts/ontology_explorer.py` → explores 6-D space interactively
8. `biopl lab inspect-state ...` → visualizes joint dynamics
9. `biopl scientist explore ...` → runs autonomous campaign
10. `biopl benchmark compare ...` → generates publication figures

**Scientific rigor restored**:
- Gradient equivalence verified in CI (EqProp vs BPTT cosine ≥ 0.5)
- Lyapunov/Control-Lyapunov proofs passing
- Determinism guaranteed for all 6-D coordinates
- Locality axioms enforced (no weight transport, seed-independent feedback)

**Time to first success**: Current ~30 min with errors → **Target: <2 min zero errors**

**Time to first insight**: Current ~hours → **Target: <5 min** (via ontology explorer + quickstart + visualizer)