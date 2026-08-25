# 6-D Joint System Composition Patterns

The 6-D ontology (S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U) enables composing systems with
plasticity mechanisms (M) that interact with standard 5-D components.
This document describes common composition patterns.

## Core API

```python
from computronium.core.system_trainer import compose_joint_system
from computronium.core.ontology import (
    DigitalSubstrate,
    FeedforwardGeometry,
    RecurrentGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    EnergyMinimizationDynamics,
    StateDynamicsConfig,
    BackpropCredit,
    ThermodynamicContrast,
    LocalGoodnessCredit,
    CreditAssignmentConfig,
    EuclideanUpdate,
    ParameterUpdateConfig,
)
from computronium.core.plasticity import (
    RoutingPlasticity,
    FastWeightPlasticity,
    RuleStatePlasticity,
)
```

---

## Pattern 1: Routing + EqProp (Meta-Learning Credit Assignment)

Combines **RoutingPlasticity** (state-dependent pathway gating) with
**EnergyMinimizationDynamics** + **ThermodynamicContrast** (Equilibrium Propagation).

The routing plasticity learns which pathways to activate based on input statistics,
while EqProp provides the contrastive credit assignment for weight updates.

```python
joint = compose_joint_system(
    substrate=DigitalSubstrate(),
    geometry=RecurrentGeometry(
        GeometryConfig.recurrent(
            input_dim=784,
            output_dim=10,
            hidden_dims=(512, 512, 512),
            init_scale=0.1,
        ),
        hidden_dim=512,
    ),
    dynamics=EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=20,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=0.1,
        )
    ),
    plasticity=RoutingPlasticity(
        gate_dim=64,
        temperature=1.0,
        decay=0.99,
        learning_rate=0.01,
    ),
    credit=ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.1)
    ),
    update=EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=0.001,
            momentum=0.9,
        )
    ),
)

# Training
for x, y in train_loader:
    metrics = joint.train_step(x, y)
    print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
```

**Use case**: Learning sparse, input-dependent routing for mixture-of-experts
or conditional computation within an EqProp framework.

---

## Pattern 2: Fast Weights + Backprop (Working Memory + Gradient Descent)

Combines **FastWeightPlasticity** (episode-local associative memory) with
**InstantaneousDynamics** + **BackpropCredit** (standard backpropagation).

The fast weights accumulate Hebbian associations within an episode, providing
a working memory buffer, while backprop handles the long-term weight updates.

```python
joint = compose_joint_system(
    substrate=DigitalSubstrate(),
    geometry=FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(256, 128),
            init_scale=0.1,
        )
    ),
    dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
    plasticity=FastWeightPlasticity(
        fast_weight_dim=512,
        decay=0.9,
        learning_rate=0.1,
        outer_product_scale=1.0,
    ),
    credit=BackpropCredit(CreditAssignmentConfig.gradient()),
    update=EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=0.001,
            momentum=0.9,
        )
    ),
)

# Training
for x, y in train_loader:
    metrics = joint.train_step(x, y)
    print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
```

**Use case**: Few-shot learning, episodic memory, or tasks requiring rapid
adaptation within episodes while maintaining stable long-term weights.

---

## Pattern 3: Rule State Plasticity for Hebbian Meta-Learning (Z3 Benchmark)

Implements the **Z3 benchmark**: frozen weights (θ) with algorithm switching
via ψ (RuleStatePlasticity). The system learns to select operators from a
fixed library, enabling task switching without weight updates.

```python
joint = compose_joint_system(
    substrate=DigitalSubstrate(),
    geometry=FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=64,
            output_dim=2,
            hidden_dims=(128,),
            init_scale=0.1,
        )
    ),
    dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
    plasticity=RuleStatePlasticity(
        num_operators=8,
        operator_dim=64,
        controller_hidden=128,
        temperature=1.0,
        learning_rate=0.01,
        decay=0.99,
    ),
    credit=LocalGoodnessCredit(CreditAssignmentConfig.local_goodness()),
    update=EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=0.01,
            momentum=0.9,
        )
    ),
)

# Meta-training phase: learn operator embeddings and controller
joint.plasticity.unfreeze_theta()
for x, y in meta_train_loader:
    metrics = joint.train_step(x, y)

# Z3 evaluation phase: freeze θ, only adapt ψ
joint.plasticity.freeze_theta()
assert joint.plasticity.verify_theta_frozen()  # Exact parameter invariance
for x, y in z3_eval_loader:
    metrics = joint.train_step(x, y)  # Only ψ updates
```

**Use case**: Z3 benchmark (frozen-θ task switching), algorithm selection,
meta-learning with exact parameter invariance guarantees.

---

## Key Concepts

### Plasticity State (ψ) vs Parameters (θ)

| Component | State Type | Updated During Training | Consolidated |
|-----------|------------|------------------------|--------------|
| `RoutingPlasticity` | `gate_logits`, `active_routes` | Yes (per step) | Yes (episode boundary) |
| `FastWeightPlasticity` | `fast_weights` | Yes (per step) | Yes (episode boundary) |
| `RuleStatePlasticity` | `operator_logits`, `controller_state` | Yes (per step) | No (θ frozen in eval) |
| `NullPlasticity` | `{}` (empty) | N/A | N/A |
| `SubstrateCoupledPlasticity` | `{}` (ψ ≡ σ) | Via substrate | Via substrate |

### Training Loop Integration

The `JointSystem.train_step(x, y)` executes:
1. **Initial ψ**: `plasticity.initial_psi(context, batch_size)`
2. **Free phase**: `dynamics.settle(state, geometry, substrate, target=None)`
3. **Nudged phase**: `dynamics.settle(state, geometry, substrate, target=y)`
4. **Credit assignment**: `credit.compute_pseudo_gradient(free, nudged, loss, geometry)`
5. **Parameter update**: `update.step(geometry.params, pseudo_grads, geometry)`
6. **Plasticity step**: `plasticity.step(psi, joint_state, context)` (if implemented)

### Zero-Extension Theorem (M=Null)

When `plasticity=NullPlasticity()`, the 6-D joint system is behaviorally
equivalent to a 5-D system:
```python
joint = compose_joint_system(
    substrate=DigitalSubstrate(),
    geometry=RecurrentGeometry(...),
    dynamics=EnergyMinimizationDynamics(...),
    plasticity=NullPlasticity(),  # M=Null
    credit=ThermodynamicContrast(...),
    update=EuclideanUpdate(...),
)
# joint.train_step() ≡ 5-D system.train_step()
```

This guarantees backward compatibility: all 5-D compositions are valid
6-D coordinates with `M=Null`.

---

## Creating Custom Plasticity

Implement the `PlasticityPrimitive` protocol:

```python
from computronium.core.joint.transition import PlasticityPrimitive, PlasticityConfig
from computronium.core.joint.context import SystemContext
from computronium.core.joint.state import CompositeState
from torch import Tensor

class MyPlasticity:
    config = PlasticityConfig.routing(gate_dim=32)  # or custom type
    
    def initial_psi(self, context: SystemContext, batch_size: int = 1) -> dict[str, Tensor]:
        return {"my_state": torch.zeros(batch_size, 32)}
    
    def step(self, psi: dict[str, Tensor], z: CompositeState, context: SystemContext) -> dict[str, Tensor]:
        # Update plastic state based on joint state z
        return updated_psi
```

Then compose:
```python
joint = compose_joint_system(
    substrate=DigitalSubstrate(),
    geometry=FeedforwardGeometry(...),
    dynamics=InstantaneousDynamics(...),
    plasticity=MyPlasticity(),
    credit=BackpropCredit(...),
    update=EuclideanUpdate(...),
)
```

---

## References

- `computronium/core/system_trainer.py::compose_joint_system` — Main factory
- `computronium/core/joint/transition.py` — PlasticityPrimitive protocol
- `computronium/core/plasticity/` — Built-in plasticity implementations
- `tests/property/joint/test_composability.py` — Property tests for 6-D composition