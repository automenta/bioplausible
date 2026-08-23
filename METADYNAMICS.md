# Bioplausible MetaDynamics  Final Specification
## Composable Coupled Dynamical Systems & Adaptive Computational Rules

**Status:** Definitive Architectural Specification
**Scope:** Mathematical formalism, API contracts, state lifecycle semantics, stability theory, and experimental campaign for extending Bioplausible into a framework for **Adaptive Computational Dynamics**.

---

## 1. Conceptual Thesis

Bioplausible v1 models learning systems as compositions of fixed transition rules acting on changing states. Bioplausible v2 elevates the framework to model systems where **the transition rule can itself be state**.

The central axiom of v2 is:
> **Computation is the evolution of computational state, where the computational state may include the parameters, routing, and physics that determine its own subsequent evolution.**

This moves the project beyond "biologically plausible learning algorithms" and transforms the AutoScientist’s search space. Instead of searching for *which learning rule works best*, the framework now searches for **which coupled dynamical systems are good computers**. The object being searched is the transition law itself.

---

## 2. The Six-Axis Ontology & State Registry

The generative physico-computational engine is defined over a 6D tensor product:

$$
\boxed{
\text{System} = S \otimes G \otimes D \otimes M \otimes C \otimes U
}
$$

| Axis | Symbol | Role in v2 |
|---|---:|---|
| **Substrate** | $S$ | Physical medium, device physics, and passivity constraints. |
| **Geometry** | $G$ | Topology, connectivity, and structural scaffold. |
| **StateDynamics** | $D$ | Activity evolution, settling, and inference projections. |
| **Plasticity (MetaDynamics)** | $M$ | The mechanism by which the computational rule becomes a dynamical variable. |
| **CreditAssignment** | $C$ | Error, goodness, or contrastive signal construction. |
| **ParameterUpdate** | $U$ | Slow, persistent parameter consolidation. |

### 2.1 The State Registry & Lifecycle Semantics
To resolve ontological overlaps (e.g., a memristor's conductance acting simultaneously as physical state, effective parameter, and plastic state), v2 abandons rigid semantic typing in favor of a **State Registry** with explicit lifecycle flags.

Every state variable $v$ in the system is registered with metadata:
```python
class StateVariable:
    name: str
    persistent: bool  # Survives episode boundaries (traditionally \theta)
    fast_plastic: bool  # Evolves via intra-episode plastic law (traditionally \psi)
    substrate_owned: (
        bool  # Subject to physical device constraints (traditionally \sigma)
    )
    consolidatable: bool  # Can be promoted to persistent state at episode end
```

**Crucial Distinction:** The designation of a variable as "persistent $\theta$" is an **operational lifecycle designation**, not an ontological requirement of the computation. A substrate variable can migrate from fast to persistent state seamlessly if the registry and consolidation policies permit it.

---

## 3. The Joint Dynamical System (Mathematical Core)

In v1, the mathematical center was the isolated state transition $x_{t+1} = T(x_t)$.
In v2, the mathematical center is the **joint transition operator** acting on a composite state vector.

Let the joint intra-episode state be:
$$
z_t = (x_t, \psi_t, \sigma_t)
$$
where $x$ is activity, $\psi$ is the plastic rule state, and $\sigma$ is the substrate state.

The coupled dynamical system is defined as:
$$
\boxed{
z_{t+1} = F_\theta(z_t; G, S)
}
$$

`StateDynamics`, `Plasticity`, and `Substrate` are no longer strictly independent siblings; they are **projections and decompositions of this single coupled transition law** $F_\theta$.

Slow learning operates on the persistent parameter state $\theta$ at episode boundaries:
$$
\boxed{
\theta_{e+1} = U\left(\theta_e, C(\tau_e)\right)
}
$$
where $\tau_e$ is the joint trajectory.

---

## 4. API Contract: The Composite Transition Protocol

To prevent `Plasticity` from degrading into a mere "weight pre-processor," v2 introduces the `CoupledTransition` protocol. This protocol operates on a single composite state object and an immutable context, rather than passing disjoint state tensors as execution arguments.

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class CoupledTransition(Protocol):
    def step(
        self,
        z: CompositeState,  # Contains z.activity, z.plastic, z.substrate
        context: SystemContext,  # Contains immutable theta, geometry, substrate physics
    ) -> CompositeState:
        """
        Executes one step of the joint dynamical system: z_{t+1} = F_\theta(z_t; G, S).
        Returns the updated composite state.
        """
        ...
```

This abstraction is mathematically clean and essential for:
*   Automatic differentiation through the joint dynamics.
*   Jacobian estimation and stability monitoring.
*   Trajectory recording and event-driven dynamics.
*   Continuous-time system integration.

---

## 5. The Plasticity Hierarchy (Elevating Rule-State)

The $M$ axis dictates how the computational rule evolves. The hierarchy is conceptualized as:
$$
\boxed{
\text{Plasticity} \supset \{\text{routing, weights, thresholds, precision, topology, rules}\}
}
$$
where the effective transition operator is parameterized by the plastic state: $T_t = T(\psi_t)$.

### 5.1 `NullPlasticity` (The Zero-Extension Theorem)
$$ \psi_{t+1} = \psi_t $$
**Purpose:** Backward compatibility. Guarantees that $M = \text{Null} \implies \text{v2 system} \equiv \text{v1 system}$.

### 5.2 `RoutingPlasticity` & `FastWeightPlasticity`
**Purpose:** State-dependent gating, sparse pathway selection, and episode-local associative memory. These are specific parameterizations of $T(\psi_t)$.

### 5.3 `SubstrateCoupledPlasticity`
**Purpose:** Physical realization of plasticity (e.g., memristive drift). Here, $\psi_t \equiv \sigma_t$.

### 5.4 `RuleStatePlasticity` (The Z3 Primitive)
**Purpose:** The apex of the hierarchy. $\psi$ is not interpreted as weights, but as a **computational controller** or rule encoder. This primitive changes *the family of maps from which $T$ is constructed*.

$$
T_t = \sum_k g_k(\psi_t)T_k
$$

This acts as a rudimentary **neural instruction selection mechanism**. A more radical version lets $\psi_t = (o_1, o_2, \ldots, o_n)$ select primitives from an operator library. This allows the framework to experimentally ask: *Can a local dynamical system discover and rewrite an internal computational procedure without a conventional program counter?*

---

## 6. Joint Stability & The Stability-Plasticity Frontier

v1 relied on strict Lyapunov descent and global contraction. In v2, we recognize that global contraction is a *sufficient* condition for a unique fixed point, but not a *necessary* condition for useful computation. Systems can exhibit local contraction, multiple attractors, limit cycles, or metastable states.

### 6.1 The Frontier Hypothesis
We formulate the research object as:
$$
\boxed{
\text{adaptive computation} \leftrightarrow \text{controlled departure from contraction}
}
$$

The important hypothesis is that **useful rule reconfiguration may require temporarily sacrificing some of the contraction/stability margin that a fixed computational attractor would maximize.**

### 6.2 Monitoring the Frontier
The framework must measure the spectral radius of the joint Jacobian $\rho(J_F)$, local Lyapunov exponents, basin stability, and settling time. The goal is not to demand global contraction, but to map the **stability-plasticity frontier** where the system is stable enough to persist, but unstable enough to reconfigure.

---

## 7. The Experimental Campaign & Pareto Frontiers

A sufficiently expressive fixed-rule system (like a universal RNN) can theoretically simulate a plastic system by storing $\psi_t$ inside ordinary state $h_t = (x_t, \psi_t)$. Therefore, computability-wise, plasticity does not transcend fixed-rule universality.

The scientific claim of v2 is strictly about **resource scaling, locality, energy efficiency, and learnability** under constrained physical resources. We define a resource vector:
$$
\mathcal{C} = (\text{compute}, \text{memory}, \text{energy}, \text{latency}, \text{plastic-state capacity})
$$
The campaign asks whether adaptive-rule systems occupy a superior Pareto frontier in $\mathcal{C}$.

### The 5-Level Benchmark Hierarchy

#### Experiment 1: Adaptation Efficiency
Does plasticity adapt faster to non-stationary shifts than fixed-rule systems under matched compute budgets?

#### Experiment 2: Compute Efficiency
Does the coupled system accomplish the same task with fewer effective operations (e.g., via dynamic sparsity/routing)?

#### Experiment 3: Structural Robustness
Can the joint system recover functionality after topology or device damage via autonomous rerouting?

#### Experiment 3.5: Algorithm Migration
Start with a task requiring internal strategy $A_0$. Change the environment so the appropriate strategy becomes $A_1$. Measure the time and energy to transition $\text{time}(A_0 \rightarrow A_1)$ *without changing $\theta$*. This measures **internal reconfiguration of the computational process** (algorithmic plasticity).

#### Experiment 4: Fixed Weights, Changing Algorithm (The Z3 Benchmark)
This is the crown jewel of the v2 campaign.
*   **Constraint:** Hold persistent parameters $\theta = \text{constant}$.
*   **Task:** Require the system to solve several algorithmically distinct task classes ($A, B, C$) using *only* transient state $\psi$.
*   **Comparison:** Conventional recurrent system ($h_{t+1} = F_\theta(h_t, x_t)$) vs. Rule-state system ($x_{t+1} = T_{\psi_t, \theta}(x_t), \psi_{t+1} = M(\psi_t, x_t)$).
*   **Metrics:** Adaptation time, energy, effective operator diversity, degree of parameter invariance.

If the rule-state system can rapidly instantiate fundamentally different effective algorithms using almost unchanged persistent parameters, it provides a strong empirical demonstration of the central v2 thesis: **control can be represented as evolving computational state rather than an externally imposed instruction sequence.**

---

## 8. Implementation & Migration Path

1.  **Phase 1 (Core Protocol):** Implement `CompositeState`, `StateRegistry`, `CoupledTransition`, and `NullPlasticity`.
2.  **Phase 2 (System Resolver):** Update `System` to compose $D, M, S$ into a `CoupledTransition`. Default $M = \text{Null}$.
3.  **Phase 3 (Validation):** Implement joint stability monitors ($\rho(J_F)$ estimation) and energy decomposition trackers.
4.  **Phase 4 (Primitives):** Implement `Routing`, `FastWeight`, `STP`, `Precision`, and `SubstrateCoupled`.
5.  **Phase 5 (The Z3 Primitive):** Implement `RuleStatePlasticity`.
6.  **Phase 6 (Campaign):** Execute the 5-level benchmark hierarchy across the AutoScientist search space, mapping the Pareto frontiers of $\mathcal{C}$.

---

## 9. Summary

Bioplausible v2 is a paradigm shift from searching over fixed learning algorithms to searching over **composable coupled dynamical systems**.

By defining the joint transition operator $z_{t+1} = F_\theta(z_t; G, S)$, introducing the State Registry to handle complex physical lifecycles, and elevating `RuleStatePlasticity` to the apex of the primitive hierarchy, the framework gains the capacity to scientifically investigate the **stability-plasticity frontier**.

The ultimate research question is no longer "How do we approximate backprop locally?" but rather: **"Which coupled dynamical systems, operating under strict physical and thermodynamic constraints, yield the most efficient, robust, and adaptable computational substrates?"**

