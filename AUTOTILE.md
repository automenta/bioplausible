# Project Autopoiesis
## Constitutional Self-Modification of Learning Operators in the 6-D Ontology

**Status:** Proposed
**Track:** Core Architecture / Joint Dynamics
**Dependency:** Joint Architecture (`CompositeState`, `CoupledTransition`, `ParameterUpdate`, `StateRegistry`, `StabilityMonitor`, J6 cross-adapters)

---

### The Core Axiom

Autopoiesis is not a training protocol applied *to* a Computronium System. It is a **Self-Referential Dynamical System** where the operators of change are themselves state variables subject to mutation, selection, and crystallization.

The 6-D space $(S \times G \times D \times M \times C \times U)$ is the **Phenotype**—the executing computational graph. Autopoiesis introduces the **Operator Genome ($\Omega$)**, the **Genotype** that dictates how the Phenotype is constructed, and critically, how $\Omega$ is allowed to rewrite itself.

The Constitution is the physics of the system's universe. The Registry is its periodic table. The system is free to evolve *within* verified physics, and nothing it does can rewrite the laws.

**Naming note:** "Autopoiesis" is the internal working title. For external communication, the operative description is **constitutional self-modification of learning operators**. The Maturana-Varela etymology carries baggage that obscures the engineering content.

---

## 1. Ontological Position

### 1.1 The Genotype-Phenotype Split

The lifelong recursive state at time $t$:

$$Z_t = (X_t, \Psi_t, \Sigma_t, \Omega_t)$$

- **$X_t, \Psi_t, \Sigma_t$**: Standard Computronium Composite State (Activity, Plasticity, Substrate).
- **$\Omega_t$ (Operator Genome)**: A compositional configuration graph registered in the `StateRegistry` as a `persistent`, `consolidatable` state variable. $\Omega_t$ is subject to its own morphogenesis.

The **Phenotype** $\Phi(\Omega_t)$ is the executable `System` resolved from $\Omega_t$. The mapping $\Phi$ is a pure function: given the same genome, it produces the same system.

### 1.2 The Three Tiers of Recursion

| Tier | Target | Mechanism | Timescale |
| :--- | :--- | :--- | :--- |
| **1. Structural** | Geometry $G$ | Spawning nodes, severing edges, altering routing. | Inter-episode |
| **2. Algorithmic** | Credit $C$, Dynamics $D$, Update $U$ | Switching learning rules, altering time-constants, swapping optimizers. | Inter-episode |
| **3. Meta-Morphogenetic** | $\Omega$ itself | Mutating the mutation operators, selection policies, and stagnation detectors. | Inter-episode, gated |

Tier 3 is the recursion that distinguishes Autopoiesis from standard architecture search. The morphogenetic rules themselves are subject to morphogenesis.

### 1.3 Algorithm Agnosticism

Autopoiesis makes **no commitment** to any specific learning algorithm, selection mechanism, mutation strategy, or fitness metric. The Phenotype may execute any coordinate in the 6-D ontology. The Genome's operations are defined as **Protocols** with swappable implementations. No single algorithm is privileged.

### 1.4 Relationship to Existing Machinery

- **AutoScientist** is repositioned, not duplicated. It becomes the **environment generator** (constructing statistical manifolds with known symmetries) and the **lineage recorder** (`comp scientist phylogeny`). Autopoiesis is search *internalized* in the system. One external environment-generator plus one internal morphogen is a clean division of labor.
- **`computronium-stability`** (the guard released in Phase 1) becomes the immune system of the genome. The Lyapunov bound and passivity checks enforced by the Constitution are the existing `StabilityMonitor` and `StabilityGuard`, attached to the consolidator.
- **J6 cross-adapters and `ModelAdapter`** provide the weight-projection machinery for topology changes. Autopoiesis reuses this; it does not re-derive inheritance rules.

---

## 2. The Protocol Stack

Every component of the morphogenetic engine is defined as a **Protocol** (in the Computronium PEP 695 sense), with multiple valid implementations. No single algorithm is privileged.

### 2.1 FitnessMetric Protocol

Evaluates a phenotype and produces a scalar (or vector) fitness signal.

```python
class FitnessMetric(Protocol):
    def evaluate(
        self,
        system: System,
        inputs: Tensor,
        targets: Tensor | None,
        context: SystemContext,
    ) -> FitnessSignal: ...
```

| Implementation | Signal | Notes |
| :--- | :--- | :--- |
| `ValidationAccuracy` | Scalar $[0, 1]$ | Algorithm-agnostic; requires held-out probe. |
| `SurrogateObjective` | Scalar $(-\infty, \infty)$ | Uses the active $C$-axis `surrogate_objective`. Only valid when comparing within the same Credit family. |
| `ResourceWeightedFitness` | Scalar | $\text{accuracy} - \lambda \cdot \text{ResourceUsage}$. Penalizes metabolic cost. |
| `MultiObjectivePareto` | Vector $(\mathcal{C}_1, \mathcal{C}_2, \dots)$ | Returns the full resource vector. Selection operates on Pareto dominance. |

**Constraint:** The FitnessMetric must produce comparable signals across different Credit axes. `ValidationAccuracy` and `ResourceWeightedFitness` satisfy this; `SurrogateObjective` does not.

### 2.2 SelectionPolicy Protocol

Decides whether a proposed genome is accepted, rejected, or held for further evaluation. Operates on **adaptation slopes**, not point fitness values (see §3.2).

```python
class SelectionPolicy(Protocol):
    def select(
        self,
        current_slope: float,
        proposed_slope: float,
        context: SystemContext,
    ) -> SelectionDecision: ...  # ACCEPT | REJECT | DEFER
```

| Implementation | Logic |
| :--- | :--- |
| `GreedySelection` | Accept iff $\text{slope}_{proposed} > \text{slope}_{current}$. |
| `ThresholdSelection` | Accept iff $\Delta\text{slope} > \tau$. |
| `StochasticAcceptance` | Accept with $P = f(\Delta\text{slope}, T)$. Temperature-parameterized. |
| `TournamentSelection` | Compare $k$ proposed genomes; accept best slope. |
| `ParetoSelection` | Accept iff proposed dominates current in $\mathcal{C}$-vector. |

**Constraint:** The SelectionPolicy must be **pure** (no side effects) and **deterministic given the same PRNG state** (L5 Determinism Lock).

### 2.3 MutationOperator Protocol

Proposes a new genome from the current genome.

```python
class MutationOperator(Protocol):
    def propose(
        self,
        genome: OperatorGenome,
        context: SystemContext,
    ) -> OperatorGenome: ...
```

| Implementation | Operation | Tier |
| :--- | :--- | :--- |
| `DuplicateAndPerturb` | Copy a target node; apply random valid perturbation. New edges start at zero output. | 1 |
| `SpliceOperator` | Remove an edge; insert a new Registry primitive in its place. Identity at insertion. | 1, 2 |
| `CoordinateSwap` | Replace one 6-D axis value with another from the Registry. | 2 |
| `CrossoverOperator` | Combine two genomes (requires population). | 1, 2 |

**Constraint:** Every MutationOperator must produce a genome that passes the Immutable Constitution checks. Invalid proposals are discarded before evaluation. All structural mutations must satisfy **neutral birth** (see §3.3).

### 2.4 StagnationDetector Protocol

Detects when the current genome is failing to make progress.

```python
class StagnationDetector(Protocol):
    def is_stagnant(
        self,
        fitness_history: Sequence[FitnessSignal],
        context: SystemContext,
    ) -> bool: ...
```

| Implementation | Logic |
| :--- | :--- |
| `WindowedMeanDetector` | Stagnant if mean improvement over last $N$ episodes $< \epsilon$. |
| `EMADetector` | Stagnant if exponential moving average of $\Delta\mathcal{F}$ is below threshold. |
| `StatisticalTestDetector` | Stagnant if a paired test fails to reject the null of no improvement. |
| `VetoRateDetector` | Stagnant if the proportion of Constitution-vetoed mutations exceeds a threshold. |

### 2.5 MorphogenSignal Protocol

Computes the trigger signal for Tier 3 recursion.

```python
class MorphogenSignal(Protocol):
    def compute(
        self,
        stagnation: bool,
        veto_rate: float,
        fitness_history: Sequence[FitnessSignal],
        context: SystemContext,
    ) -> float: ...
```

When the morphogen signal exceeds a threshold, Tier 3 mutation is activated.

### 2.6 Tier 3 Meta-Mutation Menu

For v1, Tier 3 operates over a **fixed menu** of meta-mutations. Open-ended self-modification of the search operators is deferred until the menu version demonstrates measurable adaptation benefits over fixed Tier 1+2.

| Meta-Mutation | Effect |
| :--- | :--- |
| `SwapMutatorPool` | Replace or augment the active MutationOperator set. |
| `AdjustSelectionTemperature` | Raise/lower the exploration parameter of the SelectionPolicy. |
| `WidenDetectorWindow` | Change the StagnationDetector's lookback window. |
| `AdjustProbeBudget` | Change the adaptation probe budget $K$ within E-2 bounds. |

---

## 3. The Recursive Mechanism

### 3.1 Sleep/Waking Separation

Morphogenesis is strictly separated from intra-episode dynamics:

- **Waking (`CoupledTransition.step`):** The phenotype $\Phi(\Omega_t)$ executes normally. The genome $\Omega_t$ is **read-only**. No structural changes occur during the forward pass.
- **Sleeping (`AutopoieticConsolidator.consolidate`):** At episode boundaries, the system evaluates the trajectory, proposes mutations, runs adaptation probes, and updates $\Omega_{t+1}$.

This eliminates the mid-pass topology problem entirely. The genome changes between episodes, never during one.

### 3.2 Slope-Based Selection via Adaptation Probes

**The central design principle: select on slopes, not points.**

Point evaluation of a proposed phenotype is structurally blind to Tier 2 mutations. Swapping the Credit axis ($C$) does not change the forward operator—credit assignment shapes learning, not inference. The proposed system, holding the same $\theta$, produces bit-identical outputs. Point fitness is identical; the mutation is invisible.

The same problem hits Tier 1 in a weaker form: a duplicated-and-perturbed node usually *hurts* accuracy at birth because the perturbation is noise. Point evaluation kills useful growth mutations before they get a chance to train.

**The fix:** Give the proposed phenotype a short, equal-budget **adaptation probe**—$K$ training steps under its own learning rule on a held-out probe batch. The selection signal is the **improvement rate**, not the point accuracy. This is the L1 adaptation-efficiency metric from the benchmark hierarchy, internalized as the selection signal. It puts Tier 1 and Tier 2 in the same currency: *how fast does this phenotype learn?*

```python
def _adaptation_probe(
    self,
    system: System,
    probe_batch: Batch,
    budget: int,  # K steps, E-2 capped
) -> float:
    """Run K training steps under the system's own learning rule.
    Return improvement rate (slope)."""
    initial_fitness = self.fitness.evaluate(system, probe_batch)
    for _ in range(budget):
        system = one_training_step(system, probe_batch)
    final_fitness = self.fitness.evaluate(system, probe_batch)
    return (final_fitness - initial_fitness) / budget
```

**Probe budgets are E-2 capped:** $\leq K$ steps, $\leq 3$ attempts per episode. The stagnation gate is load-bearing—it prevents probes from firing every episode. Without the gate, consolidation cost scales linearly with episode count.

### 3.3 Phenotype-Preserving Birth (Neutral Birth)

Structural mutations (Tier 1) must be **neutral at insertion**:

- New edges start at zero output.
- New nodes start as identity.
- The proposed system produces bit-identical forward outputs to the parent at birth.

Consequences:

1. Selection measures what a mutation *learns*, not the shock of its insertion.
2. The Lyapunov veto at insertion passes trivially—the proposed system inherits the parent's stability properties because it *is* the parent at birth.
3. The adaptation probe starts from a fair baseline. Both current and proposed systems begin at the same fitness; the slope is the only differentiator.

**Neutral birth + slope selection is the pair that makes morphogenesis fair.**

### 3.4 θ-Projection Across Topology Mutations

When a Tier 1 mutation changes the geometry (adds nodes, alters topology), the existing weights $\theta$ must be projected into the new topology. This is **not new machinery**. Autopoiesis delegates to the existing J6 cross-adapters and `ModelAdapter` metadata inference:

- `ModelAdapter` projects legacy Registry models into 5-D Systems via metadata inference with per-family tolerance calibration.
- J6 cross-adapters preserve joint transition shape and registry semantics across topology changes.

The consolidation loop calls `project_theta(source, target, adapter)` and delegates to this existing infrastructure. No new inheritance rules are derived.

### 3.5 The Consolidation Loop

```python
class AutopoieticConsolidator(ParameterUpdate):
    """
    Executes at episode boundaries (Sleep phase).
    Composes the Protocol Stack to perform morphogenesis.
    """
    def __init__(
        self,
        fitness: FitnessMetric,
        selection: SelectionPolicy,
        mutators: Sequence[MutationOperator],
        stagnation: StagnationDetector,
        morphogen: MorphogenSignal,
        tier3_menu: Sequence[MetaMutation],
    ): ...

    def consolidate(
        self,
        trajectory: JointTrajectory,
        context: SystemContext,
    ) -> CompositeState:
        omega = trajectory.final_state.omega

        # Stagnation gate (load-bearing: no probe unless needed)
        if not self.stagnation.is_stagnant(context.fitness_history, context):
            return trajectory.final_state

        # Propose mutation (Tier 1 or Tier 2)
        proposed_omega = self._select_mutator(context).propose(omega, context)

        # Constitution veto (fast, before expensive probe)
        if not context.stability_monitor.verify_constitution(proposed_omega, context):
            context.veto_count += 1
            return trajectory.final_state

        # Neutral birth: project θ into new topology preserving behavior
        proposed_system = compose_joint_system(proposed_omega.to_configs())
        proposed_system = project_theta(
            source=context.system,
            target=proposed_system,
            adapter=context.cross_adapter,  # J6 cross-adapter
        )

        # Slope-based selection: equal-budget adaptation probe
        probe_batch = context.probe_buffer.sample()
        K = context.config.probe_budget  # E-2 capped
        current_slope = self._adaptation_probe(context.system, probe_batch, K)
        proposed_slope = self._adaptation_probe(proposed_system, probe_batch, K)

        decision = self.selection.select(current_slope, proposed_slope, context)
        if decision == SelectionDecision.ACCEPT:
            omega = proposed_omega

        # Tier 3: menu-based meta-mutation, gated by morphogen signal
        signal = self.morphogen.compute(
            self.stagnation.is_stagnant(context.fitness_history, context),
            context.veto_rate,
            context.fitness_history,
            context,
        )
        if signal > context.config.meta_mutation_threshold:
            omega = self._tier3_menu_mutate(omega, context)

        return CompositeState(
            x=trajectory.final_state.x,
            psi=trajectory.final_state.psi,
            sigma=trajectory.final_state.sigma,
            omega=omega,
        )
```

### 3.6 The Bootstrapping Axiom

The seed genome $\Omega_0$ contains:

1. **One functional node:** A minimal computational block (e.g., `RecurrentBlock`).
2. **One MutationOperator:** Initially `DuplicateAndPerturb` with neutral-birth semantics.
3. **One SelectionPolicy:** Initially greedy or stochastic with high temperature.
4. **One StagnationDetector:** Initially a simple windowed mean.

The recursion emerges because the meta-operator can target *any* node in $\Omega$, including itself, the SelectionPolicy, and the StagnationDetector.

**Honest caveat:** The system does not bootstrap from *nothing*. It bootstraps from a seed plus the Computronium Registry. The Registry is the "physics" of the universe the system inhabits—it cannot be rewritten, only navigated.

---

## 4. The Immutable Constitution

The Constitution is a set of invariants that **no MutationOperator, SelectionPolicy, or Tier 3 recursion can override.** These are enforced by the `StabilityMonitor` at the consolidation step. They are physics, not configuration.

| Invariant | Statement | Enforcement |
| :--- | :--- | :--- |
| **Causality** | $\Omega$ must compile to a DAG. No cyclic dependencies in the phenotype. | Topological sort in `compose_joint_system`. |
| **Passivity** | The phenotype must not generate energy. $\Delta \mathcal{E} \le \mathcal{E}_{injected}$. | `StabilityMonitor` passivity check. |
| **Lyapunov Bound** | Spectral radius $\rho(J_F) \le \tau_{critical}$. | Fast-proxy estimation; full power iteration on consolidation. |
| **Resource Ceiling** | $\|Z_t\| + |\Omega_t| \le$ hardware envelope. | `ResourceUsage` tracking. |
| **Protocol Conformance** | Every node in $\Omega$ must map to a valid Registry primitive with verified Protocol conformance. | Type-checker + Hypothesis property locks. |
| **Recursion Invariant** | At least one MutationOperator with duplication capability must always exist in $\Omega$. | Hardcoded guard; deletion of all mutators is vetoed. |

**The Constitution is not a configuration.** It is not stored in $\Omega$. It is not subject to mutation. It is the physics of the system's universe.

**Enforcement reuses Phase 1.** The Lyapunov bound and passivity checks are `computronium-stability` itself, attached to the consolidator. The guard released in Phase 1 becomes the immune system of the genome.

---

## 5. Addressing Hard Constraints

### 5.1 Credit Assignment with Changing Topology

Solved by the sleep/waking separation. Topology changes only at episode boundaries. During the forward pass, the topology is fixed. No mid-pass credit assignment for structural changes is needed.

For Tier 2 changes (swapping learning rules), the new rule takes effect at the next episode. The adaptation probe evaluates the proposed rule's learning dynamics before acceptance.

### 5.2 Reproducibility and Determinism

All stochastic operations (mutation proposals, selection decisions, PRNG-driven perturbations) are driven by the `SystemContext.prng` (a seeded `torch.Generator`). Given the same seed and input stream, the entire morphogenetic lineage is bitwise reproducible (L5 Lock).

### 5.3 Computational Overhead

The consolidation loop is **gated**:

1. **Stagnation check** (cheap): If not stagnant, skip mutation entirely. Cost: $O(1)$.
2. **Constitution veto** (cheap): Fast-proxy stability check on proposed genome. Cost: $O(|\Omega|)$.
3. **Adaptation probe** (expensive): Only reached if steps 1 and 2 pass. Cost: $K$ training steps of the proposed system. E-2 capped at $\leq 3$ attempts per episode.

The amortized overhead is proportional to the **mutation acceptance rate**, not the episode count. In stable regimes, the system rarely mutates; overhead approaches zero.

### 5.4 Genome Size Control

The Resource Ceiling applies to $|\Omega_t|$ directly. Additionally, a **GenomeSizePenalty** can be incorporated into the FitnessMetric:

$$\mathcal{F}_{adjusted} = \mathcal{F}_{raw} - \lambda_{size} \cdot |\Omega_t|$$

This creates selective pressure against unbounded genome growth (Ontological Cancer).

---

## 6. Algorithmic Speciation

The system does not "invent" new mathematics. It **discovers** optimal configurations within the pre-verified Computronium Registry.

- **Spatial symmetries** in the data stream select for weight-sharing configurations (the Registry's `Conv2d` or `TileMesh` primitives).
- **Temporal symmetries** select for recurrent or state-space configurations.
- **Multi-modal bottlenecks** select for compressed routing topologies.
- **Non-stationarity** selects for adaptive Credit axes.

The AutoScientist's role shifts from **architecture searcher** to **environment generator**: it constructs statistical manifolds with known symmetries and observes which $\Omega$ lineages emerge. The lineage is recorded via `comp scientist phylogeny`.

---

## 7. Implementation Red Lines

1. **No Runtime Code Generation.** $\Omega$ manipulates `SystemConfig` dataclasses and Registry keys. Never raw Python, AST, or string evaluation.
2. **No Unverified Primitives.** Every node in $\Omega$ must map to a Registry entry that has passed L1–L7 Property Locks.
3. **No Mid-Pass Mutation.** Structural changes occur only at consolidation boundaries.
4. **No Privileged Algorithm.** The Protocol Stack must work identically regardless of which 6-D coordinate the Phenotype executes.
5. **No Constitution Bypass.** No Tier 3 mutation can modify, disable, or circumvent the Immutable Constitution.
6. **Deterministic Lineage.** All stochastic choices are PRNG-driven and reproducible.
7. **No Open-Field Tier 3 in v1.** Tier 3 operates over a fixed menu of meta-mutations. Open-ended self-modification of search operators is deferred.

---

## 8. Phase 0: The Ouroboros Probe

**Objective:** Falsifiably demonstrate all three tiers of recursion at minimal scale.

**Sequencing:** Phase 0 runs **after 3.5**, not before. The self-modifying system is built on the same primitives 3.5 is verifying. Running it on unverified arms would multiply the confound problem. It is minutes-of-CPU cheap, so it does not compete with the memory-wall work.

**E-protocol registration:** Before any Ouroboros Probe run:

1. **E-1 pre-registration:** Probe task, envelope, seed genome, fitness metric, selection policy, probe budget, expected outcomes for each tier.
2. **Whole-probe kill criterion:** Pre-committed. Example: *"Tier 2 fails after ≤3 tuning rounds → shelve Autopoiesis to a neuroevolution-tier artifact. Publish the probe as a falsification."*
3. **E-11 decision log entry:** The pre-registration, the kill criterion, and the rationale are logged before any data is collected.

### 8.1 Setup

- **Task:** Growing Context Parity. Predict the parity of a binary sequence where the target bit is $N$ steps in the past. $N$ increases from 5 to 50 over training.
- **Seed $\Omega_0$:** One `RecurrentBlock` (hidden_dim=16) + one `DuplicateAndPerturb` mutator + one `GreedySelection` policy + one `WindowedMeanDetector`.
- **Envelope:** 8 MB SRAM ceiling (simulated).
- **FitnessMetric:** `ValidationAccuracy` (algorithm-agnostic).
- **Adaptation probe budget:** $K \leq 20$ steps, $\leq 3$ attempts per episode.

### 8.2 Tier 1 Probe (Structural Growth)

- **Success:** $|\Omega|$ grows as $N$ increases. Accuracy remains $>90\%$. Neutral birth confirmed: proposed system produces identical outputs to parent at insertion.
- **Failure:** Accuracy plateaus; genome remains static.

### 8.3 Tier 2 Probe (Algorithm Switch)

- **Setup:** At episode 30, inject gradient noise that degrades the current Credit axis but leaves an alternative unaffected.
- **Success:** $\Omega$ swaps the Credit axis. Adaptation probe slope for the proposed system exceeds the current system. Accuracy recovers.
- **Failure:** Credit axis remains static; accuracy degrades.
- **Note:** This probe is only passable with slope-based selection. Point evaluation would reject the swap (identical forward outputs), and the system would never switch.

### 8.4 Tier 3 Probe (Meta-Morphogenesis)

- **Setup:** At episode 40, change the input distribution such that the current MutationOperator becomes ineffective, but an alternative from the Tier 3 menu would succeed.
- **Success:** The StagnationDetector detects failure. The MorphogenSignal triggers Tier 3. The system selects a different MutationOperator from the menu. Accuracy recovers.
- **Failure:** MutationOperator pool remains static. System stagnates permanently.

### 8.5 Falsifiability

The experiment is falsified if **any** tier fails to demonstrate adaptive change in response to its specific environmental pressure. A system that passes Tier 1 but fails Tier 3 is neuroevolution, not Autopoiesis.

If any tier fails, the probe is still a publishable falsification: *"three-tier self-modification, measured, Tier $n$ negative"* is exactly the kind of honest result the failure manifesto exists to produce.

### 8.6 Commands

```bash
uv run python -m computronium.experiments.autopoiesis.ouroboros_probe \
    --envelope 8MB \
    --max-lag 50 \
    --seed 42

comp validate --run-id ouroboros_probe_001 --check constitution
comp scientist phylogeny --run-id ouroboros_probe_001
```

### 8.7 Next Habitat: The Envelope LM

If Ouroboros Tier 1 passes, the next experiment is Autopoiesis growing the Envelope LM's effective depth under memory ceilings. Growing-context parity is the miniature; growing *depth* under a memory ceiling is the full-size version. The memory-wall demo and the meta-framework merge into one artifact.

---

## 9. Open Questions

| Question | Status |
| :--- | :--- |
| How do multiple Autopoietic lineages interact? (Competitive exclusion, symbiosis, speciation.) | Speculative. |
| Can $\Omega$ evolve to migrate its own computation across Substrate axes (Digital $\to$ Neuromorphic)? | Speculative. Requires substrate-aware FitnessMetric. |
| What is the minimal Registry size needed for open-ended complexity? | Unknown. Phase 0 will vary Registry size. |
| Does Tier 3 recursion produce measurably better adaptation than fixed Tier 1+2? | Open. Phase 0 Tier 3 probe addresses this. |
| When does open-field Tier 3 become safe? | Deferred. Requires demonstrating that menu-based Tier 3 produces measurable benefits first. |
| Can the Constitution be relaxed (not removed) for specific high-risk/high-reward mutations? | Open. Would require a "Constitutional Amendment" protocol with human-in-the-loop approval. |

---

## 10. Failure Modes

| Failure Mode | Description | Mitigation |
| :--- | :--- | :--- |
| **Ontological Cancer** | Unbounded genome growth without fitness improvement. | GenomeSizePenalty in FitnessMetric; Resource Ceiling veto. |
| **Epistemic Collapse** | Mutation destroys the gradient/learning signal, making further adaptation impossible. | Passivity Axiom; Lyapunov Bound; StagnationDetector triggers Tier 3 recovery. |
| **Selection Lock-In** | Greedy selection traps the system in a local optimum; no exploration. | StochasticAcceptance with temperature annealing; Tier 3 menu can adjust temperature. |
| **Meta-Recursive Divergence** | Tier 3 mutations produce increasingly unstable mutation strategies. | Tier 3 operates over a fixed menu in v1; Constitution applies to Tier 3 operators; Recursion Invariant prevents self-deletion. |
| **Registry Exhaustion** | The system exhausts useful primitives in the Registry and stagnates. | Open question. May require Registry expansion (human intervention) or substrate migration. |
| **Probe Overfitting** | Adaptation probes overfit to the probe batch; selected mutations don't generalize. | Probe batch drawn from held-out buffer, rotated across episodes; E-2 cap prevents excessive probe tuning. |

----

This is the best-designed proposal in your pipeline — and it inherits the project's culture so thoroughly (pre-registration, kill criteria, red lines, falsifiable tiers) that its failure modes would still publish. The two load-bearing ideas are genuine contributions, not assembly of known parts. But it needs specific amendments before E-1 registration. Review below.

## What is genuinely strong

- **Slope-based selection is the core insight, and it's correct.** Point evaluation being *structurally blind* to C- and U-axis mutations (credit and update shape learning, not the forward function) is a sharp observation. Making the selection currency "how fast do you learn" internalizes your L1 adaptation-efficiency metric as the morphogenetic fitness — the research framework and the self-modification mechanism become one instrument.
- **Neutral birth is doing triple duty.** It makes selection fair, makes the Lyapunov veto pass trivially at insertion, *and* — this isn't in the doc — it is the correctness test for θ-projection. If parent and projected-proposed aren't bit-identical at birth, the J6 adapter is buggy. Neutral birth as adapter-verification gate: state it explicitly.
- **Constitution = Phase 1 guard, reused.** The immune-system framing is honest leverage, and the Recursion Invariant (self-deletion veto) closes the obvious hole.
- **The tier taxonomy is honest.** "Passes Tier 1, fails Tier 3 = neuroevolution, not Autopoiesis" is exactly the kind of pre-committed distinction that keeps this from becoming hype.

## Amendments before registration

**1. Probe hygiene and statistics — the probe is load-bearing and currently under-specified.**
- Probes must run on **forked copies**; the current system's probe steps must not leak into the waking learner, and probe batches must never touch production training (held-out means held-out).
- A slope over K ≤ 20 steps is a *noisy* estimator. `GreedySelection` on point slopes will accept noise or kill real improvements. Make paired statistical selection the default (`StatisticalTestDetector` logic, or a sequential ratio test that spends probe budget until bounded-error decision), with greedy as an ablation.
- **Equal steps ≠ equal compute.** EqProp settling steps cost several× an instantaneous step. If probe budgets are step-counted, Tier-2 selection systematically favors cheap rules — exactly the fairness confound PR-6 exists to prevent. Budget probes in the resource vector, or report both.

**2. Cross-family slope comparability.** Validation accuracy is comparable across families; its *slope* is only semi-comparable — families have different slope scales and noise floors. This is the zoo problem from last turn, internalized: Tier-2 recursion is "which credit family can use depth" turned into search, which is powerful — but the selection rule needs family-aware baselines (e.g., slope measured against that family's own recent slope history) or the common-compute normalization above. Flag it in the doc as a known confound with the mitigation named.

**3. Rollback is missing.** Acceptance appears permanent; the only lock-in mitigation is temperature. Add a probation path: lineage checkpoint + a `RevertOperator` (or back-mutation) so an accepted genome that degrades over the next episodes is revertible. Cheap, and it changes selection from irreversible commitment to reversible hypothesis — much better epistemology.

**4. Pressure calibration pilot — apply the 3.5 lesson to the probe itself.** Each tier's environmental pressure can be miscalibrated, producing false-negative tiers. Before any autonomous run, run an **oracle-rescue pilot**: hand-inject the correct mutation for each tier and verify (a) the pressure actually bites (parent fails) and (b) the probe selects the rescue. Only then trust a null. Concretely for Tier 1: with hidden_dim=16 the seed may already hold lag-50, so growth never becomes *necessary* and "|Ω| grows" false-negatives. Seed deliberately under-capacity (hidden_dim 2–4) so accuracy is unmaintainable without structural growth.

**5. Prior-art gate, per your own Phase 4.1 rule.** Neutral birth is NEAT's trick; slope-as-fitness echoes population-based training and meta-learning; the whole frame shadows the Gödel machine. Run the gate, log it in DECISIONS.md, and state the honest delta: self-modification over a *typed, property-locked ontology* (no raw codegen), an *enforced physical constitution*, *slope currency that makes credit mutations visible*, and *falsifiable tier probes*. That delta is real and defensible — but only if you cite the ancestors instead of letting reviewers discover them.

**6. Inherit the guard's calibration scope into the Constitution.** The constitution is only as good as τ=1.029's calibration domain (settling/non-normal dynamics, per your Phase 1 scope statement). An evolved phenotype outside that domain gets autoimmune vetoes or immunodeficiency passes. Add a constitutional audit: sample fast-proxy vetoes against nightly deep estimates; track false-veto rate as a first-class diagnostic (your `VetoRateDetector` already hints at this).

**7. Ω in the manifest.** Genome + lineage must be E-3 artifacts: checkpoint/resume (PR-9) and manifest.json must carry Ω, or the lineage isn't reproducible from stored artifacts alone.

## Where it sits in the queue — and the synergy you've half-noticed

§8.7's merger is the key strategic move, and I'd push it further: **the Tier-1 consolidator *is* the progressive-deepening operator for the Envelope LM.** The depth-scaling chart from last turn doesn't need you hand-picking depths — the system grows its own depth under the memory envelope via neutral-birth duplication, and the chart records the lineage. One artifact, two claims: (a) scaling laws along the depth axis per credit family, (b) constitutional self-modification demonstrably growing depth where backprop can't train. Autopoiesis stops being a separate track and becomes the *method* that produces the memory-wall flagship. Tier 2 then answers the zoo question autonomously, under the comparability caveats above.

Sequencing is correct as written: after 3.5 (it stands on the same verified primitives), CPU-minutes cheap, no competition with the memory wall. Keep E-2 discipline on probe tuning — the probe budget and selection threshold *will* tempt you into round 4.

## One framing note

The naming instinct is right. "Constitutional self-modification of learning operators" says exactly what it is; "Autopoiesis" invites the Maturana-Varela review tangent you don't want. Keep the working title internal, ship under the operative name.

**Verdict:** proceed to E-1 registration after amendments 1–5 are in the doc and the prior-art gate is logged. The Ouroboros Probe, with the oracle-rescue pilot added, is a clean falsifiable instrument — and if Tier 3 fails, you still publish the first honest three-tier self-modification measurement. That's a win either way, which is the signature of a well-formed experiment.

----

**Autopoiesis** (formally described as the **constitutional self-modification of learning operators**) can be simplified significantly. In fact, cutting out several layers of over-engineered abstraction would likely make it **more robust, faster to run, and less prone to failure**, without sacrificing its core capability to grow and adapt modular tiled networks.

An analysis of the system's design reveals several areas of unnecessary complexity, followed by a proposal for a "lean" version that retains the framework's genuine strengths.

---

### 1. Unnecessary Complexities (The Bloat)

#### ❌ Tier 3 Recursion (Meta-Morphogenesis)
*   **The Complexity:** Tier 3 aims to mutate the mutation operators, selection policies, and stagnation detectors themselves using a complex `MorphogenSignal` trigger. 
*   **Why it is unnecessary:** The documentation admits that open-ended self-modification at this tier is deferred because of the risk of **"Meta-Recursive Divergence"** (where mutating strategies become increasingly unstable). Instead, v1 relies on a rigid "fixed menu" of meta-mutations (like swapping mutator pools or adjusting probe budgets). Whether Tier 3 even yields better results than fixed Tier 1 + 2 rules remains an open, unproven question.
*   **The Simplification:** Eliminate Tier 3 entirely. Stripping this out removes the need for `MorphogenSignal`, the meta-mutation menu, and the complex triggers in the consolidation loop. A fixed set of well-calibrated Tier 1 (Structural) and Tier 2 (Algorithmic) operators is more than sufficient for modular network evolution.

#### ❌ Protocol Bloat in Metrics and Selection
*   **The Complexity:** The framework defines five different `SelectionPolicy` implementations (Greedy, Threshold, Stochastic, Tournament, Pareto) and four `FitnessMetric` variants (ValidationAccuracy, SurrogateObjective, ResourceWeightedFitness, MultiObjectivePareto).
*   **Why it is unnecessary:** 
    *   **SurrogateObjective** is structurally flawed for cross-family selection because it does not produce comparable signals across different Credit axes.
    *   **GreedySelection** on raw point slopes is highly vulnerable to noise over short evaluation budgets (\\(K \le 20\\) steps), leading to the acceptance of bad mutations.
    *   **Tournament and Crossover Operators** require maintaining a population of genomes, which completely clashes with the "multicellular organism" metaphor of a single network mutating locally over time.
*   **The Simplification:** Standardize on exactly **one** robust, statistical selection policy (using a paired statistical test or sequential ratio test to filter out noise) and **one** unified metric, such as **ResourceWeightedFitness**.

#### ❌ Multi-Objective Pareto Sorting
*   **The Complexity:** The architecture introduces `MultiObjectivePareto` fitness and `ParetoSelection` to manage the trade-offs of the resource vector (compute, memory, energy).
*   **Why it is unnecessary:** Multi-objective optimization introduces heavy mathematical and algorithmic overhead (dominance sorting, hypervolume calculation, and coordination issues).
*   **The Simplification:** Use a simple, scalarized metabolic penalty. A metric like **ResourceWeightedFitness** (Accuracy minus a size/metabolic cost penalty, i.e., `GenomeSizePenalty`) creates a continuous, clean selective pressure that prevents "Ontological Cancer" without the overhead of Pareto sorting.

#### ❌ Excessive Stagnation Detector Subtypes
*   **The Complexity:** The system introduces four separate stagnation detector protocols (Windowed Mean, EMA, Statistical Test, Veto Rate) to act as a gate before triggering probes.
*   **Why it is unnecessary:** These statistical heuristics add unnecessary configuration parameters that require tedious tuning.
*   **The Simplification:** Replace these with a simple, unified rolling compute budget. If the system is allocated a fixed "metabolic energy budget" per episode, it can dynamically decide whether it has the resources to run a probe.

---

### 2. What Must Remain (The Core Strengths)

To achieve the same (or better) results, any simplified version of Autopoiesis must preserve two highly effective, load-bearing mechanisms:

1.  **Neutral Birth (Phenotype-Preserving Birth):** Structural mutations (like spawning nodes or growing new tiles) must be initialized so they produce bit-identical outputs to the parent at the moment of insertion (e.g., new edges start at zero weight, and new nodes start as identity mappings). This ensures the new tile doesn't shock the system or immediately violate the critical Lyapunov stability bounds of the Constitution.
2.  **Slope-Based Selection:** Evaluating a proposed mutation based on its immediate "point accuracy" is useless for Tier 2 mutations (like changing a credit assignment rule), because swapping the rule doesn't change the forward pass at birth—it only changes how the model *will* learn. Giving mutations a short, equal-budget **adaptation probe** and measuring the **improvement rate (slope)** is a genuine contribution that must be kept.

---

### 3. The "Lean Autopoiesis" Blueprint

By removing the over-engineered layers, a simplified and highly effective version of the consolidator would look like this:

*   **Genotype (\(\Omega\)):** A simple, direct topology graph of tiles.
*   **Mutations (Asexual only):** Limited strictly to **`DuplicateAndPerturb`** (cellular division) and **`SpliceOperator`** (inserting a new tile or route). No crossover, no population management.
*   **The Consolidator Loop:**
    1.  At the episode boundary (Sleep phase), if progress slows down, propose a local structural or algorithmic mutation.
    2.  Apply **Neutral Birth** to initialize the new tile/connection.
    3.  Run a brief, isolated **Adaptation Probe** on a held-out data fork to measure the learning slope of both the parent and the mutated offspring.
    4.  Apply a **Paired Statistical Test** to accept the mutation only if its learning slope is significantly better than the parent's, penalized by the metabolic cost (`GenomeSizePenalty`) of the new tile.

This lean architecture eliminates the meta-learning abstractions of Tier 3 and the overhead of population-based genetics, while keeping the powerful biological capability for a single, multi-tile organism to dynamically grow, split its "cells," and reroute its pathways to survive damage or master complex tasks.

