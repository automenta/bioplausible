### Research Action Plan: From Refactoring to Physical Computation

The current `REFACTOR.md` sprints (0.6 through 0.9) and the completed substrate extractions are strictly **engineering prerequisites**. They build the unified "physics engine" required to run rigorous experiments. Once that foundation is stable, the project must pivot from *implementing algorithms* to *discovering computational regimes*.

The central research question shifts from *"Can we replace backpropagation?"* to:
> **"What computational regimes become possible when credit assignment is constrained by locality, asynchronous dynamics, finite memory, and physical noise?"**

Below is the complete research action plan, organized into five phases, explicitly merging the strategic roadmap while subtracting the engineering tasks already slated for `REFACTOR.md`.

---

### Phase I: Scientific Foundation & The "Honest" Benchmark
*Goal: Establish falsifiable claims and measure physical cost, not just FLOPs.*

**1. The "Physicality & Locality" Taxonomy**
Extend the existing Registry metadata to classify every algorithm on a 6-level plausibility ladder:
*   **L0:** Mathematically local (No global gradient).
*   **L1:** Computationally local (Updates access only local state).
*   **L2:** Temporally local (No global synchronization/barrier).
*   **L3:** Physically local (Only spatial neighbors communicate).
*   **L4:** Device plausible (Robust to analog noise/finite precision).
*   **L5:** Biologically plausible (Matches known neurobiology).
*Action:* Annotate the Registry and filter AutoScientist queries by these levels.

**2. The Canonical "Honest" Benchmark Suite**
Establish a single, reproducible benchmark comparing BP, EqProp, FA, PC, and Forward-Forward across shallow MLPs, deep networks, CNNs, RNNs, GNNs, and small LMs.
*   **The Metric Shift:** Stop optimizing solely for epochs or accuracy. The primary dashboard must track: $\text{Energy/sample}$, $\text{Wall-clock latency}$, $\text{Peak activation memory}$, and $\text{Communication volume}$.
*   *Note: This relies on the unified `supervised_step` (Sprint 0.6) and Strategy-Optimizer wiring (Sprint 0.9) to ensure fair comparison.*

**3. Gradient Alignment Tracking**
Instrument the training loop to explicitly measure the cosine similarity ($\cos \theta$) between local gradients (EqProp/FA/PC) and true backprop gradients.
*   **Experiment:** Plot alignment against depth ($L$), nudge strength ($\beta$), equilibrium tolerance ($\epsilon$), and feedback asymmetry. This provides a mechanistic explanation for *why* an algorithm succeeds or fails, moving beyond black-box accuracy metrics.

**4. API Freeze & Reproducibility**
*   Pin benchmark datasets and random seeds.
*   Implement deterministic regression tests for every core algorithm to ensure refactoring doesn't silently alter physics.
*   Cut the first tagged release to establish a baseline for the scientific community.

---

### Phase II: The Depth & Settling Bottleneck
*Goal: Prove the memory advantage and solve the relaxation cost.*

**5. The Deep Scaling Flagship**
The project claims EqProp avoids depth-dependent activation memory issues. Test this explicitly.
*   **Experiment:** Train networks at $L \in \{10, 100, 1000, 3000, 10000\}$.
*   **Baselines:** Compare against Checkpointed Backprop, Reversible Networks, and Feedback Alignment.
*   **Metrics:** Measure activation memory scaling ($M \propto L^\beta$), training stability, and the number of equilibrium iterations required.

**6. Adaptive Equilibrium Solving**
EqProp replaces explicit backward passes with dynamical relaxation. If settling takes too long, the physical advantage is erased.
*   **Action:** Replace fixed-step relaxation with adaptive solvers: Anderson acceleration, multigrid-like coarse-to-fine relaxation, and residual-based early stopping.
*   **Hypothesis:** Adaptive settling reduces $N_{\text{settle}}$ by orders of magnitude without sacrificing gradient alignment.

**7. Muon/MEP Geometry Investigation**
Treat the MEP (Muon Equilibrium Propagation) family as a serious research branch.
*   **Experiment:** Compare $\text{EP Gradient} \to \text{SGD}$ vs $\text{EP Gradient} \to \text{Muon}$ vs $\text{EP Gradient} \to \text{Natural Gradient}$.
*   **Question:** Why do specific parameter-space geometries (orthogonalized updates) stabilize equilibrium dynamics? Measure convergence in terms of wall-clock time and gradient alignment.

---

### Phase III: The Asynchronous & Physical Regimes
*Goal: Move from "simulated local learning" to "physical dynamical systems."*

**8. Eliminate the Global Clock (Asynchronous Execution)**
The most compelling hardware story is learning as an asynchronous physical dynamical system.
*   **Action:** Build an event-driven execution backend in `core/execution/`. Remove global timesteps, synchronous layer execution, and centralized phase transitions.
*   **Paradigm:** $\text{Event} \to \text{Local State Update} \to \text{Neighbor Events} \to \text{Local Plasticity}$. Prove that learning converges without a global barrier.

**9. The "Ugly" Physics Simulator Backend**
If the thesis is physical realizability, algorithms must not silently rely on ideal PyTorch assumptions.
*   **Action:** Develop a deliberately constrained backend that injects: finite precision, asymmetric conductance, synaptic delay, state leakage, and conductance drift.
*   **Validation:** An algorithm is only truly "bioplausible" if it converges on this backend.

**10. The Physicality Ladder Sweeps**
Run the canonical benchmark across increasingly difficult hardware regimes:
1.  Digital Ideal (FP32)
2.  Quantized Digital (INT8/Ternary)
3.  Noisy Digital (Gaussian state/weight noise)
4.  Analog-like (Device mismatch, limited dynamic range, asymmetric updates)
*   **Goal:** Identify the exact crossover point where EqProp/Tile architectures outperform Backpropagation due to fault tolerance.

---

### Phase IV: EquiTile & Continual Intelligence
*Goal: Explore what emerges from locally-coupled, dynamic modules.*

**11. EquiTile Topological Dynamics**
With the substrate classes shipped (Sprint 0.7), research the architectural properties of the Tile substrate.
*   **Experiment:** Allow tiles to dynamically $\text{split}$, $\text{merge}$, $\text{grow}$, and $\text{prune}$ based on local error signals.
*   **Constraint:** Restrict tiles to sparse $k$-neighbor communication. Can global intelligence emerge from purely local predictive objectives?

**12. Aggressive Continual Learning**
Biological systems excel at continual adaptation. Static supervised benchmarks miss this entirely.
*   **Experiment:** Evaluate replay-free learning using local plasticity, homeostatic constraints, and EWC.
*   **Metrics:** Measure catastrophic forgetting ($F_{\text{forget}}$) and adaptation time ($T_{\text{adapt}}$) in streaming task environments.

**13. Scaling Laws**
Fit fundamental relationships for the Tile and EqProp families:
*   Time to convergence: $T \propto L^\alpha$
*   Memory footprint: $M \propto L^\beta$
*   Energy cost: $E \propto N^\gamma$
Compare these exponents directly against standard Backpropagation.

---

### Phase V: Autonomous Scientific Discovery
*Goal: Transition the LLM from hyperparameter tuner to hypothesis generator.*

**14. Hypothesis-Driven AutoScientist**
Constrain the `AutoScientist` to the strict Scientific Method loop to prevent it from generating thousands of weak, random experiments.
*   **Workflow:** $\text{Hypothesis} \to \text{Prediction} \to \text{Minimal Experiment} \to \text{Ablation} \to \text{Statistical Test} \to \text{Conclusion}$.
*   *Example:* "Increasing spectral contraction should improve EqProp stability at depth but increase settling time."

**15. The Negative Results Chronicle**
Systematically log *why* algorithms fail into the Knowledge Base (e.g., "Gradient variance exploded because the energy landscape lacked spectral normalization"). Use these structured negative results to guide the AutoScientist's next hypotheses, ensuring the system learns from dead ends.

----

## Identify the "Killer Application" (The Edge & Hostile Environments)
Backpropagation has already won the cloud. It is highly optimized for massive GPUs with abundant memory and power. Bioplausible algorithms will likely lose on standard cloud benchmarks (like ImageNet on A100s) due to the overhead of simulating physics on digital hardware.
You must define the environments where backpropagation physically cannot go:
Always-On Edge AI / IoT: Devices that must learn continuously from streaming sensor data without ever connecting to the cloud, operating on microwatts of power.
Space & Radiation-Hardened Computing: In space, cosmic rays cause single-event upsets (bit flips). Backprop networks collapse when weights are flipped. Equilibrium Propagation’s contractive dynamics are mathematically proven to "self-heal" and relax back to the correct state despite noise. This is a massive selling point for aerospace.
Implantable BCIs (Brain-Computer Interfaces): Hardware that must operate locally inside the body, adapting to neural drift in real-time without offloading data to a server.

## The Mathematical & Theoretical Frontier
Empirical benchmarking (Phase I & II) is necessary, but to convince the theoretical ML community, you need rigorous mathematical proofs regarding the energy landscapes of these new architectures.
Energy Landscape Topology: Map the loss landscapes of EqProp and EquiTile. Does local learning introduce spurious local minima that backpropagation avoids?
Lyapunov Stability for Asynchronous Tiles: When you remove the global clock (Phase III), how do you mathematically guarantee that a network of asynchronous, dynamically splitting/merging tiles won't enter chaotic oscillatory states? Developing a Lyapunov stability proof for asynchronous EquiTile dynamics would be a landmark theoretical paper.
Information Bottleneck in Local Learning: Use Information Bottleneck theory to measure exactly how much mutual information is preserved across synapses when global gradients are removed.

## AI Safety and "Local Alignment"
The AI safety community is currently obsessed with the dangers of global objective functions (which lead to reward hacking and deceptive alignment). Bioplausible learning offers a fundamentally different safety paradigm.
No Global Objective to Hack: If an EquiTile network learns purely through local predictive coding and local Hebbian plasticity, there is no global loss function for the system to "hack."
Research Direction: Frame bioplausible as a testbed for Inherently Safe/Aligned Architectures. Investigate whether local learning rules naturally prevent the emergence of deceptive mesa-optimizers because the network lacks the global credit assignment required to plan long-term deceptive strategies.
