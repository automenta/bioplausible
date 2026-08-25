

### The Z3 Experiment

Once the baseline gate is closed, pivot entirely to producing **one flagship empirical result**. 

**The Target: Level 4 — Z3 Fixed Weights**
This is your most novel contribution. The question—*"Can frozen $\theta$ solve multiple tasks via $\psi$-mediated rule selection?"*—has no direct analogue in the literature. It directly tests the thesis that elevating the computational rule to a dynamical variable (the M-axis) yields a qualitatively different capability.

**Concrete Protocol:**
1.  **Setup:** Initialize a small MLP (e.g., 64→64→64). **Freeze $\theta$** entirely (requires exact parameter invariance: $\|\theta_{after} - \theta_{before}\| = 0$).
2.  **Tasks:** Implement 3 distinct operators: `Identity`, `Threshold`, and `Parity`.
3.  **Adaptation:** Train $\psi$ (`RuleStatePlasticity`) to switch between these tasks.
4.  **Metrics:** Measure adaptation time, energy cost, and operator diversity.
5.  **Baselines:** Compare strictly against (a) fine-tuning $\theta$ (the standard approach) and (b) random $\psi$ initialization.

*Fallback:* If Z3 proves too difficult to converge, pivot to **Level 1 (Adaptation Efficiency)**—comparing Null vs. FastWeight vs. Routing on a switching distribution. This is more conventional but guarantees a clean figure (adaptation time vs. compute cost).

---

### The AutoScientist Ablation Campaign

With one manual experiment result in hand, give the **AutoScientist** a concrete target. Use it to run the *ablations* around your Z3 or Adaptation result.

*   **Targeted Campaign:** Run an M-axis ablation (Null vs. Routing vs. FastWeight) to build the **Resource-Vector Pareto Frontier** $\mathcal{C} = (\text{compute}, \text{memory}, \text{energy}, \text{latency}, \text{plastic-state capacity})$.
*   **Runtime Verification:** Shift verification from "Proof" to "Monitoring." Elevate the `StabilityMonitor` ($\rho(J_F)$ spectral radius, local Lyapunov exponents) to a **runtime campaign guard**. If the AutoScientist generates a 6-D coordinate where $\rho(J_F) > 1.0$, the framework should automatically kill the run, log it to the `failure_manifesto`, and mutate the hyperparameters.

---


*   **The "Continual Learning" Proof (Solving Catastrophic Forgetting):**
    *   **The Problem:** Standard backprop + SGD overwrites old knowledge when learning new tasks (catastrophic forgetting), requiring massive replay buffers.
    *   **The Experiment:** Use `FastWeightPlasticity` (episode-local memory) or `ElasticConsolidationUpdate` (EWC) to train a system on a stream of non-stationary tasks (e.g., Split-MNIST or Continual RL). 
    *   **The Claim:** "Computronium's M-axis natively solves catastrophic forgetting without replay buffers by decoupling fast plastic states ($\psi$) from slow consolidated weights ($\theta$)."
*   **The "Physics-Informed" Proof (Strict Conservation Laws):**
    *   **The Problem:** Standard neural networks struggle to strictly obey physical conservation laws (energy, mass, momentum) when solving PDEs, because backprop treats physics as just another soft loss term.
    *   **The Experiment:** Lean heavily into your `Scientific` domain (Navier-Stokes, Heat/Wave equations). Use `EnergyMinimizationDynamics` (EqProp) where the network's Lyapunov function *is* the physical Hamiltonian/Lagrangian of the system.
    *   **The Claim:** "Energy-based 6-D systems natively conserve physical invariants with zero penalty overhead, outperforming Physics-Informed Neural Networks (PINNs) on long-horizon simulations."


*   **Pivot to the "De Facto Non-Backprop Benchmark":**
    *   **The Strategy:** Stop competing on general tasks. Position Computronium as the absolute standard for evaluating *alternatives to backpropagation* (Forward-Forward, Equilibrium Prop, Predictive Coding, Feedback Alignment).
    *   **The Output:** Publish a massive benchmark paper: *"The Computronium Benchmark: A Fair, 6-D Evaluation of 20 Local Learning Rules."* Because you have the `SystemTrainer` API, you are the only ones who can evaluate these fairly on equal footing.
*   **Pivot to "Algorithm Discovery" (AI for AI):**
    *   **The Strategy:** Use the AutoScientist not just to tune hyperparameters, but to *invent new learning rules*. Frame the 6-D space as a search space for an evolutionary algorithm or LLM.
    *   **The Output:** The AutoScientist discovers a novel, undocumented combination of (CreditAssignment + ParameterUpdate) that empirically beats Adam+Backprop on a specific toy task. You publish the *discovered algorithm*, not just the framework.
*   **Pivot to Edge/Green AI:**
    *   **The Strategy:** Focus entirely on the resource vector $\mathcal{C}$ (compute, memory, energy). Position Computronium as the ultimate framework for ultra-low-power edge computing, where global backward passes and infinite memory are physically impossible.
    *   **The Output:** A suite of models specifically optimized for deployment on microcontrollers (via your ONNX/Ternary export pipelines), proving that local learning rules yield better accuracy-per-watt than quantized MobileNets.


*   **The "Drop-in PyTorch Wrapper":**
    *   **The Strategy:** The 6-D ontology and `SystemTrainer` are powerful but require users to rewrite their training loops. Build a `torch.nn.ComputroniumLinear` wrapper. 
    *   **The Execution:** Allow PyTorch users to swap exactly *one line of code* in their existing scripts to replace `nn.Linear` and `Adam` with an EqProp or Forward-Forward coordinate. Let the wrapper handle the free/nudged phases under the hood.

*   **The "Biological Twin" Project:** Create a 1:1 mapped simulation of a specific, well-documented biological microcircuit (e.g., the *C. elegans* connectome or a specific cortical column) using the 6-D ontology. Use it to predict biological responses to stimuli or lesions, bridging computational neuroscience and ML.



