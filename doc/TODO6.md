This is a comprehensive, multi-track strategic research plan designed to elevate **Toreq** from a research repository to a foundational AI paradigm.

The strategy is organized by **Maturity Horizons** rather than calendar dates. Each horizon represents a distinct leap in capability, requiring the completion of the previous horizon's objectives.

---

### **Master Strategy: The "Thermodynamic AI" Roadmap**

**Core Thesis:** Move AI from "Geometric Optimization" (Backprop/Gradient Descent) to "Physical Equilibration" (EqProp/Energy Minimization).

---

### **Track 1: Algorithmic Scaling (The "Intelligence" Track)**
*Goal: Prove EqProp is the superior mechanism for sequence modeling and reasoning at scale.*

#### **Horizon 1: The Attention Primitive**
*   **Objective:** Stabilize Energy-Based Attention.
*   **The Problem:** Standard Softmax attention assumes a discrete forward pass. EqProp requires attention to be a dynamical system that settles.
*   **The Deliverable:** A robust **`EqAttention`** layer that functions as a continuous Hopfield Network, allowing information to be retrieved via energy minimization rather than matrix multiplication.
*   **Success Metric:** Perplexity parity with standard Transformers on WikiText-103 using 50% less memory.

#### **Horizon 2: The "Infinite-Context" LLM**
*   **Objective:** Leverage O(1) memory for infinite sequence handling.
*   **The Innovation:** Unlike BPTT (Backprop Through Time), which stores history to calculate gradients, EqProp calculates gradients based solely on the *current* equilibrium state.
*   **The Deliverable:** A **Toreq-7B** (Foundation Model) capable of processing distinct, effectively infinite context windows without exploding memory usage.
*   **Success Metric:** Successful training of a standard LLM benchmark on a single consumer GPU where a standard Transformer would OOM (Out of Memory).

#### **Horizon 3: The 1-Bit "Integer Brain"**
*   **Objective:** Extreme Quantization.
*   **The Innovation:** Combining Spectral Normalization with Ternary Weights ($\{-1, 0, +1\}$).
*   **The Deliverable:** A high-performance Large Language Model that requires **zero floating-point multiplication**, relying entirely on integer addition and sign flips.
*   **Success Metric:** State-of-the-art inference speed on CPUs (not GPUs) due to the removal of float math overhead.

---

### **Track 2: Hardware Physicality (The "Substrate" Track)**
*Goal: Transition from "simulating physics" on GPUs to "exploiting physics" on native hardware.*

#### **Horizon 1: The "Lazy" Event Engine**
*   **Objective:** Logical Efficiency.
*   **The Innovation:** Refine `LazyEqProp` to be strictly event-driven. Neurons only update when their input energy changes beyond a threshold ( $\Delta E > \epsilon$ ).
*   **The Deliverable:** A CUDA kernel that demonstrates **90%+ sparse updates** during training, proving that most of the network can "sleep" while learning.

#### **Horizon 2: FPGA & Neuromorphic Synthesis**
*   **Objective:** Digital-Physical Verification.
*   **The Innovation:** Port the update logic to Verilog/VHDL. Since EqProp relies on local updates, it eliminates the "memory bottleneck" of fetching global weights.
*   **The Deliverable:** An FPGA implementation (e.g., Xilinx UltraScale) where thousands of "physical neurons" operate in parallel, asynchronously finding equilibrium.
*   **Success Metric:** Demonstrate >100x energy efficiency (Joules/Op) compared to an H100 GPU running the same network.

#### **Horizon 3: The Analog/Optical Chip**
*   **Objective:** Speed-of-Light Inference.
*   **The Innovation:** Implementing the contraction mapping using optical attenuation. Light passing through a medium naturally "relaxes" to a stable state instantly.
*   **The Deliverable:** A "Photonic Toreq" design where the forward/backward pass happens at the speed of light, and the "spectral normalization" is enforced by the laws of thermodynamics, not code.

---

### **Track 3: Unified Dynamics (The "Generative" Track)**
*Goal: Unify Classification, Generation, and Robustness into a single energy landscape.*

#### **Horizon 1: Bidirectional "Dreaming"**
*   **Objective:** Zero-Shot Generation.
*   **The Innovation:** Standard classifiers are $f(x) \rightarrow y$. Toreq models define an energy $E(x,y)$. By clamping $y$ and minimizing $E$ with respect to $x$, the model generates data.
*   **The Deliverable:** A "Dual-Mode" Vision Model. It achieves Top-1 accuracy on ImageNet (Discriminative) and generates recognizable class-conditional images (Generative) *using the exact same weights*.

#### **Horizon 2: The Diffusion Link**
*   **Objective:** High-Fidelity Synthesis.
*   **The Innovation:** Formally linking Equilibrium Propagation with Langevin Dynamics (used in Diffusion Models). The "settling" process of EqProp *is* the denoising process of Diffusion.
*   **The Deliverable:** A generative model that does not require 1,000 denoising steps. It simply "falls" into the high-probability image state via equilibrium dynamics.

#### **Horizon 3: Homeostatic "Self-Healing"**
*   **Objective:** Provable Safety.
*   **The Innovation:** Exploiting the Lipschitz Constraints ($L < 1$) to prove that adversarial perturbations *decay* exponentially as they propagate.
*   **The Deliverable:** A certified "Safety-Critical" model for autonomous driving that is mathematically proven to be immune to standard adversarial attacks (FGSM/PGD).

---

### **Track 4: Architecture & Topology (The "Biomimetic" Track)**
*Goal: Move beyond "Layers" to "Neural Mediums."*

#### **Horizon 1: JAX/PyTorch Abstraction**
*   **Objective:** Frictionless Adoption.
*   **The Deliverable:** A library (`toreq.nn`) where the physics is hidden.
    *   `model = toreq.nn.Sequential(...)`
    *   `loss.backward()` (Handles the Free Phase, Nudge Phase, and Weight Update internally).
*   **Success Metric:** A standard researcher can convert a ResNet to a Toreq-ResNet by changing import statements, with zero modification to their training loop.

#### **Horizon 2: Meta-Plasticity (AutoML)**
*   **Objective:** Eliminate Hyperparameters.
*   **The Innovation:** The network learns its own "nudge factor" ($\beta$) and learning rates locally at each synapse, mimicking biological homeostasis.
*   **The Deliverable:** A "Self-Tuning" optimizer that adjusts the energy landscape shape dynamically, preventing exploding/vanishing gradients without user intervention.

#### **Horizon 3: The "Neural Gel" (Continuous Medium)**
*   **Objective:** Topology Independence.
*   **The Innovation:** Abolishing the concept of "Layers." Neurons exist in a continuous 3D coordinate space $(x, y, z)$. Connectivity is determined by spatial proximity, not matrix indices.
*   **The Deliverable:** A spatially aware "Neural Gel" that allows for **Neurogenesis** (growing new neurons in high-error regions) and **Pruning** (dissolving useless connections) in real-time 3D space.

---

### **The "Killer Demo" Integration Goal**

To prove supremacy, the final output of this roadmap is a single integrated demonstration:

**The "Living" Neural Cube:**
1.  **Runs on Edge Hardware:** A low-power FPGA or embedded chip.
2.  **Continuous Learning:** Learns a video game in real-time (Online Learning) without a replay buffer.
3.  **Generative Dreaming:** Can "imagine" future frames of the game to plan ahead.
4.  **Self-Repairing:** Can sustain "damage" (randomly zeroing out 20% of neurons) and recover performance within seconds via equilibrium dynamics.