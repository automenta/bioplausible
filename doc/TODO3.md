This specification transforms **toreq** from a code repository into a **Living Research Laboratory**. We are moving away from "training a model" toward "observing a dynamical system."

This spec, the **"TorEq Dynamic Observatory" (TDO)**, integrates the spatial heatmap with fractal/hierarchical architectures and asynchronous "lazy" logic.

---

### Phase 1: The Spatial Heatmap Spec (The "Synapse Eye")
Instead of a single loss curve, we visualize the entire network state as a **multi-scale density map**.

*   **View Layers as Grids:** Every hidden layer is reshaped into a square grid (e.g., 1024 neurons = 32x32).
*   **The Three-Channel View (RGB Mapping):**
    *   **Red Channel:** **Activation Magnitude ($s$)**. Shows which neurons are "awake."
    *   **Green Channel:** **Equilibrium Velocity ($\Delta s_t$)**. Shows which neurons are still "settling." When the screen turns black/dim, the network has reached a fixed point.
    *   **Blue Channel:** **Nudge Magnitude ($s_{\text{nudged}} - s_{\text{free}}$)**. Shows the "Credit Assignment" flowing backward. This is the **most important channel**—it visualizes the gradient literally "bleeding" through the layers.
*   **Stability Overlays:** Highlight neurons that exceed the Lipschitz bound (saturated or oscillating) in bright white.

---

### Phase 2: The Fractal/Hierarchical Architecture
We move beyond the "Flat MLP" and implement a **Nested Recursive Structure**.

*   **The "Recursive Block" Concept:** Instead of `Layer(Input, Output)`, we use `Block(N)`. A block is a self-contained recurrent loop.
*   **Fractal Connectivity:** 
    *   Layer 1 connects to Layer 2.
    *   Layer 2 contains a "Mini-TorEq" inside it that iterates 5 times for every 1 iteration of the main loop.
    *   **The Goal:** Test if Spectral Normalization can stabilize a network where "sub-brains" are reaching their own equilibria inside a larger global equilibrium.
*   **Experiment:** Does this "Deep Nesting" allow the network to learn complex features (like loops or textures) better than a flat stack?

---

### Phase 3: The Asynch/Lazy "Event-Driven" Engine
We break the "Global Clock" to save energy and simulate hardware.

*   **Activity-Gated Relaxation:**
    *   A neuron (or block) only updates its state if its input has changed by more than $\epsilon$.
    *   **The Visualization:** The Heatmap will show "Avalanches." An input pulse hits the first layer, and you see the "settling" green light ripple through the hierarchy.
*   **Asynchronous Phases:**
    *   Instead of "Free Phase for 30 steps, then Nudged Phase for 20," the Output Layer is *constantly* providing a weak "Nudge."
    *   The network is in a continuous state of **"Persistent Relaxation."** It is never fully "free" and never fully "nudged." It is always chasing a moving target.
*   **Experiment:** Prove that this "Lazy" version achieves the same accuracy as the "Clocked" version but with **70% fewer FLOPs**.

---

### Phase 4: The "Undeniable" Multi-Layer Challenge
This is the "Boss Fight" for EqProp.

*   **The "100-Layer" Test:**
    *   Build a 100-layer recursive network.
    *   Use the Heatmap to watch the "Nudge Blue" travel from Layer 100 back to Layer 1.
    *   **The Innovation:** If the blue light vanishes by Layer 50, the experiment fails. If it reaches Layer 1, you have achieved **Infinite Depth Credit Assignment**.
*   **Lipschitz-Tuning (The Vibe-Knob):**
    *   Add a real-time slider to adjust the **Spectral Norm Constraint ($\sigma$)**.
    *   As you slide $\sigma$ above 1.0, the Heatmap should show the network "exploding" into white noise. 
    *   As you slide it below 1.0, you see the "Contraction" happen—the chaos settles into a crystal-clear equilibrium.

---

### Implementation Spec (The "How-To")

1.  **Frontend:** Use **PyGame** or **VisPy** (high-performance GPU-bound visualization). Matplotlib is too slow for real-time dynamics.
2.  **Kernel:** Modify your **CuPy kernel** to export the "Delta-State" ($s_t - s_{t-1}$) at every iteration. This is the data that feeds the Green channel.
3.  **Metrics to Track:**
    *   **Settling Time ($T_{relax}$):** How many steps until Green channel < threshold?
    *   **Nudge Depth ($D_{nudge}$):** How many layers deep is the Blue channel visible?
    *   **Energy Consumption (Estimated):** Based on the % of "Lazy" neurons skipped.

---

### Why this makes the research "Undeniable":

If you send a link to a researcher and they see a **live video** of a 100-layer network "breathing," settling into equilibrium, and visualizing its own gradients as a blue ripple—all while being stabilized by a simple Spectral Norm constraint—**they cannot ignore it.**

It moves the conversation from:
*"Does this math hold up?"* 
to 
*"I can see it working in real-time. The stability is physical."*
