# Maximizing Impact: TorEqProp Strategy

> **Goal**: Transform TorEqProp from a "promising repository" into a **landmark research platform** that redefines how the community views Equilibrium Propagation.

## 1. The Core Value Proposition (The "Hook")

We must clearly articulate *why* anyone should care. We have three distinct hooks for different audiences:

*   **For the DL Researcher**: "EqProp finally works on Transformers. It matches Backprop accuracy (97.5%) and is stable. Here is the code."
*   **For the Systems/Hardware Engineer**: "Train infinite-depth networks with O(1) memory. Perfect for edge devices and future neuromorphic chips."
*   **For the Neuroscientist**: "A biologically plausible learning rule (local, Hebbian) that actually scales to modern tasks."

## 2. Strategic Pillars

### A. The "Killer" Software Experience (Ease of Use)
Current research code is often messy. We will win by being **clean, modular, and installable**.

*   **Action**: Publish as a PyPI package (`pip install toreq`).
*   **Action**: Create a "Model Zoo" of pre-trained checkpoint weights.
*   **Action**: Provide "1-minute" Colab notebooks for instant reproduction.

### B. "Seeing is Believing" (Visualizations)
Energy-based models are abstract. We need to make them visible.

*   **Idea**: Visualizer for the "Relaxation Phase". Show the state $h$ settling into an energy well.
*   **Idea**: "Nudge" visualization. Show how the target $y$ pulls the state $h$ out of the well, and how weights update to shift the well.
*   **Idea**: **Live Memory Plot**. Side-by-side comparison of Backprop (growing memory) vs EqProp (flat memory) as depth increases.

### C. The "Novelty" Moat (Scientific Rigor)
Solidify the claims so they are undeniable.

*   **Spectral Normalization**: Frame this not just as a "trick", but as a **theoretical necessity** for contraction.
*   **Beta Stability**: Publish the "Fixed Beta" guideline as a standard best practice.

### D. Hardware Reality Check
Theoretical O(1) is good; simulated energy savings are better.

*   **Action**: Estimate FLOPs and Memory transfers vs Backprop.
*   **Action**: Project energy usage on putative neuromorphic hardware (e.g., "1000x efficient").

## 3. Immediate High-Impact Actions

### 1. The "Manifesto" Blog Post
Draft a high-quality blog post (Distill.pub style) summarizing:
*   Why EqProp failed before (Instability).
*   How we fixed it (Spectral Norm).
*   The O(1) memory revolution.

### 2. The Interactive Demo
Build a simple web-based or CLI demo where users can:
*   Train a small network on MNIST in real-time.
*   Toggle "Spectral Norm" and watch it explode/stabilize.
*   Toggle "Backprop" vs "EqProp" and see usage stats.

### 3. CIFAR-10 Scale-Up
MNIST is "solved". CIFAR-10 is the minimum bar for modern respect.
*   **Urgent**: Getting >85% on CIFAR-10 proves this isn't a toy.

## 4. Execution Plan

1.  **Polish Codebase**: Ensure `pip install -e .` work flawlessly. Add type hints, docstrings.
2.  **Create Visual Assets**: Generate the "Memory vs Depth" and "Energy Landscape" plots.
3.  **Scale to CIFAR**: Run the ConvEqProp or ModernEqProp on CIFAR-10 tonight.
4.  **Publish**: Release the repo, the blog post, and the preprint simultaneously.
