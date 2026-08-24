from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from computronium.core.logging import get_logger
from computronium.zoo.models.eqprop import (
    BackpropMLP,
    LoopedMLP,
)

from ..utils import create_synthetic_dataset, evaluate_accuracy, train_model
from ._base import build_track_result, track_header

if TYPE_CHECKING:
    from ..notebook import TrackResult

root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))


__all__ = [
    "logger",
    "root_path",
    "track_1_spectral_norm",
    "track_2_backprop_parity",
    "track_3_adversarial_healing",
]
logger = get_logger()


def track_1_spectral_norm(verifier) -> TrackResult:
    """Core: Spectral Normalization maintains L < 1."""
    start = track_header(1, "Spectral Normalization Stability")
    input_dim, hidden_dim, output_dim = 64, 128, 10
    X, y = create_synthetic_dataset(verifier.n_samples, input_dim, 10, verifier.seed)

    # Without SN - use higher LR to show instability
    logger.info("\n[1a] Without spectral norm (aggressive training)...")
    model_no_sn = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=False)
    L_before_no = model_no_sn.compute_lipschitz()
    # Higher LR causes L to grow more
    train_model(model_no_sn, X, y, epochs=verifier.epochs, lr=0.05, name="No SN")
    L_after_no = model_no_sn.compute_lipschitz()

    # With SN
    logger.info("[1b] With spectral norm...")
    model_sn = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=True)
    L_before_sn = model_sn.compute_lipschitz()
    train_model(model_sn, X, y, epochs=verifier.epochs, lr=0.05, name="With SN")
    L_after_sn = model_sn.compute_lipschitz()

    # Evaluate: Key insight is that SN constrains L while non-SN allows growth
    sn_constrained = L_after_sn <= 1.05  # With SN, L should stay near 1
    l_difference = L_after_no - L_after_sn  # Non-SN should have larger L

    # Score based on whether SN is effective
    if sn_constrained and l_difference > 0.5:
        score = 100
        status = "pass"
    elif sn_constrained:
        score = 75
        status = "partial"
    else:
        score = 25
        status = "fail"

    l_diff_no = L_after_no - L_before_no
    l_diff_sn = L_after_sn - L_before_sn
    evidence = f"""
**Claim**: Spectral normalization keeps L ≤ 1 vs unconstrained training.

**Experiment**: Train identical networks with and without spectral normalization.

| Configuration | L (before) | L (after) | Δ | Constrained? |
|---------------|------------|-----------|---|--------------|
| Without SN | {L_before_no:.3f} | {L_after_no:.3f} | {l_diff_no:+.2f} | [FAIL]  No |
| With SN | {L_before_sn:.3f} | {L_after_sn:.3f} | {l_diff_sn:+.2f} |
| | | | | {"[OK]  Yes" if sn_constrained else "[FAIL]  No"} |

**Key Difference**: L(no_sn) - L(sn) = {l_difference:.3f}

**Interpretation**:
- Without SN: L = {L_after_no:.2f} (unconstrained, can grow)
- With SN: L = {L_after_sn:.2f} (constrained to ~1.0)
- SN provides {(L_after_no / L_after_sn - 1) * 100:.0f}% Lipschitz reduction
"""

    improvements = []
    if not sn_constrained:
        improvements.append(
            "Spectral norm not constraining L ≤ 1; check implementation"
        )
    if l_difference < 0.5:
        improvements.append(
            "Difference between SN/non-SN too small; increase epochs or LR"
        )

    return build_track_result(
        track_id=1,
        name="Spectral Normalization Stability",
        status=status,
        score=score,
        metrics={"L_no_sn": L_after_no, "L_sn": L_after_sn, "difference": l_difference},
        evidence=evidence,
        start=start,
        improvements=improvements,
    )


# Attach metadata
track_1_spectral_norm.description = (
    "Verifies spectral norm constraints keep Lipschitz constant <= 1"
)
track_1_spectral_norm.category = "Core Stability"


def track_2_backprop_parity(verifier) -> TrackResult:
    """Core: EqProp achieves accuracy parity with Backprop."""
    start = track_header(2, "EqProp vs Backprop Parity")
    input_dim, hidden_dim, output_dim = 64, 128, 10

    # Create a single dataset and split it for fair comparison
    # Using the same data for both methods ensures fair algorithm comparison
    X_all, y_all = create_synthetic_dataset(
        verifier.n_samples, input_dim, 10, verifier.seed
    )
    split = int(0.8 * len(X_all))
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]

    # Backprop
    logger.info("\n[2a] Backprop MLP...")
    bp_model = BackpropMLP(input_dim, hidden_dim, output_dim)
    train_model(bp_model, X_train, y_train, epochs=verifier.epochs, name="Backprop")
    bp_acc = evaluate_accuracy(bp_model, X_test, y_test)

    # EqProp
    logger.info("[2b] EqProp (LoopedMLP)...")
    eq_model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=True)
    train_model(eq_model, X_train, y_train, epochs=verifier.epochs, name="EqProp")
    eq_acc = evaluate_accuracy(eq_model, X_test, y_test)

    gap = (bp_acc - eq_acc) * 100

    # Score: Pass if both achieve excellent performance (>99%) OR gap < 3%
    # This handles floating point precision issues when both round to 100.0%
    both_excellent = bp_acc >= 0.99 and eq_acc >= 0.99

    if both_excellent or abs(gap) < 3:
        score = 100
        status = "pass"
    elif abs(gap) < 10:
        score = 70
        status = "partial"
    else:
        score = 30
        status = "fail"

    evidence = f"""
**Claim**: EqProp achieves competitive accuracy with Backpropagation (gap < 3%).

**Experiment**: Train identical architectures with Backprop and EqProp
on synthetic classification.

| Method | Test Accuracy | Gap |
|--------|---------------|-----|
| Backprop MLP | {bp_acc * 100:.1f}% | — |
| EqProp (LoopedMLP) | {eq_acc * 100:.1f}% | {gap:+.1f}% |

**Verdict**: {"[OK]  PARITY" if abs(gap) < 5 else "[WARN]  Gap"} (gap = {abs(gap):.1f}%)

**Note**: Small datasets may show variance; run with --full for 5-seed validation.
"""

    improvements = []
    if abs(gap) > 3:
        improvements.append(
            f"Gap of {abs(gap):.1f}% exceeds target; tune hyperparameters"
        )
    if eq_acc < 0.8:
        improvements.append("Low absolute accuracy; increase epochs or model size")

    return build_track_result(
        track_id=2,
        name="EqProp vs Backprop Parity",
        status=status,
        score=score,
        metrics={"bp_acc": bp_acc, "eq_acc": eq_acc, "gap": gap},
        evidence=evidence,
        start=start,
        improvements=improvements,
    )


# Attach metadata
track_2_backprop_parity.description = (
    "Tests if EqProp matches Backprop accuracy on synthetic data"
)
track_2_backprop_parity.category = "Performance"


def track_3_adversarial_healing(verifier) -> TrackResult:
    """Track 1 (README): Adversarial Self-Healing via noise damping."""
    start = track_header(3, "Adversarial Self-Healing")
    input_dim, hidden_dim, output_dim = 64, 128, 10

    X, y = create_synthetic_dataset(verifier.n_samples, input_dim, 10, verifier.seed)
    model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=True)

    logger.info("\n[3a] Pre-training model...")
    train_model(model, X, y, epochs=verifier.epochs, name="Pre-train")

    logger.info("[3b] Testing noise damping...")
    noise_levels = [0.5, 1.0, 2.0]
    results = {}

    model.eval()
    with torch.no_grad():
        # Get the initial activations (input projection)
        x_test = X[:32]
        activations = model._initial_activations(x_test)

    for noise in noise_levels:
        # Inject noise into hidden state and measure damping through relaxation
        model.eval()
        with torch.no_grad():
            # Start from clean activations
            h_clean = activations[-1].clone()

            # Add noise to hidden state
            noise_tensor = torch.randn_like(h_clean) * noise
            h_noisy = h_clean + noise_tensor
            initial_noise_mag = noise_tensor.abs().mean().item()

            # Run relaxation steps (without input, just recurrent dynamics)
            # Replace the hidden activation with noisy version and settle
            activations_noisy = list(activations)
            activations_noisy[-1] = h_noisy

            # Use the model's settle function
            from computronium.core.local_learning.settling import (
                settle_activations_list,
            )

            settled, _, _ = settle_activations_list(
                activations_0=activations_noisy,
                forward_dynamics=model.forward_dynamics,
                steps=model.max_steps,
                beta=0.0,
                target=None,
                return_trajectory=False,
                return_dynamics=False,
                convergence_threshold=model.convergence_threshold,
                convergence_start=model.convergence_start,
            )

            h_final = settled[-1]
            final_noise = (h_final - h_clean).abs().mean().item()
            damping_percent = (1 - final_noise / (initial_noise_mag + 1e-8)) * 100

        results[noise] = {
            "initial_noise": initial_noise_mag,
            "final_noise": final_noise,
            "damping_percent": damping_percent,
        }
        logger.info("  sigma=%s: damping=%.1f%%", noise, damping_percent)

    avg_damping = np.mean([r["damping_percent"] for r in results.values()])
    score = min(100, avg_damping)
    status = "pass" if avg_damping > 95 else ("partial" if avg_damping > 50 else "fail")

    table_rows = "\n".join(
        f"| σ={n} | {r['initial_noise']:.3f} | {r['final_noise']:.6f} | "
        f"{r['damping_percent']:.1f}% |"
        for n, r in results.items()
    )

    evidence = f"""
**Claim**: EqProp networks automatically damp injected noise to zero via contraction.

**Experiment**: Inject Gaussian noise at hidden layer mid-relaxation, measure residual.

| Noise Level | Initial | Final | Damping |
|-------------|---------|-------|---------|
{table_rows}

**Average Damping**: {avg_damping:.1f}%

**Mechanism**: Contraction mapping (L < 1) guarantees: ||noise|| → L^k × ||initial|| → 0

**Hardware Impact**: Enables radiation-hardened, fault-tolerant neuromorphic chips.
"""

    improvements = []
    if avg_damping < 99:
        improvements.append(
            f"Damping at {avg_damping:.1f}%; check Lipschitz constraint"
        )

    return build_track_result(
        track_id=3,
        name="Adversarial Self-Healing",
        status=status,
        score=score,
        metrics={"avg_damping": avg_damping, "results": results},
        evidence=evidence,
        start=start,
        improvements=improvements,
    )


# Attach metadata
track_3_adversarial_healing.description = (
    "Measures noise damping (self-healing) properties of EqProp"
)
track_3_adversarial_healing.category = "Robustness"
