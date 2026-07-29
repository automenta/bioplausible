"""
Research Tracks (42-44) for 2025 EqProp Research Landscape.

Validates new research directions:
- Holomorphic EP (Complex-valued)
- Directed EP (Asymmetric weights)
- Finite-Nudge EP (Large beta)
"""

import logging
import time

import torch

from bioplausible.zoo.models.eqprop import DirectedEP, FiniteNudgeEP, HolomorphicEP

from ..notebook import TrackResult

logger = logging.getLogger(__name__)


def _get_synthetic_data(n=32, input_dim=64, output_dim=10):
    x = torch.randn(n, input_dim)
    y = torch.randint(0, output_dim, (n,))
    return x, y


def track_42_holomorphic_ep(verifier) -> TrackResult:
    """Track 42: Holomorphic Equilibrium Propagation."""
    logger.info("\n%s", "=" * 60)
    logger.info("TRACK 42: Holomorphic EP (Complex)")
    logger.info("%s", "=" * 60)

    start = time.time()

    # 1. Setup
    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    x, y = _get_synthetic_data(
        n=verifier.n_samples if verifier.quick_mode else 1000,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # 2. Model
    model = HolomorphicEP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        eq_steps=10,
        learning_rate=0.01,
    )

    # 3. Training Loop
    logger.info("\n[42a] Training HolomorphicEP...")
    initial_metrics = model.train_step(x[:32], y[:32])
    initial_loss = initial_metrics["loss"]
    logger.info("  Initial Loss: %.4f", initial_loss)

    losses = []
    epochs = 30 if verifier.quick_mode else 50
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(x.size(0))
        epoch_loss = 0
        batches = 0
        for i in range(0, x.size(0), batch_size):
            idx = perm[i : i + batch_size]
            metrics = model.train_step(x[idx], y[idx])
            epoch_loss += metrics["loss"]
            batches += 1

        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            logger.info("  Epoch %d: Loss %.4f", epoch + 1, avg_loss)

    final_loss = losses[-1]

    # Check learning
    learned = final_loss < initial_loss * 0.95

    # Check complex weights
    is_complex = model.layers[0].weight.is_complex()

    score = 100 if learned and is_complex else 0
    status = "pass" if score == 100 else "fail"

    evidence = f"""
**Claim**: Holomorphic EP learns using complex-valued states and weights.

**Results**:
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
- Complex Weights: {"[OK]  Yes" if is_complex else "[FAIL]  No"}
- Learning: {"[OK]  Yes" if learned else "[FAIL]  No"}
"""

    return TrackResult(
        track_id=42,
        name="Holomorphic EP",
        status=status,
        score=score,
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
        evidence=evidence,
        time_seconds=time.time() - start,
    )


def track_43_directed_ep(verifier) -> TrackResult:
    """Track 43: Directed Equilibrium Propagation."""
    logger.info("\n%s", "=" * 60)
    logger.info("TRACK 43: Directed EP (Asymmetric)")
    logger.info("%s", "=" * 60)

    start = time.time()

    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    x, y = _get_synthetic_data(
        n=verifier.n_samples if verifier.quick_mode else 1000,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    model = DirectedEP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        eq_steps=10,
        learning_rate=0.01,
    )

    # Verify asymmetry
    w_fwd = model.forward_layers[0].weight
    w_bwd = model.feedback_layers[0].weight  # Corresponds to layer 0 connection?
    # In my implementation:
    # forward_layers[0] connects input -> h1 (dim 0 -> 1)
    # feedback_layers[0] connects h1 -> input (dim 1 -> 0)
    # Check shapes
    logger.info("  Forward W shape: %s", w_fwd.shape)
    logger.info("  Feedback B shape: %s", w_bwd.shape)

    # Check if tied (should NOT be tied/shared memory)
    is_tied = w_fwd.data_ptr() == w_bwd.data_ptr()
    logger.info("  Weights Tied: %s", is_tied)

    # Train
    logger.info("\n[43a] Training DirectedEP...")
    metrics = model.train_step(x[:32], y[:32])
    initial_loss = metrics["loss"]
    logger.info("  Initial Loss: %.4f", initial_loss)

    epochs = 30 if verifier.quick_mode else 50
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch_size):
            idx = perm[i : i + batch_size]
            model.train_step(x[idx], y[idx])

    metrics = model.train_step(x[:32], y[:32])
    final_loss = metrics["loss"]
    logger.info("  Final Loss: %.4f", final_loss)

    learned = final_loss < initial_loss * 0.95

    score = 100 if learned and not is_tied else 0
    status = "pass" if score == 100 else "fail"

    evidence = f"""
**Claim**: Directed EP learns with asymmetric forward/feedback weights.

**Results**:
- Asymmetric: {"[OK]  Yes" if not is_tied else "[FAIL]  No"}
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
"""

    return TrackResult(
        track_id=43,
        name="Directed EP",
        status=status,
        score=score,
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
        evidence=evidence,
        time_seconds=time.time() - start,
    )


def track_44_finite_nudge_ep(verifier) -> TrackResult:
    """Track 44: Finite-Nudge Equilibrium Propagation."""
    logger.info("\n%s", "=" * 60)
    logger.info("TRACK 44: Finite-Nudge EP (Large Beta)")
    logger.info("%s", "=" * 60)

    start = time.time()

    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    x, y = _get_synthetic_data(
        n=verifier.n_samples if verifier.quick_mode else 1000,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # Use Beta = 1.0 (Very large compared to standard 0.1 or 0.5/sqrt(N))
    model = FiniteNudgeEP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        beta=1.0,
        eq_steps=10,
        learning_rate=0.01,
    )
    logger.info("  Using Beta: %s", model.beta)

    # Train
    logger.info("\n[44a] Training FiniteNudgeEP...")
    metrics = model.train_step(x[:32], y[:32])
    initial_loss = metrics["loss"]
    logger.info("  Initial Loss: %.4f", initial_loss)

    epochs = 30 if verifier.quick_mode else 50
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch_size):
            idx = perm[i : i + batch_size]
            model.train_step(x[idx], y[idx])

    metrics = model.train_step(x[:32], y[:32])
    final_loss = metrics["loss"]
    logger.info("  Final Loss: %.4f", final_loss)

    learned = final_loss < initial_loss * 0.95

    score = 100 if learned else 0
    status = "pass" if score == 100 else "fail"

    evidence = f"""
**Claim**: Finite-Nudge EP learns stably with large beta ({model.beta}).

**Results**:
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
- Stability: {"[OK]  Stable" if final_loss < 100 else "[FAIL]  Unstable"}
"""

    return TrackResult(
        track_id=44,
        name="Finite-Nudge EP",
        status=status,
        score=score,
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
        evidence=evidence,
        time_seconds=time.time() - start,
    )
