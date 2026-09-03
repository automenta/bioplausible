"""
NEBC (Nobody Ever Bothered to Check) training utilities.

Canonical NEBC training utilities moved from zoo/nebc_base.py to core/ per REFACTOR4.
"""

import torch

from computronium.core.logging import get_logger
from computronium.core.losses import compute_accuracy
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer

logger = get_logger()


def train_nebc_model(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    verbose: bool = True,
) -> list[float]:
    """
    Standard training loop for NEBC models.

    Returns list of losses for analysis.
    """
    optimizer = create_optimizer(
        model, OptimizerConfig(name="adam", lr=lr, weight_decay=0.0)
    )
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            acc = compute_accuracy(out, y, scale=100)
            L = model.compute_lipschitz()
            logger.info(
                "  [%s] Epoch %d/%d: loss=%.3f, acc=%.1f%%, L=%.3f",
                model.algorithm_name,
                epoch + 1,
                epochs,
                loss.item(),
                acc,
                L,
            )

    return losses


def evaluate_nebc_model(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    """
    Evaluate an NEBC model and return comprehensive metrics.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(X)
            loss = torch.nn.functional.cross_entropy(out, y).item()
            acc = compute_accuracy(out, y)
            L = model.compute_lipschitz()
    finally:
        if was_training:
            model.train()

    return {"accuracy": acc, "loss": loss, "lipschitz": L, **model.get_stats()}
