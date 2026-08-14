"""
NEBC (Nobody Ever Bothered to Check) training utilities.

Canonical NEBC training utilities moved from zoo/nebc_base.py to core/ per REFACTOR4.
"""

import torch

from bioplausible.core.losses import compute_accuracy
from bioplausible.core.logging import get_logger
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer

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


def run_nebc_ablation(
    algorithm_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    epochs: int = 50,
    **kwargs,
) -> dict[str, dict]:
    """Run ablation study comparing algorithm with/without spectral norm.

    Returns dict with 'with_sn' and 'without_sn' results.
    """
    from bioplausible.core.registry import ComponentCategory, Registry

    algorithm_cls = Registry.get(ComponentCategory.MODEL, algorithm_name)

    results = {}
    for use_sn in [True, False]:
        label = "with_sn" if use_sn else "without_sn"
        logger.info("  Training %s (%s)...", algorithm_name, label)

        model = algorithm_cls(
            input_dim, hidden_dim, output_dim, use_spectral_norm=use_sn, **kwargs
        )

        train_nebc_model(model, X_train, y_train, epochs=epochs)
        metrics = evaluate_nebc_model(model, X_test, y_test)
        results[label] = metrics

    # Compute delta
    results["delta"] = {
        "accuracy": results["with_sn"]["accuracy"] - results["without_sn"]["accuracy"],
        "lipschitz": results["without_sn"]["lipschitz"]
        - results["with_sn"]["lipschitz"],
        "sn_stabilizes": results["with_sn"]["lipschitz"] <= 1.05,
    }

    return results
