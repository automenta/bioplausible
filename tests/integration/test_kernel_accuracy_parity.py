"""Kernel accuracy parity tests on digits/MNIST (REFACTOR7 Phase 2-9).

Verifies that kernel-driven training reaches accuracy within 1% of the
PyTorch reference for each algorithm family with a registered KernelBackend.
"""

import pytest
import torch
from torch import nn

from bioplausible.acceleration import get_algorithm_kernels
from bioplausible.core.trainer import CoreTrainer, TrainerConfig

# Ensure kernel registry is populated
get_algorithm_kernels()


@pytest.fixture(scope="module", autouse=True)
def _populate_kernel_registry():
    get_algorithm_kernels()
    yield


# Families with registered KernelBackends and models that expose Linear stacks
# for the generic consumer. Each entry: (model_name, family, model_kwargs)
KERNEL_FAMILIES = [
    (
        "standard_fa",
        "fa",
        {
            "input_dim": 64,
            "hidden_dim": 64,
            "output_dim": 10,
            "num_layers": 2,
            "use_spectral_norm": False,
        },
    ),
    (
        "backprop_mlp",
        "backprop",
        {
            "input_dim": 64,
            "hidden_dim": 64,
            "output_dim": 10,
            "num_layers": 2,
        },
    ),
    # PEPITA is the first bespoke-dynamics family consumed through the
    # dispatch seam (``kernel_train_step`` instead of the uniform consumer).
    # It learns in-place at the model's own ``lr`` (forward-only, no optimizer);
    # pass a tuned LR so the parity gate runs a real learning comparison.
    (
        "pepita",
        "pepita",
        {
            "input_dim": 64,
            "hidden_dim": 64,
            "output_dim": 10,
            "num_layers": 2,
            "learning_rate": 0.3,
        },
    ),
    # Difference Target Propagation is consumed through the same bespoke seam:
    # ``TPKernelBackend.kernel_train_step`` mirrors the reference DTP dynamics
    # (output-layer Adam + inverse-net target propagation + per-layer fitting),
    # so kernel and reference must land on the same accuracy.
    (
        "diff_target_prop",
        "tp",
        {
            "input_dim": 64,
            "hidden_dim": 64,
            "output_dim": 10,
            "num_layers": 2,
            "learning_rate": 1e-3,
            "target_lr": 0.1,
        },
    ),
]

# Families that require the reference to use the model's own ``train_step``
# dynamics (forward-only learners like PEPITA route their updates in-place and
# never touch the trainer optimizer). The generic ``_train_and_eval`` helper
# handles them identically via ``_train_step`` → ``dispatch_train_step``.
BESPOKE_FAMILIES = {"pepita", "tp"}


def _train_and_eval(
    model_name: str,
    model_kwargs: dict,
    use_kernel: bool,
    kernel_backend: str = "cpu",
    epochs: int = 3,
    seed: int = 42,
) -> float:
    """Train model for `epochs` and return final test accuracy."""
    torch.manual_seed(seed)

    config = TrainerConfig(
        model=model_name,
        task="digits",
        model_kwargs=model_kwargs,
        epochs=epochs,
        use_kernel=use_kernel,
        kernel_backend=kernel_backend,
        track_energy=False,
        batches_per_epoch=50,
        batch_size=32,
        optimizer="adam",
        optimizer_kwargs={"lr": 1e-3},
        allow_bptt_fallback=True,
        run_validation=True,
    )

    trainer = CoreTrainer(config)
    history = trainer.fit()

    # Return final validation accuracy from history
    if history:
        return history[-1].val_acc or 0.0
    return 0.0


@pytest.mark.slow
@pytest.mark.parametrize("model_name,family,model_kwargs", KERNEL_FAMILIES)
def test_kernel_accuracy_parity_digits(model_name, family, model_kwargs):
    """Kernel-driven training accuracy within 1% of reference on digits.

    This is the end-to-end learning parity gate for each kernel backend family.
    """
    ref_acc = _train_and_eval(model_name, model_kwargs, use_kernel=False, seed=42)
    kernel_acc = _train_and_eval(
        model_name, model_kwargs, use_kernel=True, kernel_backend="cpu", seed=42
    )

    diff = abs(kernel_acc - ref_acc)
    print(
        f"\n{model_name} ({family}): ref_acc={ref_acc:.4f}, kernel_acc={kernel_acc:.4f}, diff={diff:.4f}"
    )

    # Allow kernel to be within 1% OR better than reference (kernel can outperform)
    assert diff <= 0.01 or kernel_acc >= ref_acc, (
        f"{model_name} kernel accuracy {kernel_acc:.4f} drifted >1% below "
        f"reference {ref_acc:.4f} (diff={diff:.4f})"
    )


# ---------------------------------------------------------------------------
# Synthetic separable task parity (fast, no data download)
# ---------------------------------------------------------------------------


def _train_synthetic(
    model_name: str,
    model_kwargs: dict,
    use_kernel: bool,
    kernel_backend: str = "cpu",
    seed: int = 42,
) -> float:
    """Train on synthetic separable task and return final accuracy."""
    torch.manual_seed(seed)

    n, dim, classes = 400, 64, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, classes, (n,))
    for c in range(classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    config = TrainerConfig(
        model=model_name,
        task="digits",
        model_kwargs=model_kwargs,
        epochs=1,
        use_kernel=use_kernel,
        kernel_backend=kernel_backend,
        track_energy=False,
        batches_per_epoch=100,
        batch_size=32,
        optimizer="adam",
        optimizer_kwargs={"lr": 1e-3},
        allow_bptt_fallback=True,
    )

    trainer = CoreTrainer(config)
    trainer.setup()
    dev = next(trainer.model.parameters()).device
    xd, yd = x.to(dev), y.to(dev)

    for _ in range(8):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i : i + 32]
            trainer._train_step(xd[idx], yd[idx])

    trainer.model.eval()
    with torch.no_grad():
        return (trainer.model(xd).argmax(1) == yd).float().mean().item()


@pytest.mark.parametrize("model_name,family,model_kwargs", KERNEL_FAMILIES)
def test_kernel_accuracy_parity_synthetic(model_name, family, model_kwargs):
    """Fast synthetic-task parity gate (same as test_kernel_dispatch but DRY)."""
    ref_acc = _train_synthetic(model_name, model_kwargs, use_kernel=False, seed=42)
    kernel_acc = _train_synthetic(
        model_name, model_kwargs, use_kernel=True, kernel_backend="cpu", seed=42
    )

    diff = abs(kernel_acc - ref_acc)
    print(
        f"\n{model_name} ({family}) synthetic: ref={ref_acc:.4f}, kernel={kernel_acc:.4f}, diff={diff:.4f}"
    )

    # Allow kernel to be within 1% OR better than reference
    assert diff <= 0.01 or kernel_acc >= ref_acc, (
        f"{model_name} kernel accuracy {kernel_acc:.4f} drifted >1% below "
        f"reference {ref_acc:.4f} on synthetic task (diff={diff:.4f})"
    )
