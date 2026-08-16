"""Kernel dispatch integration (REFACTOR7 Phase 2-9 opt-in wiring).

Verifies that :meth:`CoreTrainer._wrap_with_kernel` attaches a ``KernelBackend``
instance to the model when ``use_kernel=True`` and the model's family has a
registered backend, and that the plain (kernel-off) path leaves the model
untouched. This exercises the trainer-level dispatch seam that the per-family
parity suites validate at the backend level.
"""

from __future__ import annotations

import pytest

from bioplausible.acceleration import get_algorithm_kernels
from bioplausible.core.trainer import CoreTrainer, TrainerConfig


@pytest.fixture(scope="module", autouse=True)
def _populate_kernel_registry():
    get_algorithm_kernels()
    yield

@pytest.fixture(autouse=True)
def _clear_kernel_cache():
    """Clear kernel registry cache between tests to avoid state pollution."""
    from bioplausible.acceleration.kernel_backend import KernelRegistry
    KernelRegistry.clear_cache()
    yield
    KernelRegistry.clear_cache()


@pytest.mark.parametrize(
    ("model_name", "family"),
    [
        ("feedback_alignment", "fa"),
        ("backprop_mlp", "backprop"),
    ],
)
def test_kernel_dispatch_attaches_backend(model_name, family):
    """``use_kernel=True`` wraps the model with the family's kernel backend."""
    config = TrainerConfig(
        model=model_name,
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    backend = getattr(trainer.model, "_kernel_backend", None)
    assert backend is not None, (
        f"{model_name} (family {family}) was not wrapped with a kernel backend"
    )
    assert backend._config is not None
    assert backend.name == family


@pytest.mark.parametrize("model_name", ["feedback_alignment", "backprop_mlp"])
def test_kernel_off_leaves_model_untouched(model_name):
    """Default ``use_kernel=False`` must not attach a kernel backend."""
    config = TrainerConfig(
        model=model_name,
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=False,
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    assert getattr(trainer.model, "_kernel_backend", None) is None


def test_kernel_backend_consumed_in_train_step():
    """A uniform-interface backend attached by ``use_kernel=True`` is actually
    driven during training (forward → backward → update_weights), so the
    ``dispatch_train_step`` kernel seam is not a dead path.

    Uses ``standard_fa`` (plain ``nn.Linear`` layer stack) so the generic
    consumer binds via ``set_model_ref`` and updates the live weights.
    """
    import torch

    from bioplausible.acceleration import (
        AlgorithmFamily,
        HardwareTarget,
        KernelConfig,
        KernelRegistry,
    )

    config = TrainerConfig(
        model="standard_fa",
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    backend = getattr(trainer.model, "_kernel_backend", None)
    assert backend is not None

    before = [p.clone() for p in trainer.model.layers[0].parameters()]
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    metrics = trainer._train_step(x, y)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    after = list(trainer.model.layers[0].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "kernel train step did not update the model weights"
    )
    assert trainer._training_path_counts.get("kernel", 0) >= 1


def test_kernel_backend_matches_reference_learning():
    """The kernel-driven FA path learns to match the PyTorch reference.

    ``standard_fa`` trained through ``_train_step`` with ``use_kernel=True``
    must reach the same accuracy as the same model trained without the kernel
    (the backend's fixed feedback weights are shared with the model). This
    guards the consumption seam against silent parity drift.
    """
    import torch

    from bioplausible.acceleration import get_algorithm_kernels

    get_algorithm_kernels()

    torch.manual_seed(42)
    n, dim, classes = 400, 64, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, classes, (n,))
    for c in range(classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    def _final_acc(use_kernel: bool) -> float:
        cfg = TrainerConfig(
            model="standard_fa",
            task="digits",
            model_kwargs={
                "input_dim": dim,
                "hidden_dim": 64,
                "output_dim": classes,
                "num_layers": 2,
            },
            epochs=1,
            use_kernel=use_kernel,
            kernel_backend="cpu",
            track_energy=False,
        )
        trainer = CoreTrainer(cfg)
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

    kernel_acc = _final_acc(True)
    ref_acc = _final_acc(False)
    assert abs(kernel_acc - ref_acc) <= 0.01, (
        f"kernel acc={kernel_acc:.3f} vs reference acc={ref_acc:.3f} "
        f"drifted beyond 1% parity"
    )


def test_backprop_kernel_consumed_in_train_step():
    """The generalized consumer binds ``backprop_mlp``'s ``.net`` Linear stack
    and drives it through the attached Backprop kernel (REFACTOR7 Phase 10).

    ``backprop_mlp`` has no ``.layers`` attribute — its stack lives on
    ``.net`` — so this guards the consumer's generic ``_resolve_kernel_layers``
    fallback (uniform backends are now consumable beyond ``standard_fa``).
    """
    import torch

    config = TrainerConfig(
        model="backprop_mlp",
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    backend = getattr(trainer.model, "_kernel_backend", None)
    assert backend is not None

    before = [p.clone() for p in trainer.model.net[0].parameters()]
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    metrics = trainer._train_step(x, y)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    after = list(trainer.model.net[0].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "backprop kernel train step did not update the model weights"
    )
    assert trainer._training_path_counts.get("kernel", 0) >= 1


def test_backprop_kernel_learns():
    """The kernel-driven backprop path genuinely learns on a separable task.

    ``backprop_mlp`` is trained through ``_train_step`` with ``use_kernel=True``
    (the generic consumer routes the Backprop kernel's exact BPTT gradients
    into the model's weights). Accuracy must rise well above chance, proving
    the uniform-interface consumer is a real learning path for a second family.
    """
    import torch

    torch.manual_seed(7)
    n, dim, classes = 400, 64, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, classes, (n,))
    for c in range(classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    cfg = TrainerConfig(
        model="backprop_mlp",
        task="digits",
        model_kwargs={
            "input_dim": dim,
            "hidden_dim": 64,
            "output_dim": classes,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(cfg)
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
        acc = (trainer.model(xd).argmax(1) == yd).float().mean().item()
    assert acc >= 0.2, (
        f"kernel-driven backprop reached only {acc:.3f} accuracy (chance=0.1)"
    )


def test_kernel_uses_trainer_optimizer_lr():
    """The consumed kernel path trains through the trainer's optimizer, not a
    raw-SGD fallback at ``config.learning_rate``.

    Guards the documented REFACTOR7 open item (uniform models **without** a
    model-side ``optimizer`` — here ``backprop_mlp`` — should learn at the
    reference optimizer's LR, so ``_train_step`` eagerly ensures the standard
    optimizer when a kernel backend drives the model). We set ``optimizer=
    "sgd"`` at lr=0.5: a raw-SGD fallback at ``lr_for(model)`` (config LR,
    ~1e-3) would move weights ~500x less, proving the kernel gradients route
    through the trainer optimizer.
    """
    import torch

    config = TrainerConfig(
        model="backprop_mlp",
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
        optimizer="sgd",
        optimizer_kwargs={"lr": 0.5},
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    assert getattr(trainer.model, "_kernel_backend", None) is not None
    assert getattr(trainer.model, "optimizer", None) is None

    before = trainer.model.net[0].weight.data.clone()
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    trainer._train_step(x, y)
    assert trainer.optimizer is not None
    assert type(trainer.optimizer).__name__ == "SGD"
    assert trainer.optimizer.param_groups[0]["lr"] == 0.5
    delta = (trainer.model.net[0].weight.data - before).abs().max().item()
    assert delta > 0.05, (
        f"kernel path did not apply the trainer optimizer LR (max delta={delta:.3e})"
    )


def test_bespoke_kernel_train_step_consumed():
    """A bespoke-dynamics backend (``kernel_train_step``) is driven through the
    dispatch seam, not the uniform consumer.

    PEPITA is a forward-only local learner: its two-pass dynamics can't fit
    ``forward(x) → backward(acts, error) → update_weights``, so the backend
    exposes ``kernel_train_step`` and the dispatcher prefers it. Guards the
    REFACTOR7 bespoke-family consumption wiring.
    """
    import torch

    config = TrainerConfig(
        model="pepita",
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
            "learning_rate": 0.3,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    backend = getattr(trainer.model, "_kernel_backend", None)
    assert backend is not None
    assert hasattr(backend, "kernel_train_step")

    before = [p.data.clone() for p in trainer.model.layers[0].parameters()]
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    metrics = trainer._train_step(x, y)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    after = [p.data.clone() for p in trainer.model.layers[0].parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "kernel_train_step did not update the model weights"
    )
    assert trainer._training_path_counts.get("kernel", 0) >= 1


def test_kernel_pepita_learns():
    """Kernel-driven PEPITA learns on a separable task (above chance), matching
    the reference's forward-only update dynamics via ``kernel_train_step``.

    The Bespoke family mirrors the model's ``train_step`` exactly, so its
    accuracy must rise well above chance — proving the bespoke seam is a real
    learning path, not a no-op.
    """
    import torch

    torch.manual_seed(7)
    n, dim, classes = 400, 64, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, classes, (n,))
    for c in range(classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    cfg = TrainerConfig(
        model="pepita",
        task="digits",
        model_kwargs={
            "input_dim": dim,
            "hidden_dim": 64,
            "output_dim": classes,
            "num_layers": 2,
            "learning_rate": 0.3,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(cfg)
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
        acc = (trainer.model(xd).argmax(1) == yd).float().mean().item()
    assert acc >= 0.3, (
        f"kernel-driven PEPITA reached only {acc:.3f} accuracy (chance=0.1)"
    )


def test_tp_kernel_train_step_consumed():
    """The ``TPKernelBackend`` bespoke step is driven through the dispatch
    seam for ``diff_target_prop``.

    Target Propagation is a layerwise learner whose per-layer optimizers and
    inverse-net target propagation can't fit the uniform forward/backward
    contract, so its backend exposes ``kernel_train_step`` mirroring the
    reference DTP dynamics, and the dispatcher prefers it.
    """
    import torch

    config = TrainerConfig(
        model="diff_target_prop",
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
            "learning_rate": 1e-3,
            "target_lr": 0.1,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    backend = getattr(trainer.model, "_kernel_backend", None)
    assert backend is not None
    assert hasattr(backend, "kernel_train_step")

    # diff_target_prop owns per-layer optimizers; grab a sampled first-layer
    # forward weight before / after the step.
    layer0 = trainer.model.layers[0]
    before = [p.data.clone() for p in layer0.forward_net.parameters()]
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    metrics = trainer._train_step(x, y)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    after = [p.data.clone() for p in layer0.forward_net.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "TP kernel_train_step did not update the model weights"
    )
    assert trainer._training_path_counts.get("kernel", 0) >= 1


def test_kernel_tp_learns():
    """Kernel-driven Difference Target Propagation learns on a separable task.

    ``diff_target_prop`` is consumed through the bespoke ``kernel_train_step``
    seam (which mirrors the reference DTP target-propagation dynamics). Its
    accuracy must rise above chance, and stay within 1% of the reference path.
    """
    import torch

    torch.manual_seed(7)
    n, dim, classes = 400, 64, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, classes, (n,))
    for c in range(classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    def _final_acc(use_kernel: bool) -> float:
        cfg = TrainerConfig(
            model="diff_target_prop",
            task="digits",
            model_kwargs={
                "input_dim": dim,
                "hidden_dim": 64,
                "output_dim": classes,
                "num_layers": 2,
                "learning_rate": 1e-3,
                "target_lr": 0.1,
            },
            epochs=1,
            use_kernel=use_kernel,
            kernel_backend="cpu",
            track_energy=False,
        )
        trainer = CoreTrainer(cfg)
        trainer.setup()
        dev = next(trainer.model.parameters()).device
        xd, yd = x.to(dev), y.to(dev)
        for _ in range(6):
            perm = torch.randperm(n)
            for i in range(0, n, 32):
                idx = perm[i : i + 32]
                trainer._train_step(xd[idx], yd[idx])
        trainer.model.eval()
        with torch.no_grad():
            return (trainer.model(xd).argmax(1) == yd).float().mean().item()

    kernel_acc = _final_acc(True)
    ref_acc = _final_acc(False)
    assert abs(kernel_acc - ref_acc) <= 0.01, (
        f"TP kernel acc={kernel_acc:.3f} vs reference acc={ref_acc:.3f} "
        f"drifted beyond 1% parity"
    )
    assert kernel_acc >= 0.2, (
        f"kernel-driven TP reached only {kernel_acc:.3f} accuracy (chance=0.1)"
    )
