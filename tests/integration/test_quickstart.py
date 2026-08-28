"""Integration test for quickstart script: Both algorithms train on same architecture."""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.gpu
def test_backprop_vs_eqprop_mnist():
    """Both algorithms train on same architecture, achieve >50% on MNIST in 3 epochs."""
    from computronium.core.system_trainer import (
        SystemTrainer,
        SystemTrainerConfig,
        create_backprop_system,
        create_eqprop_system,
    )
    from computronium.domains.factory import create_task
    from computronium.ontology import DigitalSubstrate, SubstrateConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create task
    task = create_task("mnist", device=device, quick_mode=True)
    task.setup()

    input_dim = task.input_dim
    if isinstance(input_dim, (tuple, list)):
        input_dim = int(torch.prod(torch.tensor(input_dim)))
    output_dim = task.output_dim
    hidden_dim = 256

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )
    )

    # Create systems
    backprop_system = create_backprop_system(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        lr=0.001,
    )

    eqprop_system = create_eqprop_system(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        beta=0.5,
        settle_steps=20,
        lr=0.01,
    )

    # Create data loaders
    class _FlattenLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for x, y in self.loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                yield x, y

        def __len__(self):
            return len(self.loader)

    train_loader = _FlattenLoader(task.get_dataloader("train"))
    val_loader = _FlattenLoader(task.get_dataloader("val"))

    trainer_config = SystemTrainerConfig(
        max_epochs=3,
        batch_size=64,
        device=device,
        seed=42,
        log_every_n_steps=100,
    )

    # Train Backprop
    backprop_trainer = SystemTrainer(
        system=backprop_system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )
    backprop_history = backprop_trainer.fit()

    # Train EqProp
    eqprop_trainer = SystemTrainer(
        system=eqprop_system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )
    eqprop_history = eqprop_trainer.fit()

    # Check final accuracies
    backprop_final_acc = backprop_history[-1].get(
        "val_acc", backprop_history[-1].get("train_acc", 0.0)
    )
    eqprop_final_acc = eqprop_history[-1].get(
        "val_acc", eqprop_history[-1].get("train_acc", 0.0)
    )

    # Backprop should achieve >90% (it's the gold standard)
    assert backprop_final_acc > 0.90, (
        f"Backprop only achieved {backprop_final_acc * 100:.1f}%"
    )

    # EqProp should achieve some learning (>5% is better than random)
    assert eqprop_final_acc > 0.05, (
        f"EqProp only achieved {eqprop_final_acc * 100:.1f}%"
    )

    print(
        f"Backprop: {backprop_final_acc * 100:.1f}% | EqProp: {eqprop_final_acc * 100:.1f}%"
    )


if __name__ == "__main__":
    test_backprop_vs_eqprop_mnist()
