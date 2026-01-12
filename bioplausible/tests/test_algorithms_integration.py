
import pytest
import torch
from bioplausible.core import EqPropTrainer
from bioplausible.algorithms.eqprop import StandardEqProp
from bioplausible.algorithms.base import AlgorithmConfig

def test_eqprop_algorithm_integration():
    """
    Test that EqPropTrainer can train a StandardEqProp algorithm model.
    """
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 5

    # Create synthetic data
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Configure Algorithm
    config = AlgorithmConfig(
        name='eqprop',
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=output_dim,
        learning_rate=0.01,
        equilibrium_steps=5, # Short for testing
        beta=0.1
    )

    model = StandardEqProp(config)

    # Initialize Trainer
    # We don't need to specify optimizer because StandardEqProp handles it internally
    trainer = EqPropTrainer(model, use_kernel=False, use_compile=False)

    # Initial loss
    initial_metrics = trainer.evaluate(loader)
    initial_loss = initial_metrics['loss']

    # Train
    history = trainer.fit(loader, epochs=5)

    # Final loss
    final_metrics = trainer.evaluate(loader)
    final_loss = final_metrics['loss']

    print(f"Initial Loss: {initial_loss}, Final Loss: {final_loss}")

    # Basic check: Loss should not be NaN and preferably decrease (though with random data/small steps it might vary)
    assert not torch.isnan(torch.tensor(final_loss))
    assert 'train_loss' in history
    assert len(history['train_loss']) == 5

def test_eqprop_dynamics_shapes():
    """Verify shapes during dynamics."""
    config = AlgorithmConfig(
        name='eqprop',
        input_dim=5,
        hidden_dims=[10, 8],
        output_dim=3,
        equilibrium_steps=2
    )
    model = StandardEqProp(config)

    x = torch.randn(2, 5) # Batch 2

    # Forward
    out = model(x)
    assert out.shape == (2, 3)

    # Check states
    activations = model._last_activations
    # Should have: Input, Hidden1, Hidden2, Output
    # Total 4 tensors
    assert len(activations) == 4
    assert activations[0].shape == (2, 5)
    assert activations[1].shape == (2, 10)
    assert activations[2].shape == (2, 8)
    assert activations[3].shape == (2, 3)
