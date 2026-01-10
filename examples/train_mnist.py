#!/usr/bin/env python3
"""
Example: Training on MNIST with EqProp

Demonstrates the high-level EqPropTrainer API with torch.compile acceleration.
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add parent directory to path to allow importing without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from eqprop_torch library
from eqprop_torch import EqPropTrainer, LoopedMLP
from eqprop_torch.datasets import get_vision_dataset, create_data_loaders


def main():
    # Configuration
    HIDDEN_DIM = 256
    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 0.001
    
    print("=" * 60)
    print("EqProp-Torch: MNIST Training Example")
    print("=" * 60)
    
    # Create model with spectral normalization (required for stability)
    model = LoopedMLP(
        input_dim=784,
        hidden_dim=HIDDEN_DIM,
        output_dim=10,
        use_spectral_norm=True,  # Guarantees L < 1
        max_steps=30,
    )
    print(f"Model: LoopedMLP({784} → {HIDDEN_DIM} → {10})")
    print(f"Initial Lipschitz: {model.compute_lipschitz():.4f}")
    
    # Create trainer with torch.compile for 2x speedup
    trainer = EqPropTrainer(
        model,
        lr=LR,
        use_compile=True,  # Enable torch.compile (portable to CPU/CUDA/MPS)
    )
    print(f"Device: {trainer.device}")
    
    # Load MNIST dataset
    train_loader, test_loader = create_data_loaders(
        'mnist', 
        batch_size=BATCH_SIZE, 
        flatten=True,  # Flatten for MLP
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print("\nTraining...")
    print("-" * 60)
    
    # Callback for progress
    def on_epoch(metrics):
        print(f"Epoch {metrics['epoch']:3d}/{EPOCHS} | "
              f"Loss: {metrics['train_loss']:.4f} | "
              f"Acc: {metrics['train_acc']:.1%}")
    
    # Train
    history = trainer.fit(
        train_loader,
        epochs=EPOCHS,
        val_loader=test_loader,
        callback=on_epoch,
    )
    
    print("-" * 60)
    
    # Final evaluation
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Final Test Loss: {test_metrics['loss']:.4f}")
    print(f"Final Lipschitz: {trainer.compute_lipschitz():.4f}")
    
    # Save model
    trainer.save_checkpoint("mnist_eqprop.pt")
    print("\nCheckpoint saved to: mnist_eqprop.pt")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
