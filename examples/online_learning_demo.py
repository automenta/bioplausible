"""
Online Learning Demo

Demonstrates how to use the SupervisedTrainer for online learning (streaming data)
without the rigid Task/Dataset structure. This allows integration into other systems
where data arrives sequentially.
"""

import torch
import numpy as np
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.models.standard_eqprop import StandardEqProp

def main():
    print("Initializing Online Learning Demo...")

    # 1. Define Model
    # We use StandardEqProp which is a multi-layer EqProp network
    input_dim = 20
    output_dim = 2
    hidden_dim = 64

    model = StandardEqProp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        # Config via kwargs for convenience in BioModel
        learning_rate=0.01,
        beta=0.5,
        equilibrium_steps=20
    )

    # 2. Initialize Trainer without a Task
    # We specify task_type="vision" (generic vector input) so it knows how to handle input
    trainer = SupervisedTrainer(
        model=model,
        task=None,         # No Task object needed!
        task_type="vision",
        device="cpu",      # or "cuda"
        use_compile=False  # Disable compile for simple demo speed
    )

    print("Trainer initialized. Starting online training loop...")

    # 3. Simulated Data Stream
    # Let's learn a simple XOR-like pattern or random projection
    # Target: Class 0 if sum(x) > 0 else Class 1

    losses = []
    accuracies = []

    n_samples = 1000

    for i in range(n_samples):
        # Generate single sample (Online)
        x = torch.randn(1, input_dim)
        y_val = 1 if x.sum().item() > 0 else 0
        y = torch.tensor([y_val])

        # Train on single sample (or small batch)
        metrics = trainer.train_batch(x, y)

        loss = metrics["loss"]
        acc = metrics["accuracy"]

        losses.append(loss)
        accuracies.append(acc)

        if i % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if i > 0 else loss
            avg_acc = np.mean(accuracies[-100:]) if i > 0 else acc
            print(f"Step {i}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

    print("\nTraining Complete.")
    print(f"Final Average Accuracy (last 100): {np.mean(accuracies[-100:]):.4f}")

    # Verify inference
    test_x = torch.randn(10, input_dim)
    test_y = (test_x.sum(dim=1) > 0).long()

    model.eval()
    with torch.no_grad():
        out = model(test_x)
        preds = out.argmax(dim=1)
        test_acc = (preds == test_y).float().mean().item()

    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
