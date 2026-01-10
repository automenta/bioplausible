
"""
Simple Transfer Learning Example using TorEqProp
================================================

This script demonstrates how to:
1. Train an Equilibrium Propagation model on a source task.
2. Save the model.
3. Load the model and fine-tune it on a target task.

Usage:
    python simple_transfer.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import local modules
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models import LoopedMLP
from validation.utils import create_synthetic_dataset, train_model, evaluate_accuracy

def main():
    print("🚀 TorEqProp Transfer Learning Demo")
    print("-----------------------------------")
    
    # 1. Setup Data
    # Task A: 5 classes
    # Task B: 5 classes
    print("\n[1] Generating Synthetic Data...")
    X, y = create_synthetic_dataset(n_samples=500, input_dim=64, n_classes=10)
    
    mask_A = y < 5
    X_A, y_A = X[mask_A], y[mask_A]
    
    mask_B = y >= 5
    X_B, y_B = X[mask_B], y[mask_B] - 5 # Shift labels to 0-4
    
    # 2. Pre-train Task A
    print("\n[2] Pre-training on Task A (Classes 0-4)...")
    model = LoopedMLP(input_dim=64, hidden_dim=128, output_dim=5, use_spectral_norm=True)
    train_model(model, X_A, y_A, epochs=5, lr=0.01)
    acc_A = evaluate_accuracy(model, X_A, y_A)
    print(f"    Task A Accuracy: {acc_A*100:.1f}%")
    
    # 3. Simulate Save/Load
    print("\n[3] Saving Model...")
    torch.save(model.state_dict(), "toreq_pretrained.pt")
    
    print("\n[4] Loading Model for Task B...")
    new_model = LoopedMLP(input_dim=64, hidden_dim=128, output_dim=5, use_spectral_norm=True)
    new_model.load_state_dict(torch.load("toreq_pretrained.pt"))
    
    # Reset readout layer (transfer learning standard practice)
    # Note: In EqProp, we might want to keep weights if semantically related, 
    # but here classes are disjoint.
    nn.init.xavier_uniform_(new_model.W_out.parametrizations.weight.original)
    nn.init.zeros_(new_model.W_out.bias)
    
    # 4. Fine-tune on Task B
    print("\n[5] Fine-tuning on Task B (Classes 5-9)...")
    # Optional: Freeze lower layers
    # new_model.W_in.requires_grad_(False)
    
    train_model(new_model, X_B, y_B, epochs=5, lr=0.01)
    acc_B = evaluate_accuracy(new_model, X_B, y_B)
    print(f"    Task B Accuracy: {acc_B*100:.1f}%")
    
    print("\n✨ Transfer Complete!")

if __name__ == "__main__":
    main()
