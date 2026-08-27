#!/usr/bin/env python3
"""Verify eqprop_mlp passing config consistently."""

import torch
from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP


def train_and_eval(hebbian_lr=0.01, beta=0.05, max_steps=20, seed=456):
    torch.manual_seed(42)
    n_samples = 500
    input_dim = 64
    n_classes = 10
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(input_dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    # Backprop baseline
    torch.manual_seed(123)
    bp_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_classes),
    )
    opt = torch.optim.Adam(bp_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    bp_model.train()
    for _ in range(3):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i : i + 32]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            logits = bp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    bp_model.eval()
    with torch.no_grad():
        logits = bp_model(x)
        backprop_baseline = (logits.argmax(1) == y).float().mean().item()

    # EqProp
    torch.manual_seed(seed)
    model = LoopedMLP(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=n_classes,
        use_spectral_norm=True,
        max_steps=max_steps,
        gradient_method="contrastive",
        backend="pytorch",
    )
    model.hebbian_lr = hebbian_lr
    model.beta = beta

    model.train()
    for epoch in range(3):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i : i + 32]
            xb, yb = x[idx], y[idx]
            model.train_step(xb, yb)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        bio_acc = (logits.argmax(1) == y).float().mean().item()

    diff = backprop_baseline - bio_acc
    return {
        "hebbian_lr": hebbian_lr,
        "beta": beta,
        "max_steps": max_steps,
        "seed": seed,
        "bio_acc": bio_acc,
        "backprop_baseline": backprop_baseline,
        "diff": diff,
        "passed": diff <= 0.05,
    }


# Test the passing config with different seeds
print("Testing passing config with different seeds:")
for seed in [456, 457, 458, 459, 460, 123, 789, 999]:
    result = train_and_eval(hebbian_lr=0.01, beta=0.05, max_steps=20, seed=seed)
    status = "✓ PASS" if result["passed"] else "✗ FAIL"
    print(
        f"  {status} seed={seed} diff={result['diff']:.3f} bio={result['bio_acc']:.3f}"
    )

print("\nTesting nearby configs with seed=456:")
for hebbian_lr in [0.005, 0.008, 0.01, 0.015, 0.02]:
    for beta in [0.03, 0.05, 0.08, 0.1]:
        for max_steps in [15, 20, 25, 30]:
            result = train_and_eval(
                hebbian_lr=hebbian_lr, beta=beta, max_steps=max_steps, seed=456
            )
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(
                f"  {status} hebbian_lr={hebbian_lr} beta={beta} max_steps={max_steps} diff={result['diff']:.3f}"
            )
