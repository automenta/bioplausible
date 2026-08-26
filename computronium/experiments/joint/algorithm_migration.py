"""Algorithm Migration Benchmark (Level 3.5).

Question: Can ψ switch strategy without θ update?

Toy Task: Task A0 -> Task A1 migration
- Task A0: Classify by cumulative sum
- Task A1: Classify by last symbol

Measure: time(A0->A1), energy(A0->A1), ||θ_after - θ_before|| == 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import Tensor

from computronium.core.profiling import measure_suite_resources
from computronium.experiments.joint import CLAIMS_SCOPE_PLUMBING_ONLY


def create_task_a0(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Task A0: Classify by cumulative sum (sum > 0 -> class 1)."""
    device = torch.device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    cumsum = x.sum(dim=1).mean(dim=-1)  # [batch]
    y = (cumsum > 0).long()
    return x, y


def create_task_a1(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Task A1: Classify by last symbol (last > 0 -> class 1)."""
    device = torch.device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    last = x[:, -1, :].mean(dim=-1)  # [batch]
    y = (last > 0).long()
    return x, y


def evaluate_migration(
    coordinate: str,
    epochs_a0: int = 30,
    epochs_a1: int = 30,
    batch_size: int = 64,
    seq_len: int = 10,
    input_dim: int = 32,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Evaluate algorithm migration for a coordinate."""

    from torch import nn

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device)
    start_time = time.perf_counter()

    parts = coordinate.split("/")
    if len(parts) != 6:
        raise ValueError(f"Invalid coordinate: {coordinate}")

    plasticity_type = parts[3]

    # Build plasticity primitive
    from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
    from computronium.core.plasticity import (
        create_fast_weight_plasticity,
        create_routing_plasticity,
        create_rule_state_plasticity,
        create_substrate_coupled_plasticity,
    )

    plasticity_config = PlasticityConfig(
        plasticity_type=plasticity_type,
        plastic_state_dims={"gate_logits": 64, "active_routes": 64}
        if plasticity_type == "routing"
        else {"fast_weights": 512}
        if plasticity_type == "fast_weights"
        else {"operator_logits": 8}
        if plasticity_type == "rule_state"
        else None,
    )

    if plasticity_type == "null":
        plasticity = NullPlasticity()
    elif plasticity_type == "routing":
        plasticity = create_routing_plasticity(plasticity_config)
    elif plasticity_type == "fast_weights":
        plasticity = create_fast_weight_plasticity(plasticity_config)
    elif plasticity_type == "substrate_coupled":
        plasticity = create_substrate_coupled_plasticity(plasticity_config)
    elif plasticity_type == "rule_state":
        plasticity = create_rule_state_plasticity(plasticity_config)
    else:
        raise ValueError(f"Unknown plasticity: {plasticity_type}")

    # Model with plasticity
    class PlasticityModel(nn.Module):
        def __init__(self, plasticity_primitive):
            super().__init__()
            self.plasticity = plasticity_primitive
            self.psi = plasticity_primitive.initial_psi(None)
            self.input_proj = nn.Linear(input_dim, 64)
            self.hidden = nn.Linear(64, 64)
            self.output = nn.Linear(64, 2)

        def forward(self, x):
            # x: [batch, seq_len, input_dim] -> average over seq
            x = x.mean(dim=1)  # [batch, input_dim]
            x = torch.relu(self.input_proj(x))
            x = torch.relu(self.hidden(x))
            return self.output(x)

        def get_theta_norm(self):
            """Get norm of persistent parameters (θ)."""
            total = 0
            for p in self.parameters():
                if p.requires_grad:
                    total += p.data.norm().item() ** 2
            return total**0.5

    model = PlasticityModel(plasticity).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Record θ before Task A0
    theta_before_a0 = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Train on Task A0
    model.train()
    a0_losses = []
    for epoch in range(epochs_a0):
        x, y = create_task_a0(batch_size, seq_len, input_dim, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        a0_losses.append(loss.item())

    # Record θ after Task A0 (before migration)
    theta_after_a0 = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Evaluate on Task A0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(10):
            x, y = create_task_a0(batch_size, seq_len, input_dim, device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    a0_accuracy = correct / total

    # Migrate to Task A1 (ψ adapts, θ should stay frozen for pure migration)
    # For true migration test, we freeze θ and only allow ψ to adapt
    # But here we allow full training to see how fast it adapts
    model.train()
    a1_losses = []
    migration_times = []

    for epoch in range(epochs_a1):
        x, y = create_task_a1(batch_size, seq_len, input_dim, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        a1_losses.append(loss.item())

        if loss.item() < 0.5 and len(migration_times) == 0:
            migration_times.append(epoch)

    migration_time = migration_times[0] if migration_times else epochs_a1

    # Record θ after Task A1
    theta_after_a1 = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Evaluate on Task A1
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(10):
            x, y = create_task_a1(batch_size, seq_len, input_dim, device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    a1_accuracy = correct / total

    # Compute θ change (should be 0 for pure ψ-mediated migration)
    theta_change = 0
    for name in theta_after_a0:
        if name in theta_after_a1:
            diff = (theta_after_a1[name] - theta_after_a0[name]).norm().item()
            theta_change += diff**2
    theta_change = theta_change**0.5

    # Also check if we can recover Task A0 after Task A1 (catastrophic forgetting)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(10):
            x, y = create_task_a0(batch_size, seq_len, input_dim, device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    a0_accuracy_after_a1 = correct / total

    return {
        "claims_scope": CLAIMS_SCOPE_PLUMBING_ONLY,
        "coordinate": coordinate,
        "a0_accuracy": a0_accuracy,
        "a1_accuracy": a1_accuracy,
        "a0_accuracy_after_a1": a0_accuracy_after_a1,
        "migration_time": migration_time,
        "theta_change": theta_change,
        "theta_change_normalized": theta_change / model.get_theta_norm()
        if model.get_theta_norm() > 0
        else 0,
        "a0_losses": a0_losses,
        "a1_losses": a1_losses,
        "resources": measure_suite_resources(
            model=model,
            coordinate=coordinate,
            device=str(device),
            batch_size=batch_size,
            elapsed_s=time.perf_counter() - start_time,
        ).to_dict(),
    }


def run_algorithm_migration_suite(
    coordinates: list[str],
    output_dir: Path,
    epochs_a0: int = 30,
    epochs_a1: int = 30,
    batch_size: int = 64,
    seeds: int = 3,
    device: str = "auto",
) -> list[dict]:
    """Run algorithm migration benchmark suite."""
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device

    all_results = []

    for coord in coordinates:
        print(f"\nEvaluating: {coord}")
        coord_results = {"coordinate": coord, "seeds": []}

        for seed in range(seeds):
            print(f"  Seed {seed}...")
            result = evaluate_migration(
                coordinate=coord,
                epochs_a0=epochs_a0,
                epochs_a1=epochs_a1,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
            coord_results["seeds"].append(result)
            print(
                f"    A0 acc: {result['a0_accuracy']:.4f}, A1 acc: {result['a1_accuracy']:.4f}, "
                f"Migration: {result['migration_time']}, θ-change: {result['theta_change']:.6f}"
            )

        # Aggregate
        if coord_results["seeds"]:
            a0_accs = [s["a0_accuracy"] for s in coord_results["seeds"]]
            a1_accs = [s["a1_accuracy"] for s in coord_results["seeds"]]
            a0_after = [s["a0_accuracy_after_a1"] for s in coord_results["seeds"]]
            mig_times = [s["migration_time"] for s in coord_results["seeds"]]
            theta_changes = [s["theta_change"] for s in coord_results["seeds"]]

            coord_results["mean_a0_accuracy"] = sum(a0_accs) / len(a0_accs)
            coord_results["mean_a1_accuracy"] = sum(a1_accs) / len(a1_accs)
            coord_results["mean_a0_after_a1"] = sum(a0_after) / len(a0_after)
            coord_results["mean_migration_time"] = sum(mig_times) / len(mig_times)
            coord_results["mean_theta_change"] = sum(theta_changes) / len(theta_changes)

        all_results.append(coord_results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "algorithm_migration_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Algorithm Migration Benchmark Summary (Level 3.5)")
    print("=" * 80)
    print(
        f"{'Coordinate':<50} {'A0 Acc':<8} {'A1 Acc':<8} {'A0->A1':<8} {'θ-change':<10} {'Plasticity'}"
    )
    print("-" * 80)
    for r in all_results:
        coord_short = (
            r["coordinate"][:48] + ".."
            if len(r["coordinate"]) > 50
            else r["coordinate"]
        )
        prim = r["coordinate"].split("/")[3]
        print(
            f"{coord_short:<50} {r.get('mean_a0_accuracy', 0):<8.4f} {r.get('mean_a1_accuracy', 0):<8.4f} "
            f"{r.get('mean_migration_time', 0):<8.1f} {r.get('mean_theta_change', 0):<10.6f} {prim}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Algorithm Migration Benchmark (Level 3.5)"
    )
    parser.add_argument("--coordinates", nargs="+", help="Coordinates to test")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/algorithm_migration",
        help="Output directory",
    )
    parser.add_argument("--epochs-a0", type=int, default=30, help="Epochs for Task A0")
    parser.add_argument("--epochs-a1", type=int, default=30, help="Epochs for Task A1")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (5 epochs each, 1 seed)"
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs_a0 = 5
        args.epochs_a1 = 5
        args.seeds = 1

    coordinates = args.coordinates or [
        "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
        "digital/recurrent/energy_minimization/fast_weights/thermodynamic_contrast/euclidean",
        "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
    ]

    run_algorithm_migration_suite(
        coordinates=coordinates,
        output_dir=Path(args.output_dir),
        epochs_a0=args.epochs_a0,
        epochs_a1=args.epochs_a1,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
