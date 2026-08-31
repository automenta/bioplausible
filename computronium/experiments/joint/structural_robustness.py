"""Structural Robustness Benchmark (Level 3).

Question: Can system recover after damage?

Toy Task: Damage recovery
- Zeroed weights, removed nodes, dead channels, noisy memristive states

Compare: Null vs Routing vs SubstrateCoupled
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from computronium.core.utils.device import get_device

from computronium.core.profiling import measure_suite_resources
from computronium.experiments.joint import CLAIMS_SCOPE_PLUMBING_ONLY


def create_damage_scenarios(
    model: torch.nn.Module, damage_type: str, severity: float = 0.5
) -> dict:
    """Apply damage to model and return original state for recovery.

    Args:
        model: Model to damage
        damage_type: Type of damage ("zero_weights", "remove_nodes", "noise")
        severity: Fraction of weights/nodes to damage (0-1)

    Returns:
        Dict with original state for recovery
    """
    original_state = {}

    for name, param in model.named_parameters():
        original_state[name] = param.data.clone()

        if damage_type == "zero_weights":
            # Zero out random weights
            mask = torch.rand_like(param) < severity
            param.data[mask] = 0

        elif damage_type == "remove_nodes":
            # Zero out entire output neurons (for Linear layers)
            if param.dim() == 2:  # weight matrix [out_features, in_features]
                num_neurons = param.shape[0]
                num_damage = int(num_neurons * severity)
                damage_indices = torch.randperm(num_neurons)[:num_damage]
                param.data[damage_indices] = 0

        elif damage_type == "noise":
            # Add noise to weights
            noise = torch.randn_like(param) * severity * param.data.std()
            param.data += noise

        elif damage_type == "dead_channels":
            # For conv-like: zero out entire channels
            if param.dim() >= 2:
                num_channels = param.shape[0]
                num_damage = int(num_channels * severity)
                damage_indices = torch.randperm(num_channels)[:num_damage]
                param.data[damage_indices] = 0

    return original_state


def evaluate_recovery(
    model: torch.nn.Module,
    original_state: dict,
    damage_type: str,
    recovery_steps: int,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
) -> dict:
    """Evaluate recovery after damage."""

    # Measure initial performance after damage
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    initial_accuracy = correct / total if total > 0 else 0

    # Recovery training
    model.train()
    recovery_losses = []
    recovery_accuracies = []

    for step in range(recovery_steps):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=-1) == y).sum().item()
            epoch_total += y.shape[0]

        recovery_losses.append(epoch_loss / len(train_loader))
        recovery_accuracies.append(
            epoch_correct / epoch_total if epoch_total > 0 else 0
        )

    # Final accuracy
    final_accuracy = recovery_accuracies[-1] if recovery_accuracies else 0

    # Recovery metric: how much accuracy recovered relative to original
    # We need original accuracy - approximate from pre-damage
    recovery_ratio = final_accuracy / initial_accuracy if initial_accuracy > 0 else 0

    return {
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "recovery_ratio": recovery_ratio,
        "recovery_losses": recovery_losses,
        "recovery_accuracies": recovery_accuracies,
    }


def evaluate_structural_robustness(
    coordinate: str,
    epochs: int = 10,
    batch_size: int = 64,
    input_dim: int = 64,
    hidden_dim: int = 128,
    output_dim: int = 10,
    recovery_steps: int = 20,
    damage_severity: float = 0.3,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Evaluate structural robustness for a coordinate."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    random.seed(seed)
    device = get_device(device)
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

    # Simple MLP model
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleMLP().to(device)

    # Pre-train on synthetic task
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate synthetic training data
    train_x = torch.randn(1000, input_dim, device=device)
    train_y = (train_x.sum(dim=-1) > 0).long() % output_dim
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Save original model state after pre-training
    original_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Measure pre-damage accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in train_loader:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    pre_damage_accuracy = correct / total

    # Apply multiple damage types and measure recovery
    damage_types = ["zero_weights", "remove_nodes", "noise"]
    damage_results = {}

    for damage_type in damage_types:
        # Restore original model
        model.load_state_dict(original_model_state)

        # Apply damage
        original_state = create_damage_scenarios(model, damage_type, damage_severity)

        # Evaluate recovery
        recovery = evaluate_recovery(
            model,
            original_state,
            damage_type,
            recovery_steps,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        damage_results[damage_type] = recovery

    # Average recovery across damage types
    avg_recovery_ratio = sum(
        r["recovery_ratio"] for r in damage_results.values()
    ) / len(damage_results)
    avg_final_accuracy = sum(
        r["final_accuracy"] for r in damage_results.values()
    ) / len(damage_results)

    return {
        "claims_scope": CLAIMS_SCOPE_PLUMBING_ONLY,
        "coordinate": coordinate,
        "pre_damage_accuracy": pre_damage_accuracy,
        "damage_severity": damage_severity,
        "damage_results": damage_results,
        "avg_recovery_ratio": avg_recovery_ratio,
        "avg_final_accuracy": avg_final_accuracy,
        "resources": measure_suite_resources(
            model=model,
            coordinate=coordinate,
            device=str(device),
            batch_size=batch_size,
            elapsed_s=time.perf_counter() - start_time,
        ).to_dict(),
    }


def run_structural_robustness_suite(
    coordinates: list[str],
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    recovery_steps: int = 20,
    damage_severity: float = 0.3,
    seeds: int = 3,
    device: str = "auto",
) -> list[dict]:
    """Run structural robustness benchmark suite."""
    device = get_device(device)

    all_results = []

    for coord in coordinates:
        print(f"\nEvaluating: {coord}")
        coord_results = {"coordinate": coord, "seeds": []}

        for seed in range(seeds):
            print(f"  Seed {seed}...")
            result = evaluate_structural_robustness(
                coordinate=coord,
                epochs=epochs,
                batch_size=batch_size,
                recovery_steps=recovery_steps,
                damage_severity=damage_severity,
                device=device,
                seed=seed,
            )
            coord_results["seeds"].append(result)
            print(
                f"    Pre-damage: {result['pre_damage_accuracy']:.4f}, Recovery: {result['avg_recovery_ratio']:.4f}"
            )

        # Aggregate
        if coord_results["seeds"]:
            recovery_ratios = [s["avg_recovery_ratio"] for s in coord_results["seeds"]]
            final_accs = [s["avg_final_accuracy"] for s in coord_results["seeds"]]
            coord_results["mean_recovery_ratio"] = sum(recovery_ratios) / len(
                recovery_ratios
            )
            coord_results["std_recovery_ratio"] = (
                (
                    sum(
                        (r - coord_results["mean_recovery_ratio"]) ** 2
                        for r in recovery_ratios
                    )
                    / len(recovery_ratios)
                )
                ** 0.5
                if len(recovery_ratios) > 1
                else 0
            )
            coord_results["mean_final_accuracy"] = sum(final_accs) / len(final_accs)

        all_results.append(coord_results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "structural_robustness_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Structural Robustness Benchmark Summary")
    print("=" * 80)
    print(
        f"{'Coordinate':<50} {'Pre-Dmg Acc':<12} {'Recovery Ratio':<14} {'Final Acc':<10} {'Plasticity'}"
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
            f"{coord_short:<50} {r['seeds'][0]['pre_damage_accuracy']:<12.4f} {r.get('mean_recovery_ratio', 0):<14.4f} {r.get('mean_final_accuracy', 0):<10.4f} {prim}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Structural Robustness Benchmark (Level 3)"
    )
    parser.add_argument("--coordinates", nargs="+", help="Coordinates to test")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/structural_robustness",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Pre-training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--recovery-steps", type=int, default=20, help="Recovery training steps"
    )
    parser.add_argument(
        "--damage-severity", type=float, default=0.3, help="Damage severity (0-1)"
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (3 epochs, 1 seed)"
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 3
        args.recovery_steps = 5
        args.seeds = 1

    coordinates = args.coordinates or [
        "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
        "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
        "memristive/recurrent/energy_minimization/substrate_coupled/thermodynamic_contrast/euclidean",
        "neuromorphic/recurrent/spike_integration/null/thermodynamic_contrast/euclidean",
    ]

    run_structural_robustness_suite(
        coordinates=coordinates,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        recovery_steps=args.recovery_steps,
        damage_severity=args.damage_severity,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
