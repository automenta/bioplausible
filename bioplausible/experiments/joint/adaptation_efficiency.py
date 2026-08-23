"""Adaptation Efficiency Benchmark (Level 1).

Question: Does plasticity adapt faster than Null?

Toy Task: Switching input distribution
- Phase A: y = f_A(x) (e.g., classify by cumulative sum)
- Phase B: y = f_B(x) (e.g., classify by last symbol)

Compare: Null vs FastWeight vs Routing vs SubstrateCoupled
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn


def create_switching_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    phase: str = "A",
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create switching task data.

    Phase A: Classify by cumulative sum (sum of sequence > 0 -> class 1)
    Phase B: Classify by last symbol (last element > 0 -> class 1)
    """
    device = torch.device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    if phase == "A":
        cumsum = x.sum(dim=1)
        y = (cumsum.mean(dim=-1) > 0).long()
    else:
        last = x[:, -1, :]
        y = (last.mean(dim=-1) > 0).long()

    return x, y


class CompositeState:
    """Simple composite state for benchmarking."""

    def __init__(self, activity, plastic, substrate):
        self.activity = activity
        self.plastic = plastic
        self.substrate = substrate


class PlasticityModulatedModel(nn.Module):
    """Model where plasticity directly modulates the forward pass."""

    def __init__(self, input_dim: int, plasticity, hidden_dim: int = 64):
        super().__init__()
        self.plasticity = plasticity
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.psi = plasticity.initial_psi(
            None, batch_size=1
        )  # Will be expanded in forward

        # Base weights (θ)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

        # Plasticity-specific components
        if hasattr(plasticity, "gate_dim"):
            # Routing plasticity: gating on hidden units
            self.route_gate = nn.Linear(hidden_dim, plasticity.gate_dim)
        elif hasattr(plasticity, "fast_weight_dim"):
            # Fast weights: additive modulation
            self.fast_weight_proj = nn.Linear(plasticity.fast_weight_dim, hidden_dim)
        elif hasattr(plasticity, "num_operators"):
            # Rule state: operator selection
            self.operator_proj = nn.Linear(plasticity.operator_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, input_dim]
        batch_size = x.shape[0]
        device = x.device

        # Ensure psi is on correct device and batch size
        if self.psi:
            new_psi = {}
            for k, v in self.psi.items():
                if v.shape[0] != batch_size:
                    if v.shape[0] == 1:
                        new_psi[k] = v.expand(batch_size, -1).to(device)
                    else:
                        new_psi[k] = v[:batch_size].to(device)
                else:
                    new_psi[k] = v.to(device)
            self.psi = new_psi

        h = torch.relu(self.fc1(x))

        # Apply plasticity modulation
        if self.psi:
            z = CompositeState(
                activity={"x": x, "h": h}, plastic=self.psi, substrate={}
            )
            self.psi = self.plasticity.step(self.psi, z, None)

            # Modulate hidden state based on plasticity type
            if hasattr(self.plasticity, "gate_dim"):
                # Routing: gate hidden units
                gate_logits = self.psi.get(
                    "gate_logits",
                    torch.zeros(batch_size, self.plasticity.gate_dim, device=device),
                )
                active_routes = self.psi.get(
                    "active_routes", torch.ones_like(gate_logits)
                )
                # Apply gating to hidden units (expand to hidden_dim)
                gate = (
                    active_routes[:, : h.shape[1]]
                    if active_routes.shape[1] >= h.shape[1]
                    else torch.cat(
                        [
                            active_routes,
                            torch.ones(
                                batch_size,
                                h.shape[1] - active_routes.shape[1],
                                device=device,
                            ),
                        ],
                        dim=1,
                    )
                )
                h = h * gate
            elif hasattr(self.plasticity, "fast_weight_dim"):
                # Fast weights: additive modulation
                fast_weights = self.psi.get(
                    "fast_weights",
                    torch.zeros(
                        batch_size, self.plasticity.fast_weight_dim, device=device
                    ),
                )
                modulation = self.fast_weight_proj(fast_weights)
                h = h + modulation
            elif hasattr(self.plasticity, "num_operators"):
                # Rule state: operator application
                operator_logits = self.psi.get(
                    "operator_logits",
                    torch.zeros(
                        batch_size, self.plasticity.num_operators, device=device
                    ),
                )
                active_operator = self.plasticity.get_active_operator(
                    operator_logits, self.training
                )
                # Use operator embeddings to modulate
                op_emb = self.plasticity.operator_embeddings  # [num_ops, op_dim]
                combined_op = active_operator @ op_emb  # [batch, op_dim]
                if combined_op.shape[1] == self.plasticity.operator_dim:
                    modulation = self.operator_proj(combined_op)
                    h = h + modulation

        return self.fc2(h)


def evaluate_adaptation(
    coordinate: str,
    epochs_per_phase: int = 50,
    batch_size: int = 64,
    seq_len: int = 10,
    input_dim: int = 32,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Evaluate adaptation efficiency for a coordinate."""
    from bioplausible.core.joint.transition import NullPlasticity, PlasticityConfig
    from bioplausible.core.plasticity import (
        create_fast_weight_plasticity,
        create_routing_plasticity,
        create_rule_state_plasticity,
        create_substrate_coupled_plasticity,
    )

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device)

    parts = coordinate.split("/")
    if len(parts) != 6:
        raise ValueError(f"Invalid coordinate: {coordinate}")

    plasticity_type = parts[3]

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

    model = PlasticityModulatedModel(input_dim, plasticity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Phase A training
    phase_a_losses = []
    for epoch in range(epochs_per_phase):
        x, y = create_switching_task(batch_size, seq_len, input_dim, "A", device)
        x = x.mean(dim=1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        phase_a_losses.append(loss.item())

    # Phase B training (adaptation phase)
    phase_b_losses = []
    adaptation_times = []

    for epoch in range(epochs_per_phase):
        x, y = create_switching_task(batch_size, seq_len, input_dim, "B", device)
        x = x.mean(dim=1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        phase_b_losses.append(loss.item())

        if loss.item() < 0.5 and len(adaptation_times) == 0:
            adaptation_times.append(epoch)

    adaptation_time = adaptation_times[0] if adaptation_times else epochs_per_phase

    # Final accuracy on Phase B
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(10):
            x, y = create_switching_task(batch_size, seq_len, input_dim, "B", device)
            x = x.mean(dim=1)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]

    final_accuracy = correct / total

    return {
        "coordinate": coordinate,
        "phase_a_final_loss": phase_a_losses[-1],
        "phase_b_final_loss": phase_b_losses[-1],
        "adaptation_time": adaptation_time,
        "final_accuracy": final_accuracy,
        "phase_a_losses": phase_a_losses,
        "phase_b_losses": phase_b_losses,
    }


def run_adaptation_efficiency_suite(
    coordinates: list[str],
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    seeds: int = 3,
    device: str = "auto",
) -> list[dict]:
    """Run adaptation efficiency benchmark suite."""
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device

    all_results = []

    for coord in coordinates:
        print(f"\nEvaluating: {coord}")
        coord_results = {"coordinate": coord, "seeds": []}

        for seed in range(seeds):
            print(f"  Seed {seed}...")
            result = evaluate_adaptation(
                coordinate=coord,
                epochs_per_phase=epochs,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
            coord_results["seeds"].append(result)
            print(
                f"    Adaptation time: {result['adaptation_time']}, Acc: {result['final_accuracy']:.4f}"
            )

        if coord_results["seeds"]:
            adapt_times = [s["adaptation_time"] for s in coord_results["seeds"]]
            accuracies = [s["final_accuracy"] for s in coord_results["seeds"]]
            coord_results["mean_adaptation_time"] = sum(adapt_times) / len(adapt_times)
            coord_results["std_adaptation_time"] = (
                (
                    sum(
                        (a - coord_results["mean_adaptation_time"]) ** 2
                        for a in adapt_times
                    )
                    / len(adapt_times)
                )
                ** 0.5
                if len(adapt_times) > 1
                else 0
            )
            coord_results["mean_accuracy"] = sum(accuracies) / len(accuracies)
            coord_results["std_accuracy"] = (
                (
                    sum((a - coord_results["mean_accuracy"]) ** 2 for a in accuracies)
                    / len(accuracies)
                )
                ** 0.5
                if len(accuracies) > 1
                else 0
            )

        all_results.append(coord_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "adaptation_efficiency_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 80)
    print("Adaptation Efficiency Benchmark Summary")
    print("=" * 80)
    print(f"{'Coordinate':<50} {'Mean Adapt Time':<15} {'Mean Acc':<10} {'Plasticity'}")
    print("-" * 80)
    for r in all_results:
        coord_short = (
            r["coordinate"][:48] + ".."
            if len(r["coordinate"]) > 50
            else r["coordinate"]
        )
        prim = r["coordinate"].split("/")[3]
        print(
            f"{coord_short:<50} {r.get('mean_adaptation_time', 0):<15.1f} {r.get('mean_accuracy', 0):<10.4f} {prim}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Adaptation Efficiency Benchmark (Level 1)"
    )
    parser.add_argument("--coordinates", nargs="+", help="Coordinates to test")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/adaptation_efficiency",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per phase")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (3 epochs, 1 seed)"
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 3
        args.seeds = 1

    coordinates = args.coordinates or [
        "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
        "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
        "digital/recurrent/energy_minimization/fast_weights/thermodynamic_contrast/euclidean",
        "digital/recurrent/energy_minimization/substrate_coupled/thermodynamic_contrast/euclidean",
    ]

    run_adaptation_efficiency_suite(
        coordinates=coordinates,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
