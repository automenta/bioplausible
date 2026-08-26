"""Compute Efficiency Benchmark (Level 2).

Question: Does routing reduce effective ops?

Toy Task: Mixture-of-experts synthetic
- Only one route needed per input
- Compare dense baseline vs routing with sparse activation

Compare: Active units, gate entropy, effective matmul FLOPs
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from computronium.core.profiling import measure_suite_resources
from computronium.experiments.joint import CLAIMS_SCOPE_PSI_WIRED_UNCONTROLLED
import torch.nn.functional as F
from torch import Tensor, nn


def create_moe_task(
    batch_size: int,
    input_dim: int,
    num_experts: int = 8,
    active_experts: int = 1,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create MoE task data.

    Each input belongs to one expert (cluster).
    Target is expert index.
    """
    device = torch.device(device)

    # Generate cluster centers
    centers = torch.randn(num_experts, input_dim, device=device) * 2

    # Assign each sample to a cluster
    expert_ids = torch.randint(0, num_experts, (batch_size,), device=device)

    # Generate samples around cluster centers
    x = centers[expert_ids] + torch.randn(batch_size, input_dim, device=device) * 0.5

    # Target is the expert ID
    y = expert_ids

    return x, y


def count_active_routes(gate_logits: Tensor, top_k: int = 1) -> tuple[float, float]:
    """Count active routes and compute gate entropy.

    Returns:
        (mean_active_routes, mean_gate_entropy)
    """
    # Softmax for entropy
    probs = torch.softmax(gate_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()

    # Hard top-k for active count
    _, indices = torch.topk(gate_logits, k=top_k, dim=-1)
    routes = torch.zeros_like(gate_logits)
    routes.scatter_(-1, indices, 1.0)
    active = routes.sum(dim=-1).float().mean().item()

    return active, entropy


class CompositeState:
    """Simple composite state for benchmarking."""

    def __init__(self, activity, plastic, substrate):
        self.activity = activity
        self.plastic = plastic
        self.substrate = substrate


class ComputeEfficiencyModel(nn.Module):
    """Model for compute efficiency benchmark."""

    def __init__(
        self, input_dim: int, num_experts: int, plasticity, hidden_dim: int = 32
    ):
        super().__init__()
        self.plasticity = plasticity
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.psi = plasticity.initial_psi(None, batch_size=1)

        if hasattr(plasticity, "gate_dim") and plasticity_type == "routing":
            # Routing model with explicit gates
            self.gate = nn.Linear(input_dim, num_experts)
            self.experts = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)
            ])
            self.output = nn.Linear(hidden_dim, num_experts)
        else:
            # Dense baseline
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts),
            )

    def forward(self, x: Tensor) -> Tensor:
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

        if hasattr(self, "gate"):
            # Routing model
            gate_logits = self.gate(x)  # [batch, num_experts]

            # Update plasticity if available
            if self.psi and hasattr(self.plasticity, "gate_dim"):
                z = CompositeState(activity={"x": x}, plastic=self.psi, substrate={})
                self.psi = self.plasticity.step(self.psi, z, None)

                # Use plasticity's gate logits instead of learned gate
                plasticity_gate = self.psi.get("gate_logits", gate_logits)
                if plasticity_gate.shape[0] != batch_size:
                    plasticity_gate = plasticity_gate[:1].expand(batch_size, -1)
                gate_logits = plasticity_gate

            # Get active routes
            is_training = self.training
            if hasattr(self.plasticity, "get_active_operator"):
                active_routes = self.plasticity.get_active_operator(
                    gate_logits, is_training
                )
            else:
                # Default: softmax
                active_routes = (
                    F.softmax(gate_logits, dim=-1)
                    if is_training
                    else F.one_hot(
                        gate_logits.argmax(dim=-1), num_classes=self.num_experts
                    ).float()
                )

            # Apply experts
            expert_outputs = torch.stack(
                [expert(x) for expert in self.experts], dim=1
            )  # [batch, num_experts, hidden]
            weighted = (active_routes.unsqueeze(-1) * expert_outputs).sum(dim=1)
            return self.output(weighted)
        else:
            # Dense model
            return self.net(x)


def evaluate_compute_efficiency(
    coordinate: str,
    epochs: int = 20,
    batch_size: int = 64,
    input_dim: int = 64,
    num_experts: int = 8,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Evaluate compute efficiency for a coordinate."""
    from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
    from computronium.core.plasticity import (
        create_fast_weight_plasticity,
        create_routing_plasticity,
        create_rule_state_plasticity,
        create_substrate_coupled_plasticity,
    )

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device)
    start_time = time.perf_counter()

    parts = coordinate.split("/")
    if len(parts) != 6:
        raise ValueError(f"Invalid coordinate: {coordinate}")

    global plasticity_type
    plasticity_type = parts[3]

    # Build plasticity primitive
    plasticity_config = PlasticityConfig(
        plasticity_type=plasticity_type,
        plastic_state_dims={"gate_logits": num_experts, "active_routes": num_experts}
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

    model = ComputeEfficiencyModel(input_dim, num_experts, plasticity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    active_routes_history = []
    gate_entropy_history = []
    losses = []

    for epoch in range(epochs):
        x, y = create_moe_task(batch_size, input_dim, num_experts, device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if plasticity_type == "routing" and hasattr(model, "gate"):
            # Track routing metrics
            with torch.no_grad():
                gate_logits = model.gate(x)
                # Also include plasticity gate if available
                if model.psi and "gate_logits" in model.psi:
                    gate_logits = model.psi["gate_logits"]
                    if gate_logits.shape[0] != batch_size:
                        gate_logits = gate_logits[:1].expand(batch_size, -1)
                active, entropy = count_active_routes(gate_logits, top_k=1)
                active_routes_history.append(active)
                gate_entropy_history.append(entropy)

    # Final evaluation
    correct = 0
    total = 0
    final_active = 0
    final_entropy = 0

    with torch.no_grad():
        for _ in range(20):
            x, y = create_moe_task(batch_size, input_dim, num_experts, device=device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]

            if plasticity_type == "routing":
                gate_logits = model.gate(x)
                if model.psi and "gate_logits" in model.psi:
                    gate_logits = model.psi["gate_logits"]
                    if gate_logits.shape[0] != batch_size:
                        gate_logits = gate_logits[:1].expand(batch_size, -1)
                active, entropy = count_active_routes(gate_logits, top_k=1)
                final_active += active
                final_entropy += entropy

    final_accuracy = correct / total
    final_active = final_active / 20 if plasticity_type == "routing" else num_experts
    final_entropy = final_entropy / 20 if plasticity_type == "routing" else 0

    # Estimate FLOPs reduction
    dense_flops = input_dim * 32 * num_experts
    routing_flops = (
        input_dim * 32 * final_active if plasticity_type == "routing" else dense_flops
    )
    flops_reduction = 1.0 - (routing_flops / dense_flops) if dense_flops > 0 else 0

    return {
        "claims_scope": CLAIMS_SCOPE_PSI_WIRED_UNCONTROLLED,
        "coordinate": coordinate,
        "final_accuracy": final_accuracy,
        "final_loss": losses[-1],
        "active_routes": final_active,
        "gate_entropy": final_entropy,
        "dense_flops": dense_flops,
        "effective_flops": routing_flops,
        "flops_reduction": flops_reduction,
        "losses": losses,
        "active_routes_history": active_routes_history,
        "gate_entropy_history": gate_entropy_history,
        "resources": measure_suite_resources(
            model=model,
            coordinate=coordinate,
            device=str(device),
            batch_size=batch_size,
            elapsed_s=time.perf_counter() - start_time,
        ).to_dict(),
    }


def run_compute_efficiency_suite(
    coordinates: list[str],
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    seeds: int = 3,
    device: str = "auto",
) -> list[dict]:
    """Run compute efficiency benchmark suite."""
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device

    all_results = []

    for coord in coordinates:
        print(f"\nEvaluating: {coord}")
        coord_results = {"coordinate": coord, "seeds": []}

        for seed in range(seeds):
            print(f"  Seed {seed}...")
            result = evaluate_compute_efficiency(
                coordinate=coord,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
            coord_results["seeds"].append(result)
            print(
                f"    Acc: {result['final_accuracy']:.4f}, Active: {result['active_routes']:.1f}, FLOPs reduction: {result['flops_reduction']:.2%}"
            )

        # Aggregate
        if coord_results["seeds"]:
            accuracies = [s["final_accuracy"] for s in coord_results["seeds"]]
            active_routes = [s["active_routes"] for s in coord_results["seeds"]]
            flops_red = [s["flops_reduction"] for s in coord_results["seeds"]]
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
            coord_results["mean_active_routes"] = sum(active_routes) / len(
                active_routes
            )
            coord_results["mean_flops_reduction"] = sum(flops_red) / len(flops_red)

        all_results.append(coord_results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "compute_efficiency_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Compute Efficiency Benchmark Summary")
    print("=" * 80)
    print(
        f"{'Coordinate':<50} {'Mean Acc':<10} {'Active Routes':<12} {'FLOPs Red.':<10} {'Plasticity'}"
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
            f"{coord_short:<50} {r.get('mean_accuracy', 0):<10.4f} {r.get('mean_active_routes', 0):<12.1f} {r.get('mean_flops_reduction', 0):<10.2%} {prim}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute Efficiency Benchmark (Level 2)"
    )
    parser.add_argument("--coordinates", nargs="+", help="Coordinates to test")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/compute_efficiency",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
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
        "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
        "digital/feedforward/instantaneous/routing/thermodynamic_contrast/euclidean",
        "sparse/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
        "ternary/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
    ]

    run_compute_efficiency_suite(
        coordinates=coordinates,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
