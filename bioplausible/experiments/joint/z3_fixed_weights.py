"""Z3 Fixed Weights Benchmark (Level 4).

Question: Can frozen θ solve multiple tasks via ψ?

Constraint: θ frozen. Tasks: parity, last-symbol, threshold.
Operator library: Identity, Threshold, Accumulate, LastSymbol, Parity, SparseTopKRoute, SignFlip, Delay

Gating: T_t = Σ_k g_k(ψ_t) T_k, g_k(ψ_t) = softmax(controller(ψ_t, x_t))
Differentiable: soft mixture during training, hard selection at eval

Parameter invariance MUST be exact: ||θ_after - θ_before|| == 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor

# ============================================================
# Z3 Task Generators
# ============================================================


def create_parity_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Parity task: classify parity of number of positive elements."""
    device = torch.device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    # Parity of positive elements across sequence
    pos_count = (x > 0).sum(dim=(1, 2))  # [batch]
    y = (pos_count % 2).long()
    return x, y


def create_last_symbol_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Last symbol task: classify by last element sign."""
    device = torch.device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    last = x[:, -1, :].mean(dim=-1)  # [batch]
    y = (last > 0).long()
    return x, y


def create_threshold_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Threshold task: classify if sum exceeds threshold."""
    device = torch.device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    total = x.sum(dim=(1, 2))  # [batch]
    y = (total > 0).long()  # threshold at 0
    return x, y


# ============================================================
# Z3 Operators (Minimal Rule Library)
# ============================================================


class Z3Operators:
    """Z3 minimal operator library.

    T_0 = Identity
    T_1 = Threshold
    T_2 = Accumulate
    T_3 = LastSymbol
    T_4 = Parity
    T_5 = SparseTopKRoute
    T_6 = SignFlip
    T_7 = Delay
    """

    @staticmethod
    def identity(x: Tensor) -> Tensor:
        """T_0: Identity - pass through unchanged."""
        return x

    @staticmethod
    def threshold(x: Tensor, threshold: float = 0.0) -> Tensor:
        """T_1: Threshold - binary activation."""
        return (x > threshold).float()

    @staticmethod
    def accumulate(x: Tensor) -> Tensor:
        """T_2: Accumulate - cumulative sum over sequence."""
        # x: [batch, seq_len, dim] -> cumsum over seq
        return x.cumsum(dim=1)

    @staticmethod
    def last_symbol(x: Tensor) -> Tensor:
        """T_3: LastSymbol - extract last timestep."""
        # x: [batch, seq_len, dim] -> [batch, dim]
        return x[:, -1, :]

    @staticmethod
    def parity(x: Tensor) -> Tensor:
        """T_4: Parity - compute parity of positive elements."""
        # x: [batch, seq_len, dim] -> parity per batch, expanded to dim
        pos = (x > 0).float()
        parity = (pos.sum(dim=(1, 2), keepdim=True) % 2).float()  # [batch, 1, 1]
        # Expand to match input_dim by repeating
        return parity.expand(-1, 1, x.shape[-1])  # [batch, 1, dim]

    @staticmethod
    def sparse_topk_route(x: Tensor, k: int = 2) -> Tensor:
        """T_5: SparseTopKRoute - route to top-k dimensions."""
        # x: [batch, dim] -> keep only top-k
        if x.dim() == 3:
            x = x.mean(dim=1)  # [batch, dim]
        topk_vals, topk_indices = torch.topk(x.abs(), k=min(k, x.shape[-1]), dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1.0)
        return x * mask

    @staticmethod
    def sign_flip(x: Tensor) -> Tensor:
        """T_6: SignFlip - flip sign of negative values."""
        return x.abs()

    @staticmethod
    def delay(x: Tensor, delay: int = 1) -> Tensor:
        """T_7: Delay - shift sequence by delay steps."""
        if x.dim() == 3:
            # [batch, seq_len, dim] -> shift right
            padded = torch.cat([torch.zeros_like(x[:, :delay]), x[:, :-delay]], dim=1)
            return padded
        return x


# ============================================================
# Z3 Controller
# ============================================================


class Z3Controller(torch.nn.Module):
    """Controller for Z3 operator selection.

    Takes (ψ_t, x_t) -> operator logits g_k
    """

    def __init__(
        self,
        operator_dim: int,
        controller_hidden: int,
        num_operators: int,
    ):
        super().__init__()
        self.num_operators = num_operators
        self.operator_dim = operator_dim

        # Controller network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(controller_hidden + operator_dim, controller_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(controller_hidden, num_operators),
        )

    def forward(self, psi: Tensor, x: Tensor) -> Tensor:
        """Compute operator logits.

        Args:
            psi: Plastic state [batch, controller_hidden]
            x: Input [batch, operator_dim] (or [batch, seq_len, operator_dim])

        Returns:
            operator_logits: [batch, num_operators]
        """
        # Flatten x if needed
        if x.dim() > 2:
            x = x.mean(dim=1)  # Average over sequence
        elif x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure psi matches batch size
        if psi.shape[0] != x.shape[0]:
            psi = psi[:1].expand(x.shape[0], -1)

        combined = torch.cat([psi, x], dim=-1)
        return self.net(combined)


# ============================================================
# Z3 Model
# ============================================================


class Z3Model(torch.nn.Module):
    """Z3 Model: Frozen θ (operator embeddings) + Adaptive ψ (controller).

    The operator embeddings are learned during meta-training and FROZEN during eval.
    The controller learns to select operators for each task.
    """

    def __init__(
        self,
        num_operators: int = 8,
        operator_dim: int = 64,
        controller_hidden: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_operators = num_operators
        self.operator_dim = operator_dim
        self.temperature = temperature

        # Operator embeddings (θ - FROZEN during eval)
        self.operator_embeddings = torch.nn.Parameter(
            torch.randn(num_operators, operator_dim) * 0.02,
            requires_grad=True,  # Trainable during meta-training
        )

        # Controller (part of ψ - ADAPTS during eval)
        self.controller = Z3Controller(operator_dim, controller_hidden, num_operators)
        self.controller_hidden = controller_hidden

        # Plastic state (ψ): controller hidden state + operator logits
        # Use Parameters with requires_grad=False to ensure proper grad tracking
        self.register_parameter(
            "psi_controller_state",
            torch.nn.Parameter(torch.zeros(1, controller_hidden), requires_grad=False),
        )
        self.register_parameter(
            "psi_operator_logits",
            torch.nn.Parameter(torch.zeros(1, num_operators), requires_grad=False),
        )

        # Operator functions
        self.operators = [
            Z3Operators.identity,
            Z3Operators.threshold,
            Z3Operators.accumulate,
            Z3Operators.last_symbol,
            Z3Operators.parity,
            Z3Operators.sparse_topk_route,
            Z3Operators.sign_flip,
            Z3Operators.delay,
        ]

    def freeze_theta(self) -> None:
        """Freeze operator embeddings (θ) for Z3 evaluation.

        Only freezes θ (operator_embeddings), NOT the controller (which is part of ψ).
        """
        self.operator_embeddings.requires_grad_(False)
        # Controller is part of ψ, should remain trainable for adaptation

    def unfreeze_theta(self) -> None:
        """Unfreeze for meta-training."""
        self.operator_embeddings.requires_grad_(True)
        for param in self.controller.parameters():
            param.requires_grad_(True)

    def verify_theta_frozen(self) -> bool:
        """Verify θ (operator_embeddings) is frozen."""
        return not self.operator_embeddings.requires_grad

    def get_theta_snapshot(self) -> dict[str, Tensor]:
        """Get snapshot of θ (operator embeddings)."""
        return {"operator_embeddings": self.operator_embeddings.data.clone()}

    def compute_theta_change(self, before: dict, after: dict) -> float:
        """Compute ||θ_after - θ_before||."""
        diff = (
            (after["operator_embeddings"] - before["operator_embeddings"]).norm().item()
        )
        return diff

    def forward(self, x: Tensor, is_training: bool = False) -> Tensor:
        """Forward pass with operator selection.

        Args:
            x: Input [batch, seq_len, operator_dim]
            is_training: If True, use soft operator mixture. If False, hard selection.

        Returns:
            Output logits [batch, num_classes]
        """
        batch_size = x.shape[0]
        device = x.device

        # Ensure plastic state is on correct device
        psi_controller = self.psi_controller_state.to(device).expand(batch_size, -1)
        psi_logits = self.psi_operator_logits.to(device).expand(batch_size, -1)

        # Controller produces operator logits update
        logits_update = self.controller(psi_controller, x)
        new_logits = psi_logits + logits_update

        # Get operator weights
        if is_training:
            # Soft mixture (differentiable)
            gumbels = -torch.empty_like(new_logits).exponential_().log()
            gumbels = (new_logits + gumbels) / self.temperature
            operator_weights = torch.softmax(gumbels, dim=-1)
        else:
            # Hard selection (argmax)
            _, indices = torch.topk(new_logits, k=1, dim=-1)
            operator_weights = torch.zeros_like(new_logits)
            operator_weights.scatter_(-1, indices, 1.0)

        # Apply operators
        # For simplicity, we apply each operator and combine
        # In practice, this would be more efficient
        operator_outputs = []
        for i, op in enumerate(self.operators):
            op_out = op(x)
            # Ensure output is [batch, operator_dim]
            if op_out.dim() == 3:
                op_out = op_out.mean(
                    dim=1
                )  # Average over sequence: [batch, operator_dim]
            elif op_out.dim() == 2 and op_out.shape[1] != self.operator_dim:
                # If 2D but wrong feature dim, project
                if op_out.shape[1] == 1:
                    op_out = op_out.expand(-1, self.operator_dim)
            elif op_out.dim() == 1:
                op_out = op_out.unsqueeze(0).expand(batch_size, -1)
            elif op_out.shape[0] != batch_size:
                op_out = op_out.expand(batch_size, -1)

            # Final check: ensure [batch, operator_dim]
            if op_out.shape != (batch_size, self.operator_dim):
                # Project or pad to correct shape
                if (
                    op_out.shape[0] == batch_size
                    and op_out.shape[1] != self.operator_dim
                ):
                    if op_out.shape[1] < self.operator_dim:
                        padding = torch.zeros(
                            batch_size,
                            self.operator_dim - op_out.shape[1],
                            device=device,
                            dtype=op_out.dtype,
                        )
                        op_out = torch.cat([op_out, padding], dim=1)
                    else:
                        op_out = op_out[:, : self.operator_dim]

            operator_outputs.append(op_out)

        # Stack: [batch, num_operators, operator_dim]
        operator_stack = torch.stack(operator_outputs, dim=1)

        # Weighted combination: [batch, operator_dim]
        combined = (operator_weights.unsqueeze(-1) * operator_stack).sum(dim=1)

        # Project operator embeddings
        # output = combined @ operator_embeddings.T
        output = combined @ self.operator_embeddings.T

        # Update plastic state (ψ)
        self.psi_operator_logits.data = new_logits.detach().cpu()
        self.psi_controller_state.data = torch.tanh(
            psi_controller + 0.1 * logits_update.detach().mean(dim=-1, keepdim=True)
        ).cpu()

        return output


# ============================================================
# Z3 Evaluation
# ============================================================


def evaluate_z3(
    coordinate: str,
    meta_train_epochs: int = 50,
    eval_epochs_per_task: int = 20,
    batch_size: int = 64,
    seq_len: int = 10,
    input_dim: int = 32,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Evaluate Z3: meta-train then freeze θ and evaluate task switching."""
    from torch import nn

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device)

    parts = coordinate.split("/")
    if len(parts) != 6:
        raise ValueError(f"Invalid coordinate: {coordinate}")

    plasticity_type = parts[3]
    if plasticity_type != "rule_state":
        # Z3 requires rule_state plasticity
        raise ValueError(f"Z3 requires rule_state plasticity, got {plasticity_type}")

    # Build Z3 model
    model = Z3Model(
        num_operators=8,
        operator_dim=input_dim,
        controller_hidden=128,
        temperature=1.0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tasks = [
        ("parity", create_parity_task),
        ("last_symbol", create_last_symbol_task),
        ("threshold", create_threshold_task),
    ]

    results = {"coordinate": coordinate, "tasks": {}}

    # ===== META-TRAINING PHASE (θ learns operator embeddings) =====
    print("  Meta-training phase...")
    model.unfreeze_theta()
    model.train()

    for epoch in range(meta_train_epochs):
        epoch_loss = 0
        for task_name, task_fn in tasks:
            x, y = task_fn(batch_size, seq_len, input_dim, device)
            optimizer.zero_grad()
            logits = model(x, is_training=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: loss={epoch_loss / len(tasks):.4f}")

    # Snapshot θ after meta-training
    theta_before = model.get_theta_snapshot()

    # ===== EVALUATION PHASE (θ FROZEN, ψ adapts) =====
    print("  Evaluation phase (θ frozen)...")
    model.freeze_theta()
    assert model.verify_theta_frozen(), "θ not frozen!"

    # For each task, reset ψ and evaluate adaptation
    for task_name, task_fn in tasks:
        print(f"    Task: {task_name}")

        # Reset plastic state (ψ) for new task
        model.psi_controller_state.zero_()
        model.psi_operator_logits.zero_()

        # Quick adaptation on this task (only ψ updates)
        model.train()
        adaptation_losses = []
        for epoch in range(eval_epochs_per_task):
            x, y = task_fn(batch_size, seq_len, input_dim, device)
            optimizer.zero_grad()
            logits = model(x, is_training=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            adaptation_losses.append(loss.item())

        # Evaluate on this task (θ frozen, ψ adapted)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(20):
                x, y = task_fn(batch_size, seq_len, input_dim, device)
                logits = model(x, is_training=False)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.shape[0]

        accuracy = correct / total
        results["tasks"][task_name] = {
            "accuracy": accuracy,
            "adaptation_losses": adaptation_losses,
        }
        print(f"      Accuracy: {accuracy:.4f}")

    # Snapshot θ after evaluation (should be identical)
    theta_after = model.get_theta_snapshot()
    theta_change = model.compute_theta_change(theta_before, theta_after)

    results["theta_change"] = theta_change
    results["theta_invariant"] = theta_change < 1e-6

    print(f"  θ change: {theta_change:.8f} (invariant: {results['theta_invariant']})")

    # Compute operator diversity (entropy of operator usage)
    model.eval()
    with torch.no_grad():
        all_logits = []
        for task_name, task_fn in tasks:
            x, _ = task_fn(batch_size, seq_len, input_dim, device)
            model.psi_controller_state.zero_()
            model.psi_operator_logits.zero_()
            # Quick adapt
            for _ in range(5):
                _ = model(x, is_training=True)
            all_logits.append(model.psi_operator_logits.clone())

        # Average operator usage across tasks
        avg_logits = torch.stack(all_logits).mean(dim=0)
        probs = torch.softmax(avg_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        results["operator_diversity"] = entropy

    return results


def run_z3_suite(
    coordinates: list[str],
    output_dir: Path,
    meta_train_epochs: int = 50,
    eval_epochs: int = 20,
    batch_size: int = 64,
    seeds: int = 3,
    device: str = "auto",
) -> list[dict]:
    """Run Z3 fixed weights benchmark suite."""
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device

    all_results = []

    for coord in coordinates:
        print(f"\nEvaluating Z3: {coord}")
        coord_results = {"coordinate": coord, "seeds": []}

        for seed in range(seeds):
            print(f"  Seed {seed}...")
            result = evaluate_z3(
                coordinate=coord,
                meta_train_epochs=meta_train_epochs,
                eval_epochs_per_task=eval_epochs,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
            coord_results["seeds"].append(result)

            # Print task accuracies
            for task_name, task_result in result["tasks"].items():
                print(f"    {task_name}: {task_result['accuracy']:.4f}")
            print(
                f"    θ-change: {result['theta_change']:.8f}, Invariant: {result['theta_invariant']}"
            )

        # Aggregate
        if coord_results["seeds"]:
            theta_changes = [s["theta_change"] for s in coord_results["seeds"]]
            diversities = [s["operator_diversity"] for s in coord_results["seeds"]]
            coord_results["mean_theta_change"] = sum(theta_changes) / len(theta_changes)
            coord_results["mean_operator_diversity"] = sum(diversities) / len(
                diversities
            )

            # Task accuracies
            for task_name in ["parity", "last_symbol", "threshold"]:
                accs = [
                    s["tasks"][task_name]["accuracy"] for s in coord_results["seeds"]
                ]
                coord_results[f"mean_{task_name}_accuracy"] = sum(accs) / len(accs)

        all_results.append(coord_results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "z3_fixed_weights_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Z3 Fixed Weights Benchmark Summary (Level 4)")
    print("=" * 80)
    print(
        f"{'Coordinate':<50} {'Parity':<8} {'LastSym':<8} {'Thresh':<8} {'θ-change':<10} {'Diversity':<10}"
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
            f"{coord_short:<50} "
            f"{r.get('mean_parity_accuracy', 0):<8.4f} "
            f"{r.get('mean_last_symbol_accuracy', 0):<8.4f} "
            f"{r.get('mean_threshold_accuracy', 0):<8.4f} "
            f"{r.get('mean_theta_change', 0):<10.8f} "
            f"{r.get('mean_operator_diversity', 0):<10.4f} {prim}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Z3 Fixed Weights Benchmark (Level 4)")
    parser.add_argument("--coordinates", nargs="+", help="Coordinates to test")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/z3_fixed_weights",
        help="Output directory",
    )
    parser.add_argument(
        "--meta-train-epochs", type=int, default=50, help="Meta-training epochs"
    )
    parser.add_argument(
        "--eval-epochs", type=int, default=20, help="Evaluation epochs per task"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (10 meta, 5 eval, 1 seed)"
    )
    args = parser.parse_args()

    if args.quick:
        args.meta_train_epochs = 10
        args.eval_epochs = 5
        args.seeds = 1

    coordinates = args.coordinates or [
        "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
    ]

    run_z3_suite(
        coordinates=coordinates,
        output_dir=Path(args.output_dir),
        meta_train_epochs=args.meta_train_epochs,
        eval_epochs=args.eval_epochs,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
