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
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from computronium.core.plasticity.theta_audit import ThetaInvarianceAudit

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


def _is_theta_param(name: str, _p: torch.nn.Parameter) -> bool:
    """Select θ (operator embeddings) — the parameters that must never move."""
    return name == "operator_embeddings"


_CRITERION_ACCURACY = 0.98


@dataclass(frozen=True, slots=True)
class TaskShape:
    """Batch geometry shared by every task-protocol helper."""

    batch_size: int
    seq_len: int
    input_dim: int
    device: torch.device

    def sample(self, task_fn) -> tuple[Tensor, Tensor]:
        return task_fn(self.batch_size, self.seq_len, self.input_dim, self.device)


def _eval_task_accuracy(
    model: Z3Model,
    shape: TaskShape,
    task_fn,
    *,
    batches: int = 20,
    soft: bool = False,
) -> float:
    """Accuracy over fresh batches; ``soft=True`` keeps the differentiable mixture."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = shape.sample(task_fn)
            logits = model(x, is_training=soft)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.shape[0]
    return correct / total


def _run_adaptation(  # noqa: PLR0913 - protocol tuple stays flat
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    criterion,
    shape: TaskShape,
    task_fn,
    *,
    epochs: int,
) -> tuple[list[float], int | None]:
    """Adam steps over whatever ``requires_grad`` currently selects.

    Returns per-epoch losses plus the 1-indexed step where batch accuracy
    first met the criterion (None if never) — the smoke-scale batch-window
    proxy for the registered 100-step definition.
    """
    model.train()
    losses: list[float] = []
    steps_to_criterion: int | None = None
    for epoch in range(epochs):
        x, y = shape.sample(task_fn)
        optimizer.zero_grad()
        logits = model(x, is_training=True)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if steps_to_criterion is None:
            acc = _eval_task_accuracy(model, shape, task_fn, batches=1)
            if acc >= _CRITERION_ACCURACY:
                steps_to_criterion = epoch + 1
    return losses, steps_to_criterion


def _meta_train(  # noqa: PLR0913 - protocol tuple stays flat
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
) -> None:
    """Joint meta-training over all tasks; θ and the controller learn together."""
    model.unfreeze_theta()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for _task_name, task_fn in tasks:
            x, y = shape.sample(task_fn)
            optimizer.zero_grad()
            logits = model(x, is_training=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: loss={epoch_loss / len(tasks):.4f}")


def _snapshot(model: Z3Model) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _reinit_psi(model: Z3Model) -> None:
    """Reset the controller to a fresh random init and zero plastic buffers."""
    for module in model.controller.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    model.psi_controller_state.zero_()
    model.psi_operator_logits.zero_()


def _adapt_all_tasks(
    model: Z3Model,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
) -> tuple[dict[str, dict], float]:
    """ψ-only adaptation protocol over the switching stream (θ stays frozen).

    One Adam over the trainable set spans all tasks, preserving PR-1
    semantics. Returns per-task result rows and elapsed wall-clock.
    """
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    started = time.perf_counter()
    rows: dict[str, dict] = {}
    for task_name, task_fn in tasks:
        model.psi_controller_state.zero_()
        model.psi_operator_logits.zero_()
        losses, steps = _run_adaptation(
            model, optimizer, criterion, shape, task_fn, epochs=epochs
        )
        rows[task_name] = {
            "accuracy": _eval_task_accuracy(model, shape, task_fn),
            "soft_eval_accuracy": _eval_task_accuracy(model, shape, task_fn, soft=True),
            "adaptation_losses": losses,
            "steps_to_criterion": steps,
        }
    return rows, time.perf_counter() - started


def _finetune_forgetting_baseline(
    model: Z3Model,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
) -> dict:
    """Baseline (a): sequential θ fine-tuning at the same per-task step budget.

    Produces the stage×task accuracy matrix whose diagonal-vs-last-column
    gap is the forgetting tax Z3 claims to avoid.
    """
    model.unfreeze_theta()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    matrix: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for stage_name, stage_fn in tasks:
        _run_adaptation(model, optimizer, criterion, shape, stage_fn, epochs=epochs)
        matrix[stage_name] = {
            name: _eval_task_accuracy(model, shape, fn) for name, fn in tasks
        }
    elapsed = time.perf_counter() - started

    final_row = matrix[tasks[-1][0]]
    forgetting = {
        name: matrix[name][name] - final_row[name] for name, _fn in tasks[:-1]
    }
    return {
        "accuracy_matrix": matrix,
        "forgetting": forgetting,
        "final_accuracy": final_row,
        "wall_clock_s": elapsed,
    }


def _run_baselines(  # noqa: PLR0913 - protocol tuple stays flat
    model: Z3Model,
    meta_state: dict[str, Tensor],
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
) -> dict:
    """E-10 control set; arms restore the meta-trained state first.

    Order matters because the fine-tune arm unfreezes θ and must run last.
    """
    # (c) floor control: meta-trained trunk, no ψ adaptation at all
    model.load_state_dict(meta_state)
    model.freeze_theta()
    floor_tasks = {
        name: {"accuracy": _eval_task_accuracy(model, shape, fn)} for name, fn in tasks
    }

    # (b) random-ψ init: isolates what meta-training bought the controller
    model.load_state_dict(meta_state)
    model.freeze_theta()
    _reinit_psi(model)
    random_psi_rows, _ = _adapt_all_tasks(model, tasks, criterion, shape, epochs=epochs)

    # (a) fine-tune θ, same step budget — the forgetting tax
    model.load_state_dict(meta_state)
    finetune = _finetune_forgetting_baseline(
        model, tasks, criterion, shape, epochs=epochs
    )

    return {
        "frozen_floor": {"tasks": floor_tasks},
        "random_psi": {
            "tasks": {
                n: {"accuracy": r["accuracy"]} for n, r in random_psi_rows.items()
            }
        },
        "finetune_forgetting": finetune,
    }


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
    shape = TaskShape(
        batch_size=batch_size, seq_len=seq_len, input_dim=input_dim, device=device
    )

    # ===== META-TRAINING PHASE (θ learns operator embeddings) =====
    print("  Meta-training phase...")
    _meta_train(model, optimizer, tasks, criterion, shape, epochs=meta_train_epochs)

    meta_state = _snapshot(model)

    # ===== EVALUATION PHASE (θ FROZEN, ψ adapts) =====
    print("  Evaluation phase (θ frozen)...")
    model.freeze_theta()
    assert model.verify_theta_frozen(), "θ not frozen!"

    # PR-2 audit: exact-diff θ across the whole switching/adaptation phase.
    # PR-1 hygiene: _adapt_all_tasks rebuilds Adam over the trainable set so
    # no meta-training momentum survives into ψ adaptation.
    with ThetaInvarianceAudit(model, selector=_is_theta_param) as audit:
        results["tasks"], psi_wall = _adapt_all_tasks(
            model, tasks, criterion, shape, epochs=eval_epochs_per_task
        )
    results["wall_clock_s"] = {"psi_adaptation": psi_wall}

    report = audit.report
    assert report is not None, "θ audit produced no report"
    results["theta_change"] = report.max_abs_change
    results["theta_invariant"] = report.is_within(1e-6)

    print(
        f"  θ change: {report.max_abs_change:.8f} "
        f"(invariant: {results['theta_invariant']})"
    )
    for task_name, row in results["tasks"].items():
        print(
            f"    {task_name}: acc={row['accuracy']:.4f} "
            f"criterion@{row['steps_to_criterion']}"
        )

    # Compute operator diversity (entropy of operator usage)
    model.eval()
    with torch.no_grad():
        all_logits = []
        for _task_name, task_fn in tasks:
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
    results["diversity_collapsed"] = bool(entropy < math.log(2))

    # ===== BASELINES (E-10 control set) =====
    print("  Baselines: frozen floor / random-ψ / θ fine-tune...")
    results["baselines"] = _run_baselines(
        model, meta_state, tasks, criterion, shape, epochs=eval_epochs_per_task
    )

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
