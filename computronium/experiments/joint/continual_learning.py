"""Continual Learning Flagship (Phase 2).

The scientific centerpiece: ψ/θ decoupling prevents catastrophic forgetting
without a replay buffer.

Arms:
- FastWeightPlasticity (ψ/θ decoupling via fast weights)
- ElasticConsolidationUpdate (EWC - θ regularization)
- Backprop+SGD (baseline)
- Replay buffer (matched total memory)
- LwF (Learning without Forgetting)
- Synaptic Intelligence

Protocols:
- Task-incremental (task boundaries signaled)
- Task-free (no boundaries, gradual shift)

Metrics:
- Backward transfer matrix
- Forgetting measure per boundary
- Memory footprint (replay storage vs ψ state)
- Stability rider (ρ(J_F), windowed growth during ψ-adaptation)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
from computronium.core.joint.state import CompositeState
from computronium.core.pipeline import run_train_step
from computronium.core.plasticity import (
    create_fast_weight_plasticity,
    create_rule_state_plasticity,
)
from computronium.core.plasticity.theta_audit import ThetaInvarianceAudit
from computronium.core.profiling import ResourceUsage, measure_suite_resources
from computronium.core.stability import StabilityGuard, GuardDecision
from computronium.core.stability.spectral_radius import SpectralRadiusEstimator
from computronium.domains.base import TaskSplit
from computronium.domains.vision import SplitMNIST
from computronium.experiments.joint import CLAIMS_SCOPE_PSI_WIRED_UNCONTROLLED


# ============================================================
# Split-MNIST Task Generators (5 binary tasks: 0/1, 2/3, 4/5, 6/7, 8/9)
# ============================================================

SPLIT_MNIST_TASKS = [
    (0, 1),  # Task 0
    (2, 3),  # Task 1
    (4, 5),  # Task 2
    (6, 7),  # Task 3
    (8, 9),  # Task 4
]

NUM_TASKS = len(SPLIT_MNIST_TASKS)
CLASSES_PER_TASK = 2


# ============================================================
# Replay Buffer (Matched Total Memory)
# ============================================================


class ReplayBuffer:
    """Fixed-capacity replay buffer for continual learning.

    Stores (input, target, task_id) tuples. When full, evicts uniformly
    to maintain balanced representation across seen tasks.
    """

    def __init__(self, capacity: int, input_shape: tuple[int, ...], device: torch.device):
        self.capacity = capacity
        self.input_shape = input_shape
        self.device = device
        self.buffer: list[tuple[Tensor, Tensor, int]] = []
        self.task_counts: dict[int, int] = {}

    def add(self, x: Tensor, y: Tensor, task_id: int) -> None:
        """Add a batch to the buffer."""
        batch_size = x.shape[0]
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()

        for i in range(batch_size):
            if len(self.buffer) >= self.capacity:
                # Evict from the task with most samples
                if self.task_counts:
                    evict_task = max(self.task_counts.keys(), key=lambda k: self.task_counts[k])
                    # Find and remove one sample from that task
                    for idx, (_, _, t) in enumerate(self.buffer):
                        if t == evict_task:
                            self.buffer.pop(idx)
                            self.task_counts[evict_task] -= 1
                            if self.task_counts[evict_task] == 0:
                                del self.task_counts[evict_task]
                            break

            self.buffer.append((x_cpu[i], y_cpu[i], task_id))
            self.task_counts[task_id] = self.task_counts.get(task_id, 0) + 1

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a batch from the buffer."""
        if not self.buffer:
            raise ValueError("Replay buffer is empty")
        indices = torch.randperm(len(self.buffer))[:batch_size]
        samples = [self.buffer[i] for i in indices]
        x = torch.stack([s[0] for s in samples]).to(self.device)
        y = torch.stack([s[1] for s in samples]).to(self.device)
        t = torch.tensor([s[2] for s in samples], device=self.device)
        return x, y, t

    def __len__(self) -> int:
        return len(self.buffer)

    def memory_bytes(self) -> int:
        """Estimate memory footprint in bytes."""
        if not self.buffer:
            return 0
        sample = self.buffer[0]
        per_sample = sample[0].numel() * sample[0].element_size() + sample[1].numel() * sample[1].element_size()
        return per_sample * len(self.buffer)


# ============================================================
# LwF (Learning without Forgetting) Loss
# ============================================================


class LwFLoss(nn.Module):
    """LwF loss: distillation from previous model + current task CE."""

    def __init__(self, temperature: float = 2.0, lambda_lwf: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_lwf = lambda_lwf
        self.prev_model: nn.Module | None = None

    def set_prev_model(self, model: nn.Module) -> None:
        """Set the previous model for distillation."""
        self.prev_model = model
        for p in self.prev_model.parameters():
            p.requires_grad_(False)
        self.prev_model.eval()

    def forward(self, x: Tensor, targets: Tensor, task_id: int, current_model: nn.Module) -> Tensor:
        """Compute loss: CE on current model + distillation from previous model.
        
        Args:
            x: Input features [batch, input_dim]
            targets: Target labels
            task_id: Current task ID
            current_model: Current model to compute logits from
        """
        # Current model logits
        logits = current_model(x, task_id=task_id)
        ce_loss = F.cross_entropy(logits, targets)
        
        if self.prev_model is None or task_id == 0:
            return ce_loss

        with torch.no_grad():
            # Previous model logits on same input
            prev_logits = self.prev_model(x, task_id=task_id)

        # Distillation loss on old task logits
        num_old_classes = task_id * CLASSES_PER_TASK
        if num_old_classes > 0:
            soft_targets = F.softmax(prev_logits[:, :num_old_classes] / self.temperature, dim=1)
            soft_logits = F.log_softmax(logits[:, :num_old_classes] / self.temperature, dim=1)
            distill_loss = F.kl_div(soft_logits, soft_targets, reduction="batchmean") * (self.temperature**2)
            return ce_loss + self.lambda_lwf * distill_loss
        return ce_loss


# ============================================================
# Synaptic Intelligence (SI) Loss
# ============================================================


class SynapticIntelligence:
    """Synaptic Intelligence: importance-weighted parameter regularization.

    Computes per-parameter importance (omega) online during training,
    then regularizes changes to important parameters.
    """

    def __init__(self, model: nn.Module, xi: float = 0.1, epsilon: float = 1e-3):
        self.model = model
        self.xi = xi
        self.epsilon = epsilon
        self.omega: dict[int, Tensor] = {}  # Parameter importance
        self.prev_params: dict[int, Tensor] = {}  # Parameters at task boundary
        self.W: dict[int, Tensor] = {}  # Accumulated parameter-specific contribution

    def start_task(self) -> None:
        """Call at the start of each new task."""
        # Store current parameters as reference for this task
        for name, param in self.model.named_parameters():
            pid = id(param)
            self.prev_params[pid] = param.data.clone()
            if pid not in self.W:
                self.W[pid] = torch.zeros_like(param.data)

    def update_importance(self) -> None:
        """Update parameter importance (omega) at task boundary."""
        for name, param in self.model.named_parameters():
            pid = id(param)
            if pid in self.prev_params:
                # Delta from task start
                delta = param.data - self.prev_params[pid]
                # Accumulate contribution: path integral of gradients * delta
                if param.grad is not None:
                    self.W[pid] += -param.grad * delta
                # Update omega (importance)
                self.omega[pid] = self.W[pid] / (delta**2 + self.epsilon)
                # Reset W for next task
                self.W[pid].zero_()

    def regularization_loss(self) -> Tensor:
        """Compute SI regularization loss."""
        if not self.omega:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            pid = id(param)
            if pid in self.omega and pid in self.prev_params:
                loss += (self.omega[pid] * (param - self.prev_params[pid]) ** 2).sum()
        return self.xi * loss


# ============================================================
# Joint System with Plasticity for Continual Learning
# ============================================================


class ContinualJointSystem(nn.Module):
    """Joint system adapted for continual learning with task-specific heads."""

    def __init__(
        self,
        joint_system,
        num_tasks: int = NUM_TASKS,
        classes_per_task: int = CLASSES_PER_TASK,
    ):
        super().__init__()
        self.joint_system = joint_system
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task

        # Register geometry as submodule so .to(device) works
        self.geometry = joint_system.geometry

        # Task-specific output heads (each is binary: 2 classes)
        # We use the joint system's forward and add a final projection
        self.task_heads = nn.ModuleList([
            nn.Linear(10, classes_per_task) for _ in range(num_tasks)
        ])

        # Current task
        self.current_task = 0

    def to(self, *args, **kwargs):
        """Override to ensure joint system components are moved to device."""
        self = super().to(*args, **kwargs)
        # Move joint system components that support .to()
        device = args[0] if args else kwargs.get("device")
        if device is not None:
            if hasattr(self.joint_system.substrate, "to"):
                self.joint_system.substrate.to(device)
            if hasattr(self.joint_system.plasticity, "to"):
                self.joint_system.plasticity.to(device)
            if hasattr(self.joint_system.credit, "to"):
                self.joint_system.credit.to(device)
            if hasattr(self.joint_system.update, "to"):
                self.joint_system.update.to(device)
            if hasattr(self.joint_system.dynamics, "to"):
                self.joint_system.dynamics.to(device)
        return self

    def forward(self, x: Tensor, task_id: int | None = None) -> Tensor:
        """Forward pass through joint system + task head."""
        task_id = task_id if task_id is not None else self.current_task
        features = self.joint_system.forward(x)
        # features shape: [batch, 10] (output_dim from joint system)
        return self.task_heads[task_id](features)

    def train_step(self, x: Tensor, y: Tensor, task_id: int | None = None) -> dict[str, float]:
        """Training step through joint system + task head."""
        task_id = task_id if task_id is not None else self.current_task
        features = self.joint_system.forward(x)
        logits = self.task_heads[task_id](features)
        loss = F.cross_entropy(logits, y)

        # Backward through joint system
        loss.backward()

        # Joint system's train_step handles parameter update
        # We need to manually call the pipeline
        metrics = run_train_step(
            self.joint_system.substrate,
            self.joint_system.geometry,
            self.joint_system.dynamics,
            self.joint_system.credit,
            self.joint_system.update,
            x, y,
        )
        metrics["loss"] = loss.item()
        return metrics

    def set_task(self, task_id: int) -> None:
        self.current_task = task_id


# ============================================================
# Arm Implementations
# ============================================================


def create_fast_weight_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    device: str = "cpu",
) -> ContinualJointSystem:
    """Create FastWeightPlasticity arm (ψ/θ decoupling)."""
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )
    from computronium.core.system_trainer import compose_joint_system

    substrate = DigitalSubstrate(
        SubstrateConfig.digital(precision="float32", noise_level=0.0, device=device)
    )

    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            init_scale=0.1,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=0.001)
    )

    plasticity = create_fast_weight_plasticity(
        PlasticityConfig.fast_weights(fast_weight_dim=512, decay=0.9, learning_rate=0.1)
    )

    joint = compose_joint_system(substrate, geometry, dynamics, plasticity, credit, update)
    return ContinualJointSystem(joint).to(device)


def create_ewc_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    device: str = "cpu",
    ewc_lambda: float = 1000.0,
) -> tuple[ContinualJointSystem, SynapticIntelligence]:
    """Create ElasticConsolidationUpdate (EWC) arm."""
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        ElasticConsolidationUpdate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )
    from computronium.core.system_trainer import compose_joint_system

    substrate = DigitalSubstrate(
        SubstrateConfig.digital(precision="float32", noise_level=0.0, device=device)
    )

    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            init_scale=0.1,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )

    update = ElasticConsolidationUpdate(
        ParameterUpdateConfig.elastic_consolidation(step_size=0.001, ewc_lambda=ewc_lambda)
    )

    joint = compose_joint_system(
        substrate, geometry, dynamics, NullPlasticity(), credit, update
    )
    system = ContinualJointSystem(joint).to(device)

    # SI tracker for EWC arm
    si = SynapticIntelligence(system, xi=ewc_lambda / 1000.0)
    return system, si


def create_backprop_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    device: str = "cpu",
) -> ContinualJointSystem:
    """Create Backprop+SGD control arm."""
    from computronium.core.ontology import (
        BackpropCredit,
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
    )
    from computronium.core.system_trainer import compose_joint_system

    substrate = DigitalSubstrate(
        SubstrateConfig.digital(precision="float32", noise_level=0.0, device=device)
    )

    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            init_scale=0.1,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = BackpropCredit(CreditAssignmentConfig.gradient())

    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=0.001)
    )

    joint = compose_joint_system(substrate, geometry, dynamics, NullPlasticity(), credit, update)
    return ContinualJointSystem(joint).to(device)


def create_replay_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    device: str = "cpu",
    buffer_capacity: int = 5000,
) -> tuple[ContinualJointSystem, ReplayBuffer]:
    """Create replay buffer arm (matched total memory)."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    # Fast weight plasticity has ~512 * 4 bytes * batch_size = ~2KB per sample
    # Match replay buffer capacity to equivalent memory
    buffer = ReplayBuffer(buffer_capacity, (input_dim,), torch.device(device))
    return system, buffer


def create_lwf_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    device: str = "cpu",
) -> tuple[ContinualJointSystem, LwFLoss]:
    """Create LwF arm."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    lwf_loss = LwFLoss(temperature=2.0, lambda_lwf=1.0)
    return system, lwf_loss


def create_si_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = 10,
    device: str = "cpu",
) -> tuple[ContinualJointSystem, SynapticIntelligence]:
    """Create Synaptic Intelligence arm."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    si = SynapticIntelligence(system, xi=0.1)
    return system, si


# ============================================================
# Evaluation Metrics
# ============================================================


@dataclass
class CLMetrics:
    """Continual learning metrics."""

    # Per-task final accuracies (after all training)
    final_accuracies: list[float] = field(default_factory=list)

    # Accuracy matrix: accuracy_matrix[i][j] = accuracy on task i after training task j
    accuracy_matrix: list[list[float]] = field(default_factory=list)

    # Backward transfer: BWT = mean(acc_i_after_all - acc_i_after_task_i)
    backward_transfer: float = 0.0

    # Forward transfer: FWT = mean(acc_i_after_task_{i-1} - random_init_acc)
    forward_transfer: float = 0.0

    # Forgetting: F_i = max_{j<i} acc_i_after_task_j - acc_i_after_all
    forgetting: list[float] = field(default_factory=list)
    avg_forgetting: float = 0.0

    # Memory footprint
    peak_memory_mb: float = 0.0
    plastic_state_bytes: float = 0.0
    replay_buffer_bytes: float = 0.0

    # Stability rider
    stability_verdicts: list[GuardDecision] = field(default_factory=list)
    max_spectral_radius: float = 0.0

    # Training time
    total_time_s: float = 0.0


def compute_cl_metrics(
    model: ContinualJointSystem,
    task_loaders: list[DataLoader],
    current_task: int,
    accuracy_matrix: list[list[float]] | None = None,
) -> CLMetrics:
    """Compute comprehensive CL metrics."""
    metrics = CLMetrics()
    metrics.accuracy_matrix = accuracy_matrix or []

    # Evaluate on all tasks up to current_task
    final_accs = []
    device = next(model.parameters()).device
    for i, loader in enumerate(task_loaders):
        if i > current_task:
            final_accs.append(0.0)
            continue
        model.set_task(i)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)
                logits = model(x, task_id=i)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
        acc = correct / total if total > 0 else 0.0
        final_accs.append(acc)

    metrics.final_accuracies = final_accs

    # Compute backward transfer (only if we have history)
    if len(metrics.accuracy_matrix) > 0 and current_task > 0:
        bwt_sum = 0.0
        for i in range(current_task):
            if i < len(metrics.accuracy_matrix) and current_task < len(metrics.accuracy_matrix[i]):
                acc_after_i = metrics.accuracy_matrix[i][i]
                acc_after_all = metrics.accuracy_matrix[i][current_task]
                bwt_sum += acc_after_all - acc_after_i
        metrics.backward_transfer = bwt_sum / current_task if current_task > 0 else 0.0

    # Compute forgetting
    if len(metrics.accuracy_matrix) > 0:
        forgetting = []
        for i in range(current_task + 1):
            if i < len(metrics.accuracy_matrix):
                row = metrics.accuracy_matrix[i]
                if len(row) > current_task:
                    max_acc = max(row[:current_task + 1])
                    final_acc = row[current_task]
                    forgetting.append(max_acc - final_acc)
        metrics.forgetting = forgetting
        metrics.avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0

    return metrics


# ============================================================
# Stability Rider
# ============================================================


def create_stability_guard(
    threshold: float = 1.029,
    statistic: str = "fast_proxy",
    window: int = 10,
) -> StabilityGuard:
    """Create stability guard."""
    estimator = SpectralRadiusEstimator(fast_mode=True)
    return StabilityGuard(
        threshold=threshold,
        estimator=estimator,
        statistic=statistic,  # type: ignore[arg-type]
        window=window,
    )


def make_transition_fn(model: nn.Module):
    """Create a simple transition function for stability checking.
    
    Returns a CompositeState with activity, plastic, and substrate.
    """
    def transition_fn(state: CompositeState, context=None):
        """Transition function that takes a CompositeState and returns CompositeState."""
        x = state.activity.get("x")
        if x is None:
            return CompositeState.empty()
        with torch.no_grad():
            y = model(x)
        # Return CompositeState: activity contains x and y, plastic is empty, substrate is empty
        return CompositeState(
            activity={"x": y, "y": y},
            plastic={},
            substrate={},
        )
    return transition_fn


def make_composite_state(x: Tensor) -> CompositeState:
    """Create a simple CompositeState for stability checking."""
    return CompositeState(
        activity={"x": x},
        plastic={},
        substrate={},
    )


def check_stability(
    guard: StabilityGuard,
    transition_fn,
    x: Tensor,
    step: int,
) -> GuardDecision:
    """Check stability at current step."""
    state = make_composite_state(x)
    return guard(transition_fn, state, None)


# ============================================================
# Main Training Loop
# ============================================================


@dataclass
class CLConfig:
    """Configuration for continual learning experiment."""

    # Model
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10

    # Training
    epochs_per_task: int = 5
    batch_size: int = 64
    lr: float = 0.001

    # Replay
    replay_capacity: int = 5000

    # LwF
    lwf_temperature: float = 2.0
    lwf_lambda: float = 1.0

    # SI
    si_xi: float = 0.1

    # EWC
    ewc_lambda: float = 1000.0

    # Stability
    stability_threshold: float = 1.029
    stability_window: int = 10

    # Experiment
    device: str = "auto"
    seed: int = 42
    protocol: str = "task_incremental"  # or "task_free"
    num_workers: int = 0  # 0 to avoid multiprocessing resource leaks


def run_continual_learning(
    arm_name: str,
    config: CLConfig,
    protocol: str = "task_incremental",
) -> CLMetrics:
    """Run continual learning for one arm."""
    device_str = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device = torch.device(device_str)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Create task loaders
    task_loaders = []
    for task_id in range(NUM_TASKS):
        task = SplitMNIST(task_id=task_id, batch_size=config.batch_size, device=device_str, num_workers=config.num_workers)
        task.setup()
        task_loaders.append(task.get_dataloader(TaskSplit.TRAIN))

    test_loaders = []
    for task_id in range(NUM_TASKS):
        task = SplitMNIST(task_id=task_id, batch_size=config.batch_size, device=device_str, num_workers=config.num_workers)
        task.setup()
        test_loaders.append(task.get_dataloader(TaskSplit.TEST))

    # Create arm
    if arm_name == "fast_weights":
        model = create_fast_weight_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra = {}
    elif arm_name == "ewc":
        model, si = create_ewc_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str, config.ewc_lambda)
        extra = {"si": si}
    elif arm_name == "backprop":
        model = create_backprop_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra = {}
    elif arm_name == "replay":
        model, buffer = create_replay_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str, config.replay_capacity)
        extra = {"buffer": buffer}
    elif arm_name == "lwf":
        model, lwf_loss = create_lwf_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra = {"lwf_loss": lwf_loss}
    elif arm_name == "si":
        model, si = create_si_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra = {"si": si}
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    # Stability guard
    guard = create_stability_guard(
        threshold=config.stability_threshold,
        statistic="fast_proxy",
        window=config.stability_window,
    )
    transition_fn = make_transition_fn(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training
    accuracy_matrix = [[0.0 for _ in range(NUM_TASKS)] for _ in range(NUM_TASKS)]
    stability_verdicts: list[GuardDecision] = []
    start_time = time.perf_counter()

    if protocol == "task_incremental":
        # Task boundaries are signaled
        for task_id in range(NUM_TASKS):
            model.set_task(task_id)

            # Arm-specific setup at task boundary
            if arm_name == "ewc" and "si" in extra:
                extra["si"].start_task()
            elif arm_name == "lwf" and "lwf_loss" in extra:
                # Save current model as previous for distillation
                import copy
                prev_model = copy.deepcopy(model)
                extra["lwf_loss"].set_prev_model(prev_model)
            elif arm_name == "si" and "si" in extra:
                extra["si"].start_task()

            loader = task_loaders[task_id]

            for epoch in range(config.epochs_per_task):
                for batch_idx, (x, y) in enumerate(loader):
                    x = x.view(x.shape[0], -1).to(device)
                    y = y.to(device)

                    optimizer.zero_grad()

                    # Forward + loss
                    logits = model(x, task_id=task_id)

                    if arm_name == "lwf" and "lwf_loss" in extra:
                        loss = extra["lwf_loss"](x, y, task_id, model)
                    elif arm_name == "si" and "si" in extra:
                        loss = F.cross_entropy(logits, y) + extra["si"].regularization_loss()
                    else:
                        loss = F.cross_entropy(logits, y)

                    loss.backward()
                    optimizer.step()

                    # Stability check
                    verdict = check_stability(guard, transition_fn, x, step=epoch * len(loader) + batch_idx)
                    stability_verdicts.append(verdict)

                    # Replay buffer update
                    if arm_name == "replay" and "buffer" in extra:
                        extra["buffer"].add(x, y, task_id)

                    # Replay training
                    if arm_name == "replay" and "buffer" in extra and len(extra["buffer"]) >= config.batch_size:
                        rx, ry, rt = extra["buffer"].sample(config.batch_size)
                        optimizer.zero_grad()
                        r_logits = model(rx, task_id=rt[0].item())  # Simplified: use first task's head
                        r_loss = F.cross_entropy(r_logits, ry)
                        r_loss.backward()
                        optimizer.step()

                # End of task: update importance for EWC/SI
                if arm_name == "ewc" and "si" in extra:
                    extra["si"].update_importance()
                elif arm_name == "si" and "si" in extra:
                    extra["si"].update_importance()

            # Evaluate on all tasks so far
            for eval_task_id in range(task_id + 1):
                model.set_task(eval_task_id)
                correct = 0
                total = 0
                model.eval()
                with torch.no_grad():
                    for x, y in test_loaders[eval_task_id]:
                        x = x.view(x.shape[0], -1).to(device)
                        y = y.to(device)
                        logits = model(x, task_id=eval_task_id)
                        pred = logits.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.shape[0]
                accuracy_matrix[eval_task_id][task_id] = correct / total if total > 0 else 0.0

    elif protocol == "task_free":
        # No task boundaries - gradual shift (simulate by mixing tasks)
        # For simplicity, we'll cycle through tasks
        all_loaders = [iter(task_loaders[i]) for i in range(NUM_TASKS)]
        total_batches = config.epochs_per_task * max(len(l) for l in task_loaders)

        for batch_idx in range(total_batches):
            task_id = batch_idx % NUM_TASKS
            model.set_task(task_id)

            try:
                x, y = next(all_loaders[task_id])
            except StopIteration:
                all_loaders[task_id] = iter(task_loaders[task_id])
                x, y = next(all_loaders[task_id])

            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, task_id=task_id)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            verdict = check_stability(guard, transition_fn, x, step=batch_idx)
            stability_verdicts.append(verdict)

            if arm_name == "replay" and "buffer" in extra:
                extra["buffer"].add(x, y, task_id)  # type: ignore[attr-defined]

            # Periodic evaluation
            if batch_idx % (total_batches // NUM_TASKS) == 0:
                eval_task = batch_idx // (total_batches // NUM_TASKS)
                if eval_task < NUM_TASKS:
                    for eval_task_id in range(eval_task + 1):
                        model.set_task(eval_task_id)
                        correct = 0
                        total = 0
                        model.eval()
                        with torch.no_grad():
                            for ex, ey in test_loaders[eval_task_id]:
                                ex = ex.view(ex.shape[0], -1).to(device)
                                ey = ey.to(device)
                                elogits = model(ex, task_id=eval_task_id)
                                epred = elogits.argmax(dim=1)
                                correct += (epred == ey).sum().item()
                                total += ey.shape[0]
                        accuracy_matrix[eval_task_id][eval_task] = correct / total if total > 0 else 0.0

    total_time = time.perf_counter() - start_time

    # Final evaluation on all tasks
    final_metrics = compute_cl_metrics(model, test_loaders, NUM_TASKS - 1, accuracy_matrix)
    final_metrics.total_time_s = total_time
    final_metrics.stability_verdicts = stability_verdicts
    final_metrics.max_spectral_radius = max(v.statistic for v in stability_verdicts) if stability_verdicts else 0.0

    # Memory footprint
    if hasattr(model.joint_system, "plasticity") and hasattr(model.joint_system.plasticity, "fast_weight_dim"):  # type: ignore[attr-defined]
        final_metrics.plastic_state_bytes = model.joint_system.plasticity.fast_weight_dim * 4 * config.batch_size  # type: ignore[attr-defined]
    if arm_name == "replay" and "buffer" in extra:
        final_metrics.replay_buffer_bytes = extra["buffer"].memory_bytes()  # type: ignore[attr-defined]

    return final_metrics


# ============================================================
# Suite Runner
# ============================================================


def run_continual_learning_suite(
    arms: list[str],
    protocols: list[str],
    output_dir: str | Path,
    config: CLConfig | None = None,
    seeds: int = 3,
) -> dict[str, dict[str, dict[str, object]]]:
    """Run continual learning benchmark suite."""
    config = config or CLConfig()
    output_dir = Path(output_dir)

    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    config.device = device

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for arm in arms:
        all_results[arm] = {}
        for protocol in protocols:
            print(f"\n=== {arm} / {protocol} ===")
            arm_results: dict[str, object] = {"seeds": []}

            for seed in range(seeds):
                print(f"  Seed {seed}...")
                config.seed = seed
                metrics = run_continual_learning(arm, config, protocol)
                arm_results["seeds"].append({  # type: ignore[attr-defined]
                    "final_accuracies": metrics.final_accuracies,
                    "accuracy_matrix": metrics.accuracy_matrix,
                    "backward_transfer": metrics.backward_transfer,
                    "forward_transfer": metrics.forward_transfer,
                    "forgetting": metrics.forgetting,
                    "avg_forgetting": metrics.avg_forgetting,
                    "peak_memory_mb": metrics.peak_memory_mb,
                    "plastic_state_bytes": metrics.plastic_state_bytes,
                    "replay_buffer_bytes": metrics.replay_buffer_bytes,
                    "max_spectral_radius": metrics.max_spectral_radius,
                    "stability_kills": sum(1 for v in metrics.stability_verdicts if v.kill),
                    "total_time_s": metrics.total_time_s,
                })
                print(f"    Avg forgetting: {metrics.avg_forgetting:.4f}, BWT: {metrics.backward_transfer:.4f}")

            # Aggregate across seeds
            seeds_list = arm_results["seeds"]
            if seeds_list:
                for key in ["avg_forgetting", "backward_transfer", "forward_transfer", "max_spectral_radius", "total_time_s"]:
                    vals = [float(s[key]) for s in seeds_list]  # type: ignore[index]
                    mean_val = sum(vals) / len(vals)
                    arm_results[f"mean_{key}"] = mean_val
                    arm_results[f"std_{key}"] = (sum((v - mean_val)**2 for v in vals) / len(vals))**0.5 if len(vals) > 1 else 0.0

            all_results[arm][protocol] = arm_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "continual_learning_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Continual Learning Flagship (Phase 2)")
    parser.add_argument("--arms", nargs="+", default=["fast_weights", "ewc", "backprop", "replay", "lwf", "si"])
    parser.add_argument("--protocols", nargs="+", default=["task_incremental", "task_free"])
    parser.add_argument("--output-dir", default="benchmark_results/continual_learning")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.seeds = 1

    config = CLConfig(
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    run_continual_learning_suite(
        arms=args.arms,
        protocols=args.protocols,
        output_dir=Path(args.output_dir),
        config=config,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()