"""SystemTrainer: Orchestrates the 5-Layer Ontology Pipeline.

Replaces the monolithic CoreTrainer with a trainer that operates on the
composable 5-D tensor product (Substrate ⊗ Geometry ⊗ StateDynamics ⊗
CreditAssignment ⊗ ParameterUpdate).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import torch
from torch import Tensor

from bioplausible.core.logging import get_logger
from bioplausible.core.ontology import (
    CreditAssignment,
    Geometry,
    ParameterUpdate,
    StateDynamics,
    Substrate,
    System,
    SystemState,
)

logger = get_logger()

TS = TypeVar("TS", bound=Substrate)
TG = TypeVar("TG", bound=Geometry)
TD = TypeVar("TD", bound=StateDynamics)
TC = TypeVar("TC", bound=CreditAssignment)
TU = TypeVar("TU", bound=ParameterUpdate)


@dataclass(slots=True)
class SystemTrainerConfig:
    """Configuration for SystemTrainer.

    Attributes:
        max_epochs: Number of training epochs
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        device: Target device ("auto", "cpu", "cuda", "mps")
        grad_clip: Gradient clipping norm (applied in ParameterUpdate if supported)
        track_energy: Track energy metrics during training
        track_flops: Track FLOPs during training
        track_memory: Track memory usage
        log_every_n_steps: Logging frequency
        seed: Random seed
        deterministic: Use deterministic algorithms
    """
    max_epochs: int = 10
    batch_size: int = 64
    val_batch_size: int | None = None
    device: str = "auto"
    grad_clip: float | None = 1.0
    track_energy: bool = True
    track_flops: bool = True
    track_memory: bool = True
    log_every_n_steps: int = 10
    seed: int = 42
    deterministic: bool = False


class _DataProvider(Protocol):
    """Protocol for data providers (DataLoader, Task, etc.)."""

    def get_batch(self) -> tuple[Tensor, Tensor]: ...
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def __len__(self) -> int: ...


@dataclass
class SystemTrainer:
    """Trainer for 5-D composable systems.

    Orchestrates the pipeline:
        Substrate.forward_op → Geometry.route → StateDynamics.settle
        → CreditAssignment.compute_pseudo_gradient → ParameterUpdate.step

    The trainer accepts a pre-composed System and data providers,
    enabling clean separation of model architecture from training loop.
    """

    system: System
    config: SystemTrainerConfig
    train_data: _DataProvider
    val_data: _DataProvider | None = None

    # Training state
    current_epoch: int = field(default=0, init=False)
    global_step: int = field(default=0, init=False)
    history: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._setup_device()
        self._set_seed()
        # Move system components to device
        if hasattr(self.system.geometry, "to"):
            self.system.geometry.to(self.device)

    def _setup_device(self) -> None:
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info("SystemTrainer using device: %s", self.device)

    def _set_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True)

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.system.geometry.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_energy = 0.0
        num_batches = 0

        for _, (x, y) in enumerate(self.train_data):
            x = x.to(self.device)
            y = y.to(self.device)

            metrics = self.system.train_step(x, y)

            epoch_loss += metrics.get("loss", 0.0)
            epoch_acc += metrics.get("accuracy", 0.0)
            epoch_energy += metrics.get("energy", 0.0)
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                logger.info(
                    "Step %d: loss=%.4f, acc=%.4f, energy=%.4f",
                    self.global_step,
                    metrics.get("loss", 0.0),
                    metrics.get("accuracy", 0.0),
                    metrics.get("energy", 0.0),
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        avg_energy = epoch_energy / max(num_batches, 1)

        epoch_metrics = {
            "epoch": self.current_epoch,
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_energy": avg_energy,
            "global_step": self.global_step,
        }

        if self.val_data is not None:
            val_metrics = self.validate()
            epoch_metrics.update(val_metrics)

        self.history.append(epoch_metrics)
        self.current_epoch += 1

        logger.info(
            "Epoch %d: train_loss=%.4f, train_acc=%.4f, train_energy=%.4f",
            self.current_epoch, avg_loss, avg_acc, avg_energy
        )

        return epoch_metrics

    def validate(self) -> dict[str, float]:
        """Run validation epoch."""
        if self.val_data is None:
            return {}

        self.system.geometry.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in self.val_data:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.system.forward(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(-1) == y).float().mean().item()

                val_loss += loss.item()
                val_acc += acc
                num_batches += 1

        return {
            "val_loss": val_loss / max(num_batches, 1),
            "val_acc": val_acc / max(num_batches, 1),
        }

    def fit(self) -> list[dict[str, float]]:
        """Run full training loop."""
        logger.info("Starting training for %d epochs", self.config.max_epochs)

        for _ in range(self.config.max_epochs):
            self.train_epoch()

        logger.info("Training complete")
        return self.history


def compose_system(
    substrate: Substrate,
    geometry: Geometry,
    dynamics: StateDynamics,
    credit: CreditAssignment,
    update: ParameterUpdate,
) -> System:
    """Compose a System from five orthogonal components.

    This is the primary factory function for creating bioplausible systems
    from the 5-D ontology primitives.

    Example:
        system = compose_system(
            substrate=DigitalSubstrate(),
            geometry=FeedforwardGeometry(GeometryConfig(input_dim=784, output_dim=10)),
            dynamics=InstantaneousDynamics(),
            credit=ThermodynamicContrast(),
            update=EuclideanUpdate(),
        )
    """
    @dataclass(frozen=True, slots=True)
    class _ComposedSystem:
        substrate: Substrate
        geometry: Geometry
        dynamics: StateDynamics
        credit: CreditAssignment
        update: ParameterUpdate

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            state = SystemState(x=x, y=y)

            # 1. Substrate + Geometry: Forward pass
            state.activations = self.geometry.forward(x, self.substrate)
            if state.activations is not None:
                state.activations = self.substrate.inject_state_noise(state.activations)

            # 2. StateDynamics: Free phase
            free_state = self.dynamics.settle(state, self.geometry, self.substrate, target=None)
            free_state.energy = self.dynamics.compute_energy(free_state, self.geometry)

            # 3. StateDynamics: Nudged phase
            nudged_state = self.dynamics.settle(state, self.geometry, self.substrate, target=y)
            nudged_state.energy = self.dynamics.compute_energy(nudged_state, self.geometry)
            nudged_state.loss = self._compute_loss(nudged_state, y)

            # 4. CreditAssignment
            pseudo_grads = self.credit.compute_pseudo_gradient(
                free_state, nudged_state, nudged_state.loss, self.geometry
            )

            # 5. ParameterUpdate
            new_params = self.update.step(self.geometry.params, pseudo_grads, self.geometry)
            self.geometry.update_params(new_params)

            return {
                "loss": float(nudged_state.loss) if nudged_state.loss is not None else 0.0,
                "energy": float(free_state.energy) if free_state.energy is not None else 0.0,
                "accuracy": free_state.metrics.get("accuracy", 0.0),
            }

        def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:
            acts = state.activations
            if acts is None:
                return torch.tensor(0.0)
            if isinstance(acts, list):
                logits = acts[-1]
            else:
                logits = acts
            return torch.nn.functional.cross_entropy(logits, y)

        def forward(self, x: Tensor) -> Tensor:
            state = SystemState(x=x)
            state.activations = self.geometry.forward(x, self.substrate)
            if state.activations is not None:
                state.activations = self.substrate.inject_state_noise(state.activations)
            state = self.dynamics.settle(state, self.geometry, self.substrate, target=None)
            acts = state.activations
            if acts is None:
                return torch.empty(0)
            if isinstance(acts, list):
                return acts[-1]
            return acts

    return _ComposedSystem(
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=update,
    )  # type: ignore[return-value]


# Convenience factory for common compositions
def create_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
) -> System:
    """Create an Equilibrium Propagation system (classic EqProp coordinate)."""
    from bioplausible.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        ThermodynamicContrast,
    )

    substrate = DigitalSubstrate()

    dims = [hidden_dim] * max(num_layers, 1)
    geometry = RecurrentGeometry(GeometryConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
    ), hidden_dim=hidden_dim)

    dynamics = EnergyMinimizationDynamics(StateDynamicsConfig(
        dynamics_type="energy_minimization",
        max_steps=settle_steps,
        beta=beta,
    ))

    credit = ThermodynamicContrast(CreditAssignmentConfig(
        credit_type="thermodynamic_contrast",
        beta=beta,
    ))

    update = EuclideanUpdate(ParameterUpdateConfig(
        update_type="euclidean",
        step_size=lr,
    ))

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_backprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
) -> System:
    """Create a standard Backprop system (baseline coordinate)."""
    from bioplausible.core.ontology import (
        BackpropCredit,
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        StateDynamicsConfig,
    )

    substrate = DigitalSubstrate()

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(GeometryConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
    ))

    dynamics = InstantaneousDynamics(StateDynamicsConfig(
        dynamics_type="instantaneous",
    ))

    credit = BackpropCredit(CreditAssignmentConfig(
        credit_type="gradient",
    ))

    update = EuclideanUpdate(ParameterUpdateConfig(
        update_type="euclidean",
        step_size=lr,
    ))

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_fa_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
) -> System:
    """Create a Feedback Alignment system."""
    from bioplausible.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        RandomProjectionsCredit,
        StateDynamicsConfig,
    )

    substrate = DigitalSubstrate()

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(GeometryConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
    ))

    dynamics = InstantaneousDynamics(StateDynamicsConfig(
        dynamics_type="instantaneous",
    ))

    credit = RandomProjectionsCredit(CreditAssignmentConfig(
        credit_type="random_projections",
    ))

    update = EuclideanUpdate(ParameterUpdateConfig(
        update_type="euclidean",
        step_size=lr,
    ))

    return compose_system(substrate, geometry, dynamics, credit, update)


__all__ = [
    "SystemTrainer",
    "SystemTrainerConfig",
    "compose_system",
    "create_backprop_system",
    "create_eqprop_system",
    "create_fa_system",
]
