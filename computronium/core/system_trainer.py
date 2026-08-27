"""SystemTrainer: Orchestrates the 5-Layer Ontology Pipeline.

Replaces the monolithic CoreTrainer with a trainer that operates on the
composable 5-D tensor product (Substrate ⊗ Geometry ⊗ StateDynamics ⊗
CreditAssignment ⊗ ParameterUpdate).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from computronium.core.joint.context import SystemContext
from computronium.core.joint.state import StateVariable
from computronium.core.joint.transition import PlasticityConfig, PlasticityPrimitive
from computronium.core.logging import get_logger
from computronium.core.ontology import (
    CreditAssignment,
    CreditAssignmentConfig,
    Geometry,
    GeometryConfig,
    ParameterUpdate,
    ParameterUpdateConfig,
    StateDynamics,
    StateDynamicsConfig,
    Substrate,
    SubstrateConfig,
    System,
    substrate_from_config,
)
from computronium.core.pipeline import run_forward, run_train_step

if TYPE_CHECKING:
    from types import TracebackType


class JointSystem[
    TS: Substrate,
    TG: Geometry,
    TD: StateDynamics,
    TP: PlasticityPrimitive,
    TC: CreditAssignment,
    TU: ParameterUpdate,
](Protocol):
    """Protocol for 6-D joint systems."""

    substrate: TS
    geometry: TG
    dynamics: TD
    plasticity: TP
    credit: TC
    update: TU

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]: ...
    def forward(self, x: Tensor) -> Tensor: ...
    @property
    def context(self) -> SystemContext: ...
    def to_spec(self) -> dict[str, object]: ...
    @classmethod
    def from_spec(cls, spec: dict) -> JointSystem: ...


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

    def __post_init__(self) -> None:
        self._setup_device()
        self._set_seed()
        # Move system components to device
        if hasattr(self.system.geometry, "to"):
            self.system.geometry.to(self.device)  # type: ignore[attr-defined]

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
            self.current_epoch,
            avg_loss,
            avg_acc,
            avg_energy,
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

    def close(self) -> None:
        """Clean up resources (e.g., move model to CPU, clear CUDA cache)."""
        if hasattr(self, "system") and self.system is not None:
            if hasattr(self.system.geometry, "cpu"):
                self.system.geometry.cpu()
        if hasattr(self, "device") and self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("SystemTrainer resources cleaned up")

    def __enter__(self) -> SystemTrainer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False


def _credit_from_config(config: CreditAssignmentConfig):
    """Instantiate the credit implementation named by ``config.credit_type``."""
    from computronium.core.ontology import (
        BackpropCredit,
        HomeostaticCredit,
        LocalGoodnessCredit,
        RandomProjectionsCredit,
        TargetInversionCredit,
        TemporalTraceCredit,
        ThermodynamicContrast,
    )

    match config.credit_type.lower():
        case "thermodynamic_contrast" | "equilibrium":
            return ThermodynamicContrast(config)
        case (
            "random_projections" | "feedback_alignment" | ("direct_feedback_alignment")
        ):
            return RandomProjectionsCredit(config)
        case "local_goodness" | "forward_only":
            return LocalGoodnessCredit(config)
        case "temporal_trace" | "spiking":
            return TemporalTraceCredit(config)
        case "target_inversion" | "target_prop":
            return TargetInversionCredit(config)
        case "homeostatic":
            return HomeostaticCredit(config)
        case "gradient" | "backprop":
            return BackpropCredit(config)
        case other:
            raise ValueError(f"Unknown credit_type: {other!r}")


def compose_system[
    TS: Substrate,
    TG: Geometry,
    TD: StateDynamics,
    TC: CreditAssignment,
    TU: ParameterUpdate,
](
    substrate: TS,
    geometry: TG,
    dynamics: TD,
    credit: TC,
    update: TU,
) -> System[TS, TG, TD, TC, TU]:
    """Compose a System from five orthogonal components.

    This is the primary factory function for creating computronium systems
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
    class _ComposedSystem[
        TS: Substrate,
        TG: Geometry,
        TD: StateDynamics,
        TC: CreditAssignment,
        TU: ParameterUpdate,
    ]:
        substrate: TS
        geometry: TG
        dynamics: TD
        credit: TC
        update: TU

        def to_spec(self) -> dict:
            """Serialize the System to a specification dictionary.

            Returns:
                Dictionary containing schema_version and all 5 axis configs.
            """
            geometry_dict = dataclasses.asdict(self.geometry.config)
            # Include recurrent_weight from geometry if present (runtime state)
            recurrent_weight = getattr(self.geometry, "_recurrent_weight", None)
            if recurrent_weight is not None:
                geometry_dict["recurrent_weight"] = recurrent_weight.tolist()

            # Include all geometry parameters for exact round-trip
            geometry_params = {}
            for name, param in self.geometry.params.items():
                geometry_params[name] = param.tolist()
            geometry_dict["params"] = geometry_params

            return {
                "schema_version": "1.0",
                "substrate": dataclasses.asdict(self.substrate.config),
                "geometry": geometry_dict,
                "dynamics": dataclasses.asdict(self.dynamics.config),
                "credit": dataclasses.asdict(self.credit.config),
                "update": dataclasses.asdict(self.update.config),
            }

        @classmethod
        def from_spec(cls, spec: dict) -> System:
            """Reconstruct a System from a specification dictionary.

            Args:
                spec: Dictionary with schema_version and 5 axis configs.

            Returns:
                A composed System instance.
            """
            if spec.get("schema_version") != "1.0":
                raise ValueError(
                    f"Unsupported schema version: {spec.get('schema_version')}"
                )

            from computronium.core.ontology import (
                CreditAssignmentConfig,
                ElasticConsolidationUpdate,
                EnergyMinimizationDynamics,
                EuclideanUpdate,
                FeedforwardGeometry,
                GeometryConfig,
                InstantaneousDynamics,
                NaturalGradientUpdate,
                ParameterUpdateConfig,
                PredictiveSettlingDynamics,
                RecurrentGeometry,
                RiemannianOrthogonalUpdate,
                SpectralConstrainedUpdate,
                SpikeIntegrationDynamics,
                StateDynamicsConfig,
                SubstrateConfig,
            )

            # Reconstruct substrate (class named by the explicit type tag)
            substrate_cfg = SubstrateConfig(**spec["substrate"])
            substrate = substrate_from_config(substrate_cfg)

            # Reconstructed geometry
            geometry_dict = spec["geometry"]
            serialized_params = geometry_dict.pop("params", None)
            # JSON serialization converts tuples to lists; restore tuple types
            if "hidden_dims" in geometry_dict and isinstance(
                geometry_dict["hidden_dims"], list
            ):
                geometry_dict["hidden_dims"] = tuple(geometry_dict["hidden_dims"])
            geometry_cfg = GeometryConfig(**geometry_dict)
            topology_type = geometry_cfg.topology_type.lower()
            if topology_type in ("recurrent", "recurrent_attractor"):
                hidden_dim = (
                    geometry_cfg.hidden_dims[-1] if geometry_cfg.hidden_dims else None
                )
                recurrent_weight = None
                if geometry_cfg.recurrent_weight is not None:
                    recurrent_weight = torch.tensor(geometry_cfg.recurrent_weight)
                geometry = RecurrentGeometry(
                    geometry_cfg,
                    hidden_dim=hidden_dim,
                    recurrent_weight=recurrent_weight,
                )
            elif topology_type in ("tile_mesh", "tile"):
                from computronium.core.ontology import TileGeometry

                geometry = TileGeometry(
                    geometry_cfg,
                    neurons_per_tile=8,
                    tiles_per_layer=2,
                )
            else:
                geometry = FeedforwardGeometry(geometry_cfg)

            # Restore serialized parameters for exact round-trip
            if serialized_params is not None:
                geometry_params = {
                    k: torch.tensor(v) for k, v in serialized_params.items()
                }
                geometry.update_params(geometry_params)

            # Reconstruct dynamics
            dynamics_cfg = StateDynamicsConfig(**spec["dynamics"])
            dynamics_type = dynamics_cfg.dynamics_type.lower()
            if dynamics_type == "energy_minimization":
                dynamics = EnergyMinimizationDynamics(dynamics_cfg)
            elif dynamics_type == "predictive_settling":
                dynamics = PredictiveSettlingDynamics(dynamics_cfg)
            elif dynamics_type == "spike_integration":
                dynamics = SpikeIntegrationDynamics(dynamics_cfg)
            else:
                dynamics = InstantaneousDynamics(dynamics_cfg)

            # Reconstruct credit
            credit_cfg = CreditAssignmentConfig(**spec["credit"])
            credit = _credit_from_config(credit_cfg)

            # Reconstruct update
            update_cfg = ParameterUpdateConfig(**spec["update"])
            update_type = update_cfg.update_type.lower()
            if update_type in ("riemannian_orthogonal", "muon"):
                update = RiemannianOrthogonalUpdate(update_cfg)
            elif update_type in ("spectral_constrained", "spectral"):
                update = SpectralConstrainedUpdate(update_cfg)
            elif update_type in ("natural_gradient", "fisher"):
                update = NaturalGradientUpdate(update_cfg)
            elif update_type in ("elastic_consolidation", "ewc"):
                update = ElasticConsolidationUpdate(update_cfg)
            else:
                update = EuclideanUpdate(update_cfg)

            return compose_system(substrate, geometry, dynamics, credit, update)

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            """Execute one training step through the family-neutral pipeline."""
            return run_train_step(
                self.substrate,
                self.geometry,
                self.dynamics,
                self.credit,
                self.update,
                x,
                y,
            )

        def forward(self, x: Tensor) -> Tensor:
            return run_forward(self.substrate, self.geometry, self.dynamics, x)

    return _ComposedSystem[TS, TG, TD, TC, TU](
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
    update_momentum: float = 0.9,
) -> System:
    """Create an Equilibrium Propagation system (classic EqProp coordinate)."""
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers, 1)
    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
        init_scale=0.1,
    )
    geometry = RecurrentGeometry(
        geometry_cfg,
        hidden_dim=hidden_dim,
    )

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=update_momentum,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_backprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
) -> System:
    """Create a standard Backprop system (baseline coordinate)."""
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

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(
        GeometryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(dims),
            num_layers=num_layers,
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = BackpropCredit(
        CreditAssignmentConfig(
            credit_type="gradient",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_fa_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.1,
) -> System:
    """Create a Feedback Alignment system."""
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        RandomProjectionsCredit,
        StateDynamicsConfig,
        SubstrateConfig,
    )

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(
        GeometryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(dims),
            num_layers=num_layers,
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = RandomProjectionsCredit(
        CreditAssignmentConfig(
            credit_type="random_projections",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=feedback_scale,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def extract_config(system: System) -> dict[str, object]:
    """Extract configuration from a composed System.

    Returns a dictionary mapping layer names to their configuration objects.
    This enables round-trip: System -> configs -> System.
    """
    return {
        "substrate": system.substrate.config,
        "geometry": system.geometry.config,
        "dynamics": system.dynamics.config,
        "credit": system.credit.config,
        "update": system.update.config,
    }


def compose_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> System:
    """Compose a System from five configuration objects.

    This is the inverse of extract_config(), enabling the round-trip:
    System --extract_config--> configs --compose_system_from_configs--> System

    Args:
        substrate: Substrate configuration
        geometry: Geometry configuration
        dynamics: StateDynamics configuration
        credit: CreditAssignment configuration
        update: ParameterUpdate configuration

    Returns:
        A composed System with default implementations for each layer.
    """
    from computronium.core.ontology import (
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        InstantaneousDynamics,
        NaturalGradientUpdate,
        PredictiveSettlingDynamics,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SpectralConstrainedUpdate,
        SpikeIntegrationDynamics,
    )

    # Instantiate substrate from config (class named by the explicit type tag)
    substrate_instance = substrate_from_config(substrate)

    # Instantiate geometry from config
    topology_type = geometry.topology_type.lower()
    if topology_type in ("recurrent", "recurrent_attractor"):
        hidden_dim = geometry.hidden_dims[-1] if geometry.hidden_dims else None
        recurrent_weight = None
        if geometry.recurrent_weight is not None:
            recurrent_weight = torch.tensor(geometry.recurrent_weight)
        geometry_instance = RecurrentGeometry(
            geometry, hidden_dim=hidden_dim, recurrent_weight=recurrent_weight
        )
    elif topology_type in ("tile_mesh", "tile"):
        from computronium.core.ontology import TileGeometry

        geometry_instance = TileGeometry(
            geometry,
            neurons_per_tile=8,
            tiles_per_layer=2,
        )
    else:
        geometry_instance = FeedforwardGeometry(geometry)

    # Instantiate dynamics from config
    dynamics_type = dynamics.dynamics_type.lower()
    if dynamics_type == "energy_minimization":
        dynamics_instance = EnergyMinimizationDynamics(dynamics)
    elif dynamics_type == "predictive_settling":
        dynamics_instance = PredictiveSettlingDynamics(dynamics)
    elif dynamics_type == "spike_integration":
        dynamics_instance = SpikeIntegrationDynamics(dynamics)
    else:
        dynamics_instance = InstantaneousDynamics(dynamics)

    # Instantiate credit from config
    credit_instance = _credit_from_config(credit)

    # Instantiate update from config
    update_type = update.update_type.lower()
    if update_type in ("riemannian_orthogonal", "muon"):
        update_instance = RiemannianOrthogonalUpdate(update)
    elif update_type in ("spectral_constrained", "spectral"):
        update_instance = SpectralConstrainedUpdate(update)
    elif update_type in ("natural_gradient", "fisher"):
        update_instance = NaturalGradientUpdate(update)
    elif update_type in ("elastic_consolidation", "ewc"):
        update_instance = ElasticConsolidationUpdate(update)
    else:
        update_instance = EuclideanUpdate(update)

    return compose_system(
        substrate_instance,
        geometry_instance,
        dynamics_instance,
        credit_instance,
        update_instance,
    )


def compose_joint_system[
    TS: Substrate,
    TG: Geometry,
    TD: StateDynamics,
    TP: PlasticityPrimitive,
    TC: CreditAssignment,
    TU: ParameterUpdate,
](
    substrate: TS,
    geometry: TG,
    dynamics: TD,
    plasticity: TP,
    credit: TC,
    update: TU,
) -> JointSystem[TS, TG, TD, TP, TC, TU]:
    """Compose a JointSystem from six orthogonal components.

    This is the primary factory function for creating computronium joint systems
    from the 6-D ontology primitives (S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U).

    The 6-D composition enables combining plasticity mechanisms with any
    credit assignment and update rule, allowing novel architectures like:

    Example 1 - Routing + EqProp (meta-learning credit assignment):
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=RecurrentGeometry(GeometryConfig.recurrent(
                input_dim=784, output_dim=10, hidden_dims=(512, 512, 512)
            ), hidden_dim=512),
            dynamics=EnergyMinimizationDynamics(StateDynamicsConfig.energy_minimization(
                max_steps=20, beta=0.1
            )),
            plasticity=RoutingPlasticity(gate_dim=64, decay=0.99),
            credit=ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast(beta=0.1)),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
        )

    Example 2 - Fast Weights + Backprop (working memory + gradient descent):
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=FeedforwardGeometry(GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256, 128)
            )),
            dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            plasticity=FastWeightPlasticity(fast_weight_dim=512, decay=0.9),
            credit=BackpropCredit(CreditAssignmentConfig.gradient()),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
        )

    Example 3 - Rule State Plasticity for Hebbian meta-learning (Z3 benchmark):
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=FeedforwardGeometry(GeometryConfig.feedforward(
                input_dim=64, output_dim=2, hidden_dims=(128,)
            )),
            dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            plasticity=RuleStatePlasticity(num_operators=8, operator_dim=64),
            credit=LocalGoodnessCredit(CreditAssignmentConfig.local_goodness()),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
        )
        # Freeze theta for Z3 evaluation: joint.plasticity.freeze_theta()

    Args:
        substrate: Physical substrate (digital, analog, memristive, etc.)
        geometry: Network topology and connectivity
        dynamics: State evolution dynamics (settling, spiking, etc.)
        plasticity: Plasticity mechanism (routing, fast weights, rule state, etc.)
        credit: Credit assignment method (thermodynamic contrast, backprop, FA, etc.)
        update: Parameter update rule (SGD, Adam, Muon, spectral, etc.)

    Returns:
        A composed 6-D JointSystem ready for training via train_step().
    """
    from computronium.core.joint.transition import NullPlasticity
    from computronium.core.plasticity import NullPlasticity as _NullPlasticity

    @dataclass(frozen=True, slots=True)
    class _JointSystem[
        TS: Substrate,
        TG: Geometry,
        TD: StateDynamics,
        TP: PlasticityPrimitive,
        TC: CreditAssignment,
        TU: ParameterUpdate,
    ]:
        substrate: TS
        geometry: TG
        dynamics: TD
        plasticity: TP
        credit: TC
        update: TU

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            """Execute one training step through the family-neutral pipeline."""
            # Preserve lazy ψ initialization side effects before the loop.
            self.plasticity.initial_psi(self.context, batch_size=x.shape[0])
            return run_train_step(
                self.substrate,
                self.geometry,
                self.dynamics,
                self.credit,
                self.update,
                x,
                y,
            )

        def forward(self, x: Tensor) -> Tensor:
            return run_forward(self.substrate, self.geometry, self.dynamics, x)

        def _make_context(self) -> SystemContext:
            """Create SystemContext from this joint system."""
            from computronium.core.joint.context import SystemContext
            from computronium.core.joint.state import StateRegistry
            from computronium.core.joint.transition import PlasticityConfig

            # Build registry from all components
            registry = StateRegistry()

            # Register persistent parameters (theta)
            for name, param in self.geometry.params.items():
                registry.register(
                    StateVariable(
                        name=name,
                        persistent=True,
                        fast_plastic=False,
                        consolidatable=False,
                    )
                )

            # Register plastic variables if any
            if (
                hasattr(self.plasticity, "config")
                and self.plasticity.config.plastic_state_dims
            ):
                for name, dim in self.plasticity.config.plastic_state_dims.items():
                    registry.register(
                        StateVariable(
                            name=name,
                            persistent=False,
                            fast_plastic=True,
                            consolidatable=True,
                        )
                    )

            # Build configs from components
            substrate_config = self.substrate.config
            geometry_config = self.geometry.config
            dynamics_config = self.dynamics.config
            credit_config = self.credit.config
            update_config = self.update.config
            plasticity_config = getattr(
                self.plasticity, "config", PlasticityConfig.null()
            )

            return SystemContext(
                theta=self.geometry.params,
                geometry=self.geometry,
                substrate=self.substrate,
                substrate_config=substrate_config,
                geometry_config=geometry_config,
                dynamics_config=dynamics_config,
                credit_config=credit_config,
                update_config=update_config,
                plasticity_config=plasticity_config,
                registry=registry,
            )

        @property
        def context(self) -> SystemContext:
            """SystemContext bound to the current θ and component configs."""
            return self._make_context()

        def to_spec(self) -> dict[str, object]:
            """Serialize the JointSystem to a specification dictionary."""
            geometry_dict = dataclasses.asdict(self.geometry.config)
            recurrent_weight = getattr(self.geometry, "_recurrent_weight", None)
            if recurrent_weight is not None:
                geometry_dict["recurrent_weight"] = recurrent_weight.tolist()

            geometry_params = {}
            for name, param in self.geometry.params.items():
                geometry_params[name] = param.tolist()
            geometry_dict["params"] = geometry_params

            return {
                "schema_version": "2.0",  # 6-D schema
                "substrate": dataclasses.asdict(self.substrate.config),
                "geometry": geometry_dict,
                "dynamics": dataclasses.asdict(self.dynamics.config),
                "plasticity": dataclasses.asdict(
                    getattr(self.plasticity, "config", PlasticityConfig.null())
                ),
                "credit": dataclasses.asdict(self.credit.config),
                "update": dataclasses.asdict(self.update.config),
            }

        @classmethod
        def from_spec(cls, spec: dict) -> JointSystem:
            """Reconstruct a JointSystem from a specification dictionary."""
            # Delegate to compose_joint_system_from_configs
            from computronium.core.system_trainer import (
                compose_joint_system_from_configs,
            )

            return compose_joint_system_from_configs(
                SubstrateConfig(**spec["substrate"]),
                GeometryConfig(**spec["geometry"]),
                StateDynamicsConfig(**spec["dynamics"]),
                PlasticityConfig(
                    **spec.get("plasticity", PlasticityConfig.null().__dict__)
                ),
                CreditAssignmentConfig(**spec["credit"]),
                ParameterUpdateConfig(**spec["update"]),
            )

    # Check if plasticity is NullPlasticity (or equivalent)
    if isinstance(plasticity, (_NullPlasticity, NullPlasticity)):
        # For NullPlasticity, we can just use the 5-D system
        from computronium.core.system_trainer import compose_system

        base_system = compose_system(substrate, geometry, dynamics, credit, update)

        # Wrap with a null plasticity interface
        class _NullJointSystem[
            TS: Substrate,
            TG: Geometry,
            TD: StateDynamics,
            TC: CreditAssignment,
            TU: ParameterUpdate,
        ]:
            def __init__(
                self,
                system: System[TS, TG, TD, TC, TU],
            ):
                self._system = system
                self.substrate = system.substrate
                self.geometry = system.geometry
                self.dynamics = system.dynamics
                self.credit = system.credit
                self.update = system.update
                self.plasticity = NullPlasticity()

            def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
                return self._system.train_step(x, y)

            def forward(self, x: Tensor) -> Tensor:
                return self._system.forward(x)

            @property
            def context(self) -> SystemContext:
                """SystemContext bound to the current θ and component configs."""
                return self._make_context()

            def _make_context(self) -> SystemContext:
                from computronium.core.joint.context import SystemContext
                from computronium.core.joint.state import StateRegistry, StateVariable
                from computronium.core.joint.transition import PlasticityConfig

                # Build registry from all components
                registry = StateRegistry()

                # Register persistent parameters (theta)
                for name, param in self.geometry.params.items():
                    registry.register(
                        StateVariable(
                            name=name,
                            persistent=True,
                            fast_plastic=False,
                            consolidatable=False,
                        )
                    )

                # Build configs from components
                substrate_config = self.substrate.config
                geometry_config = self.geometry.config
                dynamics_config = self.dynamics.config
                credit_config = self.credit.config
                update_config = self.update.config
                plasticity_config = PlasticityConfig.null()

                return SystemContext(
                    theta=self.geometry.params,
                    geometry=self.geometry,
                    substrate=self.substrate,
                    substrate_config=substrate_config,
                    geometry_config=geometry_config,
                    dynamics_config=dynamics_config,
                    credit_config=credit_config,
                    update_config=update_config,
                    plasticity_config=plasticity_config,
                    registry=registry,
                )

            def to_spec(self) -> dict[str, object]:
                spec = base_system.to_spec()
                spec["plasticity"] = dataclasses.asdict(PlasticityConfig.null())
                spec["schema_version"] = "2.0"
                return spec

            @classmethod
            def from_spec(cls, spec: dict) -> JointSystem:
                from computronium.core.system_trainer import (
                    compose_joint_system_from_configs,
                )

                return compose_joint_system_from_configs(
                    SubstrateConfig(**spec["substrate"]),
                    GeometryConfig(**spec["geometry"]),
                    StateDynamicsConfig(**spec["dynamics"]),
                    PlasticityConfig(
                        **spec.get("plasticity", PlasticityConfig.null().__dict__)
                    ),
                    CreditAssignmentConfig(**spec["credit"]),
                    ParameterUpdateConfig(**spec["update"]),
                )

        return _NullJointSystem[TS, TG, TD, TC, TU](base_system)  # type: ignore[return-value]

    return _JointSystem[TS, TG, TD, TP, TC, TU](
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        plasticity=plasticity,
        credit=credit,
        update=update,
    )  # type: ignore[return-value]


def compose_joint_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    plasticity: PlasticityConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> JointSystem[
    Substrate,
    Geometry,
    StateDynamics,
    PlasticityPrimitive,
    CreditAssignment,
    ParameterUpdate,
]:
    """Compose a JointSystem from six configuration objects.

    This is the inverse of extract_config(), enabling the round-trip:
    JointSystem --extract_config--> configs --compose_joint_system_from_configs--> JointSystem

    Args:
        substrate: Substrate configuration
        geometry: Geometry configuration
        dynamics: StateDynamics configuration
        plasticity: Plasticity configuration
        credit: CreditAssignment configuration
        update: ParameterUpdate configuration

    Returns:
        A composed JointSystem with default implementations for each layer.
    """
    from computronium.core.ontology import (
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        InstantaneousDynamics,
        NaturalGradientUpdate,
        PredictiveSettlingDynamics,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SpectralConstrainedUpdate,
        SpikeIntegrationDynamics,
    )
    from computronium.core.plasticity import (
        NullPlasticity,
        create_fast_weight_plasticity,
        create_routing_plasticity,
        create_substrate_coupled_plasticity,
    )

    # Instantiate substrate from config (class named by the explicit type tag)
    substrate_instance = substrate_from_config(substrate)

    # Instantiate geometry from config
    topology_type = geometry.topology_type.lower()
    if topology_type in ("recurrent", "recurrent_attractor"):
        hidden_dim = geometry.hidden_dims[-1] if geometry.hidden_dims else None
        recurrent_weight = None
        if geometry.recurrent_weight is not None:
            recurrent_weight = torch.tensor(geometry.recurrent_weight)
        geometry_instance = RecurrentGeometry(
            geometry, hidden_dim=hidden_dim, recurrent_weight=recurrent_weight
        )
    elif topology_type in ("tile_mesh", "tile"):
        from computronium.core.ontology import TileGeometry

        geometry_instance = TileGeometry(
            geometry,
            neurons_per_tile=8,
            tiles_per_layer=2,
        )
    else:
        geometry_instance = FeedforwardGeometry(geometry)

    # Instantiate dynamics from config
    dynamics_type = dynamics.dynamics_type.lower()
    if dynamics_type == "energy_minimization":
        dynamics_instance = EnergyMinimizationDynamics(dynamics)
    elif dynamics_type == "predictive_settling":
        dynamics_instance = PredictiveSettlingDynamics(dynamics)
    elif dynamics_type == "spike_integration":
        dynamics_instance = SpikeIntegrationDynamics(dynamics)
    else:
        dynamics_instance = InstantaneousDynamics(dynamics)

    # Instantiate credit from config
    credit_instance = _credit_from_config(credit)

    # Instantiate update from config
    update_type = update.update_type.lower()
    if update_type in ("riemannian_orthogonal", "muon"):
        update_instance = RiemannianOrthogonalUpdate(update)
    elif update_type in ("spectral_constrained", "spectral"):
        update_instance = SpectralConstrainedUpdate(update)
    elif update_type in ("natural_gradient", "fisher"):
        update_instance = NaturalGradientUpdate(update)
    elif update_type in ("elastic_consolidation", "ewc"):
        update_instance = ElasticConsolidationUpdate(update)
    else:
        update_instance = EuclideanUpdate(update)

    # Instantiate plasticity from config
    plasticity_type = plasticity.plasticity_type.lower()
    if plasticity_type == "routing":
        plasticity_instance = create_routing_plasticity(plasticity)
    elif plasticity_type == "fast_weights":
        plasticity_instance = create_fast_weight_plasticity(plasticity)
    elif plasticity_type == "substrate_coupled":
        plasticity_instance = create_substrate_coupled_plasticity(plasticity)
    elif plasticity_type == "rule_state":
        from computronium.core.plasticity import create_rule_state_plasticity

        plasticity_instance = create_rule_state_plasticity(plasticity)
    else:
        plasticity_instance = NullPlasticity()

    return compose_joint_system(
        substrate_instance,
        geometry_instance,
        dynamics_instance,
        plasticity_instance,
        credit_instance,
        update_instance,
    )


# Convenience factory for common joint compositions
def create_routing_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    gate_dim: int = 64,
    gate_init_scale: float = 0.1,
) -> JointSystem:
    """Create an EqProp system with RoutingPlasticity (6-D coordinate)."""
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )
    from computronium.core.plasticity import RoutingPlasticity

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers, 1)
    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
        init_scale=0.1,
    )
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    plasticity = RoutingPlasticity(
        gate_dim=gate_dim,
        temperature=1.0,
        decay=0.99,
        learning_rate=0.01,
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


def create_fast_weight_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    fast_weight_dim: int = 512,
) -> JointSystem:
    """Create an EqProp system with FastWeightPlasticity (6-D coordinate)."""
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )
    from computronium.core.plasticity import (
        FastWeightPlasticity,
    )

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers, 1)
    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
        init_scale=0.1,
    )
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=0.99,
        learning_rate=0.1,
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


# ============================================================
# Continual Learning Arm Factories (6-D Joint Systems)
# ============================================================

# Constants for continual learning
CL_NUM_TASKS = 5
CL_CLASSES_PER_TASK = 2
CL_TOTAL_CLASSES = CL_NUM_TASKS * CL_CLASSES_PER_TASK  # 10

SPLIT_MNIST_TASKS = [
    (0, 1),  # Task 0
    (2, 3),  # Task 1
    (4, 5),  # Task 2
    (6, 7),  # Task 3
    (8, 9),  # Task 4
]


class ContinualJointSystem(nn.Module):
    """Joint system adapted for continual learning with 10-class output.

    Uses task masking instead of task-specific heads. The joint system
    outputs 10-class logits (matching MNIST 10 digits), and we mask
    the loss to the current task's 2 classes.

    Maintains plastic state (ψ) across training steps for ψ/θ decoupling.
    """

    def __init__(self, joint_system):
        super().__init__()
        self.joint_system = joint_system
        self.num_tasks = CL_NUM_TASKS
        self.classes_per_task = CL_CLASSES_PER_TASK

        # Register geometry as submodule so .to(device) works
        self.geometry = joint_system.geometry

        # Current task
        self.current_task = 0

        # Plastic state (ψ) - maintained across steps for fast weights
        self._psi: dict[str, Tensor] | None = None

    def to(self, *args, **kwargs):
        """Override to ensure joint system components are moved to device."""
        self = super().to(*args, **kwargs)
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
        """Forward pass through joint system with plastic state modulation.

        For FastWeightPlasticity, modulates the last hidden layer with fast weights.
        Returns full 10-class logits.
        """
        # Check if we have plastic state and fast weight plasticity
        plasticity = self.joint_system.plasticity
        has_fast_weights = (
            self._psi is not None
            and "fast_weights" in self._psi
            and hasattr(plasticity, "fast_weight_dim")
        )

        if not has_fast_weights:
            return self.joint_system.forward(x)

        # Get intermediate activations from geometry
        substrate = self.joint_system.substrate
        geometry = self.joint_system.geometry
        acts = geometry.forward_with_intermediates(x, substrate)
        # acts: [input, hidden1, hidden2, ..., output]

        # Modulate last hidden layer with fast weights
        # Last hidden is acts[-2] (before output layer)
        if len(acts) >= 2:
            last_hidden = acts[-2]  # [batch, hidden_dim]
            fast_weights = self._psi["fast_weights"]  # [batch, fast_weight_dim]

            # Handle batch size mismatch - resize to current batch
            batch_size = x.shape[0]
            if fast_weights.shape[0] != batch_size:
                if fast_weights.shape[0] == 1:
                    fast_weights = fast_weights.expand(batch_size, -1)
                elif fast_weights.shape[0] > batch_size:
                    fast_weights = fast_weights[:batch_size]
                else:
                    # Cannot expand smaller batch to larger - fallback to standard forward
                    return self.joint_system.forward(x)

            # Project fast weights to hidden_dim and add
            # Need a projection layer - create if not exists
            if not hasattr(self, "_fast_weight_proj"):
                hidden_dim = last_hidden.shape[-1]
                self._fast_weight_proj = nn.Linear(
                    plasticity.fast_weight_dim, hidden_dim, bias=False
                ).to(x.device)
                # Initialize with small weights
                nn.init.normal_(self._fast_weight_proj.weight, std=0.01)

            modulation = self._fast_weight_proj(fast_weights)
            modulated_hidden = last_hidden + modulation

            # Apply output layer (last layer in geometry)
            # The output layer is the last Linear layer in geometry._layers
            output_layer = None
            for layer in reversed(geometry._layers):
                if isinstance(layer, nn.Linear):
                    output_layer = layer
                    break

            if output_layer is not None:
                logits = output_layer(modulated_hidden)
                return logits

        # Fallback to standard forward
        return self.joint_system.forward(x)

    def train_step(self, x: Tensor, y: Tensor, task_id: int | None = None) -> dict[str, float]:
        """Training step using joint system's pipeline with task-masked loss and plasticity stepping."""
        task_id = task_id if task_id is not None else self.current_task
        metrics, self._psi = run_continual_train_step(self.joint_system, x, y, task_id, self._psi)
        return metrics

    def set_task(self, task_id: int) -> None:
        self.current_task = task_id

    def reset_plastic_state(self) -> None:
        """Reset plastic state (e.g., at task boundary for new episode)."""
        self._psi = None


# ============================================================
# Continual Learning Training Step (Masked Loss)
# ============================================================


def run_continual_train_step(
    joint_system,
    x: Tensor,
    y: Tensor,
    task_id: int,
    psi: dict[str, Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, Tensor] | None]:
    """Execute one training step through the joint system with task-masked loss and plasticity stepping.

    The joint system outputs 10-class logits. We mask the loss to only the
    current task's 2 classes (task_id * 2 : task_id * 2 + 2).

    The labels y are already 0/1 (local to the task) from SplitMNIST.

    This uses the joint system's credit assignment and parameter update,
    ensuring ψ/θ decoupling (FastWeightPlasticity) and other components
    are actually invoked. Also steps the plasticity to update ψ.

    Returns:
        Tuple of (metrics, updated_psi)
    """
    # Labels are already 0/1 from SplitMNIST
    local_y = y

    # Task logit slice
    task_start = task_id * CL_CLASSES_PER_TASK
    task_end = task_start + CL_CLASSES_PER_TASK

    # Get the joint system components
    substrate = joint_system.substrate
    geometry = joint_system.geometry
    dynamics = joint_system.dynamics
    credit = joint_system.credit
    update = joint_system.update
    plasticity = joint_system.plasticity

    # Initialize plastic state if needed
    if psi is None and hasattr(plasticity, 'initial_psi') and plasticity is not None:
        psi = plasticity.initial_psi(joint_system.context, batch_size=x.shape[0])
    # Ensure psi is on the same device as x and matches batch size
    if psi is not None:
        device = x.device
        batch_size = x.shape[0]
        new_psi = {}
        for k, v in psi.items():
            if v.shape[0] != batch_size:
                if v.shape[0] == 1:
                    new_psi[k] = v.expand(batch_size, -1).to(device)
                else:
                    new_psi[k] = v[:batch_size].to(device)
            else:
                new_psi[k] = v.to(device)
        psi = new_psi

    # Run the pipeline with masked target
    from computronium.core.pipeline import forward_pass, task_loss
    from computronium.core.ontology import Phase, SystemState
    from computronium.core.joint.state import CompositeState
    from contextlib import nullcontext

    grad_ctx = nullcontext() if credit.requires_autograd else torch.no_grad()
    with grad_ctx:
        states: dict[Phase, SystemState] = {}
        initial_activations = forward_pass(substrate, geometry, x)

        for phase in credit.phases:
            state = SystemState(x=x, y=local_y)  # Use local_y for nudged phase
            state.activations = initial_activations
            target = local_y if phase is Phase.NUDGED else None
            settled = dynamics.settle(state, geometry, substrate, target=target)
            if phase is Phase.NUDGED:
                # Compute loss only on task-relevant logits
                settled.loss = _masked_task_loss(settled, local_y, task_start, task_end)
            settled.energy = dynamics.compute_energy(settled, geometry)
            states[phase] = settled

        output = states.get(Phase.NUDGED, states.get(Phase.FREE))
        if output is None:
            output = SystemState(x=x, y=local_y)
            output.activations = initial_activations
        loss = output.loss
        if loss is None:
            loss = _masked_task_loss(output, local_y, task_start, task_end)
        elif not isinstance(loss, Tensor):
            loss = torch.as_tensor(loss)
        if output.energy is None:
            output.energy = dynamics.compute_energy(output, geometry)

        pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)
        geometry.update_params(update.step(geometry.params, pseudo_grads, geometry))

        # Step plasticity if present (ψ update)
        if psi is not None and hasattr(plasticity, 'step') and plasticity is not None:
            # Create CompositeState from settled output
            acts = output.activations
            if acts is not None:
                final_acts = acts[-1] if isinstance(acts, list) else acts
                z = CompositeState(
                    activity={"x": x, "y": final_acts},
                    plastic=psi,
                    substrate={},
                )
                psi = plasticity.step(psi, z, joint_system.context)

        loss_val = loss.item() if isinstance(loss, Tensor) else float(loss)
        energy_val = output.energy.item() if isinstance(output.energy, Tensor) else float(output.energy) if output.energy is not None else 0.0
        metrics = {
            "loss": loss_val,
            "energy": energy_val,
            "accuracy": output.metrics.get("accuracy", 0.0),
        }
        metrics.update({
            k: v
            for k, v in output.metrics.items()
            if isinstance(v, (int, float)) and k != "accuracy"
        })
        return metrics, psi


def _masked_task_loss(state, local_y: Tensor, task_start: int, task_end: int) -> Tensor:
    """Compute cross-entropy loss only on task-relevant logits."""
    acts = state.activations
    if acts is None:
        return torch.tensor(0.0, device=local_y.device)
    logits = acts[-1] if isinstance(acts, list) else acts  # [batch, 10]
    task_logits = logits[:, task_start:task_end]  # [batch, 2]
    loss = F.cross_entropy(task_logits, local_y)
    with torch.no_grad():
        acc = (task_logits.argmax(dim=-1) == local_y).float().mean().item()
    state.metrics = {**state.metrics, "accuracy": acc}
    return loss


# ============================================================
# Arm Factory Functions
# ============================================================


def create_fast_weight_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,  # 10 classes
    device: str = "cpu",
) -> ContinualJointSystem:
    """Create FastWeightPlasticity arm (ψ/θ decoupling).

    Uses EnergyMinimizationDynamics with ThermodynamicContrast for proper
    free/nudged settling dynamics required by the contrastive credit assignment.
    """
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )
    from computronium.core.plasticity import create_fast_weight_plasticity
    from computronium.core.joint.transition import PlasticityConfig

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

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )

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
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
    ewc_lambda: float = 1000.0,
) -> tuple[ContinualJointSystem, "SynapticIntelligence"]:
    """Create ElasticConsolidationUpdate (EWC) arm.

    Uses EnergyMinimizationDynamics with ThermodynamicContrast for proper
    free/nudged settling dynamics required by the contrastive credit assignment.
    """
    from computronium.core.joint.transition import NullPlasticity
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
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

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )

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
    output_dim: int = CL_TOTAL_CLASSES,
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
    from computronium.core.joint.transition import NullPlasticity
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
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
    buffer_capacity: int = 5000,
) -> tuple[ContinualJointSystem, "ReplayBuffer"]:
    """Create replay buffer arm (matched total memory)."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    # Fast weight plasticity has ~512 * 4 bytes * batch_size = ~2KB per sample
    # Match replay buffer capacity to equivalent memory
    buffer = ReplayBuffer(buffer_capacity, (input_dim,), torch.device(device))
    return system, buffer


def create_lwf_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
) -> tuple[ContinualJointSystem, "LwFLoss"]:
    """Create LwF arm."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    lwf_loss = LwFLoss(temperature=2.0, lambda_lwf=1.0)
    return system, lwf_loss


def create_si_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
) -> tuple[ContinualJointSystem, "SynapticIntelligence"]:
    """Create Synaptic Intelligence arm."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    si = SynapticIntelligence(system, xi=0.1)
    return system, si


# ============================================================
# Supporting Classes for Continual Learning Arms
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

    def forward(self, logits: Tensor, targets: Tensor, task_id: int, prev_logits: Tensor | None = None) -> Tensor:
        """Compute loss: CE on current task classes + distillation from previous model.

        Args:
            logits: Full 10-class logits [batch, 10]
            targets: Target labels (0 or 1 for current task)
            task_id: Current task ID
            prev_logits: Previous model's full 10-class logits [batch, 10]
        """
        # Current task classes
        task_start = task_id * CL_CLASSES_PER_TASK
        task_end = task_start + CL_CLASSES_PER_TASK
        task_logits = logits[:, task_start:task_end]
        # Map targets 0/1 to 0/1 for the task's 2 classes
        task_targets = targets % CL_CLASSES_PER_TASK
        ce_loss = F.cross_entropy(task_logits, task_targets)

        if self.prev_model is None or task_id == 0 or prev_logits is None:
            return ce_loss

        # Distillation loss on old task logits
        num_old_classes = task_id * CL_CLASSES_PER_TASK
        if num_old_classes > 0:
            soft_targets = F.softmax(prev_logits[:, :num_old_classes] / self.temperature, dim=1)
            soft_logits = F.log_softmax(logits[:, :num_old_classes] / self.temperature, dim=1)
            distill_loss = F.kl_div(soft_logits, soft_targets, reduction="batchmean") * (self.temperature**2)
            return ce_loss + self.lambda_lwf * distill_loss
        return ce_loss


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
# CL Metrics and Configuration
# ============================================================


@dataclass
class CLConfig:
    """Configuration for continual learning experiment."""

    # Model
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = CL_TOTAL_CLASSES

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
    stability_verdicts: list = field(default_factory=list)
    max_spectral_radius: float = 0.0

    # Training time
    total_time_s: float = 0.0


def compute_cl_metrics(
    model: ContinualJointSystem,
    task_loaders: list,
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
                # Mask to task-relevant classes
                task_start = i * CL_CLASSES_PER_TASK
                task_end = task_start + CL_CLASSES_PER_TASK
                task_logits = logits[:, task_start:task_end]
                pred = task_logits.argmax(dim=1)
                # Map global labels to local (0/1)
                local_y = y % CL_CLASSES_PER_TASK
                correct += (pred == local_y).sum().item()
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


def create_stability_guard(
    threshold: float = 1.029,
    statistic: str = "fast_proxy",
    window: int = 10,
):
    """Create stability guard."""
    from computronium.core.stability import StabilityGuard
    from computronium.core.stability.spectral_radius import SpectralRadiusEstimator
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
    def transition_fn(state, context=None):
        """Transition function that takes a CompositeState and returns CompositeState."""
        x = state.activity.get("x")
        if x is None:
            from computronium.core.joint.state import CompositeState
            return CompositeState.empty()
        with torch.no_grad():
            y = model(x)
        # Return CompositeState: activity contains x and y, plastic is empty, substrate is empty
        from computronium.core.joint.state import CompositeState
        return CompositeState(
            activity={"x": y, "y": y},
            plastic={},
            substrate={},
        )
    return transition_fn


def make_composite_state(x: Tensor):
    """Create a simple CompositeState for stability checking."""
    from computronium.core.joint.state import CompositeState
    return CompositeState(
        activity={"x": x},
        plastic={},
        substrate={},
    )


def check_stability(
    guard,
    transition_fn,
    x: Tensor,
    step: int,
    context=None,
):
    """Check stability at current step."""
    state = make_composite_state(x)
    return guard(transition_fn, state, context)


def _lwf_train_step(
    model: ContinualJointSystem,
    x: Tensor,
    y: Tensor,
    task_id: int,
    lwf_loss_fn: "LwFLoss",
) -> dict[str, float]:
    """Training step for LwF arm with distillation loss."""
    # Get previous model logits for distillation (before training step)
    # Use the saved prev_model from LwFLoss
    prev_model = lwf_loss_fn.prev_model
    if prev_model is not None and task_id > 0:
        prev_model.eval()
        with torch.no_grad():
            prev_logits = prev_model(x, task_id=task_id)
    else:
        prev_logits = None
    
    model.train()
    
    # Run the joint system pipeline up to computing pseudo-gradients
    substrate = model.joint_system.substrate
    geometry = model.joint_system.geometry
    dynamics = model.joint_system.dynamics
    credit = model.joint_system.credit
    update = model.joint_system.update
    plasticity = model.joint_system.plasticity
    
    task_start = task_id * CL_CLASSES_PER_TASK
    task_end = task_start + CL_CLASSES_PER_TASK
    local_y = y  # Labels are already 0/1 from SplitMNIST
    
    # Initialize plastic state if needed
    psi = getattr(model, '_psi', None)
    if psi is None and hasattr(plasticity, 'initial_psi') and plasticity is not None:
        psi = plasticity.initial_psi(model.joint_system.context, batch_size=x.shape[0])
    if psi is not None:
        device = x.device
        batch_size = x.shape[0]
        new_psi = {}
        for k, v in psi.items():
            if v.shape[0] != batch_size:
                if v.shape[0] == 1:
                    new_psi[k] = v.expand(batch_size, -1).to(device)
                else:
                    new_psi[k] = v[:batch_size].to(device)
            else:
                new_psi[k] = v.to(device)
        psi = new_psi
    
    # Run the pipeline
    from computronium.core.pipeline import forward_pass
    from computronium.core.ontology import Phase, SystemState
    from computronium.core.joint.state import CompositeState
    from contextlib import nullcontext
    
    grad_ctx = nullcontext() if credit.requires_autograd else torch.no_grad()
    with grad_ctx:
        states: dict[Phase, SystemState] = {}
        initial_activations = forward_pass(substrate, geometry, x)
        
        for phase in credit.phases:
            state = SystemState(x=x, y=local_y)
            state.activations = initial_activations
            target = local_y if phase is Phase.NUDGED else None
            settled = dynamics.settle(state, geometry, substrate, target=target)
            if phase is Phase.NUDGED:
                # Compute loss only on task-relevant logits
                settled.loss = _masked_task_loss(settled, local_y, task_start, task_end)
            settled.energy = dynamics.compute_energy(settled, geometry)
            states[phase] = settled
        
        output = states.get(Phase.NUDGED, states.get(Phase.FREE))
        if output is None:
            output = SystemState(x=x, y=local_y)
            output.activations = initial_activations
        loss = output.loss
        if loss is None:
            loss = _masked_task_loss(output, local_y, task_start, task_end)
        elif not isinstance(loss, Tensor):
            loss = torch.as_tensor(loss)
        if output.energy is None:
            output.energy = dynamics.compute_energy(output, geometry)
        
        # Add LwF distillation loss
        logits = output.activations[-1] if isinstance(output.activations, list) else output.activations
        lwf_loss = lwf_loss_fn(logits, y, task_id, prev_logits)
        total_loss = lwf_loss
        
        pseudo_grads = credit.compute_pseudo_gradient(states, total_loss, geometry)
        geometry.update_params(update.step(geometry.params, pseudo_grads, geometry))
        
        # Step plasticity if present
        if psi is not None and hasattr(plasticity, 'step') and plasticity is not None:
            acts = output.activations
            if acts is not None:
                final_acts = acts[-1] if isinstance(acts, list) else acts
                z = CompositeState(
                    activity={"x": x, "y": final_acts},
                    plastic=psi,
                    substrate={},
                )
                psi = plasticity.step(psi, z, model.joint_system.context)
    
    model._psi = psi
    
    loss_val = total_loss.item() if isinstance(total_loss, Tensor) else float(total_loss)
    energy_val = output.energy.item() if isinstance(output.energy, Tensor) else float(output.energy) if output.energy is not None else 0.0
    metrics = {
        "loss": loss_val,
        "energy": energy_val,
        "accuracy": output.metrics.get("accuracy", 0.0),
    }
    metrics.update({
        k: v
        for k, v in output.metrics.items()
        if isinstance(v, (int, float)) and k != "accuracy"
    })
    return metrics


def _si_train_step(
    model: ContinualJointSystem,
    x: Tensor,
    y: Tensor,
    task_id: int,
    si_tracker: "SynapticIntelligence",
) -> dict[str, float]:
    """Training step for SI arm with importance-weighted regularization."""
    # Standard joint system step
    metrics = model.train_step(x, y, task_id=task_id)
    
    # Add SI regularization loss to gradients
    si_loss = si_tracker.regularization_loss()
    if si_loss != 0:
        si_loss.backward()
    
    return metrics


def run_continual_learning(
    arm_name: str,
    config: CLConfig,
    protocol: str = "task_incremental",
) -> CLMetrics:
    """Run continual learning for one arm."""
    import copy
    import random
    import time

    device_str = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device = torch.device(device_str)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Create task loaders
    from computronium.domains.vision import SplitMNIST
    from computronium.domains.base import TaskSplit
    from torch.utils.data import DataLoader

    task_loaders = []
    for task_id in range(CL_NUM_TASKS):
        task = SplitMNIST(task_id=task_id, batch_size=config.batch_size, device=device_str, num_workers=config.num_workers)
        task.setup()
        task_loaders.append(task.get_dataloader(TaskSplit.TRAIN))

    test_loaders = []
    for task_id in range(CL_NUM_TASKS):
        task = SplitMNIST(task_id=task_id, batch_size=config.batch_size, device=device_str, num_workers=config.num_workers)
        task.setup()
        test_loaders.append(task.get_dataloader(TaskSplit.TEST))

    # Create arm
    model: ContinualJointSystem
    extra: dict[str, object] = {}
    if arm_name == "fast_weights":
        model = create_fast_weight_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
    elif arm_name == "ewc":
        model, si = create_ewc_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str, config.ewc_lambda)
        extra["si"] = si
    elif arm_name == "backprop":
        model = create_backprop_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
    elif arm_name == "replay":
        model, buffer = create_replay_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str, config.replay_capacity)
        extra["buffer"] = buffer
    elif arm_name == "lwf":
        model, lwf_loss = create_lwf_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra["lwf_loss"] = lwf_loss
    elif arm_name == "si":
        model, si = create_si_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra["si"] = si
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    # Stability guard
    guard = create_stability_guard(
        threshold=config.stability_threshold,
        statistic="fast_proxy",
        window=config.stability_window,
    )
    transition_fn = make_transition_fn(model)
    # Get context from joint system for stability guard
    guard_context = model.joint_system.context

    # Training
    accuracy_matrix = [[0.0 for _ in range(CL_NUM_TASKS)] for _ in range(CL_NUM_TASKS)]
    stability_verdicts: list = []
    start_time = time.perf_counter()

    if protocol == "task_incremental":
        # Task boundaries are signaled
        for task_id in range(CL_NUM_TASKS):
            model.set_task(task_id)

            # Arm-specific setup at task boundary
            if arm_name == "fast_weights":
                # Reset plastic state at task boundary (new episode)
                model.reset_plastic_state()
            elif arm_name == "ewc":
                si = extra["si"]  # type: SynapticIntelligence
                si.start_task()
            elif arm_name == "lwf":
                lwf_loss = extra["lwf_loss"]  # type: LwFLoss
                # Save current model as previous for distillation
                prev_model = copy.deepcopy(model)
                lwf_loss.set_prev_model(prev_model)
            elif arm_name == "si":
                si = extra["si"]  # type: SynapticIntelligence
                si.start_task()

            loader = task_loaders[task_id]

            for epoch in range(config.epochs_per_task):
                for batch_idx, (x, y) in enumerate(loader):
                    x = x.view(x.shape[0], -1).to(device)
                    y = y.to(device)

                    # Arm-specific training step
                    if arm_name == "lwf":
                        lwf_loss_fn = extra["lwf_loss"]  # type: LwFLoss
                        metrics = _lwf_train_step(model, x, y, task_id, lwf_loss_fn)
                    elif arm_name == "si":
                        si_tracker = extra["si"]  # type: SynapticIntelligence
                        metrics = _si_train_step(model, x, y, task_id, si_tracker)
                    else:
                        # Use joint system's train_step with task-masked loss
                        metrics = model.train_step(x, y, task_id=task_id)

                    # Stability check
                    verdict = check_stability(guard, transition_fn, x, step=epoch * len(loader) + batch_idx, context=guard_context)
                    stability_verdicts.append(verdict)

                    # Replay buffer update
                    if arm_name == "replay":
                        buffer = extra["buffer"]  # type: ReplayBuffer
                        buffer.add(x, y, task_id)

                    # Replay training
                    if arm_name == "replay" and len(extra["buffer"]) >= config.batch_size:
                        buffer = extra["buffer"]  # type: ReplayBuffer
                        rx, ry, rt = buffer.sample(config.batch_size)
                        # For replay, we need to train on the replay task
                        # Use the replay sample's task_id
                        replay_task_id = rt[0].item()
                        model.train_step(rx, ry, task_id=replay_task_id)

                # End of task: update importance for EWC/SI
                if arm_name == "ewc":
                    si = extra["si"]  # type: SynapticIntelligence
                    si.update_importance()
                elif arm_name == "si":
                    si = extra["si"]  # type: SynapticIntelligence
                    si.update_importance()

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
                        task_start = eval_task_id * CL_CLASSES_PER_TASK
                        task_end = task_start + CL_CLASSES_PER_TASK
                        task_logits = logits[:, task_start:task_end]
                        pred = task_logits.argmax(dim=1)
                        local_y = y % CL_CLASSES_PER_TASK
                        correct += (pred == local_y).sum().item()
                        total += y.shape[0]
                accuracy_matrix[eval_task_id][task_id] = correct / total if total > 0 else 0.0

    elif protocol == "task_free":
        # No task boundaries - gradual shift (simulate by mixing tasks)
        all_loaders = [iter(task_loaders[i]) for i in range(CL_NUM_TASKS)]
        total_batches = config.epochs_per_task * max(len(l) for l in task_loaders)

        for batch_idx in range(total_batches):
            task_id = batch_idx % CL_NUM_TASKS
            model.set_task(task_id)

            try:
                x, y = next(all_loaders[task_id])
            except StopIteration:
                all_loaders[task_id] = iter(task_loaders[task_id])
                x, y = next(all_loaders[task_id])

            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)

            model.train_step(x, y, task_id=task_id)

            verdict = check_stability(guard, transition_fn, x, step=batch_idx, context=guard_context)
            stability_verdicts.append(verdict)

            if arm_name == "replay":
                buffer = extra["buffer"]  # type: ReplayBuffer
                buffer.add(x, y, task_id)

            # Periodic evaluation
            if batch_idx % (total_batches // CL_NUM_TASKS) == 0:
                eval_task = batch_idx // (total_batches // CL_NUM_TASKS)
                if eval_task < CL_NUM_TASKS:
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
                                task_start = eval_task_id * CL_CLASSES_PER_TASK
                                task_end = task_start + CL_CLASSES_PER_TASK
                                task_logits = elogits[:, task_start:task_end]
                                epred = task_logits.argmax(dim=1)
                                local_ey = ey % CL_CLASSES_PER_TASK
                                correct += (epred == local_ey).sum().item()
                                total += ey.shape[0]
                        accuracy_matrix[eval_task_id][eval_task] = correct / total if total > 0 else 0.0

    total_time = time.perf_counter() - start_time

    # Final evaluation on all tasks
    final_metrics = compute_cl_metrics(model, test_loaders, CL_NUM_TASKS - 1, accuracy_matrix)
    final_metrics.total_time_s = total_time
    final_metrics.stability_verdicts = stability_verdicts
    final_metrics.max_spectral_radius = max(v.statistic for v in stability_verdicts) if stability_verdicts else 0.0

    # Memory footprint
    if hasattr(model.joint_system, "plasticity") and hasattr(model.joint_system.plasticity, "fast_weight_dim"):  # type: ignore[attr-defined]
        final_metrics.plastic_state_bytes = model.joint_system.plasticity.fast_weight_dim * 4 * config.batch_size  # type: ignore[attr-defined]
    if arm_name == "replay" and "buffer" in extra:
        final_metrics.replay_buffer_bytes = extra["buffer"].memory_bytes()  # type: ignore[attr-defined]

    return final_metrics


def run_continual_learning_suite(
    arms: list[str],
    protocols: list[str],
    output_dir: str | Path,
    config: CLConfig | None = None,
    seeds: int = 3,
) -> dict[str, dict[str, dict[str, object]]]:
    """Run continual learning benchmark suite."""
    import json
    import time
    import random

    config = config or CLConfig()
    output_dir = Path(output_dir)

    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    config.device = device

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for arm in arms:
        all_results[arm] = {}
        for protocol in protocols:
            print(f"\n=== {arm} / {protocol} ===")
            arm_results: dict[str, list[dict[str, float | int]] | dict[str, float]] = {"seeds": []}

            for seed in range(seeds):
                print(f"  Seed {seed}...")
                config.seed = seed
                metrics = run_continual_learning(arm, config, protocol)
                arm_results["seeds"].append({
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
                    "stability_kills": sum(1 for v in metrics.stability_verdicts if getattr(v, 'kill', False)),
                    "total_time_s": metrics.total_time_s,
                })
                print(f"    Avg forgetting: {metrics.avg_forgetting:.4f}, BWT: {metrics.backward_transfer:.4f}")

            # Aggregate across seeds
            seeds_list = arm_results["seeds"]
            if seeds_list:
                for key in ["avg_forgetting", "backward_transfer", "forward_transfer", "max_spectral_radius", "total_time_s"]:
                    vals = [float(s[key]) for s in seeds_list]
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


__all__ = [
    "SystemTrainer",
    "SystemTrainerConfig",
    "compose_joint_system",
    "compose_joint_system_from_configs",
    "compose_system",
    "compose_system_from_configs",
    "create_backprop_system",
    "create_eqprop_system",
    "create_fa_system",
    "create_fast_weight_eqprop_system",
    "create_routing_eqprop_system",
    "extract_config",
    # Continual Learning Arms (6-D Joint Systems)
    "create_fast_weight_arm",
    "create_ewc_arm",
    "create_backprop_arm",
    "create_replay_arm",
    "create_lwf_arm",
    "create_si_arm",
    "ContinualJointSystem",
    "CLConfig",
    "CLMetrics",
    "run_continual_learning",
    "run_continual_learning_suite",
]
