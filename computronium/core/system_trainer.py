"""SystemTrainer: Orchestrates the 5-Layer Ontology Pipeline.

Replaces the monolithic CoreTrainer with a trainer that operates on the
composable 5-D tensor product (Substrate ⊗ Geometry ⊗ StateDynamics ⊗
CreditAssignment ⊗ ParameterUpdate).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch
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
    SystemState,
)

if TYPE_CHECKING:
    from computronium.config.experiment import ExperimentConfig


class JointSystem(Protocol):
    """Protocol for 6-D joint systems."""

    substrate: Substrate
    geometry: Geometry
    dynamics: StateDynamics
    plasticity: PlasticityPrimitive
    credit: CreditAssignment
    update: ParameterUpdate

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def to_spec(self) -> dict: ...
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

    @classmethod
    def from_configs(
        cls,
        experiment_config: ExperimentConfig,
        train_data: _DataProvider,
        val_data: _DataProvider | None = None,
    ) -> SystemTrainer:
        """Create a SystemTrainer from an ExperimentConfig.

        This is the primary factory for creating trainers from the unified
        configuration. It composes the 5-D ontology system from the ontology
        config and creates a SystemTrainer with the training config.

        Args:
            experiment_config: Unified ExperimentConfig with all hyperparameters.
            train_data: Training data provider (DataLoader, Task, etc.).
            val_data: Optional validation data provider.

        Returns:
            A configured SystemTrainer ready to call fit().
        """
        from computronium.core.ontology import (
            AnalogSubstrate,
            BackpropCredit,
            DigitalSubstrate,
            ElasticConsolidationUpdate,
            EnergyMinimizationDynamics,
            EuclideanUpdate,
            FeedforwardGeometry,
            InstantaneousDynamics,
            LocalGoodnessCredit,
            MemristiveSubstrate,
            NaturalGradientUpdate,
            NeuromorphicSubstrate,
            OpticalSubstrate,
            PredictiveSettlingDynamics,
            QuantumSubstrate,
            RandomProjectionsCredit,
            RecurrentGeometry,
            RiemannianOrthogonalUpdate,
            SpectralConstrainedUpdate,
            SpikeIntegrationDynamics,
            SystemConfig,
            TargetInversionCredit,
            TemporalTraceCredit,
            ThermodynamicContrast,
            TileGeometry,
        )
        from computronium.core.system_trainer import compose_system

        exp = experiment_config
        model = exp.model
        training = exp.training
        hardware = exp.hardware

        # Build validated SystemConfig from experiment
        sys_config = SystemConfig.from_experiment(exp)

        # Build substrate from validated config
        substrate_cfg = sys_config.substrate
        substrate_map = {
            "digital": DigitalSubstrate,
            "analog": AnalogSubstrate,
            "memristive": MemristiveSubstrate,
            "neuromorphic": NeuromorphicSubstrate,
            "optical": OpticalSubstrate,
            "quantum": QuantumSubstrate,
        }
        # Determine substrate type from precision field
        substrate_key = substrate_cfg.precision.lower()
        if substrate_key in (
            "float32",
            "float16",
            "bfloat16",
            "int8",
            "int4",
            "binary",
        ):
            substrate_key = "digital"
        substrate_cls = substrate_map.get(substrate_key, DigitalSubstrate)
        substrate = substrate_cls(substrate_cfg)

        # Build geometry from validated config
        geometry_cfg = sys_config.geometry
        topology_type = geometry_cfg.topology_type.lower()
        if topology_type == "feedforward":
            geometry = FeedforwardGeometry(geometry_cfg)
        elif topology_type in ("recurrent", "recurrent_attractor"):
            hidden_dim = (
                geometry_cfg.hidden_dims[-1]
                if geometry_cfg.hidden_dims
                else model.output_dim
            )
            geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)
        elif topology_type in ("tile_mesh", "tile"):
            geometry = TileGeometry(
                geometry_cfg,
                neurons_per_tile=model.neurons_per_tile,
                tiles_per_layer=model.tiles_per_layer,
            )
        else:
            geometry = FeedforwardGeometry(geometry_cfg)

        # Build dynamics from validated config
        dynamics_cfg = sys_config.dynamics
        dynamics_type = dynamics_cfg.dynamics_type.lower()
        if dynamics_type == "energy_minimization":
            dynamics = EnergyMinimizationDynamics(dynamics_cfg)
        elif dynamics_type == "predictive_settling":
            dynamics = PredictiveSettlingDynamics(dynamics_cfg)
        elif dynamics_type == "spike_integration":
            dynamics = SpikeIntegrationDynamics(dynamics_cfg)
        else:
            dynamics = InstantaneousDynamics(dynamics_cfg)

        # Build credit from validated config
        credit_cfg = sys_config.credit
        credit_type = credit_cfg.credit_type.lower()
        if credit_type in ("thermodynamic_contrast", "equilibrium"):
            credit = ThermodynamicContrast(credit_cfg)
        elif credit_type in ("random_projections", "feedback_alignment"):
            credit = RandomProjectionsCredit(credit_cfg)
        elif credit_type in ("local_goodness", "forward_only"):
            credit = LocalGoodnessCredit(credit_cfg)
        elif credit_type in ("temporal_trace", "spiking"):
            credit = TemporalTraceCredit(credit_cfg)
        elif credit_type in ("target_inversion", "target_prop"):
            credit = TargetInversionCredit(credit_cfg)
        else:
            credit = BackpropCredit(credit_cfg)

        # Build update from validated config
        update_cfg = sys_config.update
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

        # Compose the system
        system = compose_system(substrate, geometry, dynamics, credit, update)

        # Create trainer config
        trainer_config = SystemTrainerConfig(
            max_epochs=training.epochs,
            batch_size=training.batch_size,
            val_batch_size=training.val_batch_size,
            device=hardware.device,
            grad_clip=training.grad_clip,
            track_energy=training.track_energy,
            track_flops=training.track_flops,
            track_memory=training.track_memory,
            log_every_n_steps=training.log_every_n_steps,
            seed=exp.seed,
            deterministic=exp.deterministic,
        )

        return cls(
            system=system,
            config=trainer_config,
            train_data=train_data,
            val_data=val_data,
        )

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


def compose_system(
    substrate: Substrate,
    geometry: Geometry,
    dynamics: StateDynamics,
    credit: CreditAssignment,
    update: ParameterUpdate,
) -> System:
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
    class _ComposedSystem:
        substrate: Substrate
        geometry: Geometry
        dynamics: StateDynamics
        credit: CreditAssignment
        update: ParameterUpdate

        def to_spec(self) -> dict:
            """Serialize the System to a specification dictionary.

            Returns:
                Dictionary containing schema_version and all 5 axis configs.
            """
            geometry_dict = dataclasses.asdict(self.geometry.config)
            # Include recurrent_weight from geometry if present (runtime state)
            if (
                hasattr(self.geometry, "_recurrent_weight")
                and self.geometry._recurrent_weight is not None
            ):
                geometry_dict["recurrent_weight"] = (
                    self.geometry._recurrent_weight.tolist()
                )

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
                AnalogSubstrate,
                BackpropCredit,
                CreditAssignmentConfig,
                DigitalSubstrate,
                ElasticConsolidationUpdate,
                EnergyMinimizationDynamics,
                EuclideanUpdate,
                FeedforwardGeometry,
                GeometryConfig,
                InstantaneousDynamics,
                LocalGoodnessCredit,
                MemristiveSubstrate,
                NaturalGradientUpdate,
                NeuromorphicSubstrate,
                OpticalSubstrate,
                ParameterUpdateConfig,
                PredictiveSettlingDynamics,
                QuantumSubstrate,
                RandomProjectionsCredit,
                RecurrentGeometry,
                RiemannianOrthogonalUpdate,
                SpectralConstrainedUpdate,
                SpikeIntegrationDynamics,
                StateDynamicsConfig,
                SubstrateConfig,
                TargetInversionCredit,
                TemporalTraceCredit,
                ThermodynamicContrast,
            )

            # Reconstruct substrate
            substrate_cfg = SubstrateConfig(**spec["substrate"])
            substrate_map = {
                "digital": DigitalSubstrate,
                "float32": DigitalSubstrate,
                "float16": DigitalSubstrate,
                "bfloat16": DigitalSubstrate,
                "int8": DigitalSubstrate,
                "int4": DigitalSubstrate,
                "binary": DigitalSubstrate,
                "analog": AnalogSubstrate,
                "memristive": MemristiveSubstrate,
                "memristor": MemristiveSubstrate,
                "neuromorphic": NeuromorphicSubstrate,
                "optical": OpticalSubstrate,
                "quantum": QuantumSubstrate,
                "quantized": DigitalSubstrate,
                "noisy": DigitalSubstrate,
            }
            # Use device field to determine substrate type, fallback to precision
            substrate_key = (
                substrate_cfg.device.lower()
                if substrate_cfg.device != "cpu"
                else substrate_cfg.precision.lower()
            )
            substrate_cls = substrate_map.get(substrate_key, DigitalSubstrate)
            substrate = substrate_cls(substrate_cfg)

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
            credit_type = credit_cfg.credit_type.lower()
            if credit_type in ("thermodynamic_contrast", "equilibrium"):
                credit = ThermodynamicContrast(credit_cfg)
            elif credit_type in ("random_projections", "feedback_alignment"):
                credit = RandomProjectionsCredit(credit_cfg)
            elif credit_type in ("local_goodness", "forward_only"):
                credit = LocalGoodnessCredit(credit_cfg)
            elif credit_type in ("temporal_trace", "spiking"):
                credit = TemporalTraceCredit(credit_cfg)
            elif credit_type in ("target_inversion", "target_prop"):
                credit = TargetInversionCredit(credit_cfg)
            else:
                credit = BackpropCredit(credit_cfg)

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
            state = SystemState(x=x, y=y)

            # 1. Substrate + Geometry: Forward pass
            state.activations = self.geometry.forward(x, self.substrate)
            if state.activations is not None:
                state.activations = self.substrate.inject_state_noise(state.activations)

            # 2. StateDynamics: Free phase
            free_state = self.dynamics.settle(
                state, self.geometry, self.substrate, target=None
            )
            free_state.energy = self.dynamics.compute_energy(free_state, self.geometry)

            # 3. StateDynamics: Nudged phase
            nudged_state = self.dynamics.settle(
                state, self.geometry, self.substrate, target=y
            )
            nudged_state.energy = self.dynamics.compute_energy(
                nudged_state, self.geometry
            )
            nudged_state.loss = self._compute_loss(nudged_state, y)

            # 4. CreditAssignment
            pseudo_grads = self.credit.compute_pseudo_gradient(
                free_state, nudged_state, nudged_state.loss, self.geometry
            )

            # 5. ParameterUpdate
            new_params = self.update.step(
                self.geometry.params, pseudo_grads, self.geometry
            )
            self.geometry.update_params(new_params)

            return {
                "loss": nudged_state.loss.item()
                if nudged_state.loss is not None
                else 0.0,
                "energy": free_state.energy.item()
                if free_state.energy is not None
                else 0.0,
                "accuracy": nudged_state.metrics.get("accuracy", 0.0),
            }

        def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:
            acts = state.activations
            if acts is None:
                return torch.tensor(0.0)
            if isinstance(acts, list):
                logits = acts[-1]
            else:
                logits = acts
            loss = torch.nn.functional.cross_entropy(logits, y)
            # Compute accuracy and store in state.metrics
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()
            state.metrics = {"accuracy": acc}
            return loss

        def forward(self, x: Tensor) -> Tensor:
            state = SystemState(x=x)
            state.activations = self.geometry.forward(x, self.substrate)
            if state.activations is not None:
                state.activations = self.substrate.inject_state_noise(state.activations)
            state = self.dynamics.settle(
                state, self.geometry, self.substrate, target=None
            )
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
            momentum=0.9,
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
        BackpropCredit,
        DigitalSubstrate,
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        InstantaneousDynamics,
        LocalGoodnessCredit,
        NaturalGradientUpdate,
        NeuromorphicSubstrate,
        OpticalSubstrate,
        PredictiveSettlingDynamics,
        QuantumSubstrate,
        RandomProjectionsCredit,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SpectralConstrainedUpdate,
        SpikeIntegrationDynamics,
        TargetInversionCredit,
        ThermodynamicContrast,
    )

    # Instantiate substrate from config
    substrate_map = {
        "digital": DigitalSubstrate,
        "analog": "AnalogSubstrate",
        "memristive": "MemristiveSubstrate",
        "neuromorphic": NeuromorphicSubstrate,
        "optical": OpticalSubstrate,
        "quantum": QuantumSubstrate,
        "quantized": "QuantizedSubstrate",
        "noisy": "NoisySubstrate",
    }
    substrate_cls_name = substrate_map.get(
        substrate.precision.lower(), "DigitalSubstrate"
    )
    substrate_cls = globals().get(substrate_cls_name, DigitalSubstrate)
    substrate_instance = substrate_cls(substrate)

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
    credit_type = credit.credit_type.lower()
    if credit_type in ("thermodynamic_contrast", "equilibrium"):
        credit_instance = ThermodynamicContrast(credit)
    elif credit_type in ("random_projections", "feedback_alignment"):
        credit_instance = RandomProjectionsCredit(credit)
    elif credit_type in ("local_goodness", "forward_only"):
        credit_instance = LocalGoodnessCredit(credit)
    elif credit_type in ("temporal_trace", "spiking"):
        credit_instance = TargetInversionCredit(
            credit
        )  # Will use TemporalTraceCredit if available
    elif credit_type in ("target_inversion", "target_prop"):
        credit_instance = TargetInversionCredit(credit)
    else:
        credit_instance = BackpropCredit(credit)

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


def compose_joint_system(
    substrate: Substrate,
    geometry: Geometry,
    dynamics: StateDynamics,
    plasticity: PlasticityPrimitive,
    credit: CreditAssignment,
    update: ParameterUpdate,
) -> JointSystem:
    """Compose a JointSystem from six orthogonal components.

    This is the primary factory function for creating computronium joint systems
    from the 6-D ontology primitives (S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U).

    Example:
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=RecurrentGeometry(GeometryConfig(...)),
            dynamics=EnergyMinimizationDynamics(StateDynamicsConfig(...)),
            plasticity=RoutingPlasticity(RoutingPlasticityConfig(...)),
            credit=ThermodynamicContrast(CreditAssignmentConfig(...)),
            update=EuclideanUpdate(ParameterUpdateConfig(...)),
        )
    """
    from computronium.core.joint.transition import NullPlasticity
    from computronium.core.plasticity import NullPlasticity as _NullPlasticity

    @dataclass(frozen=True, slots=True)
    class _JointSystem:
        substrate: Substrate
        geometry: Geometry
        dynamics: StateDynamics
        plasticity: PlasticityPrimitive
        credit: CreditAssignment
        update: ParameterUpdate

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            """Execute one training step through the 6-layer pipeline."""
            from computronium.core.joint.state import CompositeState

            # Build initial joint state
            z = CompositeState(
                activity={"x": x, "y": y},
                plastic=self.plasticity.initial_psi(
                    self._make_context(), batch_size=x.shape[0]
                ),
                substrate={},
            )

            # Build context
            context = self._make_context()

            # Run joint transition (settling + plasticity)
            # For now, use the 5-D system train_step as the base
            # and apply plasticity within the settling loop
            state = SystemState(x=x, y=y)

            # 1. Substrate + Geometry: Forward pass (initial state)
            state.activations = self.geometry.forward(x, self.substrate)
            if state.activations is not None:
                state.activations = self.substrate.inject_state_noise(state.activations)

            # 2. StateDynamics: Free phase settling
            free_state = self.dynamics.settle(
                state, self.geometry, self.substrate, target=None
            )
            free_state.energy = self.dynamics.compute_energy(free_state, self.geometry)

            # 3. StateDynamics: Nudged phase settling
            nudged_state = self.dynamics.settle(
                state, self.geometry, self.substrate, target=y
            )
            nudged_state.energy = self.dynamics.compute_energy(
                nudged_state, self.geometry
            )
            nudged_state.loss = self._compute_loss(nudged_state, y)

            # 4. CreditAssignment: Compute pseudo-gradients
            pseudo_grads = self.credit.compute_pseudo_gradient(
                free_state, nudged_state, nudged_state.loss, self.geometry
            )

            # 5. ParameterUpdate: Apply updates
            new_params = self.update.step(
                self.geometry.params, pseudo_grads, self.geometry
            )
            self.geometry.update_params(new_params)

            return {
                "loss": nudged_state.loss.item()
                if nudged_state.loss is not None
                else 0.0,
                "energy": free_state.energy.item()
                if free_state.energy is not None
                else 0.0,
                "accuracy": nudged_state.metrics.get("accuracy", 0.0),
            }

        def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:
            """Compute task loss from final state."""
            acts = state.activations
            if acts is None:
                return torch.tensor(0.0)
            logits = acts[-1] if isinstance(acts, list) else acts
            loss = torch.nn.functional.cross_entropy(logits, y)
            # Compute accuracy and store in state.metrics
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()
            state.metrics = {"accuracy": acc}
            return loss

        def forward(self, x: Tensor) -> Tensor:
            """Inference forward pass (free phase only, no weight updates)."""
            state = SystemState(x=x)
            state.activations = self.geometry.forward(x, self.substrate)
            if state.activations is not None:
                state.activations = self.substrate.inject_state_noise(state.activations)
            state = self.dynamics.settle(
                state, self.geometry, self.substrate, target=None
            )
            acts = state.activations
            if acts is None:
                return torch.empty(0)
            if isinstance(acts, list):
                return acts[-1]
            return acts

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

        def to_spec(self) -> dict:
            """Serialize the JointSystem to a specification dictionary."""
            geometry_dict = dataclasses.asdict(self.geometry.config)
            if (
                hasattr(self.geometry, "_recurrent_weight")
                and self.geometry._recurrent_weight is not None
            ):
                geometry_dict["recurrent_weight"] = (
                    self.geometry._recurrent_weight.tolist()
                )

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
        class _NullJointSystem:
            def __init__(self, system):
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

            def _make_context(self):
                return (
                    base_system._make_context()
                    if hasattr(base_system, "_make_context")
                    else None
                )

            def to_spec(self):
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

            def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:
                return base_system._compute_loss(state, y)

        return _NullJointSystem(base_system)

    return _JointSystem(
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        plasticity=plasticity,
        credit=credit,
        update=update,
    )


def compose_joint_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    plasticity: PlasticityConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> JointSystem:
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
        BackpropCredit,
        DigitalSubstrate,
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        InstantaneousDynamics,
        LocalGoodnessCredit,
        NaturalGradientUpdate,
        NeuromorphicSubstrate,
        OpticalSubstrate,
        PredictiveSettlingDynamics,
        QuantumSubstrate,
        RandomProjectionsCredit,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SpectralConstrainedUpdate,
        SpikeIntegrationDynamics,
        TargetInversionCredit,
        TemporalTraceCredit,
        ThermodynamicContrast,
    )
    from computronium.core.plasticity import (
        NullPlasticity,
        create_fast_weight_plasticity,
        create_routing_plasticity,
        create_substrate_coupled_plasticity,
    )

    # Instantiate substrate from config
    substrate_map = {
        "digital": DigitalSubstrate,
        "analog": "AnalogSubstrate",
        "memristive": "MemristiveSubstrate",
        "neuromorphic": NeuromorphicSubstrate,
        "optical": OpticalSubstrate,
        "quantum": QuantumSubstrate,
        "quantized": "QuantizedSubstrate",
        "noisy": "NoisySubstrate",
    }
    substrate_cls_name = substrate_map.get(
        substrate.precision.lower(), "DigitalSubstrate"
    )
    substrate_cls = globals().get(substrate_cls_name, DigitalSubstrate)
    substrate_instance = substrate_cls(substrate)

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
    credit_type = credit.credit_type.lower()
    if credit_type in ("thermodynamic_contrast", "equilibrium"):
        credit_instance = ThermodynamicContrast(credit)
    elif credit_type in ("random_projections", "feedback_alignment"):
        credit_instance = RandomProjectionsCredit(credit)
    elif credit_type in ("local_goodness", "forward_only"):
        credit_instance = LocalGoodnessCredit(credit)
    elif credit_type in ("temporal_trace", "spiking"):
        credit_instance = TemporalTraceCredit(credit)
    elif credit_type in ("target_inversion", "target_prop"):
        credit_instance = TargetInversionCredit(credit)
    else:
        credit_instance = BackpropCredit(credit)

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
    from computronium.core.plasticity import RoutingPlasticity, RoutingPlasticityConfig

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
        RoutingPlasticityConfig(
            gate_dim=gate_dim,
            temperature=1.0,
            decay=0.99,
            learning_rate=0.01,
        )
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
        FastWeightPlasticityConfig,
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
        FastWeightPlasticityConfig(
            fast_weight_dim=fast_weight_dim,
            decay=0.99,
            learning_rate=0.1,
        )
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


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
]
