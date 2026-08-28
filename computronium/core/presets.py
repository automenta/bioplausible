"""Preset Factory Functions for Common Bioplausible Systems.

Provides one-line system construction for common 5-D and 6-D coordinates.
Instead of 20 lines of config, users can call:
    system = create_backprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_eqprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10, beta=0.5, n_iters=20)
    system = create_fa_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_routing_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_fast_weight_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from computronium.core.plasticity import (
    FastWeightPlasticity,
    RoutingPlasticity,
)
from computronium.core.system_trainer import (
    compose_joint_system,
    compose_system,
)
from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    RandomProjectionsCredit,
    RecurrentGeometry,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    TargetInversionCredit,
    TemporalTraceCredit,
    ThermodynamicContrast,
)

if TYPE_CHECKING:
    from computronium.core.system_trainer import JointSystem
    from computronium.ontology import System


def _default_substrate(device: str = "cpu") -> DigitalSubstrate:
    """Create a default digital substrate."""
    return DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )
    )


def _mlp_geometry(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    init_scale: float = 0.1,
) -> FeedforwardGeometry:
    """Create a standard MLP feedforward geometry."""
    return FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            init_scale=init_scale,
        )
    )


def _recurrent_geometry(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    init_scale: float = 0.1,
) -> RecurrentGeometry:
    """Create a recurrent geometry for EqProp."""
    # Use the last hidden dim as the recurrent state dimension
    hidden_dim = hidden_dims[-1] if hidden_dims else output_dim
    return RecurrentGeometry(
        GeometryConfig.recurrent(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            init_scale=init_scale,
        ),
        hidden_dim=hidden_dim,
    )


def _default_credit() -> BackpropCredit:
    """Create default backprop credit assignment."""
    return BackpropCredit(CreditAssignmentConfig.gradient())


def _eqprop_credit(beta: float = 0.5) -> ThermodynamicContrast:
    """Create EqProp thermodynamic contrast credit assignment."""
    return ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=beta)
    )


def _default_update(lr: float = 0.001) -> EuclideanUpdate:
    """Create default Euclidean update."""
    return EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr))


# ============================================================
# 5-D System Factories (Standard Ontology)
# ============================================================


def create_backprop_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> System:
    """Create a standard Backprop MLP system (5-D coordinate).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (e.g., (256, 128))
        output_dim: Output dimension (e.g., 10 for MNIST)
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with Backprop credit assignment.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_eqprop_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (512, 512, 512),
    output_dim: int = 10,
    beta: float = 0.1,
    inference_steps: int = 20,
    lr: float = 0.001,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> System:
    """Create an Equilibrium Propagation MLP system (5-D coordinate).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (default: 3 layers of 512)
        output_dim: Output dimension (e.g., 10 for MNIST)
        beta: Nudge strength for EqProp (default: 0.1, matches vision parity)
        inference_steps: Number of settling iterations (default: 20)
        lr: Learning rate (default: 0.001, matches vision parity)
        init_scale: Weight initialization scale
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with ThermodynamicContrast credit assignment
        and EnergyMinimization dynamics.
    """
    substrate = _default_substrate(device)
    geometry = _recurrent_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=inference_steps,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )
    credit = _eqprop_credit(beta)
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_fa_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    feedback_scale: float = 0.01,
    device: str = "cpu",
) -> System:
    """Create a Feedback Alignment MLP system (5-D coordinate).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (e.g., (256, 128))
        output_dim: Output dimension (e.g., 10 for MNIST)
        lr: Learning rate
        init_scale: Weight initialization scale
        feedback_scale: Scale for random feedback matrix initialization
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with RandomProjections (FA) credit assignment.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = RandomProjectionsCredit(
        CreditAssignmentConfig.random_projections(
            beta=0.5,
            feedback_scale=feedback_scale,
        )
    )
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_ff_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    layer_lr: float = 0.03,
    classifier_lr: float = 0.01,
    threshold: float = 2.0,
    num_layers: int | None = None,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> System:
    """Create a Forward-Forward MLP system (5-D coordinate, native ontology).

    Implements Hinton's Forward-Forward algorithm using the 5-D ontology:
    - Two forward passes per batch (positive/negative with label injection)
    - Layer-local goodness objective (sum of squared activations)
    - Per-layer independent optimizers
    - No backward pass through the network (biologically plausible)

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (e.g., (256, 128))
        output_dim: Output dimension (e.g., 10 for MNIST)
        layer_lr: Learning rate for hidden layers
        classifier_lr: Learning rate for output classifier
        threshold: Goodness threshold for softplus loss
        num_layers: Number of hidden layers (defaults to len(hidden_dims))
        init_scale: Weight initialization scale
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with custom Forward-Forward train_step.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.optim import Adam

    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = LocalGoodnessCredit(CreditAssignmentConfig.local_goodness())
    update = _default_update(layer_lr)

    base_system = compose_system(substrate, geometry, dynamics, credit, update)

    # Determine number of hidden layers
    n_layers = num_layers if num_layers is not None else len(hidden_dims)

    # Build per-layer linear modules with ReLU and L2 normalization to match FFLayer
    # Geometry params are at even indices: 0, 2, 4... (Linear layers)
    # with ReLU at odd indices: 1, 3, 5...
    layer_dims = [input_dim] + list(hidden_dims[:n_layers])
    layers = nn.ModuleList()
    layer_opts = []
    for i in range(n_layers):
        # Custom layer with L2 normalization like FFLayer
        class _FFLayer(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
                self.relu = nn.ReLU()

            def forward(self, x):
                # L2 normalize input like FFLayer
                x_dir = x / (x.norm(2, 1, keepdim=True) + 1e-4)
                return self.relu(self.linear(x_dir))

        layer = _FFLayer(layer_dims[i], layer_dims[i + 1])
        # Copy weights from geometry (even indices: 0, 2, 4...)
        param_idx = 2 * i
        weight_key = f"{param_idx}.weight"
        bias_key = f"{param_idx}.bias"
        if weight_key in geometry.params:
            with torch.no_grad():
                layer.linear.weight.copy_(geometry.params[weight_key])
                if bias_key in geometry.params:
                    layer.linear.bias.copy_(geometry.params[bias_key])
        layers.append(layer.to(device))
        layer_opts.append(Adam(layer.parameters(), lr=layer_lr, weight_decay=0.0))

    # Classifier on top of concatenated hidden states
    classifier_in_dim = sum(hidden_dims[:n_layers])
    classifier = nn.Linear(classifier_in_dim, output_dim).to(device)
    classifier_opt = Adam(classifier.parameters(), lr=classifier_lr, weight_decay=0.0)

    # Create a wrapper system that delegates to base but overrides train_step
    class _FFSystem:
        def __init__(self, base):
            self._base = base
            self.substrate = base.substrate
            self.geometry = base.geometry
            self.dynamics = base.dynamics
            self.credit = base.credit
            self.update = base.update

        def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            x = x.to(device)
            y = y.to(device)

            batch_size = x.shape[0]

            # Create positive and negative inputs with label injection
            x_pos = x.clone()
            x_neg = x.clone()

            y_neg = torch.randint(0, output_dim, (batch_size,), device=device)
            for i in range(batch_size):
                while y_neg[i] == y[i]:
                    y_neg[i] = torch.randint(0, output_dim, (1,), device=device).item()

            x_pos[:, :output_dim] = 0.0
            x_neg[:, :output_dim] = 0.0
            x_pos[range(batch_size), y] = x.max()
            x_neg[range(batch_size), y_neg] = x.max()

            total_loss = 0.0
            h_pos, h_neg = x_pos, x_neg
            hidden_states_pos = []

            for i, (layer, opt) in enumerate(zip(layers, layer_opts)):
                h_pos = layer(h_pos)
                g_pos = (h_pos**2).mean(dim=1)

                h_neg = layer(h_neg)
                g_neg = (h_neg**2).mean(dim=1)

                loss = F.softplus(
                    torch.cat([-g_pos + threshold, g_neg - threshold])
                ).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

                hidden_states_pos.append(h_pos.detach())

                h_pos = h_pos.detach()
                h_neg = h_neg.detach()

            # Sync weights back to geometry for inference
            # Geometry params are at even indices: 0, 2, 4...
            with torch.no_grad():
                for i, layer in enumerate(layers):
                    param_idx = 2 * i
                    weight_key = f"{param_idx}.weight"
                    bias_key = f"{param_idx}.bias"
                    if weight_key in geometry.params:
                        geometry.params[weight_key].copy_(layer.linear.weight)
                        if bias_key in geometry.params:
                            geometry.params[bias_key].copy_(layer.linear.bias)

            # Train classifier on concatenated hidden states
            h_all = torch.cat(hidden_states_pos, dim=1).detach()
            logits = classifier(h_all)
            cls_loss = F.cross_entropy(logits, y)
            cls_acc = (logits.argmax(-1) == y).float().mean().item()

            classifier_opt.zero_grad()
            cls_loss.backward()
            classifier_opt.step()

            return {
                "loss": total_loss / max(n_layers, 1),
                "accuracy": cls_acc,
                "cls_loss": cls_loss.item(),
            }

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Use custom layers + classifier for inference to match training
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            x = x.to(device)
            h = x
            hidden_states = []
            for layer in layers:
                h = layer(h)
                hidden_states.append(h)
            h_all = torch.cat(hidden_states, dim=1)
            return classifier(h_all)

        def to_spec(self) -> dict:
            return self._base.to_spec()

        @classmethod
        def from_spec(cls, spec: dict) -> System:
            return base_system.from_spec(spec)

    return _FFSystem(base_system)


def create_pepita_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.01,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> System:
    """Create a PEPITA MLP system (5-D coordinate).

    PEPITA uses forward-only local learning with error-modulated input
    perturbation and layer-local contrastive updates.

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device

    Returns:
        A composed 5-D System with FeedforwardGeometry + InstantaneousDynamics
        + LocalGoodnessCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = LocalGoodnessCredit(CreditAssignmentConfig.local_goodness())
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_tp_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    beta: float = 0.1,
    settle_steps: int = 30,
    device: str = "cpu",
) -> System:
    """Create a Target Propagation MLP system (5-D coordinate).

    Target Propagation uses learned inverse mappings to propagate targets
    backwards through the network instead of gradients.

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        beta: Nudge strength for target propagation
        settle_steps: Number of settling iterations for predictive dynamics
        device: Target device

    Returns:
        A composed 5-D System with FeedforwardGeometry + PredictiveSettlingDynamics
        + TargetInversionCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = PredictiveSettlingDynamics(
        StateDynamicsConfig.predictive_settling(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = TargetInversionCredit(
        CreditAssignmentConfig.target_inversion(
            beta=beta,
            feedback_scale=0.01,
        )
    )
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_pc_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    beta: float = 0.5,
    settle_steps: int = 30,
    device: str = "cpu",
) -> System:
    """Create a Predictive Coding MLP system (5-D coordinate).

    Predictive Coding minimizes prediction errors through a hierarchy of
    top-down predictions and bottom-up error signals.

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        beta: Nudge strength for predictive coding
        settle_steps: Number of settling iterations
        device: Target device

    Returns:
        A composed 5-D System with RecurrentGeometry + PredictiveSettlingDynamics
        + LocalGoodnessCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _recurrent_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = PredictiveSettlingDynamics(
        StateDynamicsConfig.predictive_settling(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = LocalGoodnessCredit(CreditAssignmentConfig.local_goodness())
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_hebbian_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> System:
    """Create a Hebbian MLP system (5-D coordinate).

    Hebbian learning uses local correlation-based weight updates
    (neurons that fire together, wire together).

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device

    Returns:
        A composed 5-D System with FeedforwardGeometry + InstantaneousDynamics
        + LocalGoodnessCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = LocalGoodnessCredit(CreditAssignmentConfig.local_goodness())
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_snn_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> System:
    """Create a Spiking Neural Network MLP system (5-D coordinate).

    SNNs use spike-based computation with temporal integration dynamics
    and temporal trace credit assignment.

    This factory now aligns with the working YAML preset (snn_mnist.yaml)
    which uses InstantaneousDynamics + LocalGoodnessCredit for compatibility
    with SystemTrainer. For true spiking dynamics, use create_spiking_snn_mlp.

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device

    Returns:
        A composed 5-D System with FeedforwardGeometry + InstantaneousDynamics
        + LocalGoodnessCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = LocalGoodnessCredit(CreditAssignmentConfig.local_goodness())
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_spiking_snn_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    device: str = "cpu",
    max_steps: int = 30,
    beta: float = 0.1,
) -> System:
    """Create a true Spiking Neural Network MLP with SpikeIntegrationDynamics and TemporalTraceCredit.

    This factory uses:
    - SpikeIntegrationDynamics: LIF membrane potential integration with thresholding
    - TemporalTraceCredit: STDP-based credit assignment using spike timing
    - FeedforwardGeometry: Standard MLP topology

    Requires SystemTrainer with spiking support (implemented in compose_system).

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device
        max_steps: Number of simulation time steps
        beta: Nudge strength for nudged phase

    Returns:
        A composed 5-D System with FeedforwardGeometry + SpikeIntegrationDynamics
        + TemporalTraceCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = SpikeIntegrationDynamics(
        StateDynamicsConfig.spike_integration(
            max_steps=max_steps,
            beta=beta,
        )
    )
    credit = TemporalTraceCredit(CreditAssignmentConfig.temporal_trace())
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


# ============================================================
# 6-D Joint System Factories (Extended Ontology with Plasticity)
# ============================================================


def create_routing_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    gate_dim: int = 64,
    gate_init_scale: float = 0.1,
    device: str = "cpu",
) -> JointSystem:
    """Create an MLP with RoutingPlasticity (6-D coordinate).

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        gate_dim: Dimension of routing gates
        gate_init_scale: Initial scale for gate logits
        device: Target device

    Returns:
        A composed 6-D JointSystem with RoutingPlasticity.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    plasticity = RoutingPlasticity(
        gate_dim=gate_dim,
        temperature=1.0,
        decay=0.99,
        learning_rate=0.01,
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


def create_fast_weight_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    fast_weight_dim: int = 512,
    decay: float = 0.9,
    learning_rate: float = 0.1,
    device: str = "cpu",
) -> JointSystem:
    """Create an MLP with FastWeightPlasticity (6-D coordinate).

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        fast_weight_dim: Dimension of fast weight matrix
        decay: Decay factor for fast weights
        learning_rate: Hebbian learning rate for fast weights
        device: Target device

    Returns:
        A composed 6-D JointSystem with FastWeightPlasticity.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=learning_rate,
        outer_product_scale=1.0,
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


def create_tile_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    neurons_per_tile: int = 8,
    tiles_per_layer: int = 2,
    device: str = "cpu",
) -> System:
    """Create a Tile MLP system (5-D coordinate with TileGeometry).

    TileGeometry organizes neurons into tiles with local connectivity,
    enabling structured sparsity and efficient computation.

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        neurons_per_tile: Number of neurons per tile
        tiles_per_layer: Number of tiles per layer
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with TileGeometry + InstantaneousDynamics
        + BackpropCredit + EuclideanUpdate
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    # Replace with TileGeometry
    from computronium.ontology import GeometryConfig, TileGeometry

    tile_cfg = GeometryConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=len(hidden_dims) + 1,
        topology_type="tile_mesh",
        connectivity=None,
        recurrent_weight=None,
        init_scale=init_scale,
    )
    geometry = TileGeometry(
        tile_cfg,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


__all__ = [
    # 5-D factories
    "create_backprop_mlp",
    "create_eqprop_mlp",
    "create_fa_mlp",
    "create_ff_mlp",
    "create_pepita_mlp",
    "create_tp_mlp",
    "create_pc_mlp",
    "create_hebbian_mlp",
    "create_snn_mlp",
    "create_spiking_snn_mlp",
    "create_tile_mlp",
    # 6-D factories
    "create_routing_mlp",
    "create_fast_weight_mlp",
]
