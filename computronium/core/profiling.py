import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from computronium.utils import count_parameters

if TYPE_CHECKING:
    from computronium.core.system_trainer import JointSystem

__all__ = [
    "EnergyProfile",
    "EnergyTracker",
    "ResourceUsage",
    "analyze_joint_system",
    "count_flops",
    "get_gpu_memory_mb",
    "profile_run",
]


@dataclass(frozen=True, slots=True)
class EnergyProfile:
    forward_flops: int  # via torch.profiler or hook counting
    backward_flops: int  # 0 for EP/FF/PEPITA/Hebbian
    param_count: int
    activation_sparsity: float  # fraction of near-zero activations
    weight_sparsity: float  # fraction of near-zero weights
    wall_time_ms: float  # elapsed per batch
    peak_memory_mb: float  # torch.cuda.max_memory_allocated
    energy_proxy: float  # (fwd + bwd flops) * (1 - activation_sparsity) / param_count
    requires_backward: bool  # from ModelSpec


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Comprehensive resource usage for a joint system (for FrontierRecord)."""

    coordinate: str
    device: str
    batch_size: int
    # Compute
    forward_flops: int
    backward_flops: int
    total_flops: int
    # Memory
    param_count: int
    param_memory_mb: float
    activation_memory_mb: float
    peak_memory_mb: float
    # Sparsity
    activation_sparsity: float
    weight_sparsity: float
    # Latency
    wall_time_ms: float
    # Energy proxy
    energy_proxy: float
    # Stability (optional)
    spectral_radius: float | None = None
    lyapunov_exponent: float | None = None


def _build_spatial_dummy(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Build a spatial dummy input for the model's expected input format.

    For spatial models (Conv2d first layer), build a 4D tensor matching the
    expected (C, H, W) from the model's input channels and typical MNIST/CIFAR
    sizes. For flat models, return a 2D tensor.
    """
    input_channels = 1
    spatial_size = (28, 28)
    first_linear_in = None

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            input_channels = module.in_channels
            break
        if isinstance(module, nn.Linear) and first_linear_in is None:
            first_linear_in = module.in_features

    is_spatial = (
        input_channels != 1 or getattr(model, "input_format", "flat") == "spatial"
    )

    if is_spatial and first_linear_in is not None:
        if first_linear_in == 784 and input_channels == 1:
            spatial_size = (28, 28)
        elif first_linear_in == 3072 and input_channels == 3:
            spatial_size = (32, 32)
        else:
            hw = int((first_linear_in / input_channels) ** 0.5)
            spatial_size = (hw, hw)

    if is_spatial:
        return torch.zeros(1, input_channels, *spatial_size, device=device)
    else:
        inp_dim = first_linear_in or getattr(model, "input_dim", None) or 64
        return torch.zeros(1, inp_dim, device=device)


def _estimate_activation_sparsity(
    model: nn.Module,
    sample_input: torch.Tensor | None = None,
    threshold: float = 1e-5,
) -> float:
    """Run a forward pass with hooks to estimate activation sparsity.

    If ``sample_input`` is None, a proper dummy is built based on the model's
    input format (spatial vs flat), so spatial (conv) models don't break.
    """
    if sample_input is None:
        device = next(model.parameters()).device
        sample_input = _build_spatial_dummy(model, device)

    activations: list[torch.Tensor] = []

    def _hook(_module, _input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach().flatten())
        elif isinstance(output, (tuple, list)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    activations.append(t.detach().flatten())

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU)):
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    if not activations:
        return 0.0

    all_acts = torch.cat(activations)
    zero_frac = (all_acts.abs() < threshold).float().mean().item()
    return zero_frac


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Estimate FLOPs for a model using parameter counting.

    For a more accurate count, use torch.profiler with record_function.
    """
    batch_size = input_shape[0] if input_shape else 1
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 2 * params * batch_size


def count_flops_detailed(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, int]:
    """Count FLOPs per layer type using module inspection.

    Returns dict with total and breakdown by layer type.
    """
    flops = {"total": 0, "linear": 0, "conv2d": 0, "matmul": 0, "other": 0}
    batch_size = input_shape[0] if input_shape else 1

    # For FeedforwardGeometry, inspect the layers directly
    if hasattr(model, "_layers"):
        for i, layer in enumerate(model._layers):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                flops["linear"] += 2 * batch_size * in_features * out_features
            elif isinstance(layer, nn.Conv2d):
                # Estimate output size - this is rough
                flops["conv2d"] += (
                    2 * batch_size * layer.in_channels * layer.out_channels * 28 * 28
                )
    else:
        # Fallback: use hooks for standard modules
        def _hook(module: nn.Module, _input, output):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                flops["linear"] += 2 * batch_size * in_features * out_features
            elif isinstance(module, nn.Conv2d):
                out_h, out_w = output.shape[2], output.shape[3]
                kh, kw = module.kernel_size
                flops["conv2d"] += (
                    2
                    * batch_size
                    * module.in_channels
                    * module.out_channels
                    * kh
                    * kw
                    * out_h
                    * out_w
                )
            else:
                # Generic matmul estimate
                flops["other"] += 1000  # placeholder

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                device = next(model.parameters()).device
                dummy = torch.zeros(input_shape, device=device)
                model(dummy)
        finally:
            for h in hooks:
                h.remove()

    flops["total"] = sum(v for k, v in flops.items() if k != "total")
    return flops


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB using nvml if available, else torch."""
    if not torch.cuda.is_available():
        return 0.0

    # Try pynvml first for more accurate total/allocated memory
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        pass

    # Fallback to torch
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_gpu_peak_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)


def profile_run(
    model: nn.Module, input_shape: tuple[int, ...], requires_backward: bool = True
) -> EnergyProfile:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fwd_flops = count_flops(model, input_shape)
    bwd_flops = 2 * fwd_flops if requires_backward else 0

    zero_weights = sum((p.abs() < 1e-5).sum().item() for p in model.parameters())
    weight_sparsity = zero_weights / max(params, 1)

    device = next(model.parameters()).device
    sample_input = torch.zeros(*input_shape, device=device)
    activation_sparsity = _estimate_activation_sparsity(model, sample_input)

    energy_proxy = (fwd_flops + bwd_flops) * (1 - activation_sparsity) / max(params, 1)

    return EnergyProfile(
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        param_count=params,
        activation_sparsity=activation_sparsity,
        weight_sparsity=weight_sparsity,
        wall_time_ms=0.0,
        peak_memory_mb=0.0,
        energy_proxy=energy_proxy,
        requires_backward=requires_backward,
    )


class EnergyTracker:
    """Per-step energy/power measurement with throttled heavy metrics.

    The activation-sparsity forward and the GPU weight-sparsity reduction are
    expensive relative to one train step. Inside a probe (``global_step`` is
    not ``None``) they are computed **once**, on the first measured step, and
    cached on the model for reuse on every later step. Standalone use
    (``global_step=None``) always measures, preserving the original eager
    behaviour. The probe driver passes the step counter so the whole run is
    monitored without paying the heavy cost per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        requires_backward: bool = True,
        global_step: int | None = None,
    ) -> None:
        self.model = model
        self.requires_backward = requires_backward
        self.global_step = global_step
        self.start_time = 0.0
        self.wall_time_ms = 0.0
        self.profile = None

    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wall_time_ms = (time.time() - self.start_time) * 1000

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if exc_type is not None:
            return False

        params = count_parameters(self.model, trainable_only=False)

        # Heavy metrics are throttled to the first step of a probe and cached on
        # the model; standalone trackers (global_step=None) always measure.
        heavy_cached = hasattr(self.model, "_biopl_activation_sparsity")
        compute_heavy = self.global_step is None or not heavy_cached

        if compute_heavy:
            zero_weights = sum(
                (p.abs() < 1e-5).sum().item() for p in self.model.parameters()
            )
            weight_sparsity = zero_weights / max(params, 1)

            device = next(self.model.parameters()).device
            # Pass None so _estimate_activation_sparsity builds a proper
            # spatial/flat dummy matching the model's input format.
            activation_sparsity = _estimate_activation_sparsity(self.model, None)

            if self.global_step is not None:
                setattr(self.model, "_biopl_activation_sparsity", activation_sparsity)
                setattr(self.model, "_biopl_weight_sparsity", weight_sparsity)
        else:
            activation_sparsity = float(
                getattr(self.model, "_biopl_activation_sparsity")
            )
            weight_sparsity = float(getattr(self.model, "_biopl_weight_sparsity"))

        batch_size = 64
        fwd_flops = 2 * params * batch_size
        bwd_flops = 2 * fwd_flops if self.requires_backward else 0

        energy_proxy = (
            (fwd_flops + bwd_flops) * (1 - activation_sparsity) / max(params, 1)
        )

        self.profile = EnergyProfile(
            forward_flops=fwd_flops,
            backward_flops=bwd_flops,
            param_count=params,
            activation_sparsity=activation_sparsity,
            weight_sparsity=weight_sparsity,
            wall_time_ms=self.wall_time_ms,
            peak_memory_mb=peak_mem,
            energy_proxy=energy_proxy,
            requires_backward=self.requires_backward,
        )
        return False


def analyze_joint_system(
    coordinate: str | SystemConfig,
    batch_size: int = 64,
    device: str = "auto",
    iterations: int = 10,
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dims: tuple[int, ...] = (256,),
) -> ResourceUsage:
    """Empirical measurement of compute, memory, energy, plastic state capacity.

    Uses torch.profiler + nvml for GPU memory.
    Returns ResourceUsage for FrontierRecord.

    Args:
        coordinate: 6-D system coordinate as string (e.g., "digital/feedforward/instantaneous/null/backprop/euclidean")
                   or SystemConfig object
        batch_size: Batch size for profiling
        device: "auto", "cpu", or "cuda"
        iterations: Number of iterations for latency averaging
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions

    Returns:
        ResourceUsage with comprehensive metrics
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse coordinate string if needed
    if isinstance(coordinate, str):
        parts = coordinate.split("/")
        if len(parts) != 6:
            raise ValueError(f"Expected 6 parts in coordinate string, got {len(parts)}")
        (
            substrate_type,
            geometry_type,
            dynamics_type,
            plasticity_type,
            credit_type,
            update_type,
        ) = parts
    else:
        # Assume SystemConfig object
        substrate_type = coordinate.substrate.substrate_type
        geometry_type = coordinate.geometry.topology_type
        dynamics_type = coordinate.dynamics.dynamics_type
        plasticity_type = coordinate.plasticity.plasticity_type
        credit_type = coordinate.credit.credit_type
        update_type = coordinate.update.update_type

    # Create system
    system = _create_joint_system_from_parts(
        substrate_type,
        geometry_type,
        dynamics_type,
        plasticity_type,
        credit_type,
        update_type,
        input_dim,
        output_dim,
        hidden_dims,
        device,
    )

    # Move to device
    if hasattr(system.geometry, "to"):
        system.geometry.to(device)
    if hasattr(system.substrate, "to"):
        system.substrate.to(device)

    # Create test input
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = system.train_step(x, y)

    # Synchronize
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure memory before
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = get_gpu_memory_mb()
    else:
        mem_before = 0.0

    # Profile train_step (full pipeline)
    import time

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = system.train_step(x, y)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    # Measure peak memory
    if device == "cuda":
        peak_mem = get_gpu_peak_memory_mb()
    else:
        peak_mem = 0.0

    mean_latency = sum(latencies) / len(latencies)

    # Parameter memory
    params = sum(p.numel() for p in system.geometry.params.values())
    param_memory_mb = sum(
        p.numel() * p.element_size() for p in system.geometry.params.values()
    ) / (1024 * 1024)

    # Activation memory estimate
    with torch.no_grad():
        sample_out = system.geometry.forward(x, system.substrate)
        if isinstance(sample_out, list):
            activation_memory_mb = sum(
                a.numel() * a.element_size() for a in sample_out
            ) / (1024 * 1024)
        else:
            activation_memory_mb = (
                sample_out.numel() * sample_out.element_size() / (1024 * 1024)
            )

    # FLOPs estimation
    input_shape = (batch_size, input_dim)
    flops_info = count_flops_detailed(system.geometry, input_shape)
    fwd_flops = flops_info["total"]
    # For bio-plausible methods, backward may not be 2x forward
    requires_backward = credit_type in ("backprop", "gradient")
    bwd_flops = 2 * fwd_flops if requires_backward else 0

    # Sparsity
    zero_weights = sum(
        (p.abs() < 1e-5).sum().item() for p in system.geometry.params.values()
    )
    weight_sparsity = zero_weights / max(params, 1)
    activation_sparsity = _estimate_activation_sparsity(system.geometry, x)

    # Energy proxy
    energy_proxy = (fwd_flops + bwd_flops) * (1 - activation_sparsity) / max(params, 1)

    # Plastic state capacity (for joint systems)
    plastic_state_capacity = 0
    if hasattr(system, "plasticity") and system.plasticity is not None:
        if hasattr(system.plasticity, "initial_psi"):
            # Estimate plastic state size using a simple context
            try:
                from computronium.core.joint.context import SystemContext

                # Build minimal context with required config objects
                context = SystemContext(
                    theta=system.geometry.params,
                    geometry=system.geometry,
                    substrate=system.substrate,
                    substrate_config=system.substrate.config,
                    geometry_config=system.geometry.config,
                    dynamics_config=system.dynamics.config,
                    credit_config=system.credit.config,
                    update_config=system.update.config,
                    plasticity_config=system.plasticity.config,
                    registry=system._registry if hasattr(system, "_registry") else None,
                )
                psi = system.plasticity.initial_psi(context, batch_size=batch_size)
                plastic_state_capacity = sum(v.numel() for v in psi.values())
            except Exception:
                pass

    # Stability proxy: spectral radius of Jacobian (if available)
    spectral_radius = None
    try:
        from computronium.core.stability.spectral_radius import estimate_spectral_radius

        spectral_radius = estimate_spectral_radius(system, x, y)
    except Exception:
        pass

    coord_str = f"{substrate_type}/{geometry_type}/{dynamics_type}/{plasticity_type}/{credit_type}/{update_type}"
    return ResourceUsage(
        coordinate=coord_str,
        device=device,
        batch_size=batch_size,
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        total_flops=fwd_flops + bwd_flops,
        param_count=params,
        param_memory_mb=param_memory_mb,
        activation_memory_mb=activation_memory_mb,
        peak_memory_mb=peak_mem,
        activation_sparsity=activation_sparsity,
        weight_sparsity=weight_sparsity,
        wall_time_ms=mean_latency,
        energy_proxy=energy_proxy,
        spectral_radius=spectral_radius,
    )


def _create_joint_system_from_parts(
    substrate_type: str,
    geometry_type: str,
    dynamics_type: str,
    plasticity_type: str,
    credit_type: str,
    update_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    device: str,
) -> JointSystem:
    """Create a JointSystem from parsed coordinate parts."""
    from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
    from computronium.core.ontology import (
        BackpropCredit,
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        RandomProjectionsCredit,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
    )
    from computronium.core.plasticity.fast_weights import create_fast_weight_plasticity
    from computronium.core.plasticity.routing import create_routing_plasticity
    from computronium.core.system_trainer import compose_joint_system

    # Substrate
    if substrate_type == "digital":
        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
    else:
        raise ValueError(f"Unknown substrate: {substrate_type}")

    # Geometry
    if geometry_type == "feedforward":
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                init_scale=0.1,
            )
        )
    elif geometry_type == "recurrent":
        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                init_scale=0.1,
            ),
            hidden_dim=hidden_dims[0] if hidden_dims else 64,
        )
    else:
        raise ValueError(f"Unknown geometry: {geometry_type}")

    # Dynamics
    if dynamics_type == "energy_minimization":
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=10, beta=0.5, step_size=0.1
            )
        )
    elif dynamics_type == "instantaneous":
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    elif dynamics_type == "predictive_settling":
        from computronium.core.ontology import PredictiveSettlingDynamics

        dynamics = PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=10, step_size=0.1)
        )
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")

    # Plasticity
    if plasticity_type == "null":
        plasticity = NullPlasticity()
    elif plasticity_type == "routing":
        plasticity = create_routing_plasticity(PlasticityConfig.routing(gate_dim=64))
    elif plasticity_type == "fast_weights":
        plasticity = create_fast_weight_plasticity(
            PlasticityConfig.fast_weights(fast_weight_dim=512)
        )
    else:
        raise ValueError(f"Unknown plasticity: {plasticity_type}")

    # Credit
    if credit_type == "backprop":
        credit = BackpropCredit(CreditAssignmentConfig.gradient())
    elif credit_type in ("thermodynamic_contrast", "thermo"):
        credit = ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        )
    elif credit_type == "random_projections":
        credit = RandomProjectionsCredit(CreditAssignmentConfig.random_projections())
    else:
        raise ValueError(f"Unknown credit: {credit_type}")

    # Update
    if update_type == "euclidean":
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))
    else:
        raise ValueError(f"Unknown update: {update_type}")

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )
