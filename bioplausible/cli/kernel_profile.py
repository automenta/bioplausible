"""Joint Kernel Profiler for 6-D Architecture.

Profiles compute, memory, and latency for each kernel type in the joint system:
- CoupledTransition.step
- PlasticityPrimitive.step
- Stability estimators
- Adapter projections
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import torch

from bioplausible.core.joint.transition import PlasticityConfig

# Import bioplausible components
from bioplausible.core.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
)
from bioplausible.core.plasticity.fast_weights import create_fast_weight_plasticity
from bioplausible.core.plasticity.routing import create_routing_plasticity
from bioplausible.core.system_trainer import compose_joint_system


@dataclass(frozen=True, slots=True)
class KernelProfile:
    """Profile data for a single kernel."""

    name: str
    coordinate: str
    batch_size: int
    device: str
    latency_ms: float
    memory_mb: float
    flops: float | None = None
    iterations: int = 10


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Complete profile result for a coordinate."""

    coordinate: str
    device: str
    batch_sizes: list[int]
    kernels: dict[
        str, list[KernelProfile]
    ]  # kernel_name -> list of profiles per batch size
    total_latency_ms: float
    peak_memory_mb: float


def _create_joint_system(
    coordinate: str, input_dim: int, output_dim: int, hidden_dim: int, device: str
):
    """Create a JointSystem from coordinate string."""
    parts = coordinate.split("/")
    if len(parts) != 6:
        raise ValueError(f"Expected 6 parts, got {len(parts)}")

    (
        substrate_type,
        geometry_type,
        dynamics_type,
        plasticity_type,
        credit_type,
        update_type,
    ) = parts

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
                hidden_dims=(hidden_dim,),
                init_scale=0.1,
            )
        )
    elif geometry_type == "recurrent":
        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=(hidden_dim,),
                init_scale=0.1,
            ),
            hidden_dim=hidden_dim,
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
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")

    # Plasticity
    if plasticity_type == "null":
        plasticity = PlasticityConfig.null()
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
    elif credit_type == "thermo":
        credit = ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        )
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


def _profile_kernel(
    kernel_fn, *args, iterations: int = 10, warmup: int = 3
) -> tuple[float, float]:
    """Profile a kernel function.

    Returns:
        (mean_latency_ms, peak_memory_mb)
    """
    device = args[0].device if hasattr(args[0], "device") else torch.device("cpu")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = kernel_fn(*args)

    # Synchronize
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure memory before
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()
    else:
        mem_before = 0

    # Timed runs
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = kernel_fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    # Measure peak memory
    if device.type == "cuda":
        peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / (
            1024 * 1024
        )  # MB
    else:
        peak_mem = 0.0

    mean_latency = sum(latencies) / len(latencies)
    return mean_latency, peak_mem


def _profile_coordinate(
    coordinate: str,
    batch_sizes: list[int],
    device: str,
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dim: int = 256,
) -> ProfileResult:
    """Profile all kernels for a coordinate."""
    print(f"  Profiling coordinate: {coordinate} on {device}")

    kernels_data = {}
    all_latencies = []
    peak_memory = 0.0

    for batch_size in batch_sizes:
        print(f"    Batch size: {batch_size}")

        # Create system
        system = _create_joint_system(
            coordinate, input_dim, output_dim, hidden_dim, device
        )

        # Move to device
        if hasattr(system.geometry, "to"):
            system.geometry.to(device)
        if hasattr(system.substrate, "to"):
            system.substrate.to(device)

        # Create test input
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, output_dim, (batch_size,), device=device)

        # Profile train_step (full pipeline)
        def train_step_fn(x, y):
            return system.train_step(x, y)

        latency, mem = _profile_kernel(train_step_fn, x, y)
        kernel_name = "CoupledTransition.train_step"
        if kernel_name not in kernels_data:
            kernels_data[kernel_name] = []
        kernels_data[kernel_name].append(
            KernelProfile(
                name=kernel_name,
                coordinate=coordinate,
                batch_size=batch_size,
                device=device,
                latency_ms=latency,
                memory_mb=mem,
                iterations=10,
            )
        )
        all_latencies.append(latency)
        peak_memory = max(peak_memory, mem)

        # Profile plasticity step if applicable
        if hasattr(system, "plasticity") and system.plasticity is not None:
            # Create dummy plastic state and joint state
            from bioplausible.core.joint.state import CompositeState

            z = CompositeState.empty()
            z.activity["x"] = x
            z.activity["y"] = y

            if hasattr(system, "_make_context"):
                context = system._make_context()
                if hasattr(system.plasticity, "initial_psi"):
                    z.plastic = system.plasticity.initial_psi(
                        context, batch_size=batch_size
                    )
                    # Move plastic state to device
                    z.plastic = {k: v.to(device) for k, v in z.plastic.items()}

                def plasticity_step_fn(psi, z_state, ctx):
                    return system.plasticity.step(psi, z_state, ctx)

                latency, mem = _profile_kernel(
                    plasticity_step_fn, z.plastic, z, context
                )
                kernel_name = "PlasticityPrimitive.step"
                if kernel_name not in kernels_data:
                    kernels_data[kernel_name] = []
                kernels_data[kernel_name].append(
                    KernelProfile(
                        name=kernel_name,
                        coordinate=coordinate,
                        batch_size=batch_size,
                        device=device,
                        latency_ms=latency,
                        memory_mb=mem,
                        iterations=10,
                    )
                )
                all_latencies.append(latency)
                peak_memory = max(peak_memory, mem)

        # Profile geometry forward
        def geometry_forward_fn(x_input, substrate):
            return system.geometry.forward(x_input, substrate)

        latency, mem = _profile_kernel(geometry_forward_fn, x, system.substrate)
        kernel_name = "Geometry.forward"
        if kernel_name not in kernels_data:
            kernels_data[kernel_name] = []
        kernels_data[kernel_name].append(
            KernelProfile(
                name=kernel_name,
                coordinate=coordinate,
                batch_size=batch_size,
                device=device,
                latency_ms=latency,
                memory_mb=mem,
                iterations=10,
            )
        )
        all_latencies.append(latency)
        peak_memory = max(peak_memory, mem)

        # Profile dynamics settle
        if hasattr(system.dynamics, "settle"):
            from bioplausible.core.ontology import SystemState

            def dynamics_settle_fn(x_input, substrate, geometry):
                state = SystemState(x=x_input)
                state.activations = geometry.forward(x_input, substrate)
                if state.activations is not None:
                    state.activations = substrate.inject_state_noise(state.activations)
                return system.dynamics.settle(state, geometry, substrate, target=None)

            latency, mem = _profile_kernel(
                dynamics_settle_fn, x, system.substrate, system.geometry
            )
            kernel_name = "StateDynamics.settle"
            if kernel_name not in kernels_data:
                kernels_data[kernel_name] = []
            kernels_data[kernel_name].append(
                KernelProfile(
                    name=kernel_name,
                    coordinate=coordinate,
                    batch_size=batch_size,
                    device=device,
                    latency_ms=latency,
                    memory_mb=mem,
                    iterations=10,
                )
            )
            all_latencies.append(latency)
            peak_memory = max(peak_memory, mem)

        # Clear CUDA cache between batch sizes
        if device == "cuda":
            torch.cuda.empty_cache()

    total_latency = sum(all_latencies)
    return ProfileResult(
        coordinate=coordinate,
        device=device,
        batch_sizes=batch_sizes,
        kernels=kernels_data,
        total_latency_ms=total_latency,
        peak_memory_mb=peak_memory,
    )


def _save_results(results: list[ProfileResult], output_path: Path):
    """Save profile results to JSON."""
    data = {
        "results": [asdict(r) for r in results],
        "summary": {
            "num_coordinates": len(results),
            "total_kernels_profiled": sum(len(r.kernels) for r in results),
        },
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")


def _generate_html_report(results: list[ProfileResult], output_path: Path):
    """Generate HTML report with Plotly visualizations."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Joint Kernel Profile Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .plot-container { margin-bottom: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric-card { display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Joint Kernel Profile Report</h1>
"""

    # Summary
    html += "<h2>Summary</h2>"
    for r in results:
        html += f"""
        <div class="metric-card">
            <h3>{r.coordinate}</h3>
            <p>Device: {r.device}</p>
            <p>Total Latency: {r.total_latency_ms:.2f} ms</p>
            <p>Peak Memory: {r.peak_memory_mb:.2f} MB</p>
        </div>
"""

    # Latency vs Batch Size
    html += """
    <h2>Latency vs Batch Size</h2>
    <div id="latency_plot" class="plot-container"></div>
    <h2>Memory vs Batch Size</h2>
    <div id="memory_plot" class="plot-container"></div>
    <h2>Kernel Breakdown</h2>
    <div id="kernel_plot" class="plot-container"></div>

    <script>
"""

    # Prepare data for plotting
    for r in results:
        coord_short = r.coordinate.replace("/", "_")
        for kernel_name, profiles in r.kernels.items():
            batch_sizes = [p.batch_size for p in profiles]
            latencies = [p.latency_ms for p in profiles]
            memories = [p.memory_mb for p in profiles]

            html += f"""
            var {coord_short}_{kernel_name.replace(".", "_").replace(" ", "_")}_latency = {{x: {batch_sizes}, y: {latencies}, name: '{r.coordinate} - {kernel_name}', mode: 'lines+markers'}};
            var {coord_short}_{kernel_name.replace(".", "_").replace(" ", "_")}_memory = {{x: {batch_sizes}, y: {memories}, name: '{r.coordinate} - {kernel_name}', mode: 'lines+markers'}};
"""

    html += """
        // Latency plot
        var latencyTraces = [];
        var memoryTraces = [];
        var kernelTraces = [];

"""

    for r in results:
        coord_short = r.coordinate.replace("/", "_")
        for kernel_name, profiles in r.kernels.items():
            var_name = (
                f"{coord_short}_{kernel_name.replace('.', '_').replace(' ', '_')}"
            )
            html += f"""
        latencyTraces.push({var_name}_latency);
        memoryTraces.push({var_name}_memory);
        kernelTraces.push({var_name}_latency);
"""

    html += """
        Plotly.newPlot('latency_plot', latencyTraces, {
            title: 'Kernel Latency vs Batch Size',
            xaxis: {title: 'Batch Size'},
            yaxis: {title: 'Latency (ms)'},
            template: 'plotly_white'
        });

        Plotly.newPlot('memory_plot', memoryTraces, {
            title: 'Peak Memory vs Batch Size',
            xaxis: {title: 'Batch Size'},
            yaxis: {title: 'Memory (MB)'},
            template: 'plotly_white'
        });

        Plotly.newPlot('kernel_plot', kernelTraces, {
            title: 'Kernel Latency Breakdown',
            xaxis: {title: 'Batch Size'},
            yaxis: {title: 'Latency (ms)'},
            template: 'plotly_white'
        });
    </script>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"HTML report saved to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="biopl benchmark profile",
        description="Profile joint system kernels",
    )
    parser.add_argument(
        "--coordinate",
        required=True,
        help="6-D coordinate to profile (e.g., digital/recurrent/energy_minimization/routing/thermo/euclidean)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=784,
        help="Input dimension",
    )
    parser.add_argument(
        "--output",
        default="kernel_profile.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--output-html",
        help="Output HTML report file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations per measurement",
    )

    args = parser.parse_args(argv)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Profiling coordinate: {args.coordinate}")
    print(f"Device: {device}")
    print(f"Batch sizes: {args.batch_sizes}")

    # Run profile
    result = _profile_coordinate(
        args.coordinate,
        args.batch_sizes,
        device,
        input_dim=args.input_dim,
    )

    # Save results
    output_path = Path(args.output)
    _save_results([result], output_path)

    # Generate HTML report if requested
    if args.output_html:
        _generate_html_report([result], Path(args.output_html))

    # Print summary
    print("\n=== Profile Summary ===")
    print(f"Coordinate: {result.coordinate}")
    print(f"Device: {result.device}")
    print(f"Total Latency: {result.total_latency_ms:.2f} ms")
    print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")
    print("\nPer-Kernel Latencies:")
    for kernel_name, profiles in result.kernels.items():
        for p in profiles:
            print(
                f"  {kernel_name} (batch={p.batch_size}): {p.latency_ms:.2f} ms, {p.memory_mb:.2f} MB"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
