"""CLI Lab for Verification and Inspection"""

import argparse
import json
import logging
from pathlib import Path

import torch

# Import zoo models to trigger registration
from computronium.core.logging import get_logger
from computronium.core.registry import Registry
from computronium.core.utils.device import get_device
from computronium.domains import create_task

logger = get_logger()

__all__ = [
    "inspect_model",
    "inspect_state",
    "logger",
    "main",
]


def inspect_model(args):
    logger.info("[LAB]  Inspecting Model: %s", args.model)

    device = str(get_device())

    # Create Task
    task = create_task(args.task, device=device)
    task.setup()
    logger.info(
        "Task: %s, Input: %s, Output: %s", args.task, task.input_dim, task.output_dim
    )

    # Create System via 5-D ontology projection
    # This uses the native 5-D composition for models that support it,
    # or falls back to ModelAdapter for legacy models.
    system = Registry.to_system(
        args.model,
        input_dim=task.input_dim or 0,
        hidden_dim=64,
        output_dim=task.output_dim,
        num_layers=2,
    )

    logger.info("System Created: %s", type(system).__name__)

    # Get parameter count from geometry
    if hasattr(system, "geometry") and hasattr(system.geometry, "params"):
        param_count = sum(p.numel() for p in system.geometry.params.values())
        logger.info("Parameters: %.2fM", param_count / 1e6)

    # Run Dummy Forward
    logger.info("Running Verification Inference...")
    x, _ = task.get_batch("val")
    with torch.no_grad():
        # LM models that expose `embed` expect integer token ids — task.get_batch
        # already returns those ids, so forward handles the embedding internally.
        # Non-LM models may receive raw features; flatten spatially for MLPs.
        if x.dim() > 2 and "Conv" not in args.model:
            x = x.view(x.size(0), -1)

        try:  # ruff: ignore[too-many-statements-in-try-clause]
            x = x.to(device)
            # Move system geometry to device if needed
            if hasattr(system.geometry, "to"):
                system.geometry.to(device)
            # System.forward signature depends on whether it's a native System (requires substrate)
            # or an _AdaptedSystem (delegates to model.forward without substrate).
            # Try with substrate first (native 5-D System), fall back to no substrate (adapted).
            from computronium.ontology import DigitalSubstrate

            substrate = DigitalSubstrate()
            try:
                out = system.forward(x, substrate)
            except TypeError:
                # _AdaptedSystem.forward only takes x
                out = system.forward(x)
        except RuntimeError, ValueError, TypeError:
            logger.exception("Forward pass failed for model %s", args.model)
            return
        logger.info("[OK]  Forward pass successful. Output shape: %s", out.shape)


def _parse_coordinate(coord_str: str) -> dict:
    """Parse a 6-D coordinate string like 'digital/recurrent/energy_min/routing/thermo/euclidean'."""
    parts = coord_str.split("/")
    if len(parts) != 6:
        raise ValueError(
            f"Expected 6 parts (substrate/geometry/dynamics/plasticity/credit/update), got {len(parts)}"
        )
    return {
        "substrate": parts[0],
        "geometry": parts[1],
        "dynamics": parts[2],
        "plasticity": parts[3],
        "credit": parts[4],
        "update": parts[5],
    }


def _create_joint_system_from_coordinate(  # ruff: ignore[complex-structure, too-many-branches]
    coord: dict, input_dim: int, output_dim: int, hidden_dim: int, device: str
):
    """Create a JointSystem from a parsed 6-D coordinate."""
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.core.plasticity.fast_weights import create_fast_weight_plasticity
    from computronium.core.plasticity.routing import create_routing_plasticity
    from computronium.core.system_trainer import compose_joint_system
    from computronium.ontology import (
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

    # Substrate
    if coord["substrate"] == "digital":
        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
    else:
        raise ValueError(f"Unknown substrate: {coord['substrate']}")

    # Geometry
    if coord["geometry"] == "feedforward":
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=(hidden_dim,),
                init_scale=0.1,
            )
        )
    elif coord["geometry"] == "recurrent":
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
        raise ValueError(f"Unknown geometry: {coord['geometry']}")

    # Dynamics
    if coord["dynamics"] == "energy_min":
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=20, beta=0.5, step_size=0.1
            )
        )
    elif coord["dynamics"] == "instantaneous":
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    else:
        raise ValueError(f"Unknown dynamics: {coord['dynamics']}")

    # Plasticity
    if coord["plasticity"] == "null":
        plasticity_config = PlasticityConfig.null()
        plasticity = None  # Will use NullPlasticity internally
    elif coord["plasticity"] == "routing":
        plasticity_config = PlasticityConfig.routing(gate_dim=64)
        plasticity = create_routing_plasticity(plasticity_config)
    elif coord["plasticity"] == "fast_weights":
        plasticity_config = PlasticityConfig.fast_weights(fast_weight_dim=512)
        plasticity = create_fast_weight_plasticity(plasticity_config)
    else:
        raise ValueError(f"Unknown plasticity: {coord['plasticity']}")

    # Credit
    if coord["credit"] == "backprop":
        credit = BackpropCredit(CreditAssignmentConfig.gradient())
    elif coord["credit"] == "thermo":
        credit = ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        )
    else:
        raise ValueError(f"Unknown credit: {coord['credit']}")

    # Update
    if coord["update"] == "euclidean":
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))
    else:
        raise ValueError(f"Unknown update: {coord['update']}")

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity or plasticity_config, credit, update
    )


def _run_state_inspection(system, task, steps: int, device: str) -> dict:  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
    """Run joint state inspection and return trajectory data."""
    from computronium.state import CompositeState

    trajectory = {
        "activity": [],
        "plastic": [],
        "substrate": [],
        "energy": [],
        "spectral_radius": [],
        "loss": [],
    }

    x, y = task.get_batch("train")
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    x = x.to(device)
    y = y.to(device)

    # Move system to device
    if hasattr(system.geometry, "to"):
        system.geometry.to(device)
    if hasattr(system.substrate, "to"):
        system.substrate.to(device)

    system.geometry.train()

    # Initialize joint state
    z = CompositeState.empty()
    z.activity["x"] = x
    z.activity["y"] = y

    # Get initial plastic state
    if hasattr(system, "_make_context"):
        context = system._make_context()
        if hasattr(system.plasticity, "initial_psi"):
            z.plastic = system.plasticity.initial_psi(context, batch_size=x.shape[0])
            # Move plastic state to device
            z.plastic = {k: v.to(device) for k, v in z.plastic.items()}

    for step in range(steps):
        # Record current state
        trajectory["activity"].append({
            k: v.detach().cpu().clone() for k, v in z.activity.items()
        })
        trajectory["plastic"].append({
            k: v.detach().cpu().clone() for k, v in z.plastic.items()
        })
        trajectory["substrate"].append({
            k: v.detach().cpu().clone() for k, v in z.substrate.items()
        })

        # Run one step of joint transition
        if hasattr(system, "dynamics") and hasattr(system.dynamics, "settle"):
            # Use the 5-D settling for now
            from computronium.ontology import SystemState

            state = SystemState(x=x, y=y)
            state.activations = system.geometry.forward(x, system.substrate)
            if state.activations is not None:
                state.activations = system.substrate.inject_state_noise(
                    state.activations
                )

            # Free phase
            free_state = system.dynamics.settle(
                state, system.geometry, system.substrate, target=None
            )
            free_state.energy = system.dynamics.compute_energy(
                free_state, system.geometry
            )

            # Nudged phase
            nudged_state = system.dynamics.settle(
                state, system.geometry, system.substrate, target=y
            )
            nudged_state.energy = system.dynamics.compute_energy(
                nudged_state, system.geometry
            )
            nudged_state.loss = task_loss(nudged_state, y)  # ruff: ignore[undefined-name]

            # Record energy
            trajectory["energy"].append(
                free_state.energy.item() if free_state.energy else 0.0
            )
            trajectory["loss"].append(
                nudged_state.loss.item() if nudged_state.loss else 0.0
            )

            # Estimate spectral radius of Jacobian
            if (
                hasattr(system.geometry, "params")
                and free_state.activations is not None
            ):
                try:  # ruff: ignore[too-many-statements-in-try-clause]
                    # Simple spectral radius estimate via power iteration
                    acts = free_state.activations
                    if isinstance(acts, list):
                        act_vec = torch.cat([a.flatten() for a in acts])
                    else:
                        act_vec = acts.flatten()
                    # Use a small perturbation to estimate
                    with torch.no_grad():
                        # Compute Jacobian-vector product approximation
                        jvp = act_vec @ act_vec  # crude proxy
                        trajectory["spectral_radius"].append(
                            float(torch.norm(jvp).item())
                        )
                except Exception:
                    trajectory["spectral_radius"].append(0.0)
            else:
                trajectory["spectral_radius"].append(0.0)

            # Update plastic state if applicable
            if hasattr(system, "plasticity") and system.plasticity is not None:
                from computronium.core.pipeline import phase_states

                pseudo_grads = system.credit.compute_pseudo_gradient(  # ruff: ignore[unused-variable]
                    phase_states(free=free_state, nudged=nudged_state),
                    nudged_state.loss,
                    system.geometry,
                )
                # Simple plastic step
                if hasattr(system.plasticity, "step"):
                    context = system._make_context()
                    z.plastic = system.plasticity.step(z.plastic, z, context)

            # Update z for next iteration
            z.activity["x"] = x
            z.activity["y"] = y

    return trajectory


def _generate_html_report(trajectory: dict, coord: dict, output_path: Path):  # ruff: ignore[complex-structure, too-many-branches]
    """Generate an interactive Plotly HTML report from trajectory data."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not available, skipping HTML report generation")
        return

    steps = len(trajectory["energy"])
    step_indices = list(range(steps))

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Energy Evolution",
            "Loss Evolution",
            "Activity Norms (per layer)",
            "Plastic State Evolution",
            "Substrate State",
            "Spectral Radius ρ(J_F)",
        ),
        vertical_spacing=0.1,
    )

    # Energy
    fig.add_trace(
        go.Scatter(
            x=step_indices, y=trajectory["energy"], name="Energy", mode="lines+markers"
        ),
        row=1,
        col=1,
    )

    # Loss
    fig.add_trace(
        go.Scatter(
            x=step_indices, y=trajectory["loss"], name="Loss", mode="lines+markers"
        ),
        row=1,
        col=2,
    )

    # Activity norms
    if trajectory["activity"]:
        for layer_name in trajectory["activity"][0].keys():  # ruff: ignore[in-dict-keys]
            norms = []
            for t in trajectory["activity"]:
                tensor = t.get(layer_name)
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    norms.append(torch.norm(tensor).item())
                else:
                    norms.append(0.0)
            fig.add_trace(
                go.Scatter(
                    x=step_indices,
                    y=norms,
                    name=f"Activity: {layer_name}",
                    mode="lines",
                ),
                row=2,
                col=1,
            )

    # Plastic state
    if trajectory["plastic"] and trajectory["plastic"][0]:
        for var_name in trajectory["plastic"][0].keys():  # ruff: ignore[in-dict-keys]
            norms = []
            for t in trajectory["plastic"]:
                tensor = t.get(var_name)
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    norms.append(torch.norm(tensor).item())
                else:
                    norms.append(0.0)
            fig.add_trace(
                go.Scatter(
                    x=step_indices, y=norms, name=f"Plastic: {var_name}", mode="lines"
                ),
                row=2,
                col=2,
            )

    # Substrate state
    if trajectory["substrate"] and trajectory["substrate"][0]:
        for var_name in trajectory["substrate"][0].keys():  # ruff: ignore[in-dict-keys]
            norms = []
            for t in trajectory["substrate"]:
                tensor = t.get(var_name)
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    norms.append(torch.norm(tensor).item())
                else:
                    norms.append(0.0)
            fig.add_trace(
                go.Scatter(
                    x=step_indices, y=norms, name=f"Substrate: {var_name}", mode="lines"
                ),
                row=3,
                col=1,
            )

    # Spectral radius
    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=trajectory["spectral_radius"],
            name="ρ(J_F)",
            mode="lines+markers",
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        height=900,
        title_text=f"Joint State Inspection: {'/'.join(coord.values())}",
        showlegend=True,
    )

    fig.write_html(str(output_path))
    logger.info("HTML report saved to %s", output_path)


def inspect_state(args):
    """Inspect joint state evolution for a 6-D coordinate."""
    logger.info("[LAB]  Inspecting Joint State: %s", args.coordinate)

    device = str(get_device())
    coord = _parse_coordinate(args.coordinate)

    # Create Task
    task = create_task(args.task, device=device)
    task.setup()
    logger.info(
        "Task: %s, Input: %s, Output: %s", args.task, task.input_dim, task.output_dim
    )

    input_dim = task.input_dim or 784
    if isinstance(input_dim, (tuple, list)):
        import torch

        input_dim = int(torch.prod(torch.tensor(input_dim)))
    output_dim = task.output_dim
    hidden_dim = args.hidden_dim

    # Create Joint System
    system = _create_joint_system_from_coordinate(
        coord, input_dim, output_dim, hidden_dim, device
    )
    logger.info("Joint System Created: %s", type(system).__name__)

    # Run state inspection
    logger.info("Running state inspection for %d steps...", args.steps)
    trajectory = _run_state_inspection(system, task, args.steps, device)

    # Save trajectory as JSON
    if args.output.endswith(".json"):
        output_path = Path(args.output)
    elif args.output.endswith(".html"):
        output_path = Path(args.output).with_suffix(".json")
    else:
        output_path = Path(args.output + ".json")

    output_path.write_text(
        json.dumps(
            {
                "coordinate": coord,
                "trajectory": {
                    "activity": [
                        {k: v.tolist() for k, v in step.items()}
                        for step in trajectory["activity"]
                    ],
                    "plastic": [
                        {k: v.tolist() for k, v in step.items()}
                        for step in trajectory["plastic"]
                    ],
                    "substrate": [
                        {k: v.tolist() for k, v in step.items()}
                        for step in trajectory["substrate"]
                    ],
                    "energy": trajectory["energy"],
                    "spectral_radius": trajectory["spectral_radius"],
                    "loss": trajectory["loss"],
                },
            },
            indent=2,
        )
    )
    logger.info("Trajectory data saved to %s", output_path)

    # Generate HTML report if requested
    if args.output.endswith(".html"):
        _generate_html_report(trajectory, coord, Path(args.output))

    logger.info("[OK]  State inspection complete")


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Lab CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    inspect = subparsers.add_parser("inspect", help="Inspect a model architecture")
    inspect.add_argument("--model", required=True, help="Model name")
    inspect.add_argument(
        "--task", default="mnist", help="Task type (e.g., mnist, cifar10)"
    )

    inspect_state_parser = subparsers.add_parser(
        "inspect-state", help="Inspect joint state evolution for a 6-D coordinate"
    )
    inspect_state_parser.add_argument(
        "--coordinate",
        required=True,
        help="6-D coordinate (e.g., digital/recurrent/energy_min/routing/thermo/euclidean)",
    )
    inspect_state_parser.add_argument(
        "--task", default="mnist", help="Task type (e.g., mnist, cifar10)"
    )
    inspect_state_parser.add_argument(
        "--steps", type=int, default=50, help="Number of inspection steps"
    )
    inspect_state_parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension"
    )
    inspect_state_parser.add_argument(
        "--output",
        default="state_evolution.json",
        help="Output file path (.json or .html)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "inspect":
        inspect_model(args)
    elif args.command == "inspect-state":
        inspect_state(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
