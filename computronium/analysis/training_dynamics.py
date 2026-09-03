"""Training Dynamics Visualizer for 6-D Joint Systems.

Provides visualization of joint training trajectories including:
- Energy evolution
- Loss evolution
- Activity trajectories per layer
- Plastic state evolution (gate logits, fast weights)
- Substrate state evolution
- Spectral radius ρ(J_F)
- Gate entropy (for routing)
- Fast weight matrix heatmaps
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True, slots=True)
class JointTrajectory:
    """Joint training trajectory data."""

    activity: list[dict[str, list]]  # Per-step activity tensors per layer
    plastic: list[dict[str, list]]  # Per-step plastic state
    substrate: list[dict[str, list]]  # Per-step substrate state
    energy: list[float]  # Energy per step
    loss: list[float]  # Loss per step
    spectral_radius: list[float]  # ρ(J_F) per step
    gate_entropy: list[float] | None = None  # Gate entropy for routing
    accuracy: list[float] | None = None  # Accuracy per step


def compute_gate_entropy(gate_logits: np.ndarray) -> float:
    """Compute entropy of gate distribution."""
    probs = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=-1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
    return float(np.mean(entropy))


def compute_spectral_radius_proxy(activations: dict[str, np.ndarray]) -> float:
    """Estimate spectral radius of Jacobian via power iteration proxy."""
    # Flatten all activations
    act_vec = np.concatenate([v.flatten() for v in activations.values()])
    if len(act_vec) == 0:
        return 0.0
    # Simple proxy: norm of activation vector
    return float(np.linalg.norm(act_vec) / np.sqrt(len(act_vec)))


def load_trajectory(filepath: Path) -> JointTrajectory:
    """Load trajectory from JSON file."""
    with Path(filepath).open(encoding="utf-8") as f:
        data = json.load(f)
    traj_data = data.get("trajectory", {})
    return JointTrajectory(
        activity=traj_data.get("activity", []),
        plastic=traj_data.get("plastic", []),
        substrate=traj_data.get("substrate", []),
        energy=traj_data.get("energy", []),
        loss=traj_data.get("loss", []),
        spectral_radius=traj_data.get("spectral_radius", []),
        gate_entropy=traj_data.get("gate_entropy"),
        accuracy=traj_data.get("accuracy"),
    )


def save_trajectory(
    traj: JointTrajectory, filepath: Path, coordinate: dict | None = None
):
    """Save trajectory to JSON file."""
    data = {
        "coordinate": coordinate or {},
        "trajectory": {
            "activity": traj.activity,
            "plastic": traj.plastic,
            "substrate": traj.substrate,
            "energy": traj.energy,
            "loss": traj.loss,
            "spectral_radius": traj.spectral_radius,
            "gate_entropy": traj.gate_entropy,
            "accuracy": traj.accuracy,
        },
    }
    with Path(filepath).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_training_dynamics(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
    trajectory: JointTrajectory,
    coordinate: dict[str, str] | None = None,
    save_html: str | Path | None = None,
    show: bool = False,
) -> go.Figure:
    """Create comprehensive interactive training dynamics visualization.

    Args:
        trajectory: JointTrajectory with training data.
        coordinate: Optional 6-D coordinate for title.
        save_html: Optional path to save HTML file.
        show: Whether to show the figure interactively.

    Returns:
        Plotly Figure object.
    """
    steps = len(trajectory.energy)
    step_indices = list(range(steps))

    coord_str = "/".join(coordinate.values()) if coordinate else "Unknown"

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Energy Evolution",
            "Loss Evolution",
            "Activity Norms (per layer)",
            "Plastic State Evolution",
            "Substrate State Evolution",
            "Spectral Radius ρ(J_F)",
            "Gate Entropy (Routing)",
            "Accuracy / Metrics",
        ),
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # 1. Energy Evolution
    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=trajectory.energy,
            name="Energy",
            mode="lines+markers",
            line={"color": colors[0], "width": 2},
            marker={"size": 6},
        ),
        row=1,
        col=1,
    )

    # 2. Loss Evolution
    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=trajectory.loss,
            name="Loss",
            mode="lines+markers",
            line={"color": colors[1], "width": 2},
            marker={"size": 6},
        ),
        row=1,
        col=2,
    )

    # 3. Activity Norms
    if trajectory.activity:
        layer_names = list(trajectory.activity[0].keys())
        for i, layer_name in enumerate(layer_names):
            norms = []
            for step_data in trajectory.activity:
                tensor_data = step_data.get(layer_name, [])
                if isinstance(tensor_data, list) and tensor_data:
                    arr = np.array(tensor_data)
                    norms.append(float(np.linalg.norm(arr)))
                else:
                    norms.append(0.0)
            fig.add_trace(
                go.Scatter(
                    x=step_indices,
                    y=norms,
                    name=f"Activity: {layer_name}",
                    mode="lines",
                    line={"color": colors[i % len(colors)], "width": 1.5},
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

    # 4. Plastic State Evolution
    if trajectory.plastic and trajectory.plastic[0]:
        plastic_names = list(trajectory.plastic[0].keys())
        for i, var_name in enumerate(plastic_names):
            norms = []
            for step_data in trajectory.plastic:
                tensor_data = step_data.get(var_name, [])
                if isinstance(tensor_data, list) and tensor_data:
                    arr = np.array(tensor_data)
                    norms.append(float(np.linalg.norm(arr)))
                else:
                    norms.append(0.0)
            fig.add_trace(
                go.Scatter(
                    x=step_indices,
                    y=norms,
                    name=f"Plastic: {var_name}",
                    mode="lines",
                    line={
                        "color": colors[(i + 3) % len(colors)],
                        "width": 1.5,
                        "dash": "dot",
                    },
                    showlegend=True,
                ),
                row=2,
                col=2,
            )

    # 5. Substrate State Evolution
    if trajectory.substrate and trajectory.substrate[0]:
        substrate_names = list(trajectory.substrate[0].keys())
        for i, var_name in enumerate(substrate_names):
            norms = []
            for step_data in trajectory.substrate:
                tensor_data = step_data.get(var_name, [])
                if isinstance(tensor_data, list) and tensor_data:
                    arr = np.array(tensor_data)
                    norms.append(float(np.linalg.norm(arr)))
                else:
                    norms.append(0.0)
            fig.add_trace(
                go.Scatter(
                    x=step_indices,
                    y=norms,
                    name=f"Substrate: {var_name}",
                    mode="lines",
                    line={
                        "color": colors[(i + 5) % len(colors)],
                        "width": 1.5,
                        "dash": "dash",
                    },
                    showlegend=True,
                ),
                row=3,
                col=1,
            )

    # 6. Spectral Radius
    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=trajectory.spectral_radius,
            name="ρ(J_F)",
            mode="lines+markers",
            line={"color": colors[2], "width": 2},
            marker={"size": 6, "symbol": "diamond"},
        ),
        row=3,
        col=2,
    )

    # 7. Gate Entropy (if available)
    if trajectory.gate_entropy:
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=trajectory.gate_entropy,
                name="Gate Entropy",
                mode="lines+markers",
                line={"color": colors[3], "width": 2},
                marker={"size": 6, "symbol": "square"},
            ),
            row=4,
            col=1,
        )

    # 8. Accuracy / Additional Metrics
    if trajectory.accuracy:
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=trajectory.accuracy,
                name="Accuracy",
                mode="lines+markers",
                line={"color": colors[4], "width": 2},
                marker={"size": 6, "symbol": "triangle-up"},
            ),
            row=4,
            col=2,
        )

    # Layout
    fig.update_layout(
        height=1200,
        title_text=f"Joint Training Dynamics: {coord_str}",
        title_font_size=16,
        template="plotly_white",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
    )

    # Axis labels
    for row in range(1, 5):
        for col in range(1, 3):
            fig.update_xaxes(title_text="Training Step", row=row, col=col)

    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="L2 Norm", row=2, col=1)
    fig.update_yaxes(title_text="L2 Norm", row=2, col=2)
    fig.update_yaxes(title_text="L2 Norm", row=3, col=1)
    fig.update_yaxes(title_text="Spectral Radius", row=3, col=2)
    fig.update_yaxes(title_text="Entropy", row=4, col=1)
    fig.update_yaxes(title_text="Accuracy", row=4, col=2)

    if save_html:
        fig.write_html(str(save_html))
        print(f"Saved interactive plot to {save_html}")

    if show:
        fig.show()

    return fig


def plot_plasticity_comparison(
    trajectories: dict[str, JointTrajectory],
    metrics: list[str] = ["energy", "loss", "spectral_radius", "gate_entropy"],
    save_html: str | Path | None = None,
) -> go.Figure:
    """Compare training dynamics across different plasticity types.

    Args:
        trajectories: Dict mapping plasticity name -> JointTrajectory.
        metrics: List of metrics to compare.
        save_html: Optional path to save HTML.

    Returns:
        Plotly Figure with comparison plots.
    """
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
        vertical_spacing=0.1,
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, metric in enumerate(metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1

        for j, (name, traj) in enumerate(trajectories.items()):
            values = getattr(traj, metric, [])
            if not values:
                continue

            steps = list(range(len(values)))
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    name=name,
                    mode="lines+markers",
                    line={"color": colors[j % len(colors)], "width": 2},
                    marker={"size": 5},
                    showlegend=(i == 0),  # Only show legend once
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=400 * n_rows,
        title_text="Plasticity Type Comparison",
        template="plotly_white",
        hovermode="x unified",
    )

    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            fig.update_xaxes(title_text="Training Step", row=row, col=col)

    if save_html:
        fig.write_html(str(save_html))
        print(f"Saved plasticity comparison to {save_html}")

    return fig


def plot_fast_weight_heatmap(
    fast_weights_history: list[np.ndarray],
    save_html: str | Path | None = None,
    title: str = "Fast Weight Evolution",
) -> go.Figure:
    """Plot fast weight matrix evolution as animated heatmap.

    Args:
        fast_weights_history: List of fast weight matrices [step, batch, dim] or [step, dim].
        save_html: Optional path to save HTML.
        title: Plot title.

    Returns:
        Plotly Figure with heatmap animation.
    """
    # Convert to array
    fw_array = np.array(fast_weights_history)

    if fw_array.ndim == 3:
        # [steps, batch, dim] -> average over batch
        fw_array = fw_array.mean(axis=1)

    steps, dim = fw_array.shape

    # Reshape to square if possible
    side = int(np.sqrt(dim))
    if side * side == dim:
        fw_grid = fw_array.reshape(steps, side, side)
    else:
        # Pad to square
        new_dim = side * side if side * side > dim else (side + 1) ** 2
        pad_width = new_dim - dim
        fw_padded = np.pad(fw_array, ((0, 0), (0, pad_width)), mode="constant")
        side = int(np.sqrt(new_dim))
        fw_grid = fw_padded.reshape(steps, side, side)

    # Create frames for animation
    frames = []
    for i in range(steps):
        frames.append(
            go.Frame(
                data=[go.Heatmap(z=fw_grid[i], colorscale="RdBu", zmid=0)],
                name=str(i),
            )
        )

    fig = go.Figure(
        data=[go.Heatmap(z=fw_grid[0], colorscale="RdBu", zmid=0)],
        frames=frames,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i in range(steps)
                ],
                "active": 0,
                "transition": {"duration": 0},
            }
        ],
    )

    if save_html:
        fig.write_html(str(save_html))
        print(f"Saved fast weight heatmap to {save_html}")

    return fig


def plot_resource_usage(
    resource_data: dict[str, list[float]],
    save_html: str | Path | None = None,
) -> go.Figure:
    """Plot resource usage over training (compute, memory, plastic state capacity).

    Args:
        resource_data: Dict with keys like 'compute_flops', 'memory_mb', 'plastic_state_size'.
        save_html: Optional path to save HTML.

    Returns:
        Plotly Figure.
    """
    n_metrics = len(resource_data)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(resource_data.keys()),
        vertical_spacing=0.1,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (metric, values) in enumerate(resource_data.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1

        steps = list(range(len(values)))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                name=metric,
                mode="lines+markers",
                line={"color": colors[i % len(colors)], "width": 2},
                marker={"size": 6},
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=400 * n_rows,
        title_text="Resource Usage",
        template="plotly_white",
        hovermode="x unified",
    )

    if save_html:
        fig.write_html(str(save_html))
        print(f"Saved resource usage plot to {save_html}")

    return fig


# Convenience function matching TODO2 spec
def plot_joint_training_dynamics(
    trajectory: JointTrajectory,
    save_html: str = "training_dynamics.html",
) -> go.Figure:
    """Convenience function matching TODO2 specification.

    Args:
        trajectory: Joint training trajectory.
        save_html: Output HTML file path.

    Returns:
        Plotly Figure.
    """
    return plot_training_dynamics(trajectory, save_html=save_html)


if __name__ == "__main__":
    # Demo with synthetic data
    import numpy as np

    steps = 50
    trajectory = JointTrajectory(
        activity=[
            {
                "layer_0": np.random.randn(32, 256).tolist(),
                "layer_1": np.random.randn(32, 128).tolist(),
            }
            for _ in range(steps)
        ],
        plastic=[
            {
                "gate_logits": np.random.randn(32, 64).tolist(),
                "active_routes": np.random.randn(32, 64).tolist(),
            }
            for _ in range(steps)
        ],
        substrate=[
            {"conductance": np.random.randn(32, 256).tolist()} for _ in range(steps)
        ],
        energy=[10.0 * np.exp(-i / 10) + np.random.randn() * 0.1 for i in range(steps)],
        loss=[2.0 * np.exp(-i / 15) + np.random.randn() * 0.05 for i in range(steps)],
        spectral_radius=[0.95 + 0.03 * np.sin(i / 5) for i in range(steps)],
        gate_entropy=[np.log(64) * (1 - np.exp(-i / 10)) for i in range(steps)],
        accuracy=[0.1 + 0.8 * (1 - np.exp(-i / 10)) for i in range(steps)],
    )

    fig = plot_training_dynamics(
        trajectory,
        coordinate={
            "substrate": "digital",
            "geometry": "recurrent",
            "dynamics": "energy_minimization",
            "plasticity": "routing",
            "credit": "thermo",
            "update": "euclidean",
        },
        save_html="demo_training_dynamics.html",
    )
    print("Demo plot saved to demo_training_dynamics.html")
