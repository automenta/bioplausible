"""
Analysis Tools for Bio-Plausible Models

Provides utilities for inspecting model dynamics, convergence, and alignment.
Useful for research and "microscope" style analysis.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = [
    "DynamicsAnalyzer",
    "EnergyTrajectory",
    "GradientAlignment",
    "TileHeatmapData",
]

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass(frozen=True, slots=True)
class EnergyTrajectory:
    """Energy trajectory data for energy-based models."""

    free_energy: np.ndarray  # [steps]
    nudged_energy: np.ndarray | None  # [steps]
    energy_gap: np.ndarray | None  # [steps]
    steps: np.ndarray


@dataclass(frozen=True, slots=True)
class GradientAlignment:
    """Gradient alignment analysis results."""

    cosine_similarity: float
    per_layer_alignment: dict[str, float]
    angle_degrees: float
    bio_grad_norm: float
    bp_grad_norm: float


@dataclass(frozen=True, slots=True)
class TileHeatmapData:
    """Tile heatmap data for visualization."""

    tile_activities: np.ndarray  # [num_tiles, batch, neurons]
    tile_errors: np.ndarray  # [num_tiles, batch, neurons]
    tile_importance: np.ndarray  # [num_tiles]
    layer_ids: np.ndarray  # [num_tiles]
    tile_ids: np.ndarray  # [num_tiles]


class DynamicsAnalyzer:
    """
    Analyzer for inspecting the internal dynamics of Equilibrium Propagation models.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def get_convergence_data(
        self, x: torch.Tensor, steps: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Run the model on input x and capture convergence dynamics.

        Args:
            x: Input tensor.
            steps: Number of equilibrium steps (overrides model default if provided).

        Returns:
            Dictionary containing:
            - 'trajectory': Array of hidden states [steps, batch, hidden_dim]
            - 'deltas': Array of state changes (L2 norm) per step [steps]
            - 'activities': Mean absolute activity per step [steps]
            - 'fixed_point': Final hidden state
        """
        self.model.eval()
        x = x.to(self.device)

        # Prepare input (similar to CoreTrainer logic)
        if hasattr(self.model, "has_embed") and self.model.has_embed:
            # Basic handling, assuming model has .embed
            h = self.model.embed(x).mean(dim=1)
        elif x.dim() > 2 and not any(
            k in self.model.__class__.__name__ for k in ["Conv", "Transformer"]
        ):
            h = x.reshape(x.size(0), -1)
        else:
            h = x

        with torch.no_grad():
            # Check if model supports return_trajectory
            kwargs = {"return_trajectory": True, "return_dynamics": True}
            if steps is not None:
                kwargs["steps"] = steps

            # Helper to check signature or try/except
            # We'll try passing kwargs.
            try:
                # Most EqProp models (LoopedMLP, etc) support this
                output = self.model(h, **kwargs)

                # output might be (out, trajectory) or (out, dynamics_dict)
                if isinstance(output, tuple):
                    if isinstance(output[1], dict):
                        dynamics = output[1]
                    else:
                        # Assume list of tensors
                        dynamics = {"trajectory": output[1]}
                else:
                    raise NotImplementedError(
                        "Model does not appear to return dynamics."
                    )

            except TypeError:
                # Fallback for models that might not accept return_dynamics
                warnings.warn(
                    "Model does not support 'return_dynamics'. "
                    "Attempting generic hook-based analysis."
                )
                dynamics = self._hook_based_analysis(h, steps)

        # Process trajectory to numpy
        traj_tensors = dynamics.get("trajectory", [])
        if not traj_tensors:
            return {}

        traj_np = np.stack([t.cpu().numpy() for t in traj_tensors])

        # Compute deltas (L2 diff between steps)
        deltas = []
        activities = []
        for i in range(1, len(traj_np)):
            diff = np.linalg.norm(traj_np[i] - traj_np[i - 1])
            deltas.append(diff)
            activities.append(np.mean(np.abs(traj_np[i])))

        return {
            "trajectory": traj_np,
            "deltas": np.array(deltas),
            "activities": np.array(activities),
            "fixed_point": traj_np[-1],
        }

    def _hook_based_analysis(self, h, steps):
        """Fallback: Use hooks to capture hidden states if model doesn't support explicit return."""
        # This is hard to do generically without knowing layer names.
        # For now, return empty.
        return {}

    def get_energy_trajectory(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int | None = None,
        beta: float = 0.5,
    ) -> EnergyTrajectory:
        """
        Compute energy trajectory for energy-based models (EqProp, CHL, etc.).

        Args:
            x: Input tensor.
            y: Target tensor.
            steps: Number of equilibrium steps.
            beta: Nudge strength for nudged phase.

        Returns:
            EnergyTrajectory with free/nudged energy trajectories and gap.
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        # Check if model has energy method
        if not hasattr(self.model, "energy"):
            warnings.warn("Model does not have energy() method. Cannot compute energy trajectory.")
            return EnergyTrajectory(
                free_energy=np.array([]),
                nudged_energy=None,
                energy_gap=None,
                steps=np.array([]),
            )

        # Prepare input
        if hasattr(self.model, "has_embed") and self.model.has_embed:
            h = self.model.embed(x).mean(dim=1)
        elif x.dim() > 2 and not any(
            k in self.model.__class__.__name__ for k in ["Conv", "Transformer"]
        ):
            h = x.reshape(x.size(0), -1)
        else:
            h = x

        free_energies = []
        nudged_energies = []

        with torch.no_grad():
            # Free phase
            kwargs = {"return_trajectory": True}
            if steps is not None:
                kwargs["steps"] = steps
            output = self.model(h, **kwargs)
            if isinstance(output, tuple) and isinstance(output[1], list):
                trajectory = output[1]
                for t in trajectory:
                    free_energies.append(self.model.energy(t, y).item())

            # Nudged phase
            kwargs["beta"] = beta
            output_nudged = self.model(h, **kwargs)
            if isinstance(output_nudged, tuple) and isinstance(output_nudged[1], list):
                trajectory_nudged = output_nudged[1]
                for t in trajectory_nudged:
                    nudged_energies.append(self.model.energy(t, y).item())

        free_energy = np.array(free_energies)
        nudged_energy = np.array(nudged_energies) if nudged_energies else None
        energy_gap = (
            free_energy - nudged_energy if nudged_energy is not None else None
        )
        steps_arr = np.arange(len(free_energy))

        return EnergyTrajectory(
            free_energy=free_energy,
            nudged_energy=nudged_energy,
            energy_gap=energy_gap,
            steps=steps_arr,
        )

    def compute_gradient_alignment(
        self, x: torch.Tensor, y: torch.Tensor, criterion=nn.CrossEntropyLoss()
    ) -> GradientAlignment:
        """
        Compute the cosine similarity between the true gradient (via Backprop)
        and the update proposed by the bio-plausible learning rule.

        Returns:
            GradientAlignment with detailed per-layer analysis.
        """
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        # 1. Compute Bio-Plausible Update
        self.model.zero_grad()

        # Prepare input
        if hasattr(self.model, "has_embed") and self.model.has_embed:
            h = self.model.embed(x).mean(dim=1)
        elif x.dim() > 2 and not any(
            k in self.model.__class__.__name__ for k in ["Conv", "Transformer"]
        ):
            h = x.reshape(x.size(0), -1)
        else:
            h = x

        # Run model's custom backward mechanism
        if hasattr(self.model, "train_step"):
            self.model.train_step(x, y)
        else:
            # Standard EqProp with .backward()
            out = self.model(h)
            loss = criterion(out, y)
            loss.backward()

        # Capture Bio Gradients
        bio_grads = {}
        bio_grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                bio_grads[name] = param.grad.clone()
                bio_grad_norms[name] = param.grad.norm().item()

        # 2. Compute True Backprop Gradients
        per_layer_alignment = {}
        alignment_sum = 0
        count = 0
        bp_grad_norms = {}

        if hasattr(self.model, "gradient_method"):
            original_method = self.model.gradient_method
            try:
                self.model.gradient_method = "bptt"  # Force BPTT
                self.model.zero_grad()
                out = self.model(h)
                loss = criterion(out, y)
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in bio_grads:
                        g_bio = bio_grads[name].flatten()
                        g_bp = param.grad.flatten()

                        # Cosine similarity
                        sim = torch.nn.functional.cosine_similarity(
                            g_bio.unsqueeze(0), g_bp.unsqueeze(0)
                        ).item()
                        per_layer_alignment[name] = sim
                        alignment_sum += sim
                        count += 1
                        bp_grad_norms[name] = param.grad.norm().item()

            finally:
                # Restore method
                self.model.gradient_method = original_method

        overall_alignment = alignment_sum / count if count > 0 else 0.0
        angle_degrees = np.degrees(np.arccos(np.clip(overall_alignment, -1, 1)))
        bio_grad_norm = float(np.mean(list(bio_grad_norms.values()))) if bio_grad_norms else 0.0
        bp_grad_norm = float(np.mean(list(bp_grad_norms.values()))) if bp_grad_norms else 0.0

        return GradientAlignment(
            cosine_similarity=overall_alignment,
            per_layer_alignment=per_layer_alignment,
            angle_degrees=angle_degrees,
            bio_grad_norm=bio_grad_norm,
            bp_grad_norm=bp_grad_norm,
        )

    def get_tile_heatmap_data(self) -> TileHeatmapData | None:
        """
        Extract tile activities, errors, and importance for TileAlgorithm models.

        Returns:
            TileHeatmapData or None if model is not a TileAlgorithm.
        """
        # Check if model has tile structure
        if not hasattr(self.model, "graph") or not hasattr(self.model, "tile_importance"):
            return None

        tile_activities = []
        tile_errors = []
        tile_importance = []
        layer_ids = []
        tile_ids = []

        for tile in self.model.graph.all_tiles:
            tile_ids.append(tile.id)
            layer_ids.append(tile.layer_id)

            # Activity
            if tile.activity is not None:
                act = tile.activity.cpu().numpy()
            else:
                act = np.zeros((1, tile.neurons))
            tile_activities.append(act)

            # Error
            if tile.error is not None:
                err = tile.error.cpu().numpy()
            else:
                err = np.zeros((1, tile.neurons))
            tile_errors.append(err)

            # Importance
            # Find tile index in sorted list
            sorted_ids = sorted(self.model.graph.tiles.keys())
            tile_idx = sorted_ids.index(tile.id)
            imp = torch.sigmoid(self.model.tile_importance[tile_idx]).item()
            tile_importance.append(imp)

        return TileHeatmapData(
            tile_activities=np.array(tile_activities),
            tile_errors=np.array(tile_errors),
            tile_importance=np.array(tile_importance),
            layer_ids=np.array(layer_ids),
            tile_ids=np.array(tile_ids),
        )

    def plot_convergence(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        title: str = "Convergence Dynamics",
    ):
        """
        Plot convergence metrics using Matplotlib.

        Args:
            x: Input tensor.
            steps: Number of steps.
            title: Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install it.")

        data = self.get_convergence_data(x, steps)
        if not data:
            raise ValueError("Could not extract convergence data from model.")

        deltas = data["deltas"]
        activities = data["activities"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Deltas (Convergence Speed)
        ax1.plot(deltas, marker="o", color="tab:blue")
        ax1.set_title("Equilibrium Error (State Change)")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("|| h_t - h_{t-1} ||")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # Plot Activity
        ax2.plot(activities, marker="s", color="tab:orange")
        ax2.set_title("Neural Activity")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Mean |h_t|")
        ax2.grid(True)

        fig.suptitle(title)
        plt.tight_layout()

        return fig

    def plot_energy_trajectory(
        self,
        energy_traj: EnergyTrajectory,
        title: str = "Energy Trajectory",
    ):
        """Plot energy trajectory using Matplotlib."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install it.")

        fig, ax = plt.subplots(figsize=(10, 5))

        if len(energy_traj.free_energy) > 0:
            ax.plot(
                energy_traj.steps,
                energy_traj.free_energy,
                marker="o",
                label="Free Phase",
                color="tab:blue",
            )

        if energy_traj.nudged_energy is not None:
            ax.plot(
                energy_traj.steps,
                energy_traj.nudged_energy,
                marker="s",
                label="Nudged Phase",
                color="tab:orange",
            )

        if energy_traj.energy_gap is not None:
            ax.plot(
                energy_traj.steps,
                energy_traj.energy_gap,
                marker="^",
                label="Energy Gap",
                color="tab:green",
            )

        ax.set_xlabel("Settling Step")
        ax.set_ylabel("Energy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_tile_heatmap(
        self,
        heatmap_data: TileHeatmapData,
        metric: str = "activity",
        title: str = "Tile Heatmap",
    ):
        """
        Plot tile heatmap using Matplotlib.

        Args:
            heatmap_data: TileHeatmapData from get_tile_heatmap_data()
            metric: 'activity', 'error', or 'importance'
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install it.")

        if metric == "activity":
            data = heatmap_data.tile_activities.mean(axis=1)  # Average over batch
            cmap = "viridis"
            label = "Mean Activity"
        elif metric == "error":
            data = heatmap_data.tile_errors.mean(axis=1)
            cmap = "hot"
            label = "Mean Error Norm"
        elif metric == "importance":
            data = heatmap_data.tile_importance.reshape(-1, 1)
            cmap = "plasma"
            label = "Importance"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Organize by layer
        unique_layers = np.unique(heatmap_data.layer_ids)
        max_tiles_per_layer = max(
            np.sum(heatmap_data.layer_ids == l) for l in unique_layers
        )

        heatmap_grid = np.zeros((len(unique_layers), max_tiles_per_layer))
        heatmap_grid[:] = np.nan

        for i, (tid, lid) in enumerate(
            zip(heatmap_data.tile_ids, heatmap_data.layer_ids)
        ):
            layer_idx = np.where(unique_layers == lid)[0][0]
            tile_idx = np.sum(
                (heatmap_data.layer_ids[:i] == lid) & (heatmap_data.tile_ids[:i] == tid)
            )
            if data.ndim == 2:
                heatmap_grid[layer_idx, tile_idx] = data[i, 0] if data.shape[1] == 1 else data[i].mean()
            else:
                heatmap_grid[layer_idx, tile_idx] = data[i]

        im = ax.imshow(heatmap_grid, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(len(unique_layers)))
        ax.set_yticklabels([f"Layer {l}" for l in unique_layers])
        ax.set_xlabel("Tile Index within Layer")
        ax.set_title(f"{title} ({label})")
        plt.colorbar(im, ax=ax, label=label)
        plt.tight_layout()

        return fig

    # --- Plotly Interactive Visualizations ---

    def plot_convergence_plotly(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        title: str = "Convergence Dynamics",
    ) -> go.Figure:
        """Create interactive Plotly convergence plot."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Please install it.")

        data = self.get_convergence_data(x, steps)
        if not data:
            raise ValueError("Could not extract convergence data from model.")

        deltas = data["deltas"]
        activities = data["activities"]
        steps_arr = np.arange(len(deltas))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Equilibrium Error (State Change)", "Neural Activity"),
            horizontal_spacing=0.1,
        )

        # Deltas
        fig.add_trace(
            go.Scatter(
                x=steps_arr,
                y=deltas,
                mode="lines+markers",
                name="State Change",
                line=dict(color="blue"),
                marker=dict(size=6),
            ),
            row=1, col=1,
        )

        # Activities
        fig.add_trace(
            go.Scatter(
                x=steps_arr,
                y=activities,
                mode="lines+markers",
                name="Mean Activity",
                line=dict(color="orange"),
                marker=dict(size=6),
            ),
            row=1, col=2,
        )

        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_yaxes(title_text="|| h_t - h_{t-1} ||", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Mean |h_t|", row=1, col=2)

        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False,
            height=400,
        )

        return fig

    def plot_energy_trajectory_plotly(
        self,
        energy_traj: EnergyTrajectory,
        title: str = "Energy Trajectory",
    ) -> go.Figure:
        """Create interactive Plotly energy trajectory plot."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Please install it.")

        fig = go.Figure()

        if len(energy_traj.free_energy) > 0:
            fig.add_trace(
                go.Scatter(
                    x=energy_traj.steps,
                    y=energy_traj.free_energy,
                    mode="lines+markers",
                    name="Free Phase",
                    line=dict(color="blue"),
                    marker=dict(size=6),
                )
            )

        if energy_traj.nudged_energy is not None:
            fig.add_trace(
                go.Scatter(
                    x=energy_traj.steps,
                    y=energy_traj.nudged_energy,
                    mode="lines+markers",
                    name="Nudged Phase",
                    line=dict(color="orange"),
                    marker=dict(size=6),
                )
            )

        if energy_traj.energy_gap is not None:
            fig.add_trace(
                go.Scatter(
                    x=energy_traj.steps,
                    y=energy_traj.energy_gap,
                    mode="lines+markers",
                    name="Energy Gap",
                    line=dict(color="green"),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Settling Step",
            yaxis_title="Energy",
            template="plotly_white",
            hovermode="x unified",
            height=400,
        )

        return fig

    def plot_tile_heatmap_plotly(
        self,
        heatmap_data: TileHeatmapData,
        metric: str = "activity",
        title: str = "Tile Heatmap",
    ) -> go.Figure:
        """Create interactive Plotly tile heatmap."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Please install it.")

        if metric == "activity":
            data = heatmap_data.tile_activities.mean(axis=1)  # [tiles, neurons]
            colorscale = "Viridis"
            label = "Mean Activity"
        elif metric == "error":
            data = heatmap_data.tile_errors.mean(axis=1)
            colorscale = "Hot"
            label = "Mean Error Norm"
        elif metric == "importance":
            data = heatmap_data.tile_importance.reshape(-1, 1)
            colorscale = "Plasma"
            label = "Importance"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Organize by layer for better visualization
        unique_layers = np.unique(heatmap_data.layer_ids)
        max_tiles_per_layer = max(
            np.sum(heatmap_data.layer_ids == l) for l in unique_layers
        )

        heatmap_grid = np.full((len(unique_layers), max_tiles_per_layer), np.nan)

        for i, (tid, lid) in enumerate(
            zip(heatmap_data.tile_ids, heatmap_data.layer_ids)
        ):
            layer_idx = np.where(unique_layers == lid)[0][0]
            # Find position within layer
            layer_mask = heatmap_data.layer_ids == lid
            layer_tiles = np.where(layer_mask)[0]
            tile_idx = np.where(layer_tiles == i)[0][0]
            if data.ndim == 2 and data.shape[1] > 1:
                heatmap_grid[layer_idx, tile_idx] = data[i].mean()
            else:
                heatmap_grid[layer_idx, tile_idx] = data[i, 0] if data.ndim == 2 else data[i]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_grid,
                x=[f"Tile {j}" for j in range(max_tiles_per_layer)],
                y=[f"Layer {l}" for l in unique_layers],
                colorscale=colorscale,
                colorbar=dict(title=label),
                hoverongaps=False,
                hovertemplate="Layer: %{y}<br>Tile: %{x}<br>Value: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Tile Index within Layer",
            yaxis_title="Layer",
            template="plotly_white",
            height=500,
        )

        return fig

    def plot_gradient_alignment_plotly(
        self,
        alignment: GradientAlignment,
        title: str = "Gradient Alignment Analysis",
    ) -> go.Figure:
        """Create interactive Plotly gradient alignment plot."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plotting. Please install it.")

        # Per-layer alignment bar chart
        layers = list(alignment.per_layer_alignment.keys())
        values = list(alignment.per_layer_alignment.values())

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Per-Layer Cosine Similarity (Overall: {alignment.cosine_similarity:.4f}, Angle: {alignment.angle_degrees:.1f}°)",
                "Gradient Norm Comparison",
            ),
            vertical_spacing=0.15,
        )

        # Alignment bars
        colors = ["green" if v > 0.5 else "orange" if v > 0 else "red" for v in values]
        fig.add_trace(
            go.Bar(
                x=layers,
                y=values,
                name="Cosine Similarity",
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            ),
            row=1, col=1,
        )

        # Gradient norms
        bio_norms = [
            alignment.bio_grad_norm if name in alignment.per_layer_alignment else 0
            for name in layers
        ]
        bp_norms = [
            alignment.bp_grad_norm if name in alignment.per_layer_alignment else 0
            for name in layers
        ]

        fig.add_trace(
            go.Bar(
                x=layers,
                y=bio_norms,
                name="Bio-Plausible Grad Norm",
                marker_color="blue",
                opacity=0.7,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=layers,
                y=bp_norms,
                name="Backprop Grad Norm",
                marker_color="red",
                opacity=0.7,
            ),
            row=2, col=1,
        )

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_yaxes(title_text="Cosine Similarity", range=[-1, 1], row=1, col=1)
        fig.update_yaxes(title_text="Gradient Norm", row=2, col=1)

        fig.update_layout(
            title=title,
            template="plotly_white",
            barmode="group",
            height=700,
            showlegend=True,
        )

        return fig

    def generate_full_report(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int | None = None,
        beta: float = 0.5,
        output_dir: str | Path = "results/dynamics",
    ) -> dict[str, Path]:
        """
        Generate a full dynamics analysis report with all visualizations.

        Args:
            x: Input tensor.
            y: Target tensor.
            steps: Number of settling steps.
            beta: Nudge strength.
            output_dir: Directory to save plots.

        Returns:
            Dictionary mapping plot names to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_paths = {}

        # Convergence plot
        if HAS_MATPLOTLIB:
            fig = self.plot_convergence(x, steps, "Convergence Dynamics")
            path = output_dir / "convergence.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            report_paths["convergence"] = path

        # Energy trajectory
        energy_traj = self.get_energy_trajectory(x, y, steps, beta)
        if len(energy_traj.free_energy) > 0 and HAS_MATPLOTLIB:
            fig = self.plot_energy_trajectory(energy_traj, "Energy Trajectory")
            path = output_dir / "energy_trajectory.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            report_paths["energy_trajectory"] = path

        # Gradient alignment
        alignment = self.compute_gradient_alignment(x, y)
        if HAS_MATPLOTLIB and alignment.per_layer_alignment:
            fig, ax = plt.subplots(figsize=(10, 5))
            layers = list(alignment.per_layer_alignment.keys())
            values = list(alignment.per_layer_alignment.values())
            colors = ["green" if v > 0.5 else "orange" if v > 0 else "red" for v in values]
            ax.bar(range(len(layers)), values, color=colors)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha="right")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(
                f"Gradient Alignment (Overall: {alignment.cosine_similarity:.4f})"
            )
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            plt.tight_layout()
            path = output_dir / "gradient_alignment.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            report_paths["gradient_alignment"] = path

        # Tile heatmaps
        heatmap_data = self.get_tile_heatmap_data()
        if heatmap_data is not None and HAS_MATPLOTLIB:
            for metric in ["activity", "error", "importance"]:
                fig = self.plot_tile_heatmap(heatmap_data, metric, f"Tile {metric.capitalize()} Heatmap")
                path = output_dir / f"tile_{metric}_heatmap.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                report_paths[f"tile_{metric}_heatmap"] = path

        # Plotly interactive versions
        if HAS_PLOTLY:
            plotly_dir = output_dir / "plotly"
            plotly_dir.mkdir(exist_ok=True)

            fig = self.plot_convergence_plotly(x, steps, "Convergence Dynamics")
            fig.write_html(plotly_dir / "convergence.html")
            report_paths["convergence_plotly"] = plotly_dir / "convergence.html"

            if len(energy_traj.free_energy) > 0:
                fig = self.plot_energy_trajectory_plotly(energy_traj, "Energy Trajectory")
                fig.write_html(plotly_dir / "energy_trajectory.html")
                report_paths["energy_trajectory_plotly"] = plotly_dir / "energy_trajectory.html"

            if alignment.per_layer_alignment:
                fig = self.plot_gradient_alignment_plotly(alignment, "Gradient Alignment Analysis")
                fig.write_html(plotly_dir / "gradient_alignment.html")
                report_paths["gradient_alignment_plotly"] = plotly_dir / "gradient_alignment.html"

            if heatmap_data is not None:
                for metric in ["activity", "error", "importance"]:
                    fig = self.plot_tile_heatmap_plotly(heatmap_data, metric, f"Tile {metric.capitalize()} Heatmap")
                    fig.write_html(plotly_dir / f"tile_{metric}_heatmap.html")
                    report_paths[f"tile_{metric}_heatmap_plotly"] = plotly_dir / f"tile_{metric}_heatmap.html"

        return report_paths
