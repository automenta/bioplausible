"""Pareto Frontier Analysis for Multi-Objective Optimization.

Computes and visualizes Pareto frontiers for accuracy, parameters,
FLOPs, memory, energy, time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """A point on the Pareto frontier."""

    model: str
    objectives: dict[str, float]
    config: dict | None = None
    trial_id: int | None = None


@dataclass(frozen=True, slots=True)
class ParetoFrontier:
    """A Pareto frontier with associated metadata."""

    points: list[ParetoPoint]
    objectives: list[str]
    directions: list[Literal["maximize", "minimize"]]
    dominated_points: list[ParetoPoint] = ()

    def __len__(self) -> int:
        return len(self.points)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        import pandas as pd

        rows = []
        for p in self.points:
            row = {"model": p.model}
            row.update(p.objectives)
            if p.config:
                for k, v in p.config.items():
                    row[f"config_{k}"] = v
            if p.trial_id is not None:
                row["trial_id"] = p.trial_id
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self, path: str | Path) -> None:
        """Save to JSON."""
        data = {
            "objectives": self.objectives,
            "directions": self.directions,
            "points": [
                {
                    "model": p.model,
                    "objectives": p.objectives,
                    "config": p.config,
                    "trial_id": p.trial_id,
                }
                for p in self.points
            ],
        }
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# Pareto Computation
# =============================================================================


def _dominates(
    obj1: dict[str, float],
    obj2: dict[str, float],
    directions: list[Literal["maximize", "minimize"]],
    objectives: list[str],
) -> bool:
    """Check if obj1 dominates obj2.

    obj1 dominates if at least as good in all objectives
    and strictly better in one.
    """
    better_in_any = False
    for obj, direction in zip(objectives, directions):
        v1 = obj1.get(obj, 0)
        v2 = obj2.get(obj, 0)

        if direction == "maximize":
            if v1 < v2:
                return False
            if v1 > v2:
                better_in_any = True
        else:  # minimize
            if v1 > v2:
                return False
            if v1 < v2:
                better_in_any = True

    return better_in_any


def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: list[str],
    directions: list[Literal["maximize", "minimize"]],
    model_col: str = "model",
) -> pd.DataFrame:
    """Compute Pareto frontier from a DataFrame.

    Args:
        df: DataFrame with model results
        objectives: List of objective column names
        directions: List of "maximize" or "minimize" for each objective
        model_col: Column name for model identifier

    Returns:
        DataFrame with Pareto-optimal points, with added 'is_pareto' column
    """

    # Ensure we have all objectives
    for obj in objectives:
        if obj not in df.columns:
            raise ValueError("Missing objective")  # ruff: ignore[raise-vanilla-args]

    # Create list of (index, objectives_dict, model)
    points = []
    for idx, row in df.iterrows():
        obj_dict = {obj: float(row[obj]) for obj in objectives}
        points.append((idx, obj_dict, row[model_col]))

    # Find non-dominated points
    pareto_indices = []
    for i, (idx_i, obj_i, model_i) in enumerate(points):
        dominated = False
        for j, (idx_j, obj_j, model_j) in enumerate(points):
            if i == j:
                continue
            if _dominates(obj_j, obj_i, directions, objectives):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(idx_i)

    # Create result DataFrame
    result = df.copy()
    result["is_pareto"] = result.index.isin(pareto_indices)

    return result


def pareto_frontier_from_trials(
    trials: list[dict],
    objectives: list[str],
    directions: list[Literal["maximize", "minimize"]],
    model_key: str = "model_name",
) -> ParetoFrontier:
    """Build ParetoFrontier from Optuna trial list.

    Args:
        trials: List of trial dicts with objectives and metadata
        objectives: Objective names
        directions: "maximize" or "minimize" for each objective
        model_key: Key for model name in trial dict

    Returns:
        ParetoFrontier object
    """
    points = []
    dominated = []

    for trial in trials:
        obj_dict = {obj: trial.get(obj, 0.0) for obj in objectives}
        model = trial.get(model_key, "unknown")
        config = trial.get("config")
        trial_id = trial.get("trial_id")

        # Check if dominated
        is_dominated = False
        for other in trials:
            if other is trial:
                continue
            other_obj = {obj: other.get(obj, 0.0) for obj in objectives}
            if _dominates(other_obj, obj_dict, directions, objectives):
                is_dominated = True
                break

        p = ParetoPoint(
            model=model,
            objectives=obj_dict,
            config=config,
            trial_id=trial_id,
        )

        if is_dominated:
            dominated.append(p)
        else:
            points.append(p)

    return ParetoFrontier(
        points=points,
        objectives=objectives,
        directions=directions,
        dominated_points=dominated,
    )


MIN_POINTS_FOR_KNEE = 3


def _get_secondary_objective(frontier: ParetoFrontier, primary: str) -> str | None:
    return next((o for o in frontier.objectives if o != primary), None)


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)


def _get_extreme_indices(x_norm: np.ndarray, prim_dir: str) -> tuple[int, int]:
    if prim_dir == "maximize":
        return np.argmax(x_norm), np.argmin(x_norm)
    return np.argmin(x_norm), np.argmax(x_norm)


def _line_distance(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    idx_min: int,
    idx_max: int,
) -> np.ndarray:
    x1, y1 = x_norm[idx_min], y_norm[idx_min]
    x2, y2 = x_norm[idx_max], y_norm[idx_max]
    a = y2 - y1
    b = -(x2 - x1)
    c = x2 * y1 - y2 * x1
    return np.abs(a * x_norm + b * y_norm + c) / np.sqrt(a**2 + b**2 + 1e-10)


def knee_detection(
    frontier: ParetoFrontier,
    primary_objective: str = "accuracy",
) -> ParetoPoint | None:
    """Detect knee point (maximum marginal return) on Pareto frontier.

    Uses the method of finding the point with maximum distance to the line
    connecting the extreme points in the primary vs secondary objective space.

    Args:
        frontier: ParetoFrontier object
        primary_objective: The primary objective (e.g., "accuracy")

    Returns:
        Knee point or None if not found
    """
    if len(frontier.points) < MIN_POINTS_FOR_KNEE:
        return None

    secondary = _get_secondary_objective(frontier, primary_objective)
    if secondary is None:
        return None

    prim_dir = frontier.directions[frontier.objectives.index(primary_objective)]

    x_vals = np.array([p.objectives[primary_objective] for p in frontier.points])
    y_vals = np.array([p.objectives[secondary] for p in frontier.points])

    x_norm = _normalize_array(x_vals)
    y_norm = _normalize_array(y_vals)

    idx_max, idx_min = _get_extreme_indices(x_norm, prim_dir)
    distances = _line_distance(x_norm, y_norm, idx_min, idx_max)

    return frontier.points[np.argmax(distances)]


# =============================================================================
# Visualization
# =============================================================================


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Configuration for Pareto plot."""

    color_obj: str | None = None
    size_obj: str | None = None
    title: str | None = None
    output_path: str | Path | None = None


def plot_pareto_frontier(
    frontier: ParetoFrontier,
    x_obj: str,
    y_obj: str,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot Pareto frontier using Plotly.

    Args:
        frontier: ParetoFrontier object
        x_obj: Objective for x-axis
        y_obj: Objective for y-axis
        config: Optional PlotConfig for styling and output

    Returns:
        Plotly Figure
    """
    import plotly.graph_objects as go

    if config is None:
        config = PlotConfig()

    df = frontier.to_dataframe()

    # Separate Pareto and dominated points
    pareto_df = df[df["is_pareto"]] if "is_pareto" in df.columns else df
    dominated_df = df[~df["is_pareto"]] if "is_pareto" in df.columns else None

    fig = go.Figure()

    # Dominated points (gray, small)
    if dominated_df is not None and len(dominated_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=dominated_df[x_obj],
                y=dominated_df[y_obj],
                mode="markers",
                name="Dominated",
                marker={
                    "color": "lightgray",
                    "size": 6,
                    "opacity": 0.5,
                },
                text=dominated_df["model"],
                hovertemplate=(
                "%{text}<br>"
                + x_obj
                + "=%{x:.4f}<br>"
                + y_obj
                + "=%{y:.4f}<extra></extra>"
            ),
            )
        )

    # Pareto points
    marker_dict = {"size": 10, "opacity": 0.8, "line": {"width": 1, "color": "white"}}

    if config.color_obj and config.color_obj in pareto_df.columns:
        marker_dict["color"] = pareto_df[config.color_obj]
        marker_dict["colorscale"] = "Viridis"
        marker_dict["showscale"] = True
        marker_dict["colorbar"] = {"title": config.color_obj}

    if config.size_obj and config.size_obj in pareto_df.columns:
        # Normalize sizes
        sizes = pareto_df[config.size_obj]
        denom = sizes.max() - sizes.min() + 1e-10
        sizes_norm = 8 + 20 * (sizes - sizes.min()) / denom
        marker_dict["size"] = sizes_norm

    fig.add_trace(
        go.Scatter(
            x=pareto_df[x_obj],
            y=pareto_df[y_obj],
            mode="markers+lines",
            name="Pareto Frontier",
            marker=marker_dict,
            line={"color": "red", "width": 2, "dash": "dot"},
            text=pareto_df["model"],
            hovertemplate=(
                "%{text}<br>"
                + x_obj
                + "=%{x:.4f}<br>"
                + y_obj
                + "=%{y:.4f}<extra></extra>"
            ),
        )
    )

    # Highlight knee point
    knee = knee_detection(frontier, primary_objective=x_obj)
    if knee:
        knee_marker = {
            "size": 18,
            "color": "gold",
            "symbol": "star",
            "line": {"width": 2, "color": "black"},
        }
        fig.add_trace(
            go.Scatter(
                x=[knee.objectives[x_obj]],
                y=[knee.objectives[y_obj]],
                mode="markers",
                name="Knee Point",
                marker=knee_marker,
                text=[f"KNEE: {knee.model}"],
                hovertemplate=(
                    "%{text}<br>"
                    + x_obj
                    + "=%{x:.4f}<br>"
                    + y_obj
                    + "=%{y:.4f}<extra></extra>"
                ),
            )
        )

    plot_title = config.title or f"Pareto Frontier: {y_obj} vs {x_obj}"
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_obj,
        yaxis_title=y_obj,
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="closest",
    )

    if config.output_path:
        fig.write_html(config.output_path)
        logger.info("Saved Pareto plot to %s", config.output_path)

    return fig


def plot_pareto_3d(
    frontier: ParetoFrontier,
    x_obj: str,
    y_obj: str,
    z_obj: str,
    output_path: str | Path | None = None,
) -> go.Figure:
    """3D Pareto frontier plot."""
    import plotly.graph_objects as go

    df = frontier.to_dataframe()
    pareto_df = df[df["is_pareto"]] if "is_pareto" in df.columns else df

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pareto_df[x_obj],
            y=pareto_df[y_obj],
            z=pareto_df[z_obj],
            mode="markers+lines",
            marker={"size": 8, "color": "red", "opacity": 0.8},
            line={"color": "red", "width": 3},
            text=pareto_df["model"],
            name="Pareto Frontier",
        )
    )

    fig.update_layout(
        title=f"3D Pareto Frontier: {x_obj} x {y_obj} x {z_obj}",
        scene={
            "xaxis_title": x_obj,
            "yaxis_title": y_obj,
            "zaxis_title": z_obj,
        },
        template="plotly_white",
    )

    if output_path:
        fig.write_html(output_path)

    return fig
