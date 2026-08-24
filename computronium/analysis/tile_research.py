"""Substrate-native research utilities: tracking, metrics, visualization, ablation.

Ported from equitile/analysis/research.py to work with TileAlgorithm substrate.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Literal, overload

import torch

from computronium.config.unified import ExperimentConfig
from computronium.core.checkpoint import save_checkpoint
from computronium.core.local_learning.algorithm import TileAlgorithm

__all__ = [
    "AblationConfig",
    "AblationStudy",
    "ExperimentConfig",
    "ExperimentTracker",
    "MetricCollector",
    "MetricEntry",
    "VisualizationHelper",
    "create_ablation_study",
    "create_metric_collector",
    "create_tracker",
    "create_visualization_helper",
]


# =============================================================================
# Experiment Tracker
# =============================================================================


class ExperimentTracker:
    """Tracks experiment parameters, metrics, and artifacts.

    Parameters
    ----------
    experiment_name : str
        Experiment name
    log_dir : str, optional
        Directory for logs
    config : ExperimentConfig, optional
        Experiment configuration

    Examples
    --------
    >>> tracker = ExperimentTracker("mnist_experiment")
    >>> tracker.log_params({"lr": 0.01, "batch_size": 32})
    >>> tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=100)
    >>> tracker.save()
    """

    def __init__(
        self,
        experiment_name: str = "",
        log_dir: str | None = None,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.config = config or ExperimentConfig(name=experiment_name)
        self.experiment_name = experiment_name or self.config.name

        # Set up log directory
        if log_dir is None:
            log_dir = str(Path("logs") / "equitile" / self.experiment_name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tracking state
        self._params: dict[str, object] = {}
        self._metrics: list[dict[str, object]] = []
        self._artifacts: list[str] = []
        self._start_time = time.time()

    def log_params(self, params: dict[str, object]) -> None:
        """Log experiment parameters.

        Parameters
        ----------
        params : dict
            Parameters to log
        """
        self._params.update(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log metrics.

        Parameters
        ----------
        metrics : dict
            Metrics to log
        step : int, optional
            Training step
        epoch : int, optional
            Epoch number
        """
        entry: dict[str, object] = {
            "timestamp": time.time(),
            "step": step,
            "epoch": epoch,
            **metrics,
        }
        self._metrics.append(entry)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact (file).

        Parameters
        ----------
        path : str
            Path to artifact
        name : str, optional
            Artifact name
        """
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        # Copy to log directory
        artifact_name = name or artifact_path.name
        dest_path = self.log_dir / artifact_name

        # Read and write to preserve
        with Path(artifact_path).open("rb") as f:
            content = f.read()
        with Path(dest_path).open("wb") as f:
            f.write(content)

        self._artifacts.append(str(dest_path))

    def log_model(
        self,
        model: TileAlgorithm,
        name: str = "model",
        include_graph: bool = False,
    ) -> None:
        """Log model checkpoint.

        Parameters
        ----------
        model : TileAlgorithm
            Model to save
        name : str
            Model name
        include_graph : bool
            Include model graph
        """
        path = self.log_dir / f"{name}.pt"
        config = getattr(model, "get_config", lambda: None)()
        config_dict: dict[str, object] = {}
        if is_dataclass(config) and not isinstance(config, type):
            config_dict = dict(asdict(config))
        elif isinstance(config, dict):
            config_dict = dict(config)
        save_checkpoint(
            path,
            {"model_state_dict": model.state_dict(), "config": config_dict},
        )
        self.log_artifact(str(path))

        if include_graph:
            graph_path = self.log_dir / f"{name}_graph.json"
            self._save_model_graph(model, str(graph_path))
            self.log_artifact(str(graph_path))

    def _save_model_graph(self, model: TileAlgorithm, path: str) -> None:
        """Save model graph to JSON.

        Parameters
        ----------
        model : TileAlgorithm
            Model
        path : str
            Output path
        """
        graph_data = {
            "n_tiles": len(model.graph.tiles),
            "n_edges": len(model.graph.edges),
            "tiles": [
                {
                    "id": tile.id,
                    "layer": tile.layer_id,
                    "neurons": tile.neurons,
                    "is_input": tile.is_input,
                    "is_output": tile.is_output,
                }
                for tile in model.graph.all_tiles
            ],
        }
        with Path(path).open("w") as f:
            json.dump(graph_data, f, indent=2)

    @overload
    def get_metrics(
        self, metric_name: str, as_array: Literal[True] = True
    ) -> list[float]: ...

    @overload
    def get_metrics(
        self, metric_name: str, as_array: Literal[False]
    ) -> list[dict[str, object]]: ...

    def get_metrics(
        self,
        metric_name: str,
        as_array: bool = True,
    ) -> list[float] | list[dict[str, object]]:
        """Get logged metrics.

        Parameters
        ----------
        metric_name : str
            Metric name
        as_array : bool
            Return as array of values

        Returns
        -------
        list
            Metrics
        """
        if as_array:
            values: list[float] = []
            for m in self._metrics:
                v = m.get(metric_name)
                if isinstance(v, (int, float)):
                    values.append(float(v))
            return values
        return [m for m in self._metrics if metric_name in m]

    def get_summary(self) -> dict[str, object]:
        """Get experiment summary.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self._metrics:
            return {}

        # Compute summary statistics for numeric metrics
        summary: dict[str, object] = {
            "experiment_name": self.experiment_name,
            "n_steps": len(self._metrics),
            "duration_seconds": time.time() - self._start_time,
            "params": self._params,
        }

        # Get all metric keys
        metric_keys = set()
        for m in self._metrics:
            metric_keys.update(
                k for k in m.keys() if k not in ("timestamp", "step", "epoch")
            )

        # Compute stats for each metric
        for key in metric_keys:
            values = self.get_metrics(key)
            if values and all(v is not None for v in values):
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_final"] = values[-1]

        return summary

    def save(self) -> str:
        """Save experiment data.

        Returns
        -------
        str
            Path to saved file
        """
        # Save params
        params_path = self.log_dir / "params.json"
        with Path(params_path).open("w") as f:
            json.dump(self._params, f, indent=2)

        # Save metrics
        metrics_path = self.log_dir / "metrics.json"
        with Path(metrics_path).open("w") as f:
            json.dump(self._metrics, f, indent=2)

        # Save summary
        summary_path = self.log_dir / "summary.json"
        with Path(summary_path).open("w") as f:
            json.dump(self.get_summary(), f, indent=2)

        return str(self.log_dir)

    def export_csv(self, path: str | None = None) -> str:
        """Export metrics to CSV.

        Parameters
        ----------
        path : str, optional
            Output path

        Returns
        -------
        str
            Path to CSV file
        """
        if path is None:
            path = str(self.log_dir / "metrics.csv")

        if not self._metrics:
            return path

        # Get all keys
        keys = set()
        for m in self._metrics:
            keys.update(m.keys())

        # Write CSV
        with Path(path).open("w") as f:
            # Header
            f.write(",".join(sorted(keys)) + "\n")

            # Rows
            for m in self._metrics:
                values = [str(m.get(k, "")) for k in sorted(keys)]
                f.write(",".join(values) + "\n")

        return path


# =============================================================================
# Metric Collector
# =============================================================================


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """Single metric entry.

    Attributes
    ----------
    name : str
        Metric name
    value : float
        Metric value
    step : int
            Training step
    timestamp : float
        Unix timestamp
    tags : dict
        Additional tags
    """

    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Collects and aggregates metrics.

    Parameters
    ----------
    window_size : int
        Window size for moving averages
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._metrics: dict[str, list[MetricEntry]] = {}
        self._step = 0

    def add(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Add a metric.

        Parameters
        ----------
        name : str
            Metric name
        value : float
            Metric value
        tags : dict, optional
            Additional tags
        """
        if name not in self._metrics:
            self._metrics[name] = []

        entry = MetricEntry(
            name=name,
            value=value,
            step=self._step,
            tags=tags or {},
        )
        self._metrics[name].append(entry)

        # Trim history
        if len(self._metrics[name]) > self.window_size:
            self._metrics[name].pop(0)

    def step(self) -> None:
        """Increment step counter."""
        self._step += 1

    def get(self, name: str) -> list[float]:
        """Get metric values.

        Parameters
        ----------
        name : str
            Metric name

        Returns
        -------
        list
            Metric values
        """
        if name not in self._metrics:
            return []
        return [e.value for e in self._metrics[name]]

    def get_mean(self, name: str, window: int | None = None) -> float | None:
        """Get mean of metric.

        Parameters
        ----------
        name : str
            Metric name
        window : int, optional
            Window size

        Returns
        -------
        float, optional
            Mean value
        """
        values = self.get(name)
        if not values:
            return None

        if window is not None:
            values = values[-window:]

        return sum(values) / len(values)

    def get_trend(self, name: str, window: int = 10) -> str:
        """Get metric trend.

        Parameters
        ----------
        name : str
            Metric name
        window : int
            Window size

        Returns
        -------
        str
            Trend direction
        """
        values = self.get(name)
        if len(values) < window * 2:
            return "stable"

        recent = sum(values[-window:]) / window
        older = sum(values[-window * 2 : -window]) / window

        if recent < older * 0.95:
            return "decreasing"
        elif recent > older * 1.05:
            return "increasing"
        return "stable"

    def get_all(self) -> dict[str, list[float]]:
        """Get all metrics.

        Returns
        -------
        dict
            All metrics
        """
        return {name: self.get(name) for name in self._metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._step = 0


# =============================================================================
# Visualization Helpers
# =============================================================================


class VisualizationHelper:
    """Visualization helpers for the tile substrate.

    Parameters
    ----------
    model : TileAlgorithm
        Model to visualize
    """

    def __init__(self, model: TileAlgorithm) -> None:
        self.model = model

    def get_tile_activities(self) -> dict[int, torch.Tensor]:
        """Get tile activities.

        Returns
        -------
        dict
            Activities per tile
        """
        return {
            tile.id: tile.activity
            for tile in self.model.graph.all_tiles
            if tile.activity is not None
        }

    def get_tile_errors(self) -> dict[int, torch.Tensor]:
        """Get tile errors.

        Returns
        -------
        dict
            Errors per tile
        """
        return {
            tile.id: tile.error
            for tile in self.model.graph.all_tiles
            if tile.error is not None
        }

    def get_importance_map(self) -> dict[int, float]:
        """Get tile importance map.

        Returns
        -------
        dict
            Importance per tile
        """
        return {
            tile.id: torch.sigmoid(self.model.tile_importance[i]).item()
            for i, tile in enumerate(self.model.graph.all_tiles)
        }

    def get_error_heatmap_data(self) -> list[list[float]]:
        """Get error data for heatmap visualization.

        Returns
        -------
        list
            2D error array
        """
        # Organize by layer
        layers: dict[int, list[float]] = {}
        for tile in self.model.graph.all_tiles:
            layer = tile.layer_id
            if layer not in layers:
                layers[layer] = []

            if tile.error is not None:
                error_norm = tile.error.norm(p=2).item()
            else:
                error_norm = 0.0

            layers[layer].append(error_norm)

        # Convert to 2D array
        max_tiles = max(len(tiles) for tiles in layers.values()) if layers else 1
        heatmap = []
        for layer_id in sorted(layers.keys()):
            row = layers[layer_id] + [0.0] * (max_tiles - len(layers[layer_id]))
            heatmap.append(row)

        return heatmap

    def get_graph_data(self) -> dict[str, object]:
        """Get graph data for visualization.

        Returns
        -------
        dict
            Graph data
        """
        nodes = []
        edges = []

        for tile in self.model.graph.all_tiles:
            nodes.append({
                "id": tile.id,
                "layer": tile.layer_id,
                "neurons": tile.neurons,
                "is_input": tile.is_input,
                "is_output": tile.is_output,
                "pos_x": tile.pos_x,
                "pos_y": tile.pos_y,
            })

        weights = getattr(self.model, "_tile_weights", None)
        for src, dst in self.model.graph.edges:
            weight = weights.get(f"{src}_{dst}") if weights is not None else None
            edges.append({
                "source": src,
                "target": dst,
                "weight_norm": weight.norm().item() if weight is not None else 0.0,
            })

        return {"nodes": nodes, "edges": edges}

    def plot_activities(self, ax=None):
        """Plot tile activities.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        matplotlib.axes.Axes
            Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        if ax is None:
            _, ax = plt.subplots()

        activities = self.get_tile_activities()

        tile_ids = []
        means = []
        stds = []

        for tile_id, activity in activities.items():
            tile_ids.append(tile_id)
            means.append(activity.mean().item())
            stds.append(activity.std().item())

        ax.bar(range(len(tile_ids)), means, yerr=stds, capsize=3)
        ax.set_xlabel("Tile ID")
        ax.set_ylabel("Activity")
        ax.set_title("Tile Activities")

        return ax

    def plot_errors(self, ax=None):
        """Plot tile errors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        matplotlib.axes.Axes
            Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        if ax is None:
            _, ax = plt.subplots()

        errors = self.get_tile_errors()

        tile_ids = []
        norms = []

        for tile_id, error in errors.items():
            tile_ids.append(tile_id)
            norms.append(error.norm().item())

        ax.bar(range(len(tile_ids)), norms)
        ax.set_xlabel("Tile ID")
        ax.set_ylabel("Error Norm")
        ax.set_title("Tile Errors")

        return ax


# =============================================================================
# Ablation Study Support
# =============================================================================


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Ablation study configuration.

    Attributes
    ----------
    name : str
        Study name
    baseline_params : dict
        Baseline parameters
    variants : list
            List of parameter variants
    """

    name: str
    baseline_params: dict[str, object]
    variants: list[dict[str, object]]


class AblationStudy:
    """Support for ablation studies.

    Parameters
    ----------
    config : AblationConfig
        Study configuration
    log_dir : str, optional
        Log directory
    """

    def __init__(
        self,
        config: AblationConfig,
        log_dir: str | None = None,
    ) -> None:
        self.config = config
        self.log_dir = (
            Path(log_dir) if log_dir else Path("logs") / "ablation" / config.name
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._results: dict[str, dict[str, object]] = {}

    def run_variant(
        self,
        variant_id: str,
        variant_params: dict[str, object],
        train_fn: Callable[[dict[str, object]], dict[str, object]],
    ) -> dict[str, object]:
        """Run a single variant.

        Parameters
        ----------
        variant_id : str
            Variant identifier
        variant_params : dict
            Variant parameters
        train_fn : callable
            Training function

        Returns
        -------
        dict
            Results
        """
        # Merge with baseline
        params = {**self.config.baseline_params, **variant_params}

        # Create tracker
        tracker = ExperimentTracker(
            experiment_name=f"{self.config.name}_{variant_id}",
            log_dir=str(self.log_dir / variant_id),
        )
        tracker.log_params(params)

        # Run training
        results = train_fn(params)

        # Log results (coerce to the float metric contract)
        tracker.log_metrics({
            k: float(v) for k, v in results.items() if isinstance(v, (int, float))
        })
        tracker.save()

        self._results[variant_id] = results
        return results

    def run_all(
        self,
        train_fn: Callable[[dict[str, object]], dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        """Run all variants.

        Parameters
        ----------
        train_fn : callable
            Training function

        Returns
        -------
        dict
            All results
        """
        # Run baseline
        baseline_id = "baseline"
        self._results[baseline_id] = self.run_variant(
            baseline_id,
            {},
            train_fn,
        )

        # Run variants
        for i, variant in enumerate(self.config.variants):
            variant_id = f"variant_{i}"
            self._results[variant_id] = self.run_variant(
                variant_id,
                variant,
                train_fn,
            )

        return self._results

    def get_comparison(self) -> dict[str, object]:
        """Get comparison of all variants.

        Returns
        -------
        dict
            Comparison data
        """
        comparison = {
            "study_name": self.config.name,
            "variants": list(self._results.keys()),
            "results": self._results,
        }

        # Save comparison
        path = self.log_dir / "comparison.json"
        with Path(path).open("w") as f:
            json.dump(comparison, f, indent=2)

        return comparison

    def export_table(self) -> str:
        """Export results as markdown table.

        Returns
        -------
        str
            Markdown table
        """
        if not self._results:
            return ""

        # Get all metric keys
        all_keys = set()
        for results in self._results.values():
            all_keys.update(results.keys())

        # Build table
        lines = []
        lines.append("| Variant | " + " | ".join(sorted(all_keys)) + " |")
        lines.append("|" + "|".join(["---"] * (len(all_keys) + 1)) + "|")

        for variant_id, results in self._results.items():
            values = [str(results.get(k, "N/A")) for k in sorted(all_keys)]
            lines.append(f"| {variant_id} | " + " | ".join(values) + " |")

        table = "\n".join(lines)

        # Save
        path = self.log_dir / "results.md"
        with Path(path).open("w") as f:
            f.write(table)

        return table


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracker(
    experiment_name: str,
    log_dir: str | None = None,
    tags: list[str] | None = None,
) -> ExperimentTracker:
    """Create an experiment tracker.

    Parameters
    ----------
    experiment_name : str
        Experiment name
    log_dir : str, optional
        Log directory
    tags : list, optional
        Experiment tags

    Returns
    -------
    ExperimentTracker
        Tracker
    """
    config = ExperimentConfig(
        name=experiment_name,
        tags=tuple(tags or []),
    )
    return ExperimentTracker(experiment_name, log_dir, config)


def create_metric_collector(window_size: int = 100) -> MetricCollector:
    """Create a metric collector.

    Parameters
    ----------
    window_size : int
        Window size

    Returns
    -------
    MetricCollector
        Collector
    """
    return MetricCollector(window_size)


def create_visualization_helper(model: TileAlgorithm) -> VisualizationHelper:
    """Create a visualization helper.

    Parameters
    ----------
    model : TileAlgorithm
        Model

    Returns
    -------
    VisualizationHelper
        Helper
    """
    return VisualizationHelper(model)


def create_ablation_study(
    name: str,
    baseline_params: dict[str, object],
    variants: list[dict[str, object]],
    log_dir: str | None = None,
) -> AblationStudy:
    """Create an ablation study.

    Parameters
    ----------
    name : str
        Study name
    baseline_params : dict
        Baseline parameters
    variants : list
        Variants
    log_dir : str, optional
        Log directory

    Returns
    -------
    AblationStudy
        Study
    """
    config = AblationConfig(
        name=name,
        baseline_params=baseline_params,
        variants=variants,
    )
    return AblationStudy(config, log_dir)
