"""Substrate-native TileAlgorithm Profiler: Performance Diagnostics.

Ported from equitile/analysis/profiler.py to work with TileAlgorithm substrate.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from bioplausible.core.local_learning.algorithm import TileAlgorithm
from bioplausible.core.logging import get_logger

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "LearningMonitor",
    "MemoryProfiler",
    "ProfileResult",
    "TileAlgorithmProfiler",
    "TileStats",
    "create_profiler",
    "logger",
    "run_benchmark",
]

logger = get_logger()


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class TileStats:
    """Statistics for a single tile.

    Attributes
    ----------
    tile_id : int
        Tile identifier
    layer_id : int
        Layer index
    is_input : bool
        This is an input tile
    is_output : bool
        This is an output tile

    # Timing
    predict_time : float
        Time spent in prediction (seconds)
    update_time : float
        Time spent in update (seconds)
    total_time : float
        Total processing time

    # Activity
    activity_mean : float
        Mean activity value
    activity_std : float
        Activity standard deviation
    activity_max : float
        Maximum activity value
    error_norm : float
        L2 norm of error

    # Learning
    weight_update_norm : float
        Norm of weight updates
    importance : float
        Tile importance score

    # Computed fields
    call_count : int
        Number of times profiled
    """

    tile_id: int
    layer_id: int
    is_input: bool = False
    is_output: bool = False

    # Timing
    predict_time: float = 0.0
    update_time: float = 0.0
    total_time: float = 0.0

    # Activity
    activity_mean: float = 0.0
    activity_std: float = 0.0
    activity_max: float = 0.0
    error_norm: float = 0.0

    # Learning
    weight_update_norm: float = 0.0
    importance: float = 0.0

    # Computed fields
    call_count: int = 0

    @property
    def computed_total_time(self) -> float:
        """Compute total time from predict and update times."""
        return self.predict_time + self.update_time


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Results from a profiling session.

    Attributes
    ----------
    tile_stats : dict
        Tile-level statistics
    total_time : float
        Total profiling time
    predict_time : float
        Time in prediction phase
    update_time : float
        Time in update phase
    learning_time : float
        Time in learning phase
    param_memory_mb : float
        Parameter memory in MB
    activation_memory_mb : float
        Activation memory in MB
    batch_size : int
        Batch size used
    n_tiles : int
        Number of tiles
    n_edges : int
        Number of edges
    timestamp : float
        Unix timestamp of profile
    """

    # Tile-level stats
    tile_stats: dict[int, TileStats] = field(default_factory=dict)

    # Aggregate stats
    total_time: float = 0.0
    predict_time: float = 0.0
    update_time: float = 0.0
    learning_time: float = 0.0

    # Memory
    param_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0

    # Metadata
    batch_size: int = 0
    n_tiles: int = 0
    n_edges: int = 0
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> dict[str, int | float]:
        """Get summary statistics.

        Returns
        -------
        dict
            Summary statistics
        """
        total_time = self.total_time if self.total_time > 0 else 0.001
        predict_pct = (self.predict_time / total_time * 100) if total_time > 0 else 0
        update_pct = (self.update_time / total_time * 100) if total_time > 0 else 0

        return {
            "total_time_ms": total_time * 1000,
            "predict_time_ms": self.predict_time * 1000,
            "predict_pct": predict_pct,
            "update_time_ms": self.update_time * 1000,
            "update_pct": update_pct,
            "batch_size": self.batch_size,
            "n_tiles": self.n_tiles,
            "n_edges": self.n_edges,
            "param_memory_mb": self.param_memory_mb,
            "activation_memory_mb": self.activation_memory_mb,
            "total_memory_mb": self.param_memory_mb + self.activation_memory_mb,
        }


# =============================================================================
# TileAlgorithm Profiler
# =============================================================================


class TileAlgorithmProfiler:
    """Profiler for TileAlgorithm models.

    Tracks timing, memory, and activity statistics.

    Parameters
    ----------
    model : TileAlgorithm
        Model to profile
    """

    def __init__(self, model: TileAlgorithm) -> None:
        self.model = model
        self._profiling = False
        self._current_result: ProfileResult | None = None
        self._tile_timers: dict[int, dict[str, float]] = defaultdict(
            lambda: {
                "predict": 0.0,
                "update": 0.0,
            }
        )
        self._section_timers: dict[str, float] = defaultdict(float)
        self._history: list[ProfileResult] = []
        self._start_time: float = 0.0

    @contextmanager
    def profile(self, batch_size: int = 0):
        """Context manager for profiling.

        Parameters
        ----------
        batch_size : int
            Batch size for this profile session

        Yields
        ------
        TileAlgorithmProfiler
            Self for use in context
        """
        self._start_profiling(batch_size)
        try:
            yield self
        finally:
            result = self._stop_profiling()
            if result is not None:
                self._history.append(result)

    def _start_profiling(self, batch_size: int = 0) -> None:
        """Start profiling.

        Parameters
        ----------
        batch_size : int
            Batch size
        """
        self._profiling = True
        self._current_result = ProfileResult(batch_size=batch_size)
        self._tile_timers.clear()
        self._section_timers.clear()
        self._start_time = time.perf_counter()

    def _stop_profiling(self) -> ProfileResult | None:
        """Stop profiling and return results.

        Returns
        -------
        ProfileResult, optional
            Profiling results
        """
        if not self._profiling or self._current_result is None:
            return None

        self._profiling = False
        result = self._current_result

        # Aggregate tile stats
        result.tile_stats = self._aggregate_tile_stats()
        result.total_time = time.perf_counter() - self._start_time
        result.predict_time = sum(t["predict"] for t in self._tile_timers.values())
        result.update_time = sum(t["update"] for t in self._tile_timers.values())
        result.n_tiles = len(self.model.graph.tiles)
        result.n_edges = len(self.model.graph.edges)

        # Memory stats
        result.param_memory_mb = self._measure_memory()
        result.activation_memory_mb = self._measure_activation_memory()

        self._current_result = None
        return result

    def _aggregate_tile_stats(self) -> dict[int, TileStats]:
        """Aggregate statistics per tile.

        Returns
        -------
        dict
            Tile statistics
        """
        stats: dict[int, TileStats] = {}

        for tile in self.model.graph.all_tiles:
            # Find the index of this tile in the sorted tile list
            sorted_ids = sorted(self.model.graph.tiles.keys())
            tile_idx = sorted_ids.index(tile.id)
            importance = torch.sigmoid(self.model.tile_importance[tile_idx]).item()

            activity_mean = 0.0
            activity_std = 0.0
            activity_max = 0.0
            error_norm = 0.0

            if tile.activity is not None:
                activity_mean = tile.activity.mean().item()
                activity_std = tile.activity.std().item()
                activity_max = tile.activity.abs().max().item()

            if tile.error is not None:
                error_norm = tile.error.norm(p=2).item()

            stats[tile.id] = TileStats(
                tile_id=tile.id,
                layer_id=tile.layer_id,
                is_input=tile.is_input,
                is_output=tile.is_output,
                predict_time=self._tile_timers[tile.id]["predict"],
                update_time=self._tile_timers[tile.id]["update"],
                total_time=self._tile_timers[tile.id]["predict"]
                + self._tile_timers[tile.id]["update"],
                activity_mean=activity_mean,
                activity_std=activity_std,
                activity_max=activity_max,
                error_norm=error_norm,
                importance=importance,
                call_count=1 if activity_mean != 0 else 0,
            )

        return stats

    def _measure_memory(self) -> float:
        """Measure parameter memory in MB.

        ``model.parameters()`` already includes the per-edge parameter dicts
        (``_tile_weights``/``_tile_biases``), so the full parameter footprint.

        Returns
        -------
        float
            Memory in MB
        """
        total_mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_mem / (1024 * 1024)

    def _measure_activation_memory(self) -> float:
        """Measure activation memory in MB.

        Returns
        -------
        float
            Memory in MB
        """
        activation_mem = 0

        for tile in self.model.graph.all_tiles:
            if tile.activity is not None:
                activation_mem += tile.activity.numel() * tile.activity.element_size()
            if tile.prediction is not None:
                activation_mem += (
                    tile.prediction.numel() * tile.prediction.element_size()
                )
            if tile.error is not None:
                activation_mem += tile.error.numel() * tile.error.element_size()

        return activation_mem / (1024 * 1024)

    @contextmanager
    def time_predict(self, tile_id: int):
        """Context manager for timing prediction.

        Parameters
        ----------
        tile_id : int
            Tile ID

        Yields
        ------
        None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._tile_timers[tile_id]["predict"] += elapsed

    @contextmanager
    def time_update(self, tile_id: int):
        """Context manager for timing update.

        Parameters
        ----------
        tile_id : int
            Tile ID

        Yields
        ------
        None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._tile_timers[tile_id]["update"] += elapsed

    @contextmanager
    def time_section(self, section: str):
        """Context manager for timing a section.

        Parameters
        ----------
        section : str
            Section name

        Yields
        ------
        None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._section_timers[section] += elapsed

    def print_report(self, last_n: int = 1) -> None:
        """Print profiling report.

        Parameters
        ----------
        last_n : int
            Number of recent profiles to report
        """
        if not self._history:
            logger.warning("No profiling data available.")
            return

        for result in self._history[-last_n:]:
            self._print_single_report(result)

    def _print_single_report(self, result: ProfileResult) -> None:
        """Print a single profiling report.

        Parameters
        ----------
        result : ProfileResult
            Profile result
        """
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("TileAlgorithm Profiling Report")
        lines.append("=" * 70)
        lines.append("")

        summary = result.summary()
        total_time = max(float(summary["total_time_ms"]), 0.001)

        lines.append("Summary:")
        lines.append(f"  Total time: {total_time:.2f} ms")
        lines.append(
            f"  Predict time: {summary['predict_time_ms']:.2f} ms ({summary['predict_pct']:.1f}%)"
        )
        lines.append(
            f"  Update time: {summary['update_time_ms']:.2f} ms ({summary['update_pct']:.1f}%)"
        )
        lines.append(f"  Batch size: {summary['batch_size']}")
        lines.append(f"  Tiles: {summary['n_tiles']}, Edges: {summary['n_edges']}")
        lines.append("")

        lines.append("Memory:")
        lines.append(f"  Parameters: {summary['param_memory_mb']:.2f} MB")
        lines.append(f"  Activations: {summary['activation_memory_mb']:.2f} MB")
        lines.append(f"  Total: {summary['total_memory_mb']:.2f} MB")
        lines.append("")

        lines.append("Tile Breakdown (top 5 by time):")
        sorted_tiles = sorted(
            result.tile_stats.values(), key=lambda s: s.total_time, reverse=True
        )[:5]

        lines.append(
            f"  {'ID':>4} {'Layer':>6} {'Time(ms)':>10} {'Error':>10} {'Importance':>10}"
        )
        lines.append(f"  {'-' * 4} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}")

        for tile in sorted_tiles:
            lines.append(
                f"  {tile.tile_id:>4} {tile.layer_id:>6} {tile.total_time * 1000:>10.2f} {tile.error_norm:>10.2f} {tile.importance:>10.3f}"
            )

        lines.append("")

        lines.append("Activity Statistics:")
        activities = [
            s.activity_mean for s in result.tile_stats.values() if not s.is_input
        ]
        errors = [s.error_norm for s in result.tile_stats.values() if not s.is_input]

        if activities:
            lines.append(f"  Activity mean: {sum(activities) / len(activities):.4f}")
            lines.append(
                f"  Activity max: {max(s.activity_max for s in result.tile_stats.values()):.4f}"
            )
        if errors:
            lines.append(f"  Error mean: {sum(errors) / len(errors):.4f}")
            lines.append(f"  Error max: {max(errors):.4f}")

        lines.append("")
        lines.append("=" * 70)

        for line in lines:
            logger.info(line)

    def get_history(self) -> list[ProfileResult]:
        """Get profiling history.

        Returns
        -------
        list
            List of profile results
        """
        return self._history

    def clear_history(self) -> None:
        """Clear profiling history."""
        self._history.clear()

    @property
    def is_profiling(self) -> bool:
        """Check if currently profiling."""
        return self._profiling


# =============================================================================
# Learning Monitor
# =============================================================================


class LearningMonitor:
    """Monitors learning progress over time.

    Parameters
    ----------
    model : TileAlgorithm
        Model to monitor
    window_size : int
        Window size for moving averages
    """

    def __init__(self, model: TileAlgorithm, window_size: int = 100) -> None:
        self.model = model
        self.window_size = window_size

        self._loss_history: list[float] = []
        self._accuracy_history: list[float] = []
        self._error_history: dict[int, list[float]] = defaultdict(list)
        self._importance_history: list[float] = []

    def record(self, stats: dict[str, float]) -> None:
        """Record training statistics.

        Parameters
        ----------
        stats : dict
            Training statistics
        """
        self._loss_history.append(stats.get("loss", 0.0))
        self._accuracy_history.append(stats.get("accuracy", 0.0))
        self._importance_history.append(
            torch.sigmoid(self.model.tile_importance).mean().item()
        )

        # Per-tile errors
        for tile in self.model.graph.all_tiles:
            if tile.error is not None:
                error_norm = tile.error.norm(p=2).item()
                self._error_history[tile.id].append(error_norm)

        # Trim history
        if len(self._loss_history) > self.window_size:
            self._loss_history.pop(0)
            self._accuracy_history.pop(0)
            self._importance_history.pop(0)

        for tile_id in self._error_history:
            if len(self._error_history[tile_id]) > self.window_size:
                self._error_history[tile_id].pop(0)

    def get_summary(self) -> dict[str, object]:
        """Get learning summary.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self._loss_history:
            return {}

        return {
            "loss_mean": sum(self._loss_history[-10:])
            / min(10, len(self._loss_history)),
            "loss_trend": self._compute_trend(self._loss_history),
            "accuracy_mean": sum(self._accuracy_history[-10:])
            / min(10, len(self._accuracy_history)),
            "accuracy_trend": self._compute_trend(self._accuracy_history),
            "importance_mean": sum(self._importance_history[-10:])
            / min(10, len(self._importance_history)),
            "hot_tiles": self._get_hot_tiles(),
        }

    def _compute_trend(self, values: list[float]) -> str:
        """Compute trend direction.

        Parameters
        ----------
        values : list
            Values to analyze

        Returns
        -------
        str
            Trend direction
        """
        if len(values) < 5:
            return "stable"

        recent = sum(values[-5:]) / 5
        older = sum(values[-10:-5]) / 5

        if recent < older * 0.95:
            return "decreasing"
        elif recent > older * 1.05:
            return "increasing"
        return "stable"

    def _get_hot_tiles(self) -> list[int]:
        """Get tiles with highest recent error.

        Returns
        -------
        list
            Hot tile IDs
        """
        if not self._error_history:
            return []

        avg_errors = {
            tile_id: sum(errors[-10:]) / min(10, len(errors))
            for tile_id, errors in self._error_history.items()
        }

        sorted_tiles = sorted(avg_errors.items(), key=lambda x: x[1], reverse=True)
        return [tile_id for tile_id, _ in sorted_tiles[:5]]

    def print_status(self) -> None:
        """Print current learning status."""
        summary = self.get_summary()

        if not summary:
            logger.warning("No data recorded yet.")
            return

        logger.info(
            "\nLearning Status:\n  Loss: %s (%s)\n  Accuracy: %s (%s)\n  Mean Importance: %s%s\n",
            f"{summary['loss_mean']:.4f}",
            summary["loss_trend"],
            f"{summary['accuracy_mean']:.4f}",
            summary["accuracy_trend"],
            f"{summary['importance_mean']:.4f}",
            f"\n  Hot Tiles: {summary['hot_tiles']}" if summary["hot_tiles"] else "",
        )


# =============================================================================
# Memory Profiler
# =============================================================================


class MemoryProfiler:
    """Memory profiling for TileAlgorithm models.

    Tracks memory usage over time and identifies memory bottlenecks.

    Parameters
    ----------
    model : TileAlgorithm
        Model to profile
    """

    def __init__(self, model: TileAlgorithm) -> None:
        self.model = model
        self._history: list[dict[str, float]] = []

    def snapshot(self) -> dict[str, float]:
        """Take a memory snapshot.

        Returns
        -------
        dict
            Memory snapshot
        """
        snapshot: dict[str, float] = {}

        # GPU memory (if available)
        if torch.cuda.is_available():
            snapshot["gpu_allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
            snapshot["gpu_reserved"] = torch.cuda.memory_reserved() / (1024 * 1024)
            snapshot["gpu_max_allocated"] = torch.cuda.max_memory_allocated() / (
                1024 * 1024
            )

        # Parameter memory
        param_mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
        snapshot["param_memory_mb"] = param_mem / (1024 * 1024)

        # Edge memory (per-edge parameter dicts on the substrate)
        edge_mem = 0
        for container in ("_tile_weights", "_tile_biases"):
            params = getattr(self.model, container, None)
            if params is not None:
                for p in params.values():
                    edge_mem += p.numel() * p.element_size()
        snapshot["edge_memory_mb"] = edge_mem / (1024 * 1024)

        # Activation memory
        activation_mem = 0
        for tile in self.model.graph.all_tiles:
            for attr in ["activity", "prediction", "error"]:
                tensor = getattr(tile, attr, None)
                if tensor is not None:
                    activation_mem += tensor.numel() * tensor.element_size()
        snapshot["activation_memory_mb"] = activation_mem / (1024 * 1024)

        # Total
        snapshot["total_memory_mb"] = (
            snapshot["param_memory_mb"]
            + snapshot["edge_memory_mb"]
            + snapshot["activation_memory_mb"]
        )

        self._history.append(snapshot)
        return snapshot

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB.

        Returns
        -------
        float
            Peak memory in MB
        """
        if not self._history:
            return 0.0

        return max(s["total_memory_mb"] for s in self._history)

    def get_average_memory(self) -> float:
        """Get average memory usage in MB.

        Returns
        -------
        float
            Average memory in MB
        """
        if not self._history:
            return 0.0

        return sum(s["total_memory_mb"] for s in self._history) / len(self._history)

    def print_report(self) -> None:
        """Print memory profiling report."""
        if not self._history:
            logger.warning("No memory profiling data available.")
            return

        latest = self._history[-1]
        peak = self.get_peak_memory()
        average = self.get_average_memory()

        lines: list[str] = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("Memory Profiling Report")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Current Total: {latest['total_memory_mb']:.2f} MB")
        lines.append(f"  Peak Total: {peak:.2f} MB")
        lines.append(f"  Average Total: {average:.2f} MB")
        lines.append("")
        lines.append("  Breakdown:")
        lines.append(f"    Parameters: {latest['param_memory_mb']:.2f} MB")
        lines.append(f"    Edges: {latest['edge_memory_mb']:.2f} MB")
        lines.append(f"    Activations: {latest['activation_memory_mb']:.2f} MB")

        if "gpu_allocated" in latest:
            lines.append("")
            lines.append("  GPU Memory:")
            lines.append(f"    Allocated: {latest['gpu_allocated']:.2f} MB")
            lines.append(f"    Reserved: {latest['gpu_reserved']:.2f} MB")
            lines.append(f"    Peak Allocated: {latest['gpu_max_allocated']:.2f} MB")

        lines.append("")
        lines.append("=" * 70)

        for line in lines:
            logger.info(line)

    def clear_history(self) -> None:
        """Clear memory profiling history."""
        self._history.clear()

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Benchmark Runner
# =============================================================================


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for benchmarking.

    Attributes
    ----------
    batch_sizes : list of int
        Batch sizes to test
    n_warmup : int
        Number of warmup iterations
    n_iterations : int
        Number of benchmark iterations
    """

    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])
    n_warmup: int = 5
    n_iterations: int = 20


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes
    ----------
    batch_size : int
        Batch size
    mean_time_ms : float
        Mean time per iteration
    std_time_ms : float
        Standard deviation
    min_time_ms : float
        Minimum time
    max_time_ms : float
        Maximum time
    throughput_samples_per_sec : float
        Throughput in samples/second
    """

    batch_size: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_samples_per_sec: float


class BenchmarkRunner:
    """Performance benchmarking for TileAlgorithm models.

    Parameters
    ----------
    model : TileAlgorithm
        Model to benchmark
    config : BenchmarkConfig, optional
        Benchmark configuration
    """

    def __init__(
        self,
        model: TileAlgorithm,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or BenchmarkConfig()
        self._results: list[BenchmarkResult] = []

    def run(self, input_dim: int, output_dim: int) -> list[BenchmarkResult]:
        """Run benchmarks.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension

        Returns
        -------
        list
            Benchmark results
        """
        self._results.clear()

        for batch_size in self.config.batch_sizes:
            result = self._benchmark_batch(batch_size, input_dim, output_dim)
            self._results.append(result)

        return self._results

    def _benchmark_batch(
        self,
        batch_size: int,
        input_dim: int,
        output_dim: int,
    ) -> BenchmarkResult:
        """Benchmark a specific batch size.

        Parameters
        ----------
        batch_size : int
            Batch size
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension

        Returns
        -------
        BenchmarkResult
            Benchmark result
        """
        device = next(self.model.parameters()).device

        # Create dummy data
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, output_dim, (batch_size,), device=device)

        # Warmup
        for _ in range(self.config.n_warmup):
            self.model.train_step(x, y)

        # Benchmark
        times: list[float] = []
        for _ in range(self.config.n_iterations):
            start = time.perf_counter()
            self.model.train_step(x, y)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Convert to milliseconds
        times_ms = [t * 1000 for t in times]

        mean_time = sum(times_ms) / len(times_ms)
        std_time = (sum((t - mean_time) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
        throughput = batch_size / (mean_time / 1000)

        return BenchmarkResult(
            batch_size=batch_size,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            throughput_samples_per_sec=throughput,
        )

    def print_report(self) -> None:
        """Print benchmark report."""
        if not self._results:
            logger.warning("No benchmark results available.")
            return

        lines: list[str] = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("Performance Benchmark Report")
        lines.append("=" * 70)
        lines.append("")
        lines.append(
            f"  {'Batch Size':>10} {'Mean (ms)':>12} {'Std (ms)':>10} {'Throughput':>15}"
        )
        lines.append(f"  {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 15}")

        for result in self._results:
            lines.append(
                f"  {result.batch_size:>10} {result.mean_time_ms:>12.2f} {result.std_time_ms:>10.2f} {result.throughput_samples_per_sec:>15.1f}"
            )

        lines.append("")
        lines.append("=" * 70)

        for line in lines:
            logger.info(line)

    def get_results(self) -> list[BenchmarkResult]:
        """Get benchmark results.

        Returns
        -------
        list
            Benchmark results
        """
        return self._results


# =============================================================================
# Factory Functions
# =============================================================================


def create_profiler(
    model: TileAlgorithm,
    enable_memory_profiling: bool = True,
    enable_learning_monitor: bool = True,
) -> tuple[TileAlgorithmProfiler, MemoryProfiler | None, LearningMonitor | None]:
    """Create a complete profiling setup.

    Parameters
    ----------
    model : TileAlgorithm
        Model to profile
    enable_memory_profiling : bool
        Enable memory profiling
    enable_learning_monitor : bool
        Enable learning monitoring

    Returns
    -------
    tuple
        (TileAlgorithmProfiler, MemoryProfiler, LearningMonitor)
    """
    profiler = TileAlgorithmProfiler(model)

    memory_profiler = None
    if enable_memory_profiling:
        memory_profiler = MemoryProfiler(model)

    learning_monitor = None
    if enable_learning_monitor:
        learning_monitor = LearningMonitor(model)

    return profiler, memory_profiler, learning_monitor


def run_benchmark(
    model: TileAlgorithm,
    input_dim: int,
    output_dim: int,
    batch_sizes: list[int] | None = None,
) -> list[BenchmarkResult]:
    """Run performance benchmarks.

    Parameters
    ----------
    model : TileAlgorithm
        Model to benchmark
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    batch_sizes : list of int, optional
        Batch sizes to test

    Returns
    -------
    list
        Benchmark results
    """
    config = BenchmarkConfig(batch_sizes=batch_sizes or [1, 8, 32, 64, 128])
    runner = BenchmarkRunner(model, config)
    return runner.run(input_dim, output_dim)
