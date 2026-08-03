import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .notebook import VerificationNotebook
from .tracks import track_registry

__all__ = [
    "Verifier",
    "logger",
]
logger = logging.getLogger(__name__)


class Verifier:
    """Complete verification suite for all research tracks."""

    def __init__(
        self,
        quick_mode: bool = False,
        intermediate_mode: bool = False,
        seed: int = 42,
        n_seeds_override: int | None = None,
        n_samples_override: int | None = None,
        n_epochs_override: int | None = None,
        export_data: bool = False,
        output_dir: str | None = None,
    ):
        self.quick_mode = quick_mode
        self.intermediate_mode = intermediate_mode
        self.seed = seed
        self.export_data = export_data
        self.notebook = VerificationNotebook()

        # Set output directory (default to ./results if not specified)
        if output_dir is None:
            self.output_dir = Path("results")
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Validation Mode Configuration
        # Quick:        ~2 min - mechanics only (smoke test)
        # Intermediate: ~1 hour - directional validation
        # Full:         ~4+ hr  - statistically significant claims
        if quick_mode:
            self.epochs = 5
            self.n_samples = 200
            self.n_seeds = 1
            self.evidence_level = "smoke"
        elif intermediate_mode:
            self.epochs = 50
            self.n_samples = 5000
            self.n_seeds = 3
            self.evidence_level = "intermediate"
        else:
            self.epochs = 100
            self.n_samples = 10000
            self.n_seeds = 5
            self.evidence_level = "full"

        # Allow override of seeds, samples, and epochs
        if n_seeds_override is not None:
            self.n_seeds = n_seeds_override
        if n_samples_override is not None:
            self.n_samples = n_samples_override
        if n_epochs_override is not None:
            self.epochs = n_epochs_override

        self.data_records = []  # For CSV export
        self.current_seed = seed  # Track current seed for logging

        # Track definitions loaded from central registry
        self.tracks = {}
        self._load_tracks()

    def _load_tracks(self):
        """Load tracks from the registry into local format."""
        # Convert raw functions to (name, function) tuples expected by Verifier
        for tid, func in track_registry.ALL_TRACKS.items():
            # Attempt to extract nice name from docstring or function name
            name = func.__doc__.split("\n")[0] if func.__doc__ else func.__name__
            # Clean up name (remove "Track X: " prefix if present)
            if ":" in name and "Track" in name.split(":")[0]:
                name = name.split(":", 1)[1].strip()

            self.tracks[tid] = (name, func)

    def print_header(self):
        evidence_labels = {
            "smoke": "[TEST]  Smoke Test (mechanics only)",
            "intermediate": "[DATA]  Intermediate (directional)",
            "full": "[OK]  Full Validation (statistically significant)",
        }
        mode_name = (
            "Quick"
            if self.quick_mode
            else ("Intermediate" if self.intermediate_mode else "Full")
        )
        mode_icon = (
            "[FAST] "
            if self.quick_mode
            else ("[DATA] " if self.intermediate_mode else "[LAB] ")
        )

        logger.info("=" * 70)
        logger.info("       TOREQPROP COMPREHENSIVE VERIFICATION SUITE")
        logger.info("       Undeniable Evidence for All Research Claims")
        logger.info("=" * 70)
        logger.info("\n[CONFIG]  Configuration:")
        logger.info("   Seed: %s", self.seed)
        logger.info("   Mode: %s %s", mode_icon, mode_name)
        logger.info("   Evidence: %s", evidence_labels[self.evidence_level])
        logger.info("   Epochs: %s", self.epochs)
        logger.info("   Samples: %s", self.n_samples)
        logger.info("   Seeds: %s", self.n_seeds)
        logger.info("   Tracks: %s", len(self.tracks))
        if self.export_data:
            logger.info("   Export: Enabled (results/data.csv)")
        logger.info("=" * 70)

    def record_metric(
        self, track_id: int, seed: int, step: int, metric_name: str, value: float
    ):
        """Record a data point for export."""
        if self.export_data:
            self.data_records.append({
                "track_id": track_id,
                "seed": seed,
                "step": step,
                "metric": metric_name,
                "value": value,
                "timestamp": datetime.now().isoformat(),
            })

    def evaluate_robustness(self, track_fn, n_seeds: int = 3) -> dict:
        """Run a track logic multiple times with different seeds."""
        scores = []
        metrics_list = []

        # Determine number of seeds to run
        # override rules:
        # 1. if quick_mode -> 1
        # 2. if --seeds X provided -> X
        # 3. if default (3) -> use track-specific n_seeds (arg)

        run_count = self.n_seeds
        if self.n_seeds == 3 and not self.quick_mode:
            run_count = n_seeds

        logger.info("      Running robustness check (%s seeds)...", run_count)

        for i in range(run_count):
            seed = self.seed + i * 100
            self.current_seed = seed  # Update state for loggers

            # Temporarily set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                score, metrics = track_fn()
                scores.append(score)
                metrics_list.append(metrics)

                # Record aggregations for export
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.record_metric(
                            0, seed, 0, k, v
                        )  # Track ID 0 is generic/unknown here

            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("        Seed %s: Failed (%s)", seed, e)
                import traceback

                traceback.print_exc()
                scores.append(0)
                metrics_list.append({})

        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0

        # Calculate 95% Confidence Interval
        n = len(scores)
        se = std_score / np.sqrt(n) if n > 1 else 0.0
        ci_95 = 1.96 * se

        # Aggregate metrics with confidence intervals
        agg_metrics = {}
        if metrics_list:
            keys = metrics_list[0].keys()
            for k in keys:
                vals = [
                    m[k]
                    for m in metrics_list
                    if k in m and isinstance(m[k], (int, float))
                ]
                if vals:
                    m_mean = np.mean(vals)
                    m_std = np.std(vals) if len(vals) > 1 else 0.0
                    m_se = m_std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    m_ci = 1.96 * m_se
                    agg_metrics[f"{k}_mean"] = m_mean
                    agg_metrics[f"{k}_std"] = m_std
                    agg_metrics[f"{k}_ci95"] = m_ci  # New: CI for each metric

        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "ci_95": ci_95,  # New: 95% CI half-width
            "metrics": agg_metrics,
            "all_scores": scores,
        }

    def run_tracks(
        self, track_ids: list[int] | None = None, parallel: bool = False
    ) -> dict:
        """Run specified tracks (or all if None)."""
        self.print_header()
        self.notebook.add_header(self.seed)

        if track_ids is None:
            track_ids = list(self.tracks.keys())

        # Auto-run Track 0 (Framework Validation) in intermediate/full modes
        if (self.intermediate_mode or (not self.quick_mode)) and 0 not in track_ids:
            logger.info("Running Track 0 (Framework Validation) automatically...")
            track_ids = [0] + track_ids

        results = {}
        start_time = time.time()

        # Helper to run a single track
        def _execute_track(tid):
            if tid not in self.tracks:
                return tid, None, f"Unknown track: {tid}"
            name, method = self.tracks[tid]
            try:
                # Pass self (Verifier) to the track method
                result = method(self)
                return tid, result, None
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                import traceback

                return tid, None, f"Failed: {e}\n{traceback.format_exc()}"

        if parallel and len(track_ids) > 1:
            import concurrent.futures

            logger.info("Running %s tracks in parallel...", len(track_ids))

            # ThreadPoolExecutor: tracks are I/O or CPU bound
            # (PyTorch/CUDA releases GIL).
            # ProcessPoolExecutor would require pickling Modules.
            max_workers = min(
                len(track_ids), 4
            )  # Cap at 4 to avoid resource contention
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_track = {
                    executor.submit(_execute_track, tid): tid for tid in track_ids
                }

                completed = 0
                for future in concurrent.futures.as_completed(future_to_track):
                    tid, result, error = future.result()

                    if error:
                        logger.error("Track %s error: %s", tid, error)
                    elif result:
                        results[tid] = result
                        self.notebook.add_track_result(result)

                        icon = {
                            "pass": "[OK] ",
                            "fail": "[FAIL] ",
                            "partial": "[WARN] ",
                            "stub": "[TODO] ",
                        }.get(result.status, "?")
                        name, _ = self.tracks[tid]
                        logger.info(
                            "%s Track %s: %s - %s (%.0f/100)",
                            icon,
                            tid,
                            name,
                            result.status.upper(),
                            result.score,
                        )

                    completed += 1
                    elapsed = time.time() - start_time
                    logger.info(
                        "   Progress: %s/%s | Elapsed: %.0fs",
                        completed,
                        len(track_ids),
                        elapsed,
                    )

        else:
            # Sequential Execution
            for i, track_id in enumerate(track_ids):
                tid, result, error = _execute_track(track_id)

                if error:
                    logger.error("Track %s failed: %s", track_id, error)
                elif result:
                    results[track_id] = result
                    self.notebook.add_track_result(result)
                    icon = {
                        "pass": "[OK] ",
                        "fail": "[FAIL] ",
                        "partial": "[WARN] ",
                        "stub": "[TODO] ",
                    }.get(result.status, "?")
                    name, _ = self.tracks[track_id]
                    logger.info(
                        "%s Track %s: %s - %s (%.0f/100)",
                        icon,
                        track_id,
                        name,
                        result.status.upper(),
                        result.score,
                    )

                # Progress
                elapsed = time.time() - start_time
                completed = i + 1
                remaining = len(track_ids) - completed
                if remaining > 0:
                    eta = (elapsed / completed) * remaining
                    logger.info(
                        "   Progress: %s/%s | Elapsed: %.0fs | ETA: %.0fs",
                        completed,
                        len(track_ids),
                        elapsed,
                        eta,
                    )

        # Save
        total_time = time.time() - start_time

        output_path = self.output_dir / "verification_notebook.md"
        self.notebook.save(output_path)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("[SUCCESS]  VERIFICATION COMPLETE")
        logger.info("=" * 70)
        logger.info("⏱️  Total time: %.1fs", total_time)
        logger.info("[LOG]  Output: %s", output_path)

        passed = sum(1 for r in results.values() if r.status == "pass")
        total = len(results)
        logger.info("[DATA]  Results: %s/%s tracks passed", passed, total)

        if self.export_data and self.data_records:
            import csv

            csv_path = self.output_dir / "data.csv"
            keys = self.data_records[0].keys()
            with Path(csv_path).open("w", newline="") as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.data_records)
            logger.info("[SAVE]  Data exported to: %s", csv_path)

        return results

    def list_tracks(self):
        """Print all available tracks."""
        logger.info("\nAvailable Verification Tracks:")
        logger.info("-" * 60)
        for tid, (name, _) in self.tracks.items():
            logger.info("  %2d. %s", tid, name)
        logger.info("-" * 60)
