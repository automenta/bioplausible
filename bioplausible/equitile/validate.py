#!/usr/bin/env python3
"""
Automated Validation Pipeline for EquiTile
===========================================

Comprehensive validation suite:
- Unit tests
- Integration tests
- Performance regression tests
- Reproducibility verification

Usage
-----
# Run all validations
python -m bioplausible.models.equitile.validate

# Run specific test category
python -m bioplausible.models.equitile.validate --category performance
python -m bioplausible.models.equitile.validate --category reproducibility

# Run quick smoke test
python -m bioplausible.models.equitile.validate --quick
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from bioplausible.core.logging import get_logger
from bioplausible.core.utils.device import get_device
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer
from bioplausible.data.lm import create_shakespeare_dataset
from bioplausible.equitile.utils import ReproducibilityTracker, set_reproducible_mode
from bioplausible.zoo.models.tile_lm import TileLM

__all__ = [
    "ValidationPipeline",
    "ValidationResult",
    "logger",
    "main",
]
logger = get_logger()


@dataclass
class ValidationResult:
    """Result of a validation test."""

    name: str
    passed: bool
    message: str
    duration_sec: float
    metrics: dict[str, float] | None = None


class ValidationPipeline:
    """Automated validation pipeline for EquiTile."""

    def __init__(self, quick: bool = False) -> None:
        self.quick = quick
        self.results: list[ValidationResult] = []
        self.tracker = ReproducibilityTracker(seed=42)

        # Device
        self.device = get_device()
        logger.info("Validation device: %s", self.device)

    def run_all(self) -> bool:
        """Run all validation tests."""
        logger.info("=" * 70)
        logger.info("EQUITILE VALIDATION PIPELINE")
        logger.info("=" * 70)

        # Unit tests
        self._run_category("unit", self._run_unit_tests)

        # Integration tests
        self._run_category("integration", self._run_integration_tests)

        # Performance tests
        if not self.quick:
            self._run_category("performance", self._run_performance_tests)

        # Reproducibility tests
        self._run_category("reproducibility", self._run_reproducibility_tests)

        # Summary
        self._print_summary()

        # Return overall pass/fail
        return all(r.passed for r in self.results)

    def _run_category(self, category: str, test_func) -> None:
        """Run a category of tests."""
        logger.info("=" * 70)
        logger.info("Running %s tests...", category.upper())
        logger.info("=" * 70)

        start = time.time()
        test_func()
        elapsed = time.time() - start

        passed = sum(1 for r in self.results if r.passed and category in r.name.lower())
        total = sum(1 for r in self.results if category in r.name.lower())

        logger.info(
            "%s tests: %d/%d passed (%.1fs)", category.upper(), passed, total, elapsed
        )

    def _add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        logger.info("  [%s] %s: %s", status, result.name, result.message)
        if result.metrics:
            for key, value in result.metrics.items():
                logger.info("           %s: %s", key, value)

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------

    def _run_unit_tests(self) -> None:
        """Run unit tests."""
        # Test 1: Model creation
        start = time.time()
        try:
            model = TileLM.from_lm(
                vocab_size=100,
                embed_dim=64,
                num_layers=2,
            )
            params = model.get_parameter_count()

            self._add_result(
                ValidationResult(
                    name="unit_model_creation",
                    passed=params > 0,
                    message=f"Created model with {params:,} parameters",
                    duration_sec=time.time() - start,
                    metrics={"parameters": params},
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="unit_model_creation",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

        # Test 2: Forward pass
        start = time.time()
        try:
            model = model.to(self.device)
            model.eval()
            input_ids = torch.randint(0, 100, (2, 16)).to(self.device)

            with torch.no_grad():
                output = model(input_ids)

            expected_shape = (2, 16, 100)
            passed = output.shape == expected_shape

            self._add_result(
                ValidationResult(
                    name="unit_forward_pass",
                    passed=passed,
                    message=f"Output shape: {tuple(output.shape)} (expected: {expected_shape})",
                    duration_sec=time.time() - start,
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="unit_forward_pass",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

        # Test 3: Training step
        start = time.time()
        try:
            model.train()
            input_ids = torch.randint(0, 100, (2, 16)).to(self.device)
            target_ids = input_ids.clone()

            stats = model.train_step(input_ids, target_ids)
            passed = "loss" in stats and stats["loss"] > 0

            self._add_result(
                ValidationResult(
                    name="unit_training_step",
                    passed=passed,
                    message=f"Loss: {stats.get('loss', 'N/A'):.4f}",
                    duration_sec=time.time() - start,
                    metrics={"loss": stats.get("loss", 0)},
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="unit_training_step",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def _run_integration_tests(self) -> None:
        """Run integration tests."""
        # Test 1: Dataset loading
        start = time.time()
        try:
            train_loader, val_loader, _tokenizer = create_shakespeare_dataset(
                batch_size=4,
                seq_length=32,
                num_workers=0,
            )

            passed = len(train_loader) > 0 and len(val_loader) > 0

            self._add_result(
                ValidationResult(
                    name="integration_dataset",
                    passed=passed,
                    message=f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches",
                    duration_sec=time.time() - start,
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="integration_dataset",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

        # Test 2: Full training loop (mini)
        start = time.time()
        try:
            model = TileLM.from_lm(
                vocab_size=100,
                embed_dim=64,
                num_layers=2,
            ).to(self.device)
            optimizer = create_optimizer(
                model, OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-2)
            )

            initial_loss = None
            final_loss = None

            model.train()
            for step in range(10):
                input_ids = torch.randint(0, 100, (4, 32)).to(self.device)
                target_ids = input_ids.clone()

                optimizer.zero_grad()
                output = model(input_ids)
                loss = model.compute_loss(output, target_ids)
                loss.backward()
                optimizer.step()

                if step == 0:
                    initial_loss = loss.item()
                final_loss = loss.item()

            # Loss should decrease (or at least not explode)
            passed = final_loss < initial_loss * 2 and final_loss < 100

            self._add_result(
                ValidationResult(
                    name="integration_training",
                    passed=passed,
                    message=f"Loss: {initial_loss:.4f} → {final_loss:.4f}",
                    duration_sec=time.time() - start,
                    metrics={"initial_loss": initial_loss, "final_loss": final_loss},
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="integration_training",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

        # Test 3: Generation
        start = time.time()
        try:
            model.eval()
            input_ids = torch.randint(0, 100, (1, 10)).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(input_ids, max_length=20)

            passed = output_ids.shape == (1, 20)

            self._add_result(
                ValidationResult(
                    name="integration_generation",
                    passed=passed,
                    message=f"Generated {output_ids.shape[1]} tokens",
                    duration_sec=time.time() - start,
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="integration_generation",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

    # -------------------------------------------------------------------------
    # Performance Tests
    # -------------------------------------------------------------------------

    def _run_performance_tests(self) -> None:
        """Run performance tests."""
        # Test 1: Throughput benchmark
        start = time.time()
        try:
            model = TileLM.from_lm(
                vocab_size=1000,
                embed_dim=128,
                num_layers=4,
            ).to(self.device)
            model.train()

            optimizer = create_optimizer(
                model, OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-2)
            )

            # Warmup
            for _ in range(5):
                input_ids = torch.randint(0, 1000, (16, 64)).to(self.device)
                target_ids = input_ids.clone()
                optimizer.zero_grad()
                output = model(input_ids)
                loss = model.compute_loss(output, target_ids)
                loss.backward()
                optimizer.step()

            # Measure
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            measure_start = time.time()
            total_tokens = 0

            for _ in range(20):
                input_ids = torch.randint(0, 1000, (16, 64)).to(self.device)
                target_ids = input_ids.clone()
                optimizer.zero_grad()
                output = model(input_ids)
                loss = model.compute_loss(output, target_ids)
                loss.backward()
                optimizer.step()
                total_tokens += input_ids.numel()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - measure_start
            throughput = total_tokens / elapsed

            # Minimum expected throughput (conservative)
            min_throughput = 10000 if self.device.type == "cuda" else 1000
            passed = throughput > min_throughput

            self._add_result(
                ValidationResult(
                    name="performance_throughput",
                    passed=passed,
                    message=f"Throughput: {throughput:,.0f} tok/s (min: {min_throughput:,})",
                    duration_sec=time.time() - start,
                    metrics={"throughput": throughput},
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="performance_throughput",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

        # Test 2: Memory efficiency
        start = time.time()
        try:
            model = TileLM.from_lm(
                vocab_size=1000,
                embed_dim=128,
                num_layers=4,
            ).to(self.device)
            model.train()

            input_ids = torch.randint(0, 1000, (32, 128)).to(self.device)
            target_ids = input_ids.clone()

            optimizer = create_optimizer(
                model, OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-2)
            )
            optimizer.zero_grad()

            output = model(input_ids)
            loss = model.compute_loss(output, target_ids)
            loss.backward()

            if self.device.type == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                max_memory = 4000  # 4GB max for this config
                passed = memory_mb < max_memory
            else:
                memory_mb = 0
                passed = True

            self._add_result(
                ValidationResult(
                    name="performance_memory",
                    passed=passed,
                    message=f"Memory: {memory_mb:.0f} MB (max: {max_memory} MB)",
                    duration_sec=time.time() - start,
                    metrics={"memory_mb": memory_mb},
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="performance_memory",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

    # -------------------------------------------------------------------------
    # Reproducibility Tests
    # -------------------------------------------------------------------------

    def _run_reproducibility_tests(self) -> None:
        """Run reproducibility tests."""
        # Test 1: Seed reproducibility
        start = time.time()
        try:
            set_reproducible_mode(seed=42)

            model1 = TileLM.from_lm(
                vocab_size=100,
                embed_dim=64,
                num_layers=2,
            ).to(self.device)

            input_ids = torch.randint(0, 100, (2, 16)).to(self.device)

            model1.eval()
            with torch.no_grad():
                output1 = model1(input_ids)

            # Reset and run again
            set_reproducible_mode(seed=42)

            model2 = TileLM.from_lm(
                vocab_size=100,
                embed_dim=64,
                num_layers=2,
            ).to(self.device)
            model2.eval()
            with torch.no_grad():
                output2 = model2(input_ids)

            # Check outputs match
            passed = torch.allclose(output1, output2, atol=1e-5)

            self._add_result(
                ValidationResult(
                    name="reproducibility_seeds",
                    passed=passed,
                    message="Outputs match across runs with same seed",
                    duration_sec=time.time() - start,
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="reproducibility_seeds",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

        # Test 2: Config logging
        start = time.time()
        try:
            model_cfg = TileLM.from_lm(vocab_size=100, embed_dim=64).get_config()
            self.tracker.log_config(model_cfg, "model")

            passed = len(self.tracker.config_log) > 0

            self._add_result(
                ValidationResult(
                    name="reproducibility_config_logging",
                    passed=passed,
                    message=f"Configs logged: {len(self.tracker.config_log)}",
                    duration_sec=time.time() - start,
                )
            )
        except Exception as e:  # broad: per-check best-effort
            self._add_result(
                ValidationResult(
                    name="reproducibility_config_logging",
                    passed=False,
                    message=str(e),
                    duration_sec=time.time() - start,
                )
            )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Print validation summary."""
        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        logger.info("Total tests: %d", total)
        logger.info("Passed: %d", passed)
        logger.info("Failed: %d", failed)

        if failed > 0:
            logger.info("FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    logger.info("  ✗ %s: %s", r.name, r.message)

        overall = "✓ ALL TESTS PASSED" if failed == 0 else "✗ SOME TESTS FAILED"
        logger.info("OVERALL: %s", overall)
        logger.info("=" * 70)

        # Save results
        results_data = {
            "timestamp": time.time(),
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_sec": r.duration_sec,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }

        results_path = Path("validation_results.json")
        with Path(results_path).open("w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        logger.info("Results saved to %s", results_path)


def main():
    parser = argparse.ArgumentParser(description="EquiTile Validation Pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip performance tests)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["unit", "integration", "performance", "reproducibility"],
        help="Run specific test category only",
    )

    args = parser.parse_args()

    pipeline = ValidationPipeline(quick=args.quick)

    if args.category:
        # Run specific category
        category_map = {
            "unit": pipeline._run_unit_tests,
            "integration": pipeline._run_integration_tests,
            "performance": pipeline._run_performance_tests,
            "reproducibility": pipeline._run_reproducibility_tests,
        }
        category_map[args.category]()
        pipeline._print_summary()
    else:
        # Run all
        pipeline.run_all()

    sys.exit(0 if all(r.passed for r in pipeline.results) else 1)


if __name__ == "__main__":
    main()
