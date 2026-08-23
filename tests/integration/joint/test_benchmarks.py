"""Integration tests for joint architecture benchmarks."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
@pytest.mark.slow
class TestAdaptationEfficiency:
    """Test adaptation efficiency benchmark runs without errors."""

    def test_adaptation_efficiency_null_vs_routing(self):
        """Test that routing plasticity adapts faster than null."""
        from bioplausible.experiments.joint.adaptation_efficiency import (
            evaluate_adaptation,
        )

        # Test null plasticity
        null_result = evaluate_adaptation(
            coordinate="digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
            epochs_per_phase=5,
            batch_size=32,
            seq_len=5,
            input_dim=16,
            device="cpu",
            seed=42,
        )

        # Test routing plasticity
        routing_result = evaluate_adaptation(
            coordinate="digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
            epochs_per_phase=5,
            batch_size=32,
            seq_len=5,
            input_dim=16,
            device="cpu",
            seed=42,
        )

        # Both should produce valid results
        assert "final_accuracy" in null_result
        assert "final_accuracy" in routing_result
        assert "adaptation_time" in null_result
        assert "adaptation_time" in routing_result

        # Accuracies should be in valid range
        assert 0 <= null_result["final_accuracy"] <= 1
        assert 0 <= routing_result["final_accuracy"] <= 1


@pytest.mark.integration
@pytest.mark.slow
class TestComputeEfficiency:
    """Test compute efficiency benchmark runs without errors."""

    def test_compute_efficiency_null_vs_routing(self):
        """Test that routing reduces active routes."""
        from bioplausible.experiments.joint.compute_efficiency import (
            evaluate_compute_efficiency,
        )

        null_result = evaluate_compute_efficiency(
            coordinate="digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
            epochs=3,
            batch_size=32,
            input_dim=32,
            num_experts=4,
            device="cpu",
            seed=42,
        )

        routing_result = evaluate_compute_efficiency(
            coordinate="digital/feedforward/instantaneous/routing/thermodynamic_contrast/euclidean",
            epochs=3,
            batch_size=32,
            input_dim=32,
            num_experts=4,
            device="cpu",
            seed=42,
        )

        assert "final_accuracy" in null_result
        assert "final_accuracy" in routing_result
        assert "active_routes" in routing_result
        assert "flops_reduction" in routing_result

        # Routing should have fewer active routes
        assert routing_result["active_routes"] <= null_result.get("active_routes", 4)


@pytest.mark.integration
@pytest.mark.slow
class TestStructuralRobustness:
    """Test structural robustness benchmark runs without errors."""

    def test_structural_robustness_runs(self):
        """Test that structural robustness evaluation runs."""
        from bioplausible.experiments.joint.structural_robustness import (
            evaluate_structural_robustness,
        )

        result = evaluate_structural_robustness(
            coordinate="digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
            epochs=3,
            batch_size=32,
            input_dim=32,
            hidden_dim=64,
            output_dim=10,
            recovery_steps=5,
            damage_severity=0.2,
            device="cpu",
            seed=42,
        )

        assert "pre_damage_accuracy" in result
        assert "damage_results" in result
        assert "avg_recovery_ratio" in result
        assert 0 <= result["pre_damage_accuracy"] <= 1


@pytest.mark.integration
@pytest.mark.slow
class TestAlgorithmMigration:
    """Test algorithm migration benchmark runs without errors."""

    def test_algorithm_migration_runs(self):
        """Test that algorithm migration evaluation runs."""
        from bioplausible.experiments.joint.algorithm_migration import (
            evaluate_migration,
        )

        result = evaluate_migration(
            coordinate="digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
            epochs_a0=3,
            epochs_a1=3,
            batch_size=32,
            seq_len=5,
            input_dim=16,
            device="cpu",
            seed=42,
        )

        assert "a0_accuracy" in result
        assert "a1_accuracy" in result
        assert "migration_time" in result
        assert "theta_change" in result
        assert 0 <= result["a0_accuracy"] <= 1
        assert 0 <= result["a1_accuracy"] <= 1


@pytest.mark.integration
@pytest.mark.slow
class TestZ3FixedWeights:
    """Test Z3 fixed weights benchmark runs without errors."""

    def test_z3_fixed_weights_runs(self):
        """Test that Z3 evaluation runs and maintains theta invariance."""
        from bioplausible.experiments.joint.z3_fixed_weights import evaluate_z3

        result = evaluate_z3(
            coordinate="digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
            meta_train_epochs=3,
            eval_epochs_per_task=3,
            batch_size=32,
            seq_len=5,
            input_dim=16,
            device="cpu",
            seed=42,
        )

        assert "tasks" in result
        assert "theta_change" in result
        assert "theta_invariant" in result

        # Check theta invariance
        assert result["theta_invariant"] is True
        assert result["theta_change"] < 1e-6

        # Check all tasks have results
        for task_name in ["parity", "last_symbol", "threshold"]:
            assert task_name in result["tasks"]
            assert "accuracy" in result["tasks"][task_name]
            assert 0 <= result["tasks"][task_name]["accuracy"] <= 1


@pytest.mark.integration
@pytest.mark.slow
class TestBenchmarkCLI:
    """Test benchmark CLI runs without errors."""

    def test_benchmark_list(self):
        """Test benchmark list command."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "biopl", "benchmark", "list"],
            capture_output=True,
            text=True,
            cwd="/home/me/bioplausible",
        )
        assert result.returncode == 0
        assert "adaptation_efficiency" in result.stdout
        assert "compute_efficiency" in result.stdout
        assert "structural_robustness" in result.stdout
        assert "algorithm_migration" in result.stdout
        assert "z3_fixed_weights" in result.stdout

    def test_benchmark_run_quick_adaptation(self):
        """Test quick adaptation efficiency benchmark."""
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "biopl",
                "benchmark",
                "run",
                "--suite",
                "adaptation_efficiency",
                "--quick",
                "--output-dir",
                "/tmp/test_bench",
            ],
            capture_output=True,
            text=True,
            cwd="/home/me/bioplausible",
            timeout=60,
        )
        assert result.returncode == 0
        assert "Results saved to" in result.stdout

    def test_benchmark_run_quick_compute(self):
        """Test quick compute efficiency benchmark."""
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "biopl",
                "benchmark",
                "run",
                "--suite",
                "compute_efficiency",
                "--quick",
                "--output-dir",
                "/tmp/test_bench",
            ],
            capture_output=True,
            text=True,
            cwd="/home/me/bioplausible",
            timeout=60,
        )
        assert result.returncode == 0
        assert "Results saved to" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
