"""Coverage tests for ExecutionStrategy -- plan_next, plan_batch, tier logic."""

import unittest
from unittest.mock import MagicMock

from bioplausible.execution.strategy import (
    ExecutionStrategy,
    _model_specs,
    _ModelSpec,
)
from bioplausible.hyperopt import PatientLevel


class TestStrategyCore(unittest.TestCase):
    """Test core ExecutionStrategy methods: plan_next, plan_batch, filtering."""

    def setUp(self):
        self.mock_state = MagicMock()
        self.mock_state.get_progress.return_value = {}
        self.mock_state.get_recent_models.return_value = []
        self.mock_state.get_recent_tasks.return_value = []
        self.mock_state.get_fragile_models.return_value = {}
        self.strategy = ExecutionStrategy(self.mock_state)
        _model_specs.cache = None

    def tearDown(self):
        _model_specs.cache = None

    # ---- plan_next / plan_batch ----

    def test_plan_next_returns_none_when_no_candidates(self):
        _model_specs.cache = []
        result = self.strategy.plan_next()
        self.assertIsNone(result)

    def test_plan_next_returns_smoke_task(self):
        _model_specs.cache = [_ModelSpec("test_mlp")]
        result = self.strategy.plan_next()
        self.assertIsNotNone(result)
        self.assertEqual(result.tier, PatientLevel.SMOKE)
        self.assertEqual(result.model_name, "test_mlp")

    def test_plan_batch_returns_up_to_batch_size(self):
        _model_specs.cache = [
            _ModelSpec("model_a"),
            _ModelSpec("model_b"),
            _ModelSpec("model_c"),
        ]
        # Mock out saturation so tasks get generated for all 3 models
        self.strategy._analyze_saturation = MagicMock(return_value={})
        self.strategy._analyze_failures = MagicMock(return_value={})
        self.strategy._analyze_fragility = MagicMock(return_value={})

        batch = self.strategy.plan_batch(batch_size=2)
        self.assertLessEqual(len(batch), 2)
        # All tasks should have unique model+task+tier keys
        keys = [(t.model_name, t.task_name, t.tier.value) for t in batch]
        self.assertEqual(len(keys), len(set(keys)))

    def test_plan_batch_deduplicates_by_model_task_tier(self):
        _model_specs.cache = [_ModelSpec("single_model")]
        self.strategy._analyze_saturation = MagicMock(return_value={})
        self.strategy._analyze_failures = MagicMock(return_value={})
        self.strategy._analyze_fragility = MagicMock(return_value={})

        batch = self.strategy.plan_batch(batch_size=10)
        keys = [(t.model_name, t.task_name, t.tier.value) for t in batch]
        self.assertEqual(len(keys), len(set(keys)))

    # ---- tier_limit filtering ----

    def test_filter_by_tier_limit_excludes_deep(self):
        _model_specs.cache = [_ModelSpec("test_mlp")]
        strategy = ExecutionStrategy(self.mock_state, tier_limit="shallow")
        strategy._analyze_saturation = MagicMock(return_value={})
        strategy._analyze_failures = MagicMock(return_value={})

        candidates = strategy.generate_candidates()
        tiers = {c.tier for c in candidates}
        self.assertNotIn(PatientLevel.DEEP, tiers)
        self.assertNotIn(PatientLevel.STANDARD, tiers)

    def test_filter_by_tier_limit_default_allows_all(self):
        _model_specs.cache = [_ModelSpec("test_mlp")]
        strategy = ExecutionStrategy(self.mock_state)
        strategy._analyze_saturation = MagicMock(return_value={})
        strategy._analyze_failures = MagicMock(return_value={})

        candidates = strategy.generate_candidates()
        tiers = {c.tier for c in candidates}
        self.assertTrue(len(tiers) > 0)

    # ---- _check_criterion boundary tests ----

    def test_check_criterion_smoke_baseline(self):
        # Default SMOKE threshold = 0.12
        self.assertTrue(
            self.strategy._check_criterion(PatientLevel.SMOKE, "mnist", 0.13)
        )
        self.assertFalse(
            self.strategy._check_criterion(PatientLevel.SMOKE, "mnist", 0.11)
        )

    def test_check_criterion_digits_smoke_higher_bar(self):
        # digits/usps SMOKE threshold = 0.50
        self.assertFalse(
            self.strategy._check_criterion(PatientLevel.SMOKE, "digits", 0.49)
        )
        self.assertTrue(
            self.strategy._check_criterion(PatientLevel.SMOKE, "digits", 0.51)
        )

    def test_check_criterion_cifar100_lower_bar(self):
        # cifar100 SMOKE threshold = 0.05
        self.assertTrue(
            self.strategy._check_criterion(PatientLevel.SMOKE, "cifar100", 0.06)
        )
        self.assertFalse(
            self.strategy._check_criterion(PatientLevel.SMOKE, "cifar100", 0.04)
        )

    def test_check_criterion_standard_pass(self):
        self.assertTrue(
            self.strategy._check_criterion(PatientLevel.STANDARD, "mnist", 0.61)
        )
        self.assertFalse(
            self.strategy._check_criterion(PatientLevel.STANDARD, "mnist", 0.59)
        )

    def test_check_criterion_deep_pass(self):
        self.assertTrue(
            self.strategy._check_criterion(PatientLevel.DEEP, "mnist", 0.81)
        )
        self.assertFalse(
            self.strategy._check_criterion(PatientLevel.DEEP, "mnist", 0.79)
        )

    # ---- _analyze_saturation ----

    def test_analyze_saturation_mnist_above_threshold(self):
        """Progress is keyed as: progress[model][task][tier] = {best_acc: ...}."""
        progress = {
            "test_mlp": {
                "mnist": {
                    PatientLevel.STANDARD: {"best_acc": 0.995, "trials": 20},
                }
            }
        }
        self.mock_state.get_progress.return_value = progress
        saturated = self.strategy._analyze_saturation(progress)
        self.assertIn("test_mlp", saturated)
        self.assertIn("mnist", saturated["test_mlp"])

    def test_analyze_saturation_below_threshold(self):
        progress = {
            "test_mlp": {
                "mnist": {
                    PatientLevel.STANDARD: {"best_acc": 0.95, "trials": 5},
                }
            }
        }
        self.mock_state.get_progress.return_value = progress
        saturated = self.strategy._analyze_saturation(progress)
        self.assertNotIn("test_mlp", saturated)

    def test_analyze_saturation_implicit_digits_solved(self):
        """Solving mnist implicitly marks digits and usps as solved."""
        progress = {
            "test_mlp": {
                "mnist": {
                    PatientLevel.STANDARD: {"best_acc": 0.995, "trials": 20},
                }
            }
        }
        self.mock_state.get_progress.return_value = progress
        saturated = self.strategy._analyze_saturation(progress)
        self.assertIn("test_mlp", saturated)
        self.assertIn("digits", saturated["test_mlp"])
        self.assertIn("usps", saturated["test_mlp"])


if __name__ == "__main__":
    unittest.main()
