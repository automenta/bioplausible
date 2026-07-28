"""Coverage tests for ExecutionStrategy -- tier progression, evolution, failures, prioritization."""

import unittest
from unittest.mock import MagicMock, patch

from bioplausible.execution.strategy import (
    ExecutionStrategy,
    _ModelSpec,
    _model_specs,
)
from bioplausible.execution.task import ExperimentTask
from bioplausible.hyperopt import PatientLevel


class _MockTrial:
    def __init__(self, accuracy=0.9, loss=0.5, config=None):
        self.accuracy = accuracy
        self.final_loss = loss
        self.config = config or {"lr": 0.001}


class TestStrategyEvolution(unittest.TestCase):
    """Test _check_evolution_needed and integration with generate_candidates."""

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

    def test_evolution_needed_triggers_at_100_trials_with_mnist_solved(self):
        progress = {
            "model_a": {
                "mnist": {"standard": {"count": 34, "best_acc": 0.995, "trials": []}}
            },
            "model_b": {
                "mnist": {"standard": {"count": 33, "best_acc": 0.995, "trials": []}}
            },
            "model_c": {
                "mnist": {"standard": {"count": 33, "best_acc": 0.995, "trials": []}}
            },
        }
        saturated = {"model_a": ["mnist"], "model_b": ["mnist"], "model_c": ["mnist"]}
        result = self.strategy._check_evolution_needed(progress, saturated)
        self.assertIsNotNone(result)
        self.assertEqual(result.model_name, "ASI_Evolve_Search")
        self.assertTrue(result.is_evolve)

    def test_evolution_needed_not_enough_models_solved(self):
        progress = {
            "model_a": {
                "mnist": {"standard": {"count": 100, "best_acc": 0.995, "trials": []}}
            },
        }
        saturated = {"model_a": ["mnist"]}
        result = self.strategy._check_evolution_needed(progress, saturated)
        self.assertIsNone(result)

    def test_evolution_needed_not_at_100_boundary(self):
        progress = {
            "model_a": {
                "mnist": {"standard": {"count": 50, "best_acc": 0.995, "trials": []}}
            },
        }
        saturated = {"model_a": ["mnist"]}
        result = self.strategy._check_evolution_needed(progress, saturated)
        self.assertIsNone(result)

    def test_should_consider_task_model_filter_excludes(self):
        strategy = ExecutionStrategy(self.mock_state, model_filter="bad_model")
        result = strategy._should_consider_task("bad_model", "mnist", {}, {})
        self.assertFalse(result)

    def test_should_consider_task_saturated_excludes(self):
        result = self.strategy._should_consider_task(
            "model_a", "mnist", {}, {"model_a": ["mnist"]}
        )
        self.assertFalse(result)


class TestStrategySmokeTier(unittest.TestCase):
    """Test _check_smoke_tier and _check_shallow_tier."""

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

    def test_smoke_tier_no_stats_returns_task(self):
        progress = {}
        result = self.strategy._check_smoke_tier("test_mlp", "mnist", progress, {})
        self.assertIsNotNone(result)
        self.assertEqual(result.tier, PatientLevel.SMOKE)
        self.assertEqual(result.priority, 100.0)

    def test_smoke_tier_has_some_stats_returns_task(self):
        progress = {
            "test_mlp": {
                "mnist": {"smoke": {"count": 1, "best_acc": 0.3, "trials": []}}
            }
        }
        result = self.strategy._check_smoke_tier("test_mlp", "mnist", progress, {})
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, 80.0)

    def test_smoke_tier_complete_returns_none(self):
        progress = {
            "test_mlp": {
                "mnist": {"smoke": {"count": 3, "best_acc": 0.5, "trials": []}}
            }
        }
        result = self.strategy._check_smoke_tier("test_mlp", "mnist", progress, {})
        self.assertIsNone(result)

    def test_smoke_tier_with_failure_constraints(self):
        progress = {}
        result = self.strategy._check_smoke_tier(
            "test_mlp", "mnist", progress, {"test_mlp": {"max_lr": 0.001}}
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.constraints, {"max_lr": 0.001})

    def test_shallow_tier_no_stats_generates_task(self):
        progress = {
            "test_mlp": {
                "mnist": {
                    "smoke": {"count": 3, "best_acc": 0.5, "trials": []},
                    "shallow": {"count": 0, "trials": []},
                }
            }
        }
        smoke_stats = {"count": 3, "best_acc": 0.5, "trials": []}
        result = self.strategy._check_shallow_tier(
            "test_mlp", "mnist", progress, smoke_stats, {}
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.tier, PatientLevel.SHALLOW)

    def test_shallow_tier_complete_returns_none(self):
        progress = {
            "test_mlp": {
                "mnist": {
                    "shallow": {"count": 10, "best_acc": 0.5, "trials": []},
                }
            }
        }
        smoke_stats = {"count": 3, "best_acc": 0.5, "trials": []}
        result = self.strategy._check_shallow_tier(
            "test_mlp", "mnist", progress, smoke_stats, {}
        )
        self.assertIsNone(result)

    def test_shallow_tier_first_run_logs_promotion(self):
        progress = {
            "test_mlp": {
                "mnist": {
                    "smoke": {"count": 3, "best_acc": 0.5, "trials": []},
                    "shallow": {"count": 0, "trials": []},
                }
            }
        }
        smoke_stats = {"count": 3, "best_acc": 0.5, "trials": []}
        with patch.object(self.strategy, "_log") as mock_log:
            self.strategy._check_shallow_tier(
                "test_mlp", "mnist", progress, smoke_stats, {}
            )
            mock_log.assert_called_once()


class TestStrategyTierLadder(unittest.TestCase):
    """Test _generate_candidates_for_task tier progression logic."""

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

    def test_generate_candidates_for_task_smoke_returns_early(self):
        progress = {}
        candidates = self.strategy._generate_candidates_for_task(
            "test_mlp", "mnist", progress, {}
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tier, PatientLevel.SMOKE)

    def test_generate_candidates_for_task_passed_smoke_no_shallow(self):
        progress = {
            "test_mlp": {
                "mnist": {
                    "smoke": {"count": 3, "best_acc": 0.5, "trials": []},
                    "shallow": {"count": 0, "trials": []},
                }
            }
        }
        candidates = self.strategy._generate_candidates_for_task(
            "test_mlp", "mnist", progress, {}
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tier, PatientLevel.SHALLOW)

    def test_generate_candidates_for_task_passed_shallow_generates_standard(self):
        progress = {
            "test_mlp": {
                "mnist": {
                    "smoke": {"count": 3, "best_acc": 0.5, "trials": []},
                    "shallow": {"count": 10, "best_acc": 0.5, "trials": []},
                    "standard": {"count": 0, "trials": []},
                }
            }
        }
        candidates = self.strategy._generate_candidates_for_task(
            "test_mlp", "mnist", progress, {}
        )
        self.assertEqual(len(candidates), 1)  # Standard exploration task
        self.assertEqual(candidates[0].tier, PatientLevel.STANDARD)


class TestStrategyPrioritization(unittest.TestCase):
    """Test prioritization helpers."""

    def setUp(self):
        self.mock_state = MagicMock()
        self.mock_state.get_progress.return_value = {}
        self.mock_state.get_recent_models.return_value = ["model_a"]
        self.mock_state.get_recent_tasks.return_value = ["mnist", "digits"]
        self.strategy = ExecutionStrategy(self.mock_state)
        _model_specs.cache = None

    def tearDown(self):
        _model_specs.cache = None

    def test_calculate_future_boost_no_track_returns_zero(self):
        boost = self.strategy._calculate_future_boost("nonexistent_task", 0.10)
        self.assertEqual(boost, 0.0)

    def test_complexity_penalty_default(self):
        penalty = self.strategy._calculate_complexity_penalty("unknown_model")
        self.assertEqual(penalty, 1.0)

    def test_complexity_penalty_hebbian(self):
        penalty = self.strategy._calculate_complexity_penalty(
            "Deep Hebbian (Hundred-Layer)"
        )
        self.assertEqual(penalty, 0.7)

    def test_apply_prioritization_modifies_priority(self):
        task = ExperimentTask(
            model_name="model_a",
            task_name="mnist",
            tier=PatientLevel.SMOKE,
            study_name="test",
            priority=100.0,
        )
        self.strategy._apply_prioritization([task])
        self.assertNotEqual(task.priority, 100.0)

    def test_apply_prioritization_recent_task_penalty(self):
        task = ExperimentTask(
            model_name="model_a",
            task_name="digits",
            tier=PatientLevel.SMOKE,
            study_name="test",
            priority=100.0,
        )
        original = task.priority
        self.strategy._apply_prioritization([task])
        # Priority should change (multiplied by weights and penalties)
        self.assertNotEqual(task.priority, original)


class TestStrategyFailureAnalysis(unittest.TestCase):
    """Test _analyze_failures and _analyze_fragility."""

    def setUp(self):
        self.mock_state = MagicMock()
        self.mock_state.get_progress.return_value = {}
        self.mock_state.get_fragile_models.return_value = {"fragile_model": 0.8}
        self.mock_state.get_failure_analysis.return_value = {
            "recommendations": [
                {"issue": "High NaN failure rate", "affected_models": ["model_a"]},
                {"issue": "Out of memory errors", "affected_models": []},
                {"issue": "Frequent timeouts", "affected_models": ["model_b"]},
                {"issue": "Early Training Instability"},
            ]
        }
        self.strategy = ExecutionStrategy(self.mock_state)
        _model_specs.cache = None

    def tearDown(self):
        _model_specs.cache = None

    def test_analyze_failures_nan_constrains(self):
        constraints = self.strategy._analyze_failures({"model_a": {}})
        self.assertIn("model_a", constraints)
        self.assertIn("max_lr", constraints["model_a"])
        self.assertEqual(constraints["model_a"]["max_lr"], 0.001)

    def test_analyze_failures_oom_fallback_all_models(self):
        progress = {"model_a": {}, "model_b": {}}
        constraints = self.strategy._analyze_failures(progress)
        self.assertIn("model_a", constraints)
        self.assertIn("model_b", constraints)
        self.assertIn("max_batch_size", constraints["model_a"])
        self.assertIn("max_hidden_dim", constraints["model_a"])

    def test_analyze_failures_timeout_constrains(self):
        constraints = self.strategy._analyze_failures({"model_b": {}})
        self.assertIn("model_b", constraints)

    def test_analyze_failures_soft_failure_high_rate(self):
        m = MagicMock()
        m.final_loss = 200.0
        m.accuracy = 0.05
        progress = {
            "high_fail_model": {
                "mnist": {
                    "smoke": {
                        "count": 10,
                        "best_acc": 0.05,
                        "trials": [m] * 10,
                    }
                }
            }
        }
        constraints = self.strategy._analyze_failures(progress)
        self.assertIn("high_fail_model", constraints)

    def test_analyze_failures_soft_failure_below_threshold(self):
        # Only provide progress without soft failures; use empty progress so OOM
        # doesn't add all models as fallback
        progress = {}
        # Mock get_failure_analysis to return nothing
        self.mock_state.get_failure_analysis.return_value = {"recommendations": []}
        constraints = self.strategy._analyze_failures(progress)
        # With no failures and no progress, constraints should be empty
        self.assertEqual(constraints, {})

    def test_analyze_fragility_returns_constraints(self):
        constraints = self.strategy._analyze_fragility()
        self.assertIn("fragile_model", constraints)
        self.assertIn("min_weight_decay", constraints["fragile_model"])

    def test_analyze_fragility_no_fragile_models(self):
        self.mock_state.get_fragile_models.return_value = {}
        constraints = self.strategy._analyze_fragility()
        self.assertEqual(constraints, {})


class TestStrategyRefineAndSaturation(unittest.TestCase):
    """Test _refine_search_space and _analyze_saturation."""

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

    def test_refine_search_space_fewer_than_3_trials(self):
        progress = {
            "model_a": {"mnist": {"shallow": {"count": 2, "trials": [MagicMock()] * 2}}}
        }
        result = self.strategy._refine_search_space(
            progress, "model_a", "mnist", PatientLevel.SHALLOW
        )
        self.assertIsNone(result)

    def test_refine_search_space_low_accuracy(self):
        trials = [_MockTrial(accuracy=0.15, config={"lr": 0.001}) for _ in range(5)]
        progress = {"model_a": {"mnist": {"shallow": {"count": 5, "trials": trials}}}}
        result = self.strategy._refine_search_space(
            progress, "model_a", "mnist", PatientLevel.SHALLOW
        )
        self.assertIsNone(result)

    def test_refine_search_space_returns_constraints(self):
        trials = [
            _MockTrial(accuracy=0.8, config={"lr": 0.001, "beta": 0.5})
            for _ in range(5)
        ]
        progress = {"model_a": {"mnist": {"shallow": {"count": 5, "trials": trials}}}}
        result = self.strategy._refine_search_space(
            progress, "model_a", "mnist", PatientLevel.SHALLOW
        )
        self.assertIsNotNone(result)
        self.assertIn("min_lr", result)
        self.assertIn("max_lr", result)

    def test_refine_search_space_no_beta_in_config(self):
        trials = [_MockTrial(accuracy=0.8, config={"lr": 0.001}) for _ in range(5)]
        progress = {"model_a": {"mnist": {"shallow": {"count": 5, "trials": trials}}}}
        result = self.strategy._refine_search_space(
            progress, "model_a", "mnist", PatientLevel.SHALLOW
        )
        self.assertIsNotNone(result)
        self.assertIn("min_lr", result)
        self.assertNotIn("min_beta", result)

    def test_saturation_thresholds(self):
        progress = {
            "model_a": {
                "mnist": {"standard": {"best_acc": 0.995, "count": 20, "trials": []}}
            }
        }
        self.mock_state.get_progress.return_value = progress
        saturated = self.strategy._analyze_saturation(progress)
        self.assertIn("model_a", saturated)
        self.assertIn("mnist", saturated["model_a"])
        self.assertIn("digits", saturated["model_a"])
        self.assertIn("usps", saturated["model_a"])

    def test_saturation_below_threshold(self):
        progress = {
            "model_a": {
                "mnist": {"standard": {"best_acc": 0.95, "count": 5, "trials": []}}
            }
        }
        self.mock_state.get_progress.return_value = progress
        saturated = self.strategy._analyze_saturation(progress)
        self.assertNotIn("model_a", saturated)

    def test_saturation_fashion_mnist_implicit(self):
        progress = {
            "model_a": {
                "fashion_mnist": {
                    "standard": {"best_acc": 0.95, "count": 20, "trials": []}
                }
            }
        }
        self.mock_state.get_progress.return_value = progress
        saturated = self.strategy._analyze_saturation(progress)
        self.assertIn("model_a", saturated)
        self.assertIn("fashion_mnist", saturated["model_a"])
        self.assertIn("mnist", saturated["model_a"])
        self.assertIn("kmnist", saturated["model_a"])

    def test_generate_candidates_full_pipeline(self):
        _model_specs.cache = [_ModelSpec("test_mlp", ["mnist"])]
        progress = {
            "test_mlp": {
                "mnist": {
                    "smoke": {"count": 3, "best_acc": 0.5, "trials": []},
                    "shallow": {"count": 10, "best_acc": 0.5, "trials": []},
                    "standard": {"count": 0, "trials": []},
                }
            }
        }
        self.mock_state.get_progress.return_value = progress
        self.strategy._analyze_failures = MagicMock(return_value={})
        self.strategy._analyze_fragility = MagicMock(return_value={})
        # Mock _should_consider_task to always allow
        with patch.object(self.strategy, "_should_consider_task", return_value=True):
            candidates = self.strategy.generate_candidates()
        self.assertGreater(len(candidates), 0)

    def test_plan_next_returns_highest_priority(self):
        _model_specs.cache = [_ModelSpec("test_mlp")]
        self.mock_state.get_progress.return_value = {}
        self.strategy._analyze_failures = MagicMock(return_value={})
        self.strategy._analyze_fragility = MagicMock(return_value={})
        task = self.strategy.plan_next()
        self.assertIsNotNone(task)

    def test_plan_batch_empty_when_no_candidates(self):
        _model_specs.cache = []
        batch = self.strategy.plan_batch(5)
        self.assertEqual(batch, [])

    def test_plan_batch_deduplicates(self):
        _model_specs.cache = [_ModelSpec("test_mlp")]
        self.mock_state.get_progress.return_value = {}
        self.strategy._analyze_failures = MagicMock(return_value={})
        self.strategy._analyze_fragility = MagicMock(return_value={})
        batch = self.strategy.plan_batch(10)
        self.assertGreater(len(batch), 0)
        keys = [(t.model_name, t.task_name, t.tier.value) for t in batch]
        self.assertEqual(len(keys), len(set(keys)))


class TestStrategyCurriculum(unittest.TestCase):
    """Test _check_curriculum and _resolve_tasks."""

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

    def test_resolve_tasks_expands_track(self):
        tasks = self.strategy._resolve_tasks(["vision"])
        self.assertIn("mnist", tasks)

    def test_resolve_tasks_empty_compat_returns_initial(self):
        tasks = self.strategy._resolve_tasks([])
        self.assertEqual(len(tasks), 1)

    def test_check_curriculum_first_task_allowed(self):
        result = self.strategy._check_curriculum({}, "model_a", "digits")
        self.assertTrue(result)

    def test_check_curriculum_prev_task_not_done(self):
        progress = {"model_a": {}}
        result = self.strategy._check_curriculum(progress, "model_a", "kmnist")
        self.assertFalse(result)

    def test_apply_saturation_logging(self):
        self.strategy._apply_saturation_logging({"model_a": ["mnist", "digits"]})
        # No assertion — just verifying it doesn't raise

    def test_apply_failure_logging(self):
        self.strategy._apply_failure_logging({"model_a": {"max_lr": 0.001}})
        # No assertion — just verifying it doesn't raise

    def test_filter_by_tier_limit_excludes_deep(self):
        _model_specs.cache = [_ModelSpec("test_mlp")]
        strategy = ExecutionStrategy(self.mock_state, tier_limit="shallow")
        strategy._analyze_saturation = MagicMock(return_value={})
        strategy._analyze_failures = MagicMock(return_value={})
        strategy._analyze_fragility = MagicMock(return_value={})
        candidates = strategy.generate_candidates()
        tiers = {c.tier for c in candidates}
        self.assertNotIn(PatientLevel.DEEP, tiers)
        self.assertNotIn(PatientLevel.STANDARD, tiers)

    def test_filter_by_tier_limit_invalid(self):
        strategy = ExecutionStrategy(self.mock_state, tier_limit="invalid")
        strategy._analyze_saturation = MagicMock(return_value={})
        strategy._analyze_failures = MagicMock(return_value={})
        strategy._analyze_fragility = MagicMock(return_value={})
        # Invalid tier limit should be silently ignored (limit_level stays -1)
        _model_specs.cache = []
        candidates = []
        strategy._filter_by_tier_limit(candidates)
        # Just verifying no crash


if __name__ == "__main__":
    unittest.main()
