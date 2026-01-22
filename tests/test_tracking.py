import unittest
from unittest.mock import MagicMock, patch

from bioplausible.tracking import ExperimentTracker


class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        # Patch wandb to prevent actual network calls
        self.wandb_patcher = patch("bioplausible.tracking.wandb")
        self.mock_wandb = self.wandb_patcher.start()

        # Mock HAS_WANDB to True
        self.has_wandb_patcher = patch("bioplausible.tracking.HAS_WANDB", True)
        self.has_wandb_patcher.start()

    def tearDown(self):
        self.wandb_patcher.stop()
        self.has_wandb_patcher.stop()

    def test_init(self):
        """Test initialization of ExperimentTracker."""
        tracker = ExperimentTracker(project="test_project", name="test_run")
        self.mock_wandb.init.assert_called_once_with(
            project="test_project",
            entity=None,
            config=None,
            mode="online",
            name="test_run",
            reinit=True,
        )
        self.assertEqual(tracker.backend, "wandb")

    def test_init_disabled(self):
        """Test initialization with disabled backend."""
        tracker = ExperimentTracker(backend="disabled")
        self.assertEqual(tracker.backend, "disabled")
        self.mock_wandb.init.assert_not_called()

    def test_log_hyperparams(self):
        """Test logging hyperparameters."""
        tracker = ExperimentTracker(project="test_project")
        # Ensure we have a mock run
        tracker.run = MagicMock()

        config = {"lr": 0.001, "batch_size": 32}
        tracker.log_hyperparams(config)

        # wandb.config.update called on the global object or the run object?
        # The implementation uses wandb.config.update(config) which updates the global config.
        # However, checking the implementation:
        # def log_hyperparams(self, config): ... wandb.config.update(config, ...)
        # So we check self.mock_wandb.config.update
        self.mock_wandb.config.update.assert_called_with(config, allow_val_change=True)

    def test_log_metrics(self):
        """Test logging metrics."""
        tracker = ExperimentTracker(project="test_project")
        tracker.run = MagicMock()

        metrics = {"loss": 0.5, "accuracy": 0.9}
        tracker.log_metrics(metrics, step=10)

        self.mock_wandb.log.assert_called_with(metrics, step=10)

    def test_log_validation_track(self):
        """Test logging validation track results."""
        tracker = ExperimentTracker(project="test_project")
        tracker.run = MagicMock()

        results = {"score": 95.0, "evidence_level": 3, "passed": True}
        tracker.log_validation_track(track_id=1, results=results)

        expected_log = {
            "track_1_score": 95.0,
            "track_1_evidence": 3,
            "track_1_passed": 1,
        }
        self.mock_wandb.log.assert_called_with(expected_log)

    def test_finish(self):
        """Test finishing the run."""
        tracker = ExperimentTracker(project="test_project")
        tracker.run = MagicMock()

        tracker.finish()
        tracker.run.finish.assert_called_once()
