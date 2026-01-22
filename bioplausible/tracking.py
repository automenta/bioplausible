import warnings
from typing import Any, Dict, Optional

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


class ExperimentTracker:
    """
    Unified experiment tracking interface.
    Currently supports Weights & Biases (wandb).
    """

    def __init__(
        self,
        project: str = "bioplausible",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        backend: str = "wandb",
        mode: str = "online",
        name: Optional[str] = None,
    ):
        """
        Initialize the experiment tracker.

        Args:
            project: Project name
            entity: User or team name
            config: Initial configuration dict
            backend: Tracking backend (default: "wandb")
            mode: "online", "offline", or "disabled"
            name: Run name
        """
        self.backend = backend.lower()
        self.run = None

        if self.backend == "wandb":
            if not HAS_WANDB:
                warnings.warn(
                    "wandb not installed. Tracking disabled. Install with `pip install wandb`"
                )
                self.backend = "disabled"
                return

            try:
                # If already initialized, use the existing run?
                # Or re-init. wandb.init() handles idempotency or re-init well.
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    config=config,
                    mode=mode,
                    name=name,
                    reinit=True,
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize wandb: {e}. Tracking disabled.")
                self.backend = "disabled"

        elif self.backend != "disabled":
            warnings.warn(f"Unknown backend '{backend}'. Tracking disabled.")
            self.backend = "disabled"

    def log_hyperparams(self, config: Dict[str, Any]):
        """Log hyperparameters."""
        if self.backend == "wandb" and self.run:
            wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (global step or epoch)
        """
        if self.backend == "wandb" and self.run:
            wandb.log(metrics, step=step)

    def log_lipschitz(self, L: float, step: Optional[int] = None):
        """Log Lipschitz constant (critical for EqProp)."""
        self.log_metrics({"lipschitz_constant": L}, step=step)

    def log_validation_track(self, track_id: int, results: Dict):
        """Log validation track results."""
        if self.backend == "wandb" and self.run:
            wandb.log(
                {
                    f"track_{track_id}_score": results.get("score", 0),
                    f"track_{track_id}_evidence": results.get("evidence_level", 0),
                    f"track_{track_id}_passed": int(results.get("passed", False)),
                }
            )

    def finish(self):
        """Finish the run."""
        if self.backend == "wandb" and self.run:
            self.run.finish()
