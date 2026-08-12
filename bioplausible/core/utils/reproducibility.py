"""
Reproducibility Framework
=========================

Tools for ensuring reproducible research:
- Seed management
- Configuration logging
- Result versioning
- Environment capture

Example
-------
>>> from bioplausible.core.utils.reproducibility import ReproducibilityTracker
>>> tracker = ReproducibilityTracker(seed=42)
>>> tracker.log_config(config)
>>> tracker.save_results(results)
"""

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from bioplausible.config.unified import BaseConfig, ReproducibilityConfig
from bioplausible.core.logging import get_logger

__all__ = [
    "EnvironmentInfo",
    "ReproducibilityConfig",
    "ReproducibilityTracker",
    "ReproducibleConfig",
    "create_tracker",
    "logger",
    "set_reproducible_mode",
]
logger = get_logger()


@dataclass
class EnvironmentInfo:
    """Captured environment information."""

    python_version: str
    torch_version: str
    cuda_version: str | None
    gpu_info: list[dict[str, object]]
    os_name: str
    cpu_count: int
    timestamp: str
    git_commit: str | None
    git_branch: str | None
    command_line: str


# =============================================================================
# Configuration Utilities
# =============================================================================


class ReproducibilityTracker:
    """Track and ensure reproducibility of experiments.

    Parameters
    ----------
    seed : int
        Random seed for all operations
    results_dir : str
        Directory to save results
    """

    def __init__(self, seed: int = 42, results_dir: str = "results") -> None:
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set all seeds
        self._set_seeds()

        # Capture environment
        self.env_info = self._capture_environment()

        # Experiment tracking
        self.experiment_id = self._generate_experiment_id()
        self.config_log: list[dict[str, object]] = []

    def _set_seeds(self) -> None:
        """Set all random seeds."""
        from bioplausible.core.utils.seeds import set_all_seeds

        set_all_seeds(self.seed, deterministic=True)

    def _capture_environment(self) -> EnvironmentInfo:
        """Capture current environment information."""
        # Git info
        git_commit = None
        git_branch = None
        try:
            import subprocess

            git_commit = (
                subprocess
                .check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
            git_branch = (
                subprocess
                .check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except OSError, ValueError, RuntimeError:
            logger.warning("Failed to get git branch name")

        # GPU info
        gpu_info = []
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "name": props.name,
                    "memory_gb": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        return EnvironmentInfo(
            python_version=sys.version,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            gpu_info=gpu_info,
            os_name=os.name,
            cpu_count=os.cpu_count() or 0,
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            git_branch=git_branch,
            command_line=" ".join(sys.argv),
        )

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{self.seed}_{os.getpid()}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{hash_id}"

    def log_config(self, config: object, name: str = "config") -> None:
        """Log configuration for reproducibility.

        Parameters
        ----------
        config : object
            Configuration object (dataclass, dict, etc.)
        name : str
            Configuration name
        """
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dataclass_fields__"):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config)

        self.config_log.append({
            "name": name,
            "config": config_dict,
            "timestamp": datetime.now().isoformat(),
        })

    def save_results(
        self,
        results: dict[str, object],
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save results with full reproducibility information.

        Parameters
        ----------
        results : dict
            Experimental results
        metrics : dict, optional
            Key metrics to extract

        Returns
        -------
        Path
            Path to saved results file
        """
        # Create results bundle
        bundle = {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "environment": asdict(self.env_info),
            "configs": self.config_log,
            "results": results,
            "metrics": metrics or {},
            "saved_at": datetime.now().isoformat(),
        }

        # Save to file
        filepath = self.results_dir / f"{self.experiment_id}.json"
        with Path(filepath).open("w") as f:
            json.dump(bundle, f, indent=2, default=str)

        # Also save as latest
        latest_path = self.results_dir / "latest.json"
        with Path(latest_path).open("w") as f:
            json.dump(bundle, f, indent=2, default=str)

        logger.info("Results saved to %s", filepath)
        return filepath

    def load_results(self, experiment_id: str) -> dict[str, object]:
        """Load results from a previous experiment.

        Parameters
        ----------
        experiment_id : str
            Experiment ID to load

        Returns
        -------
        dict
            Experiment results bundle
        """
        filepath = self.results_dir / f"{experiment_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        with Path(filepath).open() as f:
            return json.load(f)

    def verify_reproducibility(self, results_path: Path) -> dict[str, bool]:
        """Verify if results can be reproduced.

        Parameters
        ----------
        results_path : Path
            Path to results file

        Returns
        -------
        dict
            Verification results
        """
        with Path(results_path).open() as f:
            bundle = json.load(f)

        verification = {
            "seed_present": "seed" in bundle,
            "environment_captured": "environment" in bundle,
            "config_logged": len(bundle.get("configs", [])) > 0,
            "git_commit_present": bundle.get("environment", {}).get("git_commit")
            is not None,
            "torch_version_match": bundle.get("environment", {}).get("torch_version")
            == torch.__version__,
        }

        return verification

    def get_experiment_summary(self) -> str:
        """Get summary of current experiment."""
        lines = [
            f"Experiment ID: {self.experiment_id}",
            f"Seed: {self.seed}",
            f"Results dir: {self.results_dir}",
            "",
            "Environment:",
            f"  Python: {self.env_info.python_version.split()[0]}",
            f"  PyTorch: {self.env_info.torch_version}",
            f"  CUDA: {self.env_info.cuda_version or 'N/A'}",
            f"  GPU: {self.env_info.gpu_info[0]['name'] if self.env_info.gpu_info else 'N/A'}",
            "",
            f"Configs logged: {len(self.config_log)}",
        ]

        if self.env_info.git_commit:
            lines.append(
                f"Git: {self.env_info.git_commit[:8]} ({self.env_info.git_branch})"
            )

        return "\n".join(lines)


# =============================================================================
# Configuration Utilities
# =============================================================================


@dataclass(frozen=True)
class ReproducibleConfig(BaseConfig):
    """Base configuration with reproducibility support (REFACTOR.md §1 pattern)."""

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to file."""
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> ReproducibleConfig:
        """Load configuration from file."""
        with Path(path).open() as f:
            data = json.load(f)
        return cls(**data)

    def get_hash(self) -> str:
        """Get hash of configuration for versioning."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracker(
    seed: int = 42,
    results_dir: str = "results",
) -> ReproducibilityTracker:
    """Create reproducibility tracker.

    Parameters
    ----------
    seed : int
        Random seed
    results_dir : str
        Results directory

    Returns
    -------
    ReproducibilityTracker
        Tracker instance
    """
    return ReproducibilityTracker(seed=seed, results_dir=results_dir)


def set_reproducible_mode(seed: int = 42) -> None:
    """Set all seeds for reproducible execution.

    Parameters
    ----------
    seed : int
        Random seed
    """
    from bioplausible.core.utils.seeds import set_all_seeds

    set_all_seeds(seed, deterministic=True)
