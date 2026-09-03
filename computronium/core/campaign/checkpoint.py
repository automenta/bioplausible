"""
Fault Tolerance Checkpointing for Joint Campaigns.

Checkpoints: z (CompositeState), θ (persistent params), ψ (plastic state),
σ (substrate state), episode index, campaign state, RNG state.
Enables multi-hour AutoScientist campaigns on spot instances.
"""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from computronium.core.campaign.campaign_store import (
        CampaignState as StoredCampaignState,
    )
    from computronium.state import CompositeState, SystemContext


@dataclass(frozen=True, slots=True)
class JointCheckpoint:
    """Complete checkpoint for joint system state."""

    # Campaign identification
    campaign_id: str
    branch_name: str
    episode_index: int
    timestamp: str

    # Joint dynamical system state: z_t = (x_t, ψ_t, σ_t)
    composite_state: dict[str, Any]  # Serialized CompositeState

    # Persistent parameters (θ) - immutable intra-episode
    theta: dict[str, Any]  # Serialized tensors

    # Campaign state
    campaign_state: dict[str, Any]  # Serialized CampaignState

    # RNG states for reproducibility
    torch_rng_state: bytes
    numpy_rng_state: bytes
    python_rng_state: bytes
    cuda_rng_state: bytes | None = None

    # Metadata
    coordinate: str = ""
    task_name: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class CheckpointManager:
    """
    Manages checkpointing for joint architecture campaigns.

    Supports:
    - Periodic automatic checkpointing
    - Manual checkpoint creation
    - Resume from checkpoint
    - Checkpoint validation and integrity checks
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 10,
        checkpoint_interval: int = 5,  # episodes
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to retain
            checkpoint_interval: Episodes between automatic checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval

        self._episode_counter = 0

    def should_checkpoint(self, episode_index: int) -> bool:
        """Check if a checkpoint should be created at this episode."""
        return episode_index % self.checkpoint_interval == 0 and episode_index > 0

    def create_checkpoint(
        self,
        campaign_state: StoredCampaignState,
        episode_index: int,
        composite_state: CompositeState,
        context: SystemContext,
        coordinate: str,
        task_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Create a complete checkpoint.

        Args:
            campaign_state: Current campaign state
            episode_index: Current episode index
            composite_state: Current joint state z = (x, ψ, σ)
            context: SystemContext with θ
            coordinate: 6-D coordinate string
            task_name: Task identifier
            metadata: Additional metadata

        Returns:
            Path to the checkpoint file
        """
        import random

        import numpy as np

        # Serialize composite state
        composite_dict = {
            "activity": {
                k: v.detach().cpu().numpy() for k, v in composite_state.activity.items()
            },
            "plastic": {
                k: v.detach().cpu().numpy() for k, v in composite_state.plastic.items()
            },
            "substrate": {
                k: v.detach().cpu().numpy()
                for k, v in composite_state.substrate.items()
            },
        }

        # Serialize theta
        theta_dict = {k: v.detach().cpu().numpy() for k, v in context.theta.items()}

        # Capture RNG states
        torch_rng = torch.get_rng_state()
        numpy_rng = np.random.get_state()
        python_rng = random.getstate()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        checkpoint = JointCheckpoint(
            campaign_id=campaign_state.campaign_id,
            branch_name=campaign_state.branch_name,
            episode_index=episode_index,
            timestamp=datetime.now().isoformat(),
            composite_state=composite_dict,
            theta=theta_dict,
            campaign_state=asdict(campaign_state),
            torch_rng_state=pickle.dumps(torch_rng),
            numpy_rng_state=pickle.dumps(numpy_rng),
            python_rng_state=pickle.dumps(python_rng),
            cuda_rng_state=pickle.dumps(cuda_rng) if cuda_rng else None,
            coordinate=coordinate,
            task_name=task_name,
            metadata=metadata or {},
        )

        # Save checkpoint
        filename = f"checkpoint_{campaign_state.campaign_id}_ep{episode_index:06d}_{uuid.uuid4().hex[:8]}.pkl"
        filepath = self.checkpoint_dir / filename

        with filepath.open("wb") as f:
            pickle.dump(checkpoint, f)

        # Prune old checkpoints
        self._prune_checkpoints(campaign_state.campaign_id)

        return filepath

    def load_checkpoint(self, filepath: str | Path) -> JointCheckpoint:
        """Load a checkpoint from file."""
        with Path(filepath).open("rb") as f:
            checkpoint = pickle.load(f)  # ruff: ignore[suspicious-pickle-usage]
        return checkpoint

    def restore_rng_states(self, checkpoint: JointCheckpoint) -> None:
        """Restore RNG states from checkpoint."""
        import random

        import numpy as np

        torch.set_rng_state(pickle.loads(checkpoint.torch_rng_state))  # ruff: ignore[suspicious-pickle-usage]
        np.random.set_state(pickle.loads(checkpoint.numpy_rng_state))  # ruff: ignore[suspicious-pickle-usage]
        random.setstate(pickle.loads(checkpoint.python_rng_state))  # ruff: ignore[suspicious-pickle-usage]
        if checkpoint.cuda_rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(pickle.loads(checkpoint.cuda_rng_state))  # ruff: ignore[suspicious-pickle-usage]

    def restore_composite_state(
        self,
        checkpoint: JointCheckpoint,
        device: str | torch.device = "cpu",
    ) -> CompositeState:
        """Restore CompositeState from checkpoint."""
        from computronium.state import CompositeState

        activity = {
            k: torch.from_numpy(v).to(device)
            for k, v in checkpoint.composite_state["activity"].items()
        }
        plastic = {
            k: torch.from_numpy(v).to(device)
            for k, v in checkpoint.composite_state["plastic"].items()
        }
        substrate = {
            k: torch.from_numpy(v).to(device)
            for k, v in checkpoint.composite_state["substrate"].items()
        }

        return CompositeState(activity=activity, plastic=plastic, substrate=substrate)

    def restore_theta(
        self,
        checkpoint: JointCheckpoint,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Restore theta (persistent parameters) from checkpoint."""
        return {k: torch.from_numpy(v).to(device) for k, v in checkpoint.theta.items()}

    def list_checkpoints(self, campaign_id: str | None = None) -> list[Path]:
        """List available checkpoints, optionally filtered by campaign."""
        pattern = (
            f"checkpoint_{campaign_id}_*.pkl" if campaign_id else "checkpoint_*.pkl"
        )
        return sorted(
            self.checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime
        )

    def get_latest_checkpoint(self, campaign_id: str) -> Path | None:
        """Get the latest checkpoint for a campaign."""
        checkpoints = self.list_checkpoints(campaign_id)
        return checkpoints[-1] if checkpoints else None

    def _prune_checkpoints(self, campaign_id: str) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = self.list_checkpoints(campaign_id)
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink(missing_ok=True)

    def validate_checkpoint(self, checkpoint: JointCheckpoint) -> bool:
        """Validate checkpoint integrity."""
        required_fields = [
            "campaign_id",
            "branch_name",
            "episode_index",
            "composite_state",
            "theta",
            "campaign_state",
            "torch_rng_state",
            "numpy_rng_state",
            "python_rng_state",
        ]
        for field in required_fields:
            if not hasattr(checkpoint, field) or getattr(checkpoint, field) is None:
                return False

        # Validate composite state structure
        comp = checkpoint.composite_state
        if not all(k in comp for k in ("activity", "plastic", "substrate")):
            return False

        # Validate theta
        if not isinstance(checkpoint.theta, dict) or len(checkpoint.theta) == 0:  # ruff: ignore[needless-bool]
            return False

        return True


def create_resume_script(
    checkpoint_path: str | Path,
    output_path: str | Path,
    command: str = "biopl campaign resume",
) -> Path:
    """
    Generate a resume script for a checkpoint.

    Creates a shell script that restores the campaign from a checkpoint.
    """
    checkpoint = CheckpointManager(".").load_checkpoint(checkpoint_path)

    script = f"""#!/bin/bash
# Auto-generated resume script for campaign {checkpoint.campaign_id}
# Episode: {checkpoint.episode_index}
# Coordinate: {checkpoint.coordinate}
# Task: {checkpoint.task_name}
# Timestamp: {checkpoint.timestamp}

{command} --checkpoint {checkpoint_path}
"""

    out_path = Path(output_path)
    out_path.write_text(script, encoding="utf-8")
    out_path.chmod(0o755)
    return out_path
