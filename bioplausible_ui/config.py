"""
Central configuration for EqProp Trainer.
"""

from dataclasses import dataclass

@dataclass
class TrainerConfig:
    """Global configuration for the trainer."""
    epochs: int = 3  # Centralized default for all experiments (baseline & trials)
    quick_mode: bool = True
    max_trial_time: float = 60.0 # Total trial budget in seconds (used to derive per-epoch limit)
    task: str = 'shakespeare'

# Global instance
GLOBAL_CONFIG = TrainerConfig()
