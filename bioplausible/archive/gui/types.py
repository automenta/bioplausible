
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingState:
    """Per-algorithm training state."""
    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 1.0
    iter_time: float = 0.0
    vram_gb: float = 0.0
    sample: str = "..."
    signal_norms: List[float] = field(default_factory=list)
    step: int = 0
    best_acc: float = 0.0
    best_ppl: float = float('inf')
