"""
EqProp-Trainer: A stunning PyQt6 dashboard for training EqProp models.

Features:
- Dark cyberpunk theme with neon accents
- Live training plots (loss, accuracy, Lipschitz)
- Multi-tab interface (LM + Vision)
- Real-time text generation
- HuggingFace dataset integration
"""

from .dashboard import EqPropDashboard
from .main import main

__version__ = "0.1.0"
__all__ = ["EqPropDashboard", "main"]
