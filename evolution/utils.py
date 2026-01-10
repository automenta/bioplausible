"""
Utility functions for the evolution system.
"""

import torch
import numpy as np
from typing import Any, Dict, Optional
import logging


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def dict_to_str(d: Dict[str, Any], indent: int = 0) -> str:
    """Convert dict to readable string with indentation."""
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(dict_to_str(v, indent + 1))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


class MovingAverage:
    """Track moving average of a metric."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get(self) -> Optional[float]:
        if not self.values:
            return None
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values = []


def validate_config_dict(config: Dict[str, Any], required_keys: list, optional_keys: list = None) -> bool:
    """
    Validate that a config dict has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        optional_keys: List of optional key names (for documentation)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If required keys are missing
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return True
