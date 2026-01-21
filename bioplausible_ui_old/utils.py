"""
Utility functions for Bioplausible Trainer UI.
"""

import time
from typing import Dict, Any, Optional
import torch
from PyQt6.QtWidgets import QMessageBox


def calculate_eta(start_time: float, current_step: int, total_steps: int) -> tuple:
    """
    Calculate estimated time of arrival and speed.
    
    Args:
        start_time: Timestamp when process started
        current_step: Current step number
        total_steps: Total number of steps
    
    Returns:
        Tuple of (eta_string, speed_string)
    """
    if not start_time or current_step <= 0:
        return "--:--", "-- it/s"
        
    elapsed = time.time() - start_time
    speed = current_step / elapsed
    remaining = total_steps - current_step
    eta_seconds = remaining / speed if speed > 0 else 0
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    
    return eta_str, f"{speed:.2f} ep/s"


def safe_model_load(model_path: str, model_class, device: str = "cpu") -> Optional[Any]:
    """
    Safely load a model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        model_class: Model class to instantiate
        device: Device to load the model on
    
    Returns:
        Loaded model or None if failed
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def show_error_dialog(parent, title: str, message: str):
    """Show an error dialog with the given title and message."""
    QMessageBox.critical(parent, title, message)


def show_warning_dialog(parent, title: str, message: str):
    """Show a warning dialog with the given title and message."""
    QMessageBox.warning(parent, title, message)


def format_metric_value(metric_name: str, value: float) -> str:
    """
    Format a metric value based on its name.
    
    Args:
        metric_name: Name of the metric (loss, accuracy, etc.)
        value: Value to format
    
    Returns:
        Formatted string representation
    """
    if metric_name.lower() in ['accuracy', 'acc']:
        return f"{value:.1%}"
    elif metric_name.lower() in ['loss', 'lipschitz']:
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"


def get_device_info() -> str:
    """Get information about the available compute device."""
    import torch
    try:
        from bioplausible.kernel import HAS_TRITON_OPS
    except ImportError:
        HAS_TRITON_OPS = False

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = "CUDA"
        if HAS_TRITON_OPS:
            device_name += " (Triton Accel)"
    
    return device_name


def validate_hyperparams(hyperparams: Dict[str, Any]) -> bool:
    """
    Validate hyperparameters to ensure they are within acceptable ranges.
    
    Args:
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        True if valid, False otherwise
    """
    # Add validation logic as needed
    if 'lr' in hyperparams:
        lr = hyperparams['lr']
        if not (0.00001 <= lr <= 0.1):
            return False
    
    if 'epochs' in hyperparams:
        epochs = hyperparams['epochs']
        if not (1 <= epochs <= 1000):
            return False
            
    return True