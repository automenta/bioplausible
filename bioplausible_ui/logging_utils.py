"""
Logging utilities for Bioplausible Trainer UI.

Provides centralized logging configuration and utilities for the UI components.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path


class UILogger:
    """Centralized logger for UI components."""
    
    _instance: Optional['UILogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'UILogger':
        if cls._instance is None:
            cls._instance = super(UILogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup the centralized logger."""
        logger = logging.getLogger('bioplausible_ui')
        logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers multiple times
        if logger.handlers:
            return logger
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path.home() / '.bioplausible' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the centralized logger instance."""
        return self._logger


def get_ui_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance for UI components.
    
    Args:
        name: Name for the specific logger
        
    Returns:
        Configured logger instance
    """
    base_logger = UILogger().get_logger()
    return base_logger.getChild(name)


def log_exception(logger: logging.Logger, context: str = ""):
    """
    Log an exception with traceback.
    
    Args:
        logger: Logger instance to use
        context: Additional context about where the exception occurred
    """
    import traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type:
        logger.error(f"Exception in {context}: {exc_type.__name__}: {exc_value}")
        logger.debug(f"Traceback:\n{''.join(traceback.format_tb(exc_traceback))}")


def safe_execute(func, *args, logger: Optional[logging.Logger] = None, default_return=None, **kwargs):
    """
    Safely execute a function with error logging.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to function
        logger: Logger to use for error logging
        default_return: Value to return if function fails
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Result of function call or default_return if it fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            log_exception(logger, func.__name__)
        return default_return