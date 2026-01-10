
import sys
import unittest
from unittest.mock import MagicMock

# Mock Pyqt6 modules to allow importing dashboard without a display
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['pyqtgraph'] = MagicMock()

# Setup QTabWidget mock behavior for dashboard logic
mock_tabs = MagicMock()
mock_tabs.currentIndex.return_value = 0
sys.modules['PyQt6.QtWidgets'].QTabWidget.return_value = mock_tabs

try:
    from eqprop_trainer.dashboard import EqPropDashboard
    from eqprop_trainer.worker import TrainingWorker
    from eqprop_trainer.generation import UniversalGenerator
    from eqprop_trainer.hyperparams import get_hyperparams_for_model
    from eqprop_trainer.viz_utils import extract_weights
    print("SUCCESS: All modules imported correctly.")
except Exception as e:
    print(f"FAILURE: Import or Syntax Error: {e}")
    sys.exit(1)
