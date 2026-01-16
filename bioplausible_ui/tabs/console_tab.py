from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSlot
import torch
import os
import sys

class ConsoleTab(QWidget):
    """A tab for displaying system logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        self.status_btn = QPushButton("🩺 Check System Status")
        self.status_btn.clicked.connect(self._check_status)
        toolbar.addWidget(self.status_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setStyleSheet("background-color: #0a0a0f; color: #a0a0b0; font-family: Consolas, monospace;")
        layout.addWidget(self.console_log)

    def _check_status(self):
        """Run system diagnostics."""
        self.append_log("--- System Diagnostics ---")
        self.append_log(f"Python: {sys.version.split()[0]}")
        self.append_log(f"PyTorch: {torch.__version__}")

        cuda_avail = torch.cuda.is_available()
        self.append_log(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            self.append_log(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            self.append_log(f"CUDA Version: {torch.version.cuda}")

        # Check CuPy
        try:
            import cupy
            self.append_log(f"CuPy: Installed ({cupy.__version__})")
            if "CUDA_PATH" in os.environ:
                 self.append_log(f"CUDA_PATH: {os.environ['CUDA_PATH']}")
            else:
                 self.append_log("CUDA_PATH: Not Set (Autodetected?)")
        except ImportError:
            self.append_log("CuPy: Not Installed")

        # Check Triton
        try:
            from bioplausible.models.triton_kernel import HAS_TRITON
            self.append_log(f"Triton: {'Available' if HAS_TRITON else 'Not Available'}")
        except:
             self.append_log("Triton: Error checking")

        self.append_log("--------------------------")

    @pyqtSlot(str)
    def append_log(self, message: str):
        """Append a message to the console log."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        if not message.startswith("["): # Avoid double timestamping if already timestamped
            self.console_log.append(f"[{timestamp}] {message}")
        else:
            self.console_log.append(message)
