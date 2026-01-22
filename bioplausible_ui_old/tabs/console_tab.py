from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLineEdit, QLabel
from PyQt6.QtCore import pyqtSlot, Qt
import torch
import os
import sys

class ConsoleTab(QWidget):
    """A tab for displaying system logs and executing commands."""

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

        # Command Input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel(">"))
        self.cmd_input = QLineEdit()
        self.cmd_input.setStyleSheet("background-color: #1a1a1e; color: #00ff88; font-family: Consolas; border: 1px solid #333;")
        self.cmd_input.returnPressed.connect(self._execute_command)
        input_layout.addWidget(self.cmd_input)
        layout.addLayout(input_layout)

    def _execute_command(self):
        """Execute user command."""
        cmd = self.cmd_input.text().strip()
        self.cmd_input.clear()

        if not cmd: return

        self.append_log(f"> {cmd}")

        if cmd == "!help":
            self.append_log("Commands: !status, !clear, !model, !cuda, !python <code>")
        elif cmd == "!status":
            self._check_status()
        elif cmd == "!clear":
            self.console_log.clear()
        elif cmd == "!model":
            # Access dashboard parent?
            # We don't strictly have a reference to dashboard here unless passed.
            # But we can try to find it.
            parent = self.window()
            if hasattr(parent, 'model') and parent.model:
                self.append_log(str(parent.model))
            else:
                self.append_log("No model loaded in dashboard.")
        elif cmd == "!cuda":
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**2
                self.append_log(f"CUDA Memory: {mem:.2f} MB")
            else:
                self.append_log("CUDA not available.")
        elif cmd.startswith("!python "):
            code = cmd[8:]
            try:
                # Safe-ish execution context
                context = {'torch': torch, 'np': __import__('numpy')}
                # Add model if available
                parent = self.window()
                if hasattr(parent, 'model'):
                    context['model'] = parent.model

                exec(code, context)
                self.append_log("Executed.")
            except Exception as e:
                self.append_log(f"Error: {e}")
        else:
            self.append_log(f"Unknown command: {cmd}")

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
