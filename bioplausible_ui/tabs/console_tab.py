from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import pyqtSlot

class ConsoleTab(QWidget):
    """A tab for displaying system logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setStyleSheet("background-color: #0a0a0f; color: #a0a0b0; font-family: Consolas, monospace;")
        layout.addWidget(self.console_log)

    @pyqtSlot(str)
    def append_log(self, message: str):
        """Append a message to the console log."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        if not message.startswith("["): # Avoid double timestamping if already timestamped
            self.console_log.append(f"[{timestamp}] {message}")
        else:
            self.console_log.append(message)
