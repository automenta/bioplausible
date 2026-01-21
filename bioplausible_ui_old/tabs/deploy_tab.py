"""
Deploy Tab

Tools for exporting models and serving them for real-world use.
"""

import os
import torch
import uvicorn
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
    QInputDialog
)
from PyQt6.QtCore import pyqtSignal, QThread

from bioplausible.export import export_to_onnx, export_to_torchscript

class ExportWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, format, path, input_sample, parent=None):
        super().__init__(parent)
        self.model = model
        self.format = format
        self.path = path
        self.input_sample = input_sample

    def run(self):
        try:
            if self.format == "onnx":
                export_to_onnx(self.model, self.input_sample, self.path)
            elif self.format == "torchscript":
                export_to_torchscript(self.model, self.input_sample, self.path)
            self.finished.emit(f"Successfully exported to {self.path}")
        except Exception as e:
            self.error.emit(str(e))

class ServerWorker(QThread):
    """Worker to run Uvicorn server in background."""
    def __init__(self, model, host="0.0.0.0", port=8000):
        super().__init__()
        self.model = model
        self.host = host
        self.port = port
        self.server = None

    def run(self):
        # We need to set the global model instance in export.py
        import bioplausible.export as export
        export.model_instance = self.model
        if self.model:
            export.model_instance.eval()

        config = uvicorn.Config(export.app, host=self.host, port=self.port, log_level="info")
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        if self.server:
            self.server.should_exit = True
            self.wait()

class DeployTab(QWidget):
    """Deployment & Export Tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.server_worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left Panel: Export
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)

        export_group = QGroupBox("📦 Export Model")
        export_layout = QVBoxLayout(export_group)

        export_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ONNX (Universal)", "TorchScript (C++)", "Quantized INT8 (Edge)"])
        export_layout.addWidget(self.format_combo)

        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self._export_model)
        export_layout.addWidget(self.export_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        export_layout.addWidget(self.progress)

        export_layout.addStretch()
        left_panel.addWidget(export_group)

        # Right Panel: Serving
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)

        serve_group = QGroupBox("🚀 Serve & Demo")
        serve_layout = QVBoxLayout(serve_group)

        serve_layout.addWidget(QLabel("Local API Server:"))
        self.serve_btn = QPushButton("Start API Server (Port 8000)")
        self.serve_btn.setCheckable(True)
        self.serve_btn.clicked.connect(self._toggle_server)
        serve_layout.addWidget(self.serve_btn)

        self.server_log = QTextEdit()
        self.server_log.setReadOnly(True)
        self.server_log.setPlaceholderText("Server logs will appear here...")
        serve_layout.addWidget(self.server_log)

        serve_layout.addWidget(QLabel("Web Interface:"))
        self.demo_btn = QPushButton("Open Web Demo")
        self.demo_btn.setEnabled(False) # Enable only when server is running
        self.demo_btn.clicked.connect(self._open_web_demo)
        serve_layout.addWidget(self.demo_btn)

        right_panel.addWidget(serve_group)

    def update_model_ref(self, model):
        self.model = model
        self.server_log.append(f"Model loaded: {type(model).__name__}")
        if self.server_worker and self.server_worker.isRunning():
            # Update running server model reference
            import bioplausible.export as export
            export.model_instance = self.model
            self.server_log.append("Hot-swapped model in running server.")

    def _guess_input_shape(self):
        """Intelligently guess input shape or ask user."""
        if not self.model: return None

        # 1. Check for explicit input_dim (MLP)
        if hasattr(self.model, 'input_dim') and self.model.input_dim is not None:
             # Check if it's likely an image flattened?
             # But for export we need the tensor shape.
             # If it's a flattened MLP, (1, input_dim) is correct.
             return (1, self.model.input_dim)

        # 2. Check for LM embedding
        if hasattr(self.model, 'has_embed') and self.model.has_embed:
             # LM takes integer indices: (Batch, Seq)
             # Default to (1, 64)
             return (1, 64), torch.long

        # 3. Check for Conv layers to guess channels
        name = type(self.model).__name__
        if "Conv" in name or "Vision" in name:
             # Try to find first conv layer
             for m in self.model.modules():
                 if isinstance(m, torch.nn.Conv2d):
                     c_in = m.in_channels
                     # Assume square 28x28 (MNIST) or 32x32 (CIFAR)
                     # We can default to 32x32 as it's safe for most
                     return (1, c_in, 32, 32)

        # Fallback: Ask user
        return None

    def _export_model(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "No model available to export.")
            return

        fmt = self.format_combo.currentText().split()[0].lower()
        ext = "onnx" if "onnx" in fmt else "pt"

        fname, _ = QFileDialog.getSaveFileName(self, "Export Model", f"model.{ext}", f"{fmt.upper()} (*.{ext})")
        if not fname:
            return

        # Prepare dummy input
        shape_info = self._guess_input_shape()
        dtype = torch.float32

        if shape_info:
            if isinstance(shape_info, tuple) and len(shape_info) == 2 and isinstance(shape_info[1], torch.dtype):
                shape, dtype = shape_info
            else:
                shape = shape_info
        else:
            # Dialog
            text, ok = QInputDialog.getText(self, "Input Shape", "Enter input shape (comma separated, e.g. 1,1,28,28):")
            if not ok or not text: return
            try:
                shape = tuple(map(int, text.split(',')))
            except:
                QMessageBox.critical(self, "Error", "Invalid shape format")
                return

        try:
            if dtype == torch.long:
                input_sample = torch.randint(0, 100, shape)
            else:
                input_sample = torch.randn(shape)

            if hasattr(self.model, 'device'):
                device = next(self.model.parameters()).device
                input_sample = input_sample.to(device)

            self.progress.setVisible(True)
            self.progress.setRange(0, 0) # Indeterminate
            self.export_btn.setEnabled(False)

            self.worker = ExportWorker(self.model, fmt, fname, input_sample)
            self.worker.finished.connect(self._on_export_finished)
            self.worker.error.connect(self._on_export_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Pre-check Failed", f"Could not prepare for export: {e}")

    def _on_export_finished(self, msg):
        self.progress.setVisible(False)
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Export Success", msg)

    def _on_export_error(self, err):
        self.progress.setVisible(False)
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Export Failed", err)

    def _toggle_server(self, checked):
        if checked:
            if not self.model:
                self.serve_btn.setChecked(False)
                QMessageBox.warning(self, "No Model", "No model to serve.")
                return

            self.server_log.append("Starting server...")

            self.server_worker = ServerWorker(self.model)
            self.server_worker.start()

            self.serve_btn.setText("⏹ Stop Server")
            self.serve_btn.setStyleSheet("background-color: #c0392b;")
            self.demo_btn.setEnabled(True)
            self.server_log.append("Server running at http://localhost:8000")
            self.server_log.append("Docs at http://localhost:8000/docs")
        else:
            if self.server_worker:
                self.server_log.append("Stopping server...")
                self.server_worker.stop()
                self.server_worker = None

            self.serve_btn.setText("Start API Server (Port 8000)")
            self.serve_btn.setStyleSheet("")
            self.demo_btn.setEnabled(False)
            self.server_log.append("Server stopped.")

    def _open_web_demo(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("http://localhost:8000/docs")) # Open Swagger UI for now
