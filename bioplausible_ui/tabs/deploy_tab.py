"""
Deploy Tab

Tools for exporting models and serving them for real-world use.
"""

import os
import torch
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QTextEdit, QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, QThread

from bioplausible.export import export_to_onnx, export_to_torchscript, serve_model

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

class DeployTab(QWidget):
    """Deployment & Export Tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
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
        # We need to guess input shape.
        # This is tricky without task context.
        # We can try to infer from model attributes or ask user?
        # For MVP, assume standard shapes based on model type.

        try:
            if hasattr(self.model, 'input_dim'):
                # MLP or flattened
                input_sample = torch.randn(1, self.model.input_dim)
            elif "Conv" in type(self.model).__name__:
                # Vision Conv
                # Try standard image sizes
                input_sample = torch.randn(1, 1, 28, 28) # MNIST default
            else:
                # Fallback
                input_sample = torch.randn(1, 784)

            if hasattr(self.model, 'device'):
                input_sample = input_sample.to(self.model.device)

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
            # We would need a background process/thread for FastAPI/Uvicorn
            # For this MVP, we will simulate or use a thread.
            # Uvicorn run is blocking, needs thread.

            from threading import Thread
            self.server_thread = Thread(target=serve_model, args=(self.model,), daemon=True)
            self.server_thread.start()

            self.serve_btn.setText("⏹ Stop Server")
            self.serve_btn.setStyleSheet("background-color: #c0392b;")
            self.demo_btn.setEnabled(True)
            self.server_log.append("Server running at http://localhost:8000")
            self.server_log.append("Docs at http://localhost:8000/docs")
        else:
            # Stopping a thread running uvicorn is hard properly without uvicorn.Server object
            # For MVP, we instruct restart.
            self.server_log.append("Stopping server not fully implemented in demo (restart app).")
            self.serve_btn.setText("Start API Server (Port 8000)")
            self.serve_btn.setStyleSheet("")
            self.demo_btn.setEnabled(False)

    def _open_web_demo(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("http://localhost:8000/docs")) # Open Swagger UI for now
