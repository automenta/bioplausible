"""
Diffusion Tab

Interface for training and sampling from EqProp Diffusion models.
"""

import numpy as np
import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QProgressBar, QSplitter,
    QComboBox, QCheckBox, QMessageBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class DiffusionSamplingWorker(QThread):
    """Background worker for sampling from Diffusion model."""
    finished = pyqtSignal(np.ndarray) # Returns grid image
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, model, num_samples=16, device="cuda"):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.device = device

    def run(self):
        try:
            # Run sampling
            # samples: [B, C, H, W] in [-1, 1]
            samples = self.model.sample(
                num_samples=self.num_samples,
                img_size=(1, 28, 28), # Hardcoded for MNIST for now
                device=self.device
            )

            # Convert to numpy grid
            # Normalize to [0, 255]
            samples = (samples + 1) / 2.0
            samples = samples.clamp(0, 1)

            # Create grid (e.g. 4x4)
            grid_size = int(np.ceil(np.sqrt(self.num_samples)))
            B, C, H, W = samples.shape

            grid_h = grid_size * H
            grid_w = grid_size * W
            grid = torch.zeros(C, grid_h, grid_w)

            for i in range(B):
                row = i // grid_size
                col = i % grid_size
                grid[:, row*H:(row+1)*H, col*W:(col+1)*W] = samples[i]

            # To Numpy [H, W] (assuming grayscale)
            img_np = grid.squeeze(0).cpu().numpy()
            self.finished.emit(img_np)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()


class DiffusionTab(QWidget):
    """Tab for Generative Diffusion."""

    start_training_signal = pyqtSignal(str) # mode='diffusion'
    stop_training_signal = pyqtSignal()
    clear_plots_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None # Reference to trained model
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left Panel: Controls ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Config Group
        config_group = QGroupBox("Model Configuration")
        config_layout = QFormLayout()

        # Model Selection (Fixed for now)
        self.model_combo = QComboBox()
        self.model_combo.addItem("EqProp Diffusion")
        config_layout.addRow("Model:", self.model_combo)

        # Dataset (Fixed for now)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("MNIST")
        config_layout.addRow("Dataset:", self.dataset_combo)

        # Hidden Dim
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(32, 512)
        self.hidden_spin.setValue(64)
        self.hidden_spin.setSingleStep(32)
        config_layout.addRow("Hidden Channels:", self.hidden_spin)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # Training Control
        train_group = QGroupBox("Training Control")
        train_layout = QVBoxLayout(train_group)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        train_layout.addWidget(QLabel("Epochs:"))
        train_layout.addWidget(self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        train_layout.addWidget(QLabel("Learning Rate:"))
        train_layout.addWidget(self.lr_spin)

        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._start_training)
        self.train_btn.setStyleSheet("background-color: #27ae60; font-weight: bold;")

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_training_signal.emit)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #c0392b;")

        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.stop_btn)
        train_layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        train_layout.addWidget(self.progress)

        left_layout.addWidget(train_group)

        # Generation Control
        gen_group = QGroupBox("Generation")
        gen_layout = QVBoxLayout(gen_group)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 64)
        self.samples_spin.setValue(16)
        gen_layout.addWidget(QLabel("Num Samples:"))
        gen_layout.addWidget(self.samples_spin)

        self.gen_btn = QPushButton("✨ Generate Samples")
        self.gen_btn.clicked.connect(self._generate)
        self.gen_btn.setStyleSheet("background-color: #8e44ad; font-weight: bold; font-size: 14px;")
        self.gen_btn.setEnabled(False) # Enabled only when model exists
        gen_layout.addWidget(self.gen_btn)

        left_layout.addWidget(gen_group)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # --- Right Panel: Viz ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("Generated Samples"))

        if HAS_PYQTGRAPH:
            self.img_view = pg.ImageView()
            self.img_view.ui.histogram.hide()
            self.img_view.ui.roiBtn.hide()
            self.img_view.ui.menuBtn.hide()
            right_layout.addWidget(self.img_view)
        else:
            right_layout.addWidget(QLabel("PyQtGraph not installed."))

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

    def update_model_ref(self, model):
        """Update reference to trained model."""
        self.model = model
        if self.model:
            self.gen_btn.setEnabled(True)

    def _start_training(self):
        self.start_training_signal.emit('diffusion')
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _generate(self):
        if not self.model:
            return

        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("Generating...")

        device = next(self.model.parameters()).device

        self.gen_worker = DiffusionSamplingWorker(
            self.model,
            num_samples=self.samples_spin.value(),
            device=device
        )
        self.gen_worker.finished.connect(self._on_gen_finished)
        self.gen_worker.error.connect(self._on_gen_error)
        self.gen_worker.start()

    def _on_gen_finished(self, img):
        if HAS_PYQTGRAPH:
            # Transpose for pyqtgraph (W, H)
            self.img_view.setImage(img.T)
            self.img_view.autoRange()

        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("✨ Generate Samples")

    def _on_gen_error(self, err):
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("✨ Generate Samples")
        QMessageBox.critical(self, "Generation Error", err)

    def update_theme(self, theme_colors, plot_colors):
        # Update background if needed
        pass
