"""
Vision Training Tab for Bioplausible Trainer - Refactored Version

This module implements the vision training tab interface with training controls,
visualization, and specialized analysis tools.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel, QToolBox, QFrame,
    QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import torch
import torch.nn.functional as F
import numpy as np

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible_ui.dashboard_helpers import update_hyperparams_generic, get_current_hyperparams_generic
from bioplausible_ui.themes import PLOT_COLORS
from bioplausible_ui.common_widgets import create_plot_widget, create_control_group
from .vision_specialized_components import (
    AlignmentWorker, AlignmentDialog, CubeVisualizerDialog, DreamWorker,
    OracleWorker, OracleDialog, DreamDialog, VisionInferenceDialog, RobustnessDialog
)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class VisionTab(QWidget):
    """Vision Training Tab."""

    start_training_signal = pyqtSignal(str) # Mode ('vision')
    stop_training_signal = pyqtSignal()
    clear_plots_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_ref = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(15)

        # Left panel: Controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)

        # Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("🚀 Quick Presets:"))
        self.vis_preset_combo = QComboBox()
        self.vis_preset_combo.addItems(["Custom", "Standard Backprop", "Fast EqProp", "Deep EqProp (Accurate)"])
        self.vis_preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.vis_preset_combo)
        left_panel.addLayout(preset_layout)

        # Toolbox for sections
        self.toolbox = QToolBox()
        left_panel.addWidget(self.toolbox)

        # --- Section 1: Model ---
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)

        self.vis_model_combo = QComboBox()
        model_items = []
        for spec in MODEL_REGISTRY:
            if spec.task_compat is None or "vision" in spec.task_compat:
                model_items.append(f"{spec.name}")
        self.vis_model_combo.addItems(model_items)
        self.vis_model_combo.currentTextChanged.connect(self._update_model_desc)

        self.vis_desc_label = QLabel("")
        self.vis_desc_label.setWordWrap(True)
        self.vis_desc_label.setStyleSheet("color: #a0a0b0; font-size: 11px; font-style: italic; margin-bottom: 5px;")

        self.vis_hidden_spin = QSpinBox()
        self.vis_hidden_spin.setRange(64, 1024)
        self.vis_hidden_spin.setValue(256)
        self.vis_hidden_spin.setToolTip("Dimension of hidden state vectors")

        self.vis_steps_spin = QSpinBox()
        self.vis_steps_spin.setRange(5, 100)
        self.vis_steps_spin.setValue(30)
        self.vis_steps_spin.setToolTip("Number of equilibrium steps per forward pass")

        model_controls = [
            ("Architecture:", self.vis_model_combo),
            ("", self.vis_desc_label),
            ("Hidden Dim:", self.vis_hidden_spin),
            ("Max Steps:", self.vis_steps_spin)
        ]

        # Create model control group
        model_control_group = create_control_group("Model Architecture", model_controls)
        model_layout.addWidget(model_control_group)

        # Dynamic Hyperparameters
        self.vis_hyperparam_group = QGroupBox("Dynamic Params")
        self.vis_hyperparam_layout = QGridLayout(self.vis_hyperparam_group)
        self.vis_hyperparam_widgets = {}
        model_layout.addWidget(self.vis_hyperparam_group)
        self.vis_hyperparam_group.setVisible(False)
        self.vis_model_combo.currentTextChanged.connect(self._update_vis_hyperparams)

        model_layout.addStretch()
        self.toolbox.addItem(model_widget, "🧠 Model Architecture")

        # --- Section 2: Dataset ---
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        self.vis_dataset_combo = QComboBox()
        self.vis_dataset_combo.addItems(["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"])
        self.vis_dataset_combo.setToolTip("Image dataset for training")

        self.vis_batch_spin = QSpinBox()
        self.vis_batch_spin.setRange(16, 512)
        self.vis_batch_spin.setValue(64)
        self.vis_batch_spin.setToolTip("Number of images per training step")

        data_controls = [
            ("Dataset:", self.vis_dataset_combo),
            ("Batch Size:", self.vis_batch_spin)
        ]

        data_control_group = create_control_group("Data Configuration", data_controls)
        data_layout.addWidget(data_control_group)
        data_layout.addStretch()

        self.toolbox.addItem(data_widget, "📚 Data Configuration")

        # --- Section 3: Training ---
        train_widget = QWidget()
        train_layout = QVBoxLayout(train_widget)

        self.vis_grad_combo = QComboBox()
        self.vis_grad_combo.addItems(["BPTT (Standard)", "Equilibrium (Implicit Diff)", "Contrastive (Hebbian)"])
        self.vis_grad_combo.setToolTip("Gradient Calculation Method")

        self.vis_epochs_spin = QSpinBox()
        self.vis_epochs_spin.setRange(1, 100)
        self.vis_epochs_spin.setValue(10)
        self.vis_epochs_spin.setToolTip("Total number of passes over the dataset")

        self.vis_lr_spin = QDoubleSpinBox()
        self.vis_lr_spin.setRange(0.0001, 0.1)
        self.vis_lr_spin.setValue(0.001)
        self.vis_lr_spin.setDecimals(4)
        self.vis_lr_spin.setToolTip("Step size for optimizer")

        self.vis_compile_check = QCheckBox("torch.compile")
        self.vis_compile_check.setChecked(True)
        self.vis_compile_check.setToolTip("Use PyTorch 2.0 graph compilation for speedup")

        self.vis_kernel_check = QCheckBox("O(1) Kernel Mode (GPU)")
        self.vis_kernel_check.setToolTip("Use fused EqProp kernel for O(1) memory training")

        self.vis_micro_check = QCheckBox("Live Dynamics Analysis")
        self.vis_micro_check.setToolTip("Periodically analyze convergence dynamics during training")

        train_controls = [
            ("Gradient:", self.vis_grad_combo),
            ("Epochs:", self.vis_epochs_spin),
            ("Learning Rate:", self.vis_lr_spin),
            ("", self.vis_compile_check),
            ("", self.vis_kernel_check),
            ("", self.vis_micro_check)
        ]

        train_control_group = create_control_group("Optimization Strategy", train_controls)
        train_layout.addWidget(train_control_group)
        train_layout.addStretch()

        self.toolbox.addItem(train_widget, "⚙️ Optimization Strategy")

        # Trigger initial update
        self._update_model_desc(self.vis_model_combo.currentText())

        # Buttons
        btn_layout = QHBoxLayout()
        self.vis_train_btn = QPushButton("▶ Train")
        self.vis_train_btn.setObjectName("trainButton")
        self.vis_train_btn.clicked.connect(lambda: self.start_training_signal.emit('vision'))
        btn_layout.addWidget(self.vis_train_btn)

        self.vis_stop_btn = QPushButton("⏹ Stop")
        self.vis_stop_btn.setObjectName("stopButton")
        self.vis_stop_btn.setEnabled(False)
        self.vis_stop_btn.clicked.connect(self.stop_training_signal.emit)
        btn_layout.addWidget(self.vis_stop_btn)

        self.vis_pause_btn = QPushButton("⏸ Pause")
        self.vis_pause_btn.setObjectName("resetButton")
        self.vis_pause_btn.setCheckable(True)
        self.vis_pause_btn.setEnabled(False)
        btn_layout.addWidget(self.vis_pause_btn)

        self.vis_reset_btn = QPushButton("↺ Reset")
        self.vis_reset_btn.setObjectName("resetButton")
        self.vis_reset_btn.setToolTip("Reset all hyperparameters to default values")
        self.vis_reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.vis_reset_btn)

        # Test Tools
        self.vis_test_btn = QPushButton("👁️ Test")
        self.vis_test_btn.setObjectName("resetButton")
        self.vis_test_btn.setToolTip("Test model on random sample")
        self.vis_test_btn.clicked.connect(self._test_random_sample)
        btn_layout.addWidget(self.vis_test_btn)

        self.vis_dream_btn = QPushButton("🌙 Dream")
        self.vis_dream_btn.setObjectName("resetButton")
        self.vis_dream_btn.setToolTip("Invert the network to dream a class pattern")
        self.vis_dream_btn.clicked.connect(self._open_dream_dialog)
        btn_layout.addWidget(self.vis_dream_btn)

        self.vis_cube_btn = QPushButton("🧊 Cube Viz")
        self.vis_cube_btn.setObjectName("resetButton")
        self.vis_cube_btn.setToolTip("Visualize 3D Neural Cube Activations")
        self.vis_cube_btn.clicked.connect(self._open_cube_dialog)
        self.vis_cube_btn.setEnabled(False) # Enabled only for NeuralCube
        btn_layout.addWidget(self.vis_cube_btn)

        self.vis_oracle_btn = QPushButton("🔮 Oracle")
        self.vis_oracle_btn.setObjectName("resetButton")
        self.vis_oracle_btn.setToolTip("Measure Uncertainty vs Settling Time")
        self.vis_oracle_btn.clicked.connect(self._run_oracle_analysis)
        btn_layout.addWidget(self.vis_oracle_btn)

        self.vis_align_btn = QPushButton("📐 Alignment")
        self.vis_align_btn.setObjectName("resetButton")
        self.vis_align_btn.setToolTip("Check alignment between EqProp and Backprop gradients")
        self.vis_align_btn.clicked.connect(self._run_alignment_check)
        btn_layout.addWidget(self.vis_align_btn)

        self.vis_robust_btn = QPushButton("🛡️ Robustness")
        self.vis_robust_btn.setObjectName("resetButton")
        self.vis_robust_btn.setToolTip("Run robustness analysis against noise")
        self.vis_robust_btn.clicked.connect(self._run_robustness_check)
        btn_layout.addWidget(self.vis_robust_btn)

        self.vis_clear_btn = QPushButton("🗑️ Clear")
        self.vis_clear_btn.setObjectName("resetButton") # Re-use styling
        self.vis_clear_btn.setToolTip("Clear plot history")
        self.vis_clear_btn.clicked.connect(self.clear_plots_signal.emit)
        btn_layout.addWidget(self.vis_clear_btn)
        left_panel.addLayout(btn_layout)

        self.vis_progress = QProgressBar()
        self.vis_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.vis_progress)

        # ETA Label
        self.vis_eta_label = QLabel("ETA: --:-- | Speed: -- it/s")
        self.vis_eta_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.vis_eta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.vis_eta_label)

        # Parameter count
        self.vis_param_label = QLabel("Parameters: --")
        self.vis_param_label.setStyleSheet("color: #00d4ff; font-weight: bold; padding: 5px;")
        left_panel.addWidget(self.vis_param_label)
        left_panel.addStretch()

        # Right panel: Plots
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)

        if HAS_PYQTGRAPH:
            metrics_group = QGroupBox("📊 Training Metrics")
            metrics_layout = QVBoxLayout(metrics_group)
            self.vis_loss_plot, self.vis_loss_curve = create_plot_widget("Loss", "Loss")
            metrics_layout.addWidget(self.vis_loss_plot)
            self.vis_acc_plot, self.vis_acc_curve = create_plot_widget("Accuracy", "Accuracy", yrange=(0, 1.0))
            metrics_layout.addWidget(self.vis_acc_plot)
            self.vis_lip_plot, self.vis_lip_curve = create_plot_widget("Lipschitz L", "Lipschitz L")
            self.vis_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            metrics_layout.addWidget(self.vis_lip_plot)
            right_panel.addWidget(metrics_group)

        # Stats
        stats_group = QGroupBox("📈 Results")
        stats_layout = QGridLayout(stats_group)
        stats_layout.addWidget(QLabel("Test Accuracy:"), 0, 0)
        self.vis_acc_label = QLabel("--")
        self.vis_acc_label.setObjectName("metricLabel")
        stats_layout.addWidget(self.vis_acc_label, 0, 1)
        stats_layout.addWidget(QLabel("Final Loss:"), 1, 0)
        self.vis_loss_label = QLabel("--")
        stats_layout.addWidget(self.vis_loss_label, 1, 1)
        stats_layout.addWidget(QLabel("Lipschitz:"), 2, 0)
        self.vis_lip_label = QLabel("--")
        stats_layout.addWidget(self.vis_lip_label, 2, 1)
        right_panel.addWidget(stats_group)

        # Weight Visualization
        if HAS_PYQTGRAPH:
            viz_group = QGroupBox("🎞️ Network State")
            viz_layout = QVBoxLayout(viz_group)

            # View Toggle
            toggle_layout = QHBoxLayout()
            toggle_layout.addWidget(QLabel("View:"))
            self.viz_mode_combo = QComboBox()
            self.viz_mode_combo.addItems(["Synaptic Weights (W)", "Synaptic Flow (ΔW)"])
            self.viz_mode_combo.setToolTip("Switch between viewing static weights or real-time update magnitudes (gradients).")
            toggle_layout.addWidget(self.viz_mode_combo)
            viz_layout.addLayout(toggle_layout)

            self.vis_weight_widgets = []
            self.vis_weight_labels = []
            self.vis_weights_container = QWidget()
            self.vis_weights_layout = QVBoxLayout(self.vis_weights_container)
            viz_layout.addWidget(self.vis_weights_container)
            right_panel.addWidget(viz_group)
        right_panel.addStretch()

    def _apply_preset(self, preset_name):
        """Apply a predefined configuration preset."""
        if preset_name == "Custom":
            return

        if preset_name == "Standard Backprop":
            # Find a BP model or set gradient to BPTT
            self.vis_grad_combo.setCurrentText("BPTT (Standard)")
            self.vis_steps_spin.setValue(5) # Minimal steps as they aren't used for BP usually or minimal relaxation
            self.vis_kernel_check.setChecked(False)

        elif preset_name == "Fast EqProp":
            self.vis_grad_combo.setCurrentText("Equilibrium (Implicit Diff)")
            self.vis_steps_spin.setValue(15) # Faster
            self.vis_kernel_check.setChecked(True) # Encourage speed

        elif preset_name == "Deep EqProp (Accurate)":
            self.vis_grad_combo.setCurrentText("Equilibrium (Implicit Diff)")
            self.vis_steps_spin.setValue(50) # Deep relaxation
            self.vis_kernel_check.setChecked(True)

    def _reset_defaults(self):
        """Reset all controls to default values."""
        self.vis_hidden_spin.setValue(256)
        self.vis_steps_spin.setValue(30)
        self.vis_batch_spin.setValue(64)
        self.vis_epochs_spin.setValue(10)
        self.vis_lr_spin.setValue(0.001)
        self.vis_compile_check.setChecked(True)
        self.vis_kernel_check.setChecked(False)
        self.vis_micro_check.setChecked(False)
        self.vis_dataset_combo.setCurrentIndex(0)
        self.vis_grad_combo.setCurrentIndex(0)
        self.vis_preset_combo.setCurrentIndex(0)

    def _update_vis_hyperparams(self, model_name):
        update_hyperparams_generic(self, model_name, self.vis_hyperparam_layout, self.vis_hyperparam_widgets, self.vis_hyperparam_group)

    def _update_model_desc(self, model_name):
        """Update model description label."""
        try:
            spec = get_model_spec(model_name)
            self.vis_desc_label.setText(spec.description)

            # Enable/Disable Cube Button based on model type
            if hasattr(self, 'vis_cube_btn'):
                self.vis_cube_btn.setEnabled(spec.model_type == "neural_cube")
        except Exception:
            self.vis_desc_label.setText("")

    def get_current_hyperparams(self):
        return get_current_hyperparams_generic(self.vis_hyperparam_widgets)

    def update_model_ref(self, model):
        """Store reference to the trained model."""
        self.model_ref = model

        # Check if NeuralCube to enable button
        try:
            if "NeuralCube" in model.__class__.__name__:
                self.vis_cube_btn.setEnabled(True)
        except:
            pass

    def _open_cube_dialog(self):
        """Open Cube Visualizer."""
        if self.model_ref is None:
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return

        try:
            # Run inference on one sample to get state
            from bioplausible.datasets import get_vision_dataset
            ds_name = self.vis_dataset_combo.currentText().lower().replace('-', '_')
            dataset = get_vision_dataset(ds_name, train=False, flatten=True) # NeuralCube takes flattened input

            # Get random sample
            idx = np.random.randint(0, len(dataset))
            x, _ = dataset[idx]
            x = torch.tensor(x).unsqueeze(0).to(next(self.model_ref.parameters()).device)

            # Run forward with trajectory
            self.model_ref.eval()
            with torch.no_grad():
                # NeuralCube forward returns out or (out, traj)
                out, traj = self.model_ref(x, return_trajectory=True)
                h_final = traj[-1] # [1, n_neurons]

            dlg = CubeVisualizerDialog(h_final, self.model_ref.cube_size, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Viz Error", str(e))
            import traceback
            traceback.print_exc()

    def _test_random_sample(self):
        """Run inference on a random sample."""
        if self.model_ref is None:
            QMessageBox.warning(self, "No Model", "No trained model available. Train or load a model first.")
            return

        try:
            from bioplausible.datasets import get_vision_dataset

            # Identify dataset
            ds_name = self.vis_dataset_combo.currentText().lower().replace('-', '_')

            # Determine if we need flattened input
            # Check model spec if possible
            # Simplified check:
            use_flatten = True
            try:
                # Assuming model_ref has 'model_type' or we infer from its structure
                # But safer to check the combobox as in dashboard.py
                model_name = self.vis_model_combo.currentText()
                spec = get_model_spec(model_name)
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            # Get test dataset
            # Note: We get the raw dataset to extract a sample easily
            dataset = get_vision_dataset(ds_name, train=False, flatten=use_flatten)

            idx = np.random.randint(0, len(dataset))
            x, y = dataset[idx]

            # Prepare input
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)

            x_input = x.unsqueeze(0).to(next(self.model_ref.parameters()).device)

            # Inference
            self.model_ref.eval()
            settling_steps = None

            with torch.no_grad():
                # Some models take 'steps' arg, others don't.
                # EqProp models usually have it in forward or use default
                try:
                    # Request dynamics to measure settling time
                    out, dynamics = self.model_ref(x_input, return_dynamics=True)
                    if dynamics and 'deltas' in dynamics:
                        # Calculate settling steps
                        # assuming deltas is list of scalars
                        threshold = 1e-2
                        deltas = dynamics['deltas']
                        settling_steps = len(deltas)
                        for i, d in enumerate(deltas):
                            if d < threshold:
                                settling_steps = i + 1
                                break
                except TypeError:
                    # Fallback
                    out = self.model_ref(x_input)

                pred = out.argmax(dim=1).item()

            # Show Dialog
            # We want to show the ORIGINAL image structure (28,28) or (3,32,32) not flattened
            if use_flatten:
                if 'mnist' in ds_name:
                    x_disp = x.view(28, 28)
                else:
                    x_disp = x.view(3, 32, 32)
            else:
                x_disp = x

            dlg = VisionInferenceDialog(x_disp, pred, y, settling_steps, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Inference Failed", str(e))
            import traceback
            traceback.print_exc()

    def _open_dream_dialog(self):
        """Open the dreaming dialog."""
        if self.model_ref is None:
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return

        try:
            # Determine shape
            ds_name = self.vis_dataset_combo.currentText().lower()
            if "mnist" in ds_name:
                shape = (1, 28, 28)
                # Check model type for flattening
                use_flatten = True
                try:
                    model_name = self.vis_model_combo.currentText()
                    spec = get_model_spec(model_name)
                    use_flatten = spec.model_type != "modern_conv_eqprop"
                except:
                    pass

                if use_flatten:
                    shape = (784,)
                else:
                    shape = (1, 28, 28)

            elif "cifar" in ds_name or "svhn" in ds_name:
                use_flatten = True
                try:
                    model_name = self.vis_model_combo.currentText()
                    spec = get_model_spec(model_name)
                    use_flatten = spec.model_type != "modern_conv_eqprop"
                except:
                    pass

                if use_flatten:
                    shape = (3072,)
                else:
                    shape = (3, 32, 32)
            else:
                shape = (784,)

            dlg = DreamDialog(self.model_ref, shape, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Failed to start", str(e))
            import traceback
            traceback.print_exc()

    def _run_robustness_check(self):
        """Run robustness analysis on the model."""
        if self.model_ref is None:
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return

        try:
            from bioplausible.datasets import get_vision_dataset
            from torch.utils.data import DataLoader

            # Setup
            ds_name = self.vis_dataset_combo.currentText().lower().replace('-', '_')
            use_flatten = True
            try:
                model_name = self.vis_model_combo.currentText()
                spec = get_model_spec(model_name)
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            # Use small subset for speed
            dataset = get_vision_dataset(ds_name, train=False, flatten=use_flatten)
            subset_indices = np.random.choice(len(dataset), 200, replace=False)
            subset = torch.utils.data.Subset(dataset, subset_indices)
            loader = DataLoader(subset, batch_size=50, shuffle=False)

            device = next(self.model_ref.parameters()).device
            self.model_ref.eval()

            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            results = []

            with torch.no_grad():
                for noise_sigma in noise_levels:
                    correct = 0
                    total = 0
                    for x, y in loader:
                        x = x.to(device)
                        y = y.to(device)

                        # Add noise
                        if noise_sigma > 0:
                            x = x + torch.randn_like(x) * noise_sigma

                        try:
                            out = self.model_ref(x)
                        except TypeError:
                            out = self.model_ref(x, steps=20)

                        pred = out.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)

                    acc = correct / total
                    results.append((noise_sigma, acc))

            from PyQt6.QtGui import QColor # Ensure QColor is available if not imported
            dlg = RobustnessDialog(results, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))
            import traceback
            traceback.print_exc()

    def _run_oracle_analysis(self):
        """Run Oracle Metric analysis (Noise vs Settling Time)."""
        if self.model_ref is None:
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return

        self.vis_oracle_btn.setEnabled(False)
        self.vis_oracle_btn.setText("Analyzing...")

        self.oracle_worker = OracleWorker(self.model_ref, self.vis_dataset_combo.currentText())
        self.oracle_worker.finished.connect(self._show_oracle_dialog)
        self.oracle_worker.error.connect(self._on_oracle_error)
        self.oracle_worker.start()

    def _show_oracle_dialog(self, results):
        self.vis_oracle_btn.setEnabled(True)
        self.vis_oracle_btn.setText("🔮 Oracle")

        dlg = OracleDialog(results, self)
        dlg.exec()

    def _on_oracle_error(self, msg):
        self.vis_oracle_btn.setEnabled(True)
        self.vis_oracle_btn.setText("🔮 Oracle")
        QMessageBox.critical(self, "Oracle Error", msg)

    def _run_alignment_check(self):
        """Run Gradient Alignment Check."""
        if self.model_ref is None:
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return

        self.vis_align_btn.setEnabled(False)
        self.vis_align_btn.setText("Checking...")

        self.align_worker = AlignmentWorker(self.model_ref, self.vis_dataset_combo.currentText())
        self.align_worker.finished.connect(self._show_align_dialog)
        self.align_worker.error.connect(self._on_align_error)
        self.align_worker.start()

    def _show_align_dialog(self, results):
        self.vis_align_btn.setEnabled(True)
        self.vis_align_btn.setText("📐 Alignment")

        dlg = AlignmentDialog(results, self)
        dlg.exec()

    def _on_align_error(self, msg):
        self.vis_align_btn.setEnabled(True)
        self.vis_align_btn.setText("📐 Alignment")
        QMessageBox.critical(self, "Alignment Error", msg)

    def update_theme(self, theme_colors, plot_colors):
        """Update plot colors based on theme."""
        if not HAS_PYQTGRAPH:
            return

        bg = theme_colors.get('background', '#0a0a0f')

        # Update Plots
        if hasattr(self, 'vis_loss_plot'):
            self.vis_loss_plot.setBackground(bg)
            self.vis_loss_curve.setPen(pg.mkPen(plot_colors.get('loss', 'w'), width=2))
            # Update axis labels color
            self.vis_loss_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.vis_loss_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

        if hasattr(self, 'vis_acc_plot'):
            self.vis_acc_plot.setBackground(bg)
            self.vis_acc_curve.setPen(pg.mkPen(plot_colors.get('accuracy', 'w'), width=2))
            self.vis_acc_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.vis_acc_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

        if hasattr(self, 'vis_lip_plot'):
            self.vis_lip_plot.setBackground(bg)
            self.vis_lip_curve.setPen(pg.mkPen(plot_colors.get('lipschitz', 'w'), width=2))
            self.vis_lip_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.vis_lip_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))