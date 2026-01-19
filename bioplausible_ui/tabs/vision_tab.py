
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel, QToolBox, QFrame,
    QDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

import torch
import numpy as np

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible_ui.dashboard_helpers import update_hyperparams_generic, get_current_hyperparams_generic
from bioplausible_ui.themes import PLOT_COLORS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class VisionInferenceDialog(QDialog):
    """Dialog to show inference results."""
    def __init__(self, image_tensor, prediction, ground_truth, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inference Result")
        self.setFixedSize(400, 500)

        layout = QVBoxLayout(self)

        # Display Image
        # Convert tensor (C, H, W) or (H, W) to pixmap
        img = image_tensor.cpu().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]: # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1: # Grayscale HWC
            img = img.squeeze(2)

        # Normalize to 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)

        h, w = img.shape[:2]
        if img.ndim == 2: # Grayscale
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else: # RGB
            qimg = QImage(img.data, w, h, 3*w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio)

        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_label)

        # Display Result
        res_label = QLabel(f"Prediction: {prediction}\nGround Truth: {ground_truth}")
        res_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        res_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if prediction == ground_truth:
            res_label.setStyleSheet("color: #00ff88; margin-top: 20px;")
        else:
             res_label.setStyleSheet("color: #ff5555; margin-top: 20px;")

        layout.addWidget(res_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class RobustnessDialog(QDialog):
    """Dialog to show robustness check results."""
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robustness Analysis")
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        label = QLabel("Noise Tolerance Analysis (Gaussian Noise)")
        label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(label)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Noise Level (σ)", "Accuracy"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setRowCount(len(results))

        for i, (noise, acc) in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(f"{noise:.1f}"))

            acc_item = QTableWidgetItem(f"{acc:.1%}")
            if acc > 0.8:
                acc_item.setForeground(QColor("#00ff88"))
            elif acc > 0.5:
                acc_item.setForeground(QColor("#f1c40f"))
            else:
                acc_item.setForeground(QColor("#ff5555"))

            self.table.setItem(i, 1, acc_item)

        layout.addWidget(self.table)

        # Summary
        drops = [results[0][1] - r[1] for r in results[1:]]
        avg_drop = sum(drops) / len(drops) if drops else 0.0

        summary = "Robustness Score: "
        if avg_drop < 0.1:
            summary += "<span style='color: #00ff88'>Excellent</span>"
        elif avg_drop < 0.3:
            summary += "<span style='color: #f1c40f'>Good</span>"
        else:
            summary += "<span style='color: #ff5555'>Poor</span>"

        sum_label = QLabel(summary)
        sum_label.setFont(QFont("Segoe UI", 14))
        layout.addWidget(sum_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class VisionTab(QWidget):
    """Vision Training Tab."""

    start_training_signal = pyqtSignal(str) # Mode ('vision')
    stop_training_signal = pyqtSignal()
    clear_plots_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_ref = None
        self._setup_ui()

    def _create_control_group(self, title, controls):
        from PyQt6.QtWidgets import QGridLayout
        group = QGroupBox(title)
        layout = QGridLayout(group)
        for i, (label, widget) in enumerate(controls):
            layout.addWidget(QLabel(label), i, 0)
            layout.addWidget(widget, i, 1)
        return group

    def _create_plot_widget(self, title, ylabel, xlabel='Epoch', yrange=None):
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('#0a0a0f')
        plot_widget.setLabel('left', ylabel, color=PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'))
        plot_widget.setLabel('bottom', xlabel)
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        if yrange:
            plot_widget.setYRange(*yrange)
        curve = plot_widget.plot(pen=pg.mkPen(PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'), width=2))
        return plot_widget, curve

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

        # Helper to add grid controls
        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        for i, (label, widget) in enumerate(model_controls):
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(widget, i, 1)
        model_layout.addLayout(grid)

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

        data_grid = QGridLayout()
        for i, (label, widget) in enumerate(data_controls):
            data_grid.addWidget(QLabel(label), i, 0)
            data_grid.addWidget(widget, i, 1)
        data_layout.addLayout(data_grid)
        data_layout.addStretch()

        self.toolbox.addItem(data_widget, "📚 Data Configuration")

        # --- Section 3: Training ---
        train_widget = QWidget()
        train_layout = QVBoxLayout(train_widget)

        self.vis_epochs_spin = QSpinBox()
        self.vis_epochs_spin.setRange(1, 100)
        self.vis_epochs_spin.setValue(10)
        self.vis_epochs_spin.setToolTip("Total number of passes over the dataset")

        self.vis_lr_spin = QDoubleSpinBox()
        self.vis_lr_spin.setRange(0.0001, 0.1)
        self.vis_lr_spin.setValue(0.001)
        self.vis_lr_spin.setDecimals(4)
        self.vis_lr_spin.setToolTip("Step size for optimizer")

        self.vis_grad_combo = QComboBox()
        self.vis_grad_combo.addItems(["BPTT (Standard)", "Equilibrium (Implicit Diff)", "Contrastive (Hebbian)"])
        self.vis_grad_combo.setToolTip("Gradient Calculation Method")

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

        train_grid = QGridLayout()
        for i, (label, widget) in enumerate(train_controls):
            train_grid.addWidget(QLabel(label), i, 0)
            train_grid.addWidget(widget, i, 1)
        train_layout.addLayout(train_grid)
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

        self.vis_reset_btn = QPushButton("↺ Reset")
        self.vis_reset_btn.setObjectName("resetButton")
        self.vis_reset_btn.setToolTip("Reset all hyperparameters to default values")
        self.vis_reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.vis_reset_btn)

        self.vis_test_btn = QPushButton("👁️ Test")
        self.vis_test_btn.setObjectName("resetButton")
        self.vis_test_btn.setToolTip("Test model on random sample")
        self.vis_test_btn.clicked.connect(self._test_random_sample)
        btn_layout.addWidget(self.vis_test_btn)

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
            self.vis_loss_plot, self.vis_loss_curve = self._create_plot_widget("Loss", "Loss")
            metrics_layout.addWidget(self.vis_loss_plot)
            self.vis_acc_plot, self.vis_acc_curve = self._create_plot_widget("Accuracy", "Accuracy", yrange=(0, 1.0))
            metrics_layout.addWidget(self.vis_acc_plot)
            self.vis_lip_plot, self.vis_lip_curve = self._create_plot_widget("Lipschitz L", "Lipschitz L")
            self.vis_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            metrics_layout.addWidget(self.vis_lip_plot)
            right_panel.addWidget(metrics_group)

        # Stats
        from PyQt6.QtWidgets import QGridLayout
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
        except Exception:
            self.vis_desc_label.setText("")

    def get_current_hyperparams(self):
        return get_current_hyperparams_generic(self.vis_hyperparam_widgets)

    def update_model_ref(self, model):
        """Store reference to the trained model."""
        self.model_ref = model

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
            with torch.no_grad():
                # Some models take 'steps' arg, others don't.
                # EqProp models usually have it in forward or use default
                try:
                    out = self.model_ref(x_input)
                except TypeError:
                    # Try with steps if required, though default should handle it
                    out = self.model_ref(x_input, steps=20)

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

            dlg = VisionInferenceDialog(x_disp, pred, y, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Inference Failed", str(e))
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

            # Progress dialog could be nice, but we'll be quick hopefully

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
