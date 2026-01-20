
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel, QToolBox, QFrame,
    QDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

import torch
import torch.nn.functional as F
import numpy as np
import copy

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible_ui.dashboard_helpers import update_hyperparams_generic, get_current_hyperparams_generic
from bioplausible_ui.themes import PLOT_COLORS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class AlignmentWorker(QThread):
    """
    Background worker for Gradient Alignment Check.
    Compares model's update direction (EqProp) with Backprop gradient.
    """
    finished = pyqtSignal(dict) # Returns dict of layer -> alignment
    error = pyqtSignal(str)

    def __init__(self, model, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset_name = dataset_name

    def run(self):
        try:
            from bioplausible.datasets import get_vision_dataset
            from torch.utils.data import DataLoader

            # 1. Setup Data
            ds_name = self.dataset_name.lower().replace('-', '_')
            use_flatten = True
            try:
                spec = get_model_spec(self.model.config.name if hasattr(self.model, 'config') else "")
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            dataset = get_vision_dataset(ds_name, train=True, flatten=use_flatten)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            x, y = next(iter(loader))

            device = next(self.model.parameters()).device
            x, y = x.to(device), y.to(device)

            # 2. Backprop Reference
            # We need a clean copy to measure pure backprop gradient
            model_bp = copy.deepcopy(self.model)
            model_bp.train()
            model_bp.zero_grad()

            # Standard forward/backward
            # Handle models that require 'steps'
            try:
                out = model_bp(x)
            except TypeError:
                out = model_bp(x, steps=20)

            loss = F.cross_entropy(out, y)
            loss.backward()

            grads_bp = {}
            for name, param in model_bp.named_parameters():
                if param.grad is not None:
                    grads_bp[name] = param.grad.clone().flatten()

            # 3. EqProp Update
            # We need to capture what the model *would* update.
            # If the model has a 'train_step', it might apply optimizer step.
            # We clone to avoid messing up the main model state.
            model_eq = copy.deepcopy(self.model)
            model_eq.train()

            # Force SGD with LR=1.0 to measure update direction directly if internal optimizer used
            # If internal optimizer is created inside train_step, we might need to patch it.
            # Most EqProp models in this repo lazily create `internal_optimizer`.
            # We can force a manual SGD step if we can intercept gradients.

            # If gradient_method='contrastive', train_step computes grads.
            # If gradient_method='bptt', train_step might not exist or delegates.

            if hasattr(model_eq, 'train_step'):
                # We want to capture the update.
                # Snapshot weights
                w_before = {n: p.data.clone() for n, p in model_eq.named_parameters()}

                # Mock optimizer to simple SGD LR=1.0 to extract gradient from delta
                # But model creates its own optimizer usually.
                # Let's try to set `hebbian_lr` to 1.0 if possible
                if hasattr(model_eq, 'hebbian_lr'):
                    model_eq.hebbian_lr = 1.0
                if hasattr(model_eq, 'learning_rate'): # Config
                    model_eq.config.learning_rate = 1.0

                # Force reset internal optimizer if it exists
                if hasattr(model_eq, 'internal_optimizer'):
                    model_eq.internal_optimizer = torch.optim.SGD(model_eq.parameters(), lr=1.0)

                # Run step
                model_eq.train_step(x, y)

                # Measure update
                grads_eq = {}
                for name, param in model_eq.named_parameters():
                    w_new = param.data
                    w_old = w_before[name]
                    # delta = -lr * grad  =>  grad = -(delta/lr) = -(w_new - w_old) = w_old - w_new
                    grads_eq[name] = (w_old - w_new).flatten()
            else:
                # If no train_step, it uses external trainer (BPTT).
                # Alignment is 1.0 by definition.
                grads_eq = grads_bp

            # 4. Compute Alignment
            alignments = {}
            for name in grads_bp:
                if name in grads_eq:
                    g_bp = grads_bp[name]
                    g_eq = grads_eq[name]

                    if g_bp.numel() > 0 and g_eq.numel() > 0:
                        sim = F.cosine_similarity(g_bp.unsqueeze(0), g_eq.unsqueeze(0)).item()
                        alignments[name] = sim

            self.finished.emit(alignments)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

class AlignmentDialog(QDialog):
    """Dialog showing Gradient Alignment."""
    def __init__(self, alignments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gradient Alignment Analysis")
        self.resize(500, 600)

        layout = QVBoxLayout(self)

        header = QLabel("EqProp vs Backprop Alignment")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        desc = QLabel(
            "Cosine similarity between Equilibrium Propagation updates and standard Backpropagation gradients.\n"
            "1.0 = Perfect Alignment (Mathematically Identical)\n"
            "0.0 = Orthogonal\n"
            "-1.0 = Anti-Aligned"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a0b0;")
        layout.addWidget(desc)

        # Table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Layer", "Alignment (Cos)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Filter and sort
        items = [(k, v) for k, v in alignments.items() if "weight" in k] # Focus on weights
        table.setRowCount(len(items))

        avg_align = 0
        for i, (name, val) in enumerate(items):
            # Shorten name
            short_name = name.replace(".weight", "").replace("layers.", "L").replace("parametrizations.original", "")

            table.setItem(i, 0, QTableWidgetItem(short_name))

            val_item = QTableWidgetItem(f"{val:.4f}")
            if val > 0.9:
                val_item.setForeground(QColor("#00ff88"))
            elif val > 0.5:
                val_item.setForeground(QColor("#f1c40f"))
            else:
                val_item.setForeground(QColor("#ff5555"))
            table.setItem(i, 1, val_item)
            avg_align += val

        if items:
            avg_align /= len(items)

        layout.addWidget(table)

        # Global Score
        score_label = QLabel(f"Global Alignment: {avg_align:.4f}")
        score_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if avg_align > 0.9:
            score_label.setStyleSheet("color: #00ff88; border: 2px solid #00ff88; padding: 10px; border-radius: 5px;")
        else:
            score_label.setStyleSheet("color: #f1c40f; border: 2px solid #f1c40f; padding: 10px; border-radius: 5px;")

        layout.addWidget(score_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class CubeVisualizerDialog(QDialog):
    """Dialog to visualize 3D Neural Cube slices."""
    def __init__(self, h_tensor, cube_size, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Neural Cube Visualization ({cube_size}x{cube_size}x{cube_size})")
        self.resize(600, 600)
        self.h = h_tensor.cpu().reshape(cube_size, cube_size, cube_size).numpy()
        self.cube_size = cube_size

        layout = QVBoxLayout(self)

        header = QLabel("3D Activation Topography")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Slice control
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Z-Slice:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, cube_size - 1)
        self.slider.setValue(cube_size // 2)
        self.slider.valueChanged.connect(self._update_slice)
        slice_layout.addWidget(self.slider)
        self.slice_label = QLabel(f"{cube_size // 2}")
        slice_layout.addWidget(self.slice_label)
        layout.addLayout(slice_layout)

        if HAS_PYQTGRAPH:
            self.img_view = pg.ImageView()
            self.img_view.ui.histogram.hide()
            self.img_view.ui.roiBtn.hide()
            self.img_view.ui.menuBtn.hide()
            # Set colormap (Fire)
            self.img_view.setPredefinedGradient("thermal")
            layout.addWidget(self.img_view)

            self._update_slice(self.slider.value())
        else:
            layout.addWidget(QLabel("PyQtGraph required for visualization"))

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def _update_slice(self, z):
        self.slice_label.setText(str(z))
        if HAS_PYQTGRAPH:
            # Get slice z
            # Shape [Z, Y, X]
            slice_data = self.h[z, :, :]
            self.img_view.setImage(slice_data.T) # Transpose for display

class DreamWorker(QThread):
    """
    Background worker for 'Dreaming' (Input Optimization).
    Optimizes input image to maximize activation of target class.
    """
    progress = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, target_class, input_shape, steps=100, lr=0.1, parent=None):
        super().__init__(parent)
        self.model = model
        self.target = target_class
        self.shape = input_shape
        self.steps = steps
        self.lr = lr
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            device = next(self.model.parameters()).device
            # Start from gray noise
            x = torch.randn(1, *self.shape, device=device) * 0.1
            x.requires_grad_(True)

            optimizer = torch.optim.SGD([x], lr=self.lr)

            # Switch model to eval (we optimize x, not weights)
            self.model.eval()

            for i in range(self.steps):
                if self._stop: break

                optimizer.zero_grad()

                # Forward pass
                # Handle models that might require steps arg
                try:
                    out = self.model(x)
                except TypeError:
                    out = self.model(x, steps=30)

                # Loss: Maximize target logit
                # We minimize negative target score
                loss = -out[0, self.target]

                loss.backward()
                optimizer.step()

                # Regularization / Constraints
                with torch.no_grad():
                    # Blur or jitter could be added here for better viz
                    x.clamp_(-2.5, 2.5) # Assuming approx normalized range

                # Emit progress
                if i % 2 == 0:
                    img_np = x.detach().cpu().numpy().squeeze()
                    self.progress.emit(img_np)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

class OracleWorker(QThread):
    """
    Background worker for Oracle Metric analysis.
    Measures settling time vs uncertainty (noise).
    """
    finished = pyqtSignal(list) # List of (noise, settling_time) tuples
    error = pyqtSignal(str)

    def __init__(self, model, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset_name = dataset_name

    def run(self):
        try:
            from bioplausible.datasets import get_vision_dataset
            from torch.utils.data import DataLoader

            # Setup data
            ds_name = self.dataset_name.lower().replace('-', '_')
            use_flatten = True
            try:
                # Infer flattening from model if possible, otherwise assume flat for MLP
                spec = get_model_spec(self.model.config.name if hasattr(self.model, 'config') else "")
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            dataset = get_vision_dataset(ds_name, train=False, flatten=use_flatten)
            # Small subset
            indices = np.random.choice(len(dataset), 50, replace=False)
            subset = torch.utils.data.Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=10, shuffle=False)

            device = next(self.model.parameters()).device
            self.model.eval()

            results = []
            noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

            with torch.no_grad():
                for noise in noise_levels:
                    total_steps = 0
                    count = 0

                    for x, y in loader:
                        x = x.to(device)
                        if noise > 0:
                            x = x + torch.randn_like(x) * noise

                        # Run with dynamics tracking
                        # Ensure we request dynamics
                        try:
                            # Pass return_dynamics=True if supported
                            out, dynamics = self.model(x, return_dynamics=True)

                            deltas = dynamics.get('deltas', [])
                            if deltas:
                                # Find step where delta drops below threshold (e.g. 1e-3)
                                # Normalize delta?
                                threshold = 1e-2 # Heuristic
                                settled_at = len(deltas)
                                for i, d in enumerate(deltas):
                                    if d < threshold:
                                        settled_at = i + 1
                                        break
                                total_steps += settled_at
                            else:
                                total_steps += 30 # Max

                        except (TypeError, ValueError):
                            # Fallback if model doesn't support dynamics
                            # Just run forward
                            self.model(x)
                            total_steps += 30 # Assume max

                        count += 1

                    avg_steps = total_steps / count if count > 0 else 30
                    results.append((noise, avg_steps))

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

class OracleDialog(QDialog):
    """Dialog for Oracle Metric (Uncertainty Analysis)."""
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Oracle Metric: Uncertainty vs Time")
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        header = QLabel("🔮 Oracle Analysis")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        desc = QLabel(
            "Visualizing the correlation between Uncertainty (Noise) and Processing Time (Settling Steps).\n"
            "A Bio-Plausible network should take longer to resolve ambiguous inputs."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a0b0;")
        layout.addWidget(desc)

        if HAS_PYQTGRAPH:
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('#0a0a0f')
            plot_widget.setLabel('left', "Settling Time (Steps)")
            plot_widget.setLabel('bottom', "Noise Level (σ)")
            plot_widget.showGrid(x=True, y=True, alpha=0.3)

            # Plot data
            x = [r[0] for r in results]
            y = [r[1] for r in results]

            # Plot points and line
            plot_widget.plot(x, y, symbol='o', pen=pg.mkPen('#00d4ff', width=3), symbolBrush='#00d4ff')

            layout.addWidget(plot_widget)

            # Calculate correlation
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                corr_label = QLabel(f"Correlation: {corr:.3f}")
                corr_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                if corr > 0.5:
                    corr_label.setStyleSheet("color: #00ff88;") # Good positive correlation
                else:
                    corr_label.setStyleSheet("color: #f1c40f;")
                layout.addWidget(corr_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class DreamDialog(QDialog):
    """Dialog for Generative Dreaming."""
    def __init__(self, model, input_shape, parent=None):
        super().__init__(parent)
        self.model = model
        self.input_shape = input_shape
        self.worker = None

        self.setWindowTitle("Associative Dreaming (Generative Attractor)")
        self.setFixedSize(500, 600)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Dreaming: Invert Network")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        desc = QLabel("Optimizes the input image to maximize the network's confidence for a specific class. Reveals what features the network has learned.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a0b0; margin-bottom: 10px;")
        layout.addWidget(desc)

        # Controls
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Target Class:"))
        self.class_spin = QSpinBox()
        self.class_spin.setRange(0, 9) # Assume 10 classes for now
        ctrl_layout.addWidget(self.class_spin)

        ctrl_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 500)
        self.steps_spin.setValue(100)
        ctrl_layout.addWidget(self.steps_spin)

        layout.addLayout(ctrl_layout)

        # Image Display
        self.img_label = QLabel()
        self.img_label.setMinimumSize(300, 300)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background-color: #000; border: 1px solid #333;")
        layout.addWidget(self.img_label)

        # Start/Stop
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Dreaming")
        self.start_btn.setStyleSheet("background-color: #8e44ad; font-weight: bold; padding: 10px;")
        self.start_btn.clicked.connect(self._start_dreaming)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_dreaming)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

    def _start_dreaming(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.class_spin.setEnabled(False)

        self.worker = DreamWorker(
            self.model,
            self.class_spin.value(),
            self.input_shape,
            steps=self.steps_spin.value()
        )
        self.worker.progress.connect(self._update_image)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_dreaming(self):
        if self.worker:
            self.worker.stop()

    def _update_image(self, img):
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)

        # Handle shapes
        if img.ndim == 3 and img.shape[0] in [1, 3]: # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)

        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.data, w, h, 3*w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.img_label.setPixmap(pixmap)

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.class_spin.setEnabled(True)

    def _on_error(self, msg):
        self._on_finished()
        QMessageBox.warning(self, "Dream Error", msg)

    def closeEvent(self, event):
        self._stop_dreaming()
        event.accept()


class VisionInferenceDialog(QDialog):
    """Dialog to show inference results."""
    def __init__(self, image_tensor, prediction, ground_truth, settling_steps=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inference Result")
        self.setFixedSize(400, 550)

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

        # Oracle Metric Display
        if settling_steps is not None:
            oracle_label = QLabel(f"Settling Time: {settling_steps} steps")
            oracle_label.setFont(QFont("Segoe UI", 12))
            oracle_label.setStyleSheet("color: #00d4ff; margin-top: 5px;")
            oracle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(oracle_label)

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
