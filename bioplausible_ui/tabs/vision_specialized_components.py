"""
Specialized dialogs and workers for the Vision Tab in Bioplausible Trainer.
Separated to improve modularity and maintainability.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QProgressBar,
    QGroupBox, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
import torch
import torch.nn.functional as F
import numpy as np
import copy

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
    finished = pyqtSignal(dict)  # Returns dict of layer -> alignment
    error = pyqtSignal(str)

    def __init__(self, model, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset_name = dataset_name

    def run(self):
        try:
            from bioplausible.datasets import get_vision_dataset
            from torch.utils.data import DataLoader
            from bioplausible.models.registry import get_model_spec

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
                if hasattr(model_eq, 'learning_rate'):  # Config
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
        items = [(k, v) for k, v in alignments.items() if "weight" in k]  # Focus on weights
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
            self.img_view.setImage(slice_data.T)  # Transpose for display


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
                if self._stop:
                    break

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
                    x.clamp_(-2.5, 2.5)  # Assuming approx normalized range

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
    finished = pyqtSignal(list)  # List of (noise, settling_time) tuples
    error = pyqtSignal(str)

    def __init__(self, model, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset_name = dataset_name

    def run(self):
        try:
            from bioplausible.datasets import get_vision_dataset
            from torch.utils.data import DataLoader
            from bioplausible.models.registry import get_model_spec

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
                                threshold = 1e-2  # Heuristic
                                settled_at = len(deltas)
                                for i, d in enumerate(deltas):
                                    if d < threshold:
                                        settled_at = i + 1
                                        break
                                total_steps += settled_at
                            else:
                                total_steps += 30  # Max

                        except (TypeError, ValueError):
                            # Fallback if model doesn't support dynamics
                            # Just run forward
                            self.model(x)
                            total_steps += 30  # Assume max

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
                    corr_label.setStyleSheet("color: #00ff88;")  # Good positive correlation
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
        self.class_spin.setRange(0, 9)  # Assume 10 classes for now
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
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW -> HWC
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
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:  # Grayscale HWC
            img = img.squeeze(2)

        # Normalize to 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)

        h, w = img.shape[:2]
        if img.ndim == 2:  # Grayscale
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:  # RGB
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