
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible_ui.dashboard_helpers import update_hyperparams_generic, get_current_hyperparams_generic
from bioplausible_ui.themes import PLOT_COLORS

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

        # Model Selection
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
        model_group = self._create_control_group("🧠 Model", model_controls)
        left_panel.addWidget(model_group)

        # Trigger initial update
        self._update_model_desc(self.vis_model_combo.currentText())

        # Dynamic Hyperparameters Group
        self.vis_hyperparam_group = QGroupBox("⚙️ Model Hyperparameters")
        from PyQt6.QtWidgets import QGridLayout
        self.vis_hyperparam_layout = QGridLayout(self.vis_hyperparam_group)
        self.vis_hyperparam_widgets = {}
        left_panel.addWidget(self.vis_hyperparam_group)
        self.vis_hyperparam_group.setVisible(False)

        self.vis_model_combo.currentTextChanged.connect(self._update_vis_hyperparams)

        # Dataset
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
        data_group = self._create_control_group("📚 Dataset", data_controls)
        left_panel.addWidget(data_group)

        # Training
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
        self.vis_grad_combo.setToolTip("Method for computing gradients:\n"
                                       "BPTT: Backprop Through Time (Exact, high memory)\n"
                                       "Equilibrium: Implicit Differentiation (O(1) memory)\n"
                                       "Contrastive: Explicit Hebbian Update (Bio-plausible)")

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
        train_group = self._create_control_group("⚙️ Training", train_controls)
        left_panel.addWidget(train_group)

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

        self.vis_clear_btn = QPushButton("🗑️ Clear")
        self.vis_clear_btn.setObjectName("resetButton") # Re-use styling
        self.vis_clear_btn.setToolTip("Clear plot history")
        self.vis_clear_btn.clicked.connect(self.clear_plots_signal.emit)
        btn_layout.addWidget(self.vis_clear_btn)
        left_panel.addLayout(btn_layout)

        self.vis_progress = QProgressBar()
        self.vis_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.vis_progress)

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
            viz_group = QGroupBox("🎞️ Weight Matrices")
            viz_layout = QVBoxLayout(viz_group)
            self.vis_weight_widgets = []
            self.vis_weight_labels = []
            self.vis_weights_container = QWidget()
            self.vis_weights_layout = QVBoxLayout(self.vis_weights_container)
            viz_layout.addWidget(self.vis_weights_container)
            right_panel.addWidget(viz_group)
        right_panel.addStretch()

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
