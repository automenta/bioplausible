
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QPushButton, QProgressBar, QLabel
)
from PyQt6.QtCore import pyqtSignal

from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible_ui.themes import PLOT_COLORS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class RLTab(QWidget):
    """Reinforcement Learning Tab."""

    start_training_signal = pyqtSignal(str) # Mode ('rl')
    stop_training_signal = pyqtSignal()

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

        # Environment
        self.rl_env_combo = QComboBox()
        self.rl_env_combo.addItems(["CartPole-v1", "Acrobot-v1", "MountainCar-v0"])

        # Model/Algo Selection
        self.rl_algo_combo = QComboBox()
        rl_items = []
        for spec in MODEL_REGISTRY:
             if spec.task_compat is None or "rl" in spec.task_compat:
                 rl_items.append(spec.name)
        self.rl_algo_combo.addItems(rl_items)
        self.rl_algo_combo.currentTextChanged.connect(self._update_rl_controls)

        # Gradient Method
        self.rl_grad_combo = QComboBox()
        self.rl_grad_combo.addItems(["equilibrium", "bptt"])

        self.rl_hidden_spin = QSpinBox()
        self.rl_hidden_spin.setRange(32, 512)
        self.rl_hidden_spin.setValue(64)

        self.rl_steps_spin = QSpinBox()
        self.rl_steps_spin.setRange(5, 50)
        self.rl_steps_spin.setValue(20)

        model_controls = [
            ("Environment:", self.rl_env_combo),
            ("Algorithm:", self.rl_algo_combo),
            ("Gradient Mode:", self.rl_grad_combo),
            ("Hidden Dim:", self.rl_hidden_spin),
            ("Eq Steps:", self.rl_steps_spin)
        ]
        model_group = self._create_control_group("🎮 Task & Model", model_controls)
        left_panel.addWidget(model_group)

        # Training
        self.rl_episodes_spin = QSpinBox()
        self.rl_episodes_spin.setRange(10, 5000)
        self.rl_episodes_spin.setValue(200)

        self.rl_lr_spin = QDoubleSpinBox()
        self.rl_lr_spin.setRange(0.0001, 0.1)
        self.rl_lr_spin.setValue(0.005)
        self.rl_lr_spin.setDecimals(4)

        self.rl_gamma_spin = QDoubleSpinBox()
        self.rl_gamma_spin.setRange(0.0, 1.0)
        self.rl_gamma_spin.setValue(0.99)
        self.rl_gamma_spin.setSingleStep(0.01)

        train_controls = [
            ("Episodes:", self.rl_episodes_spin),
            ("Learning Rate:", self.rl_lr_spin),
            ("Gamma (Discount):", self.rl_gamma_spin),
        ]
        train_group = self._create_control_group("⚙️ Training", train_controls)
        left_panel.addWidget(train_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.rl_train_btn = QPushButton("▶ Train")
        self.rl_train_btn.clicked.connect(lambda: self.start_training_signal.emit('rl'))
        btn_layout.addWidget(self.rl_train_btn)

        self.rl_stop_btn = QPushButton("⏹ Stop")
        self.rl_stop_btn.setEnabled(False)
        self.rl_stop_btn.clicked.connect(self.stop_training_signal.emit)
        btn_layout.addWidget(self.rl_stop_btn)
        left_panel.addLayout(btn_layout)

        # Progress
        self.rl_progress = QProgressBar()
        self.rl_progress.setFormat("Episode %v / %m")
        left_panel.addWidget(self.rl_progress)
        left_panel.addStretch()

        # Right panel: Plots
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)

        if HAS_PYQTGRAPH:
            metrics_group = QGroupBox("📊 RL Metrics")
            metrics_layout = QVBoxLayout(metrics_group)
            self.rl_reward_plot, self.rl_reward_curve = self._create_plot_widget("Total Reward", "Reward", xlabel="Episode")
            self.rl_avg_reward_curve = self.rl_reward_plot.plot(pen=pg.mkPen('y', width=2))
            metrics_layout.addWidget(self.rl_reward_plot)
            self.rl_loss_plot, self.rl_loss_curve = self._create_plot_widget("Loss", "Loss", xlabel="Episode")
            metrics_layout.addWidget(self.rl_loss_plot)
            right_panel.addWidget(metrics_group)

        # Stats
        from PyQt6.QtWidgets import QGridLayout
        stats_group = QGroupBox("📈 Results")
        stats_layout = QGridLayout(stats_group)
        stats_layout.addWidget(QLabel("Avg Reward (last 50):"), 0, 0)
        self.rl_avg_label = QLabel("--")
        self.rl_avg_label.setObjectName("metricLabel")
        stats_layout.addWidget(self.rl_avg_label, 0, 1)
        right_panel.addWidget(stats_group)

    def _update_rl_controls(self, text):
        if "Backprop" in text:
            self.rl_grad_combo.setEnabled(False)
        else:
            self.rl_grad_combo.setEnabled(True)
            if "(EqProp)" in text:
                self.rl_grad_combo.setCurrentText("equilibrium")
            elif "(BPTT)" in text:
                self.rl_grad_combo.setCurrentText("bptt")
