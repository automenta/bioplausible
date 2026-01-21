
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QPushButton, QProgressBar, QLabel, QDialog
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont

import torch
import numpy as np
import gymnasium as gym

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible_ui_old.themes import PLOT_COLORS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class PlaybackWorker(QThread):
    finished = pyqtSignal(float, list) # total_reward, frames
    error = pyqtSignal(str)

    def __init__(self, model, env_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.env_name = env_name

    def run(self):
        try:
            env = gym.make(self.env_name, render_mode="rgb_array")
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            frames = []

            while not (done or truncated) and steps < 500:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                # Action
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    if hasattr(self.model, 'device'):
                        state_t = state_t.to(self.model.device)

                    q_values = self.model(state_t)
                    action = q_values.argmax().item()

                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

            env.close()
            self.finished.emit(total_reward, frames)
        except Exception as e:
            self.error.emit(str(e))

class RLPlaybackDialog(QDialog):
    """Dialog to playback agent performance."""
    def __init__(self, model, env_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Watching Agent: {env_name}")
        self.setFixedSize(600, 500)
        self.model = model
        self.env_name = env_name
        self.frames = []
        self.current_frame = 0

        layout = QVBoxLayout(self)

        self.image_label = QLabel("Running Episode...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label)

        self.status_label = QLabel("Simulating...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Indeterminate
        layout.addWidget(self.progress)

        btn = QPushButton("Close")
        btn.clicked.connect(self.close)
        layout.addWidget(btn)

        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

        # Run episode in background
        self.worker = PlaybackWorker(self.model, self.env_name)
        self.worker.finished.connect(self._on_episode_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_episode_finished(self, reward, frames):
        self.frames = frames
        self.progress.setVisible(False)
        self.status_label.setText(f"Episode Finished! Reward: {reward:.1f}. Replaying...")

        if self.frames:
            self.timer.start(50) # 20 FPS
        else:
            self.image_label.setText("No frames captured (rendering failed).")

    def _on_error(self, err):
        self.progress.setVisible(False)
        self.status_label.setText("Error occurred.")
        self.image_label.setText(f"Simulation Error:\n{err}")

    def _next_frame(self):
        if not self.frames: return

        frame = self.frames[self.current_frame]
        h, w, c = frame.shape
        qimg = QImage(frame.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        self.current_frame = (self.current_frame + 1) % len(self.frames)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.terminate() # Force kill if closed
            self.worker.wait()
        event.accept()

class RLTab(QWidget):
    """Reinforcement Learning Tab."""

    start_training_signal = pyqtSignal(str) # Mode ('rl')
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

        # Environment
        self.rl_env_combo = QComboBox()
        self.rl_env_combo.addItems(["CartPole-v1", "Acrobot-v1", "MountainCar-v0"])
        self.rl_env_combo.setToolTip("Gymnasium environment to train on")

        # Model/Algo Selection
        self.rl_algo_combo = QComboBox()
        rl_items = []
        for spec in MODEL_REGISTRY:
             if spec.task_compat is None or "rl" in spec.task_compat:
                 rl_items.append(spec.name)
        self.rl_algo_combo.addItems(rl_items)
        self.rl_algo_combo.currentTextChanged.connect(self._update_rl_controls)
        self.rl_algo_combo.currentTextChanged.connect(self._update_model_desc)

        self.rl_desc_label = QLabel("")
        self.rl_desc_label.setWordWrap(True)
        self.rl_desc_label.setStyleSheet("color: #a0a0b0; font-size: 11px; font-style: italic; margin-bottom: 5px;")

        # Gradient Method
        self.rl_grad_combo = QComboBox()
        self.rl_grad_combo.addItems(["equilibrium", "bptt"])
        self.rl_grad_combo.setToolTip("Method for computing gradients:\n"
                                      "equilibrium: Implicit Differentiation (Memory efficient)\n"
                                      "bptt: Backprop Through Time (Exact)")

        self.rl_hidden_spin = QSpinBox()
        self.rl_hidden_spin.setRange(32, 512)
        self.rl_hidden_spin.setValue(64)
        self.rl_hidden_spin.setToolTip("Dimension of hidden state vectors")

        self.rl_steps_spin = QSpinBox()
        self.rl_steps_spin.setRange(5, 50)
        self.rl_steps_spin.setValue(20)
        self.rl_steps_spin.setToolTip("Number of equilibrium steps per forward pass")

        model_controls = [
            ("Environment:", self.rl_env_combo),
            ("Algorithm:", self.rl_algo_combo),
            ("", self.rl_desc_label),
            ("Gradient Mode:", self.rl_grad_combo),
            ("Hidden Dim:", self.rl_hidden_spin),
            ("Eq Steps:", self.rl_steps_spin)
        ]
        model_group = self._create_control_group("🎮 Task & Model", model_controls)
        left_panel.addWidget(model_group)

        # Trigger initial update
        self._update_model_desc(self.rl_algo_combo.currentText())

        # Training
        self.rl_episodes_spin = QSpinBox()
        self.rl_episodes_spin.setRange(10, 5000)
        self.rl_episodes_spin.setValue(200)
        self.rl_episodes_spin.setToolTip("Total number of episodes to train")

        self.rl_lr_spin = QDoubleSpinBox()
        self.rl_lr_spin.setRange(0.0001, 0.1)
        self.rl_lr_spin.setValue(0.005)
        self.rl_lr_spin.setDecimals(4)
        self.rl_lr_spin.setToolTip("Learning rate for optimizer")

        self.rl_gamma_spin = QDoubleSpinBox()
        self.rl_gamma_spin.setRange(0.0, 1.0)
        self.rl_gamma_spin.setValue(0.99)
        self.rl_gamma_spin.setSingleStep(0.01)
        self.rl_gamma_spin.setToolTip("Discount factor for future rewards")

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

        self.rl_reset_btn = QPushButton("↺ Reset")
        self.rl_reset_btn.setObjectName("resetButton")
        self.rl_reset_btn.setToolTip("Reset all hyperparameters to default values")
        self.rl_reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.rl_reset_btn)

        self.rl_watch_btn = QPushButton("👁️ Watch")
        self.rl_watch_btn.setObjectName("resetButton")
        self.rl_watch_btn.setToolTip("Watch agent play one episode")
        self.rl_watch_btn.clicked.connect(self._watch_agent)
        btn_layout.addWidget(self.rl_watch_btn)

        self.rl_clear_btn = QPushButton("🗑️ Clear")
        self.rl_clear_btn.setObjectName("resetButton")
        self.rl_clear_btn.setToolTip("Clear plot history")
        self.rl_clear_btn.clicked.connect(self.clear_plots_signal.emit)
        btn_layout.addWidget(self.rl_clear_btn)
        left_panel.addLayout(btn_layout)

        # Progress
        self.rl_progress = QProgressBar()
        self.rl_progress.setFormat("Episode %v / %m")
        left_panel.addWidget(self.rl_progress)

        # ETA Label
        self.rl_eta_label = QLabel("ETA: --:-- | Speed: -- ep/s")
        self.rl_eta_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.rl_eta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.rl_eta_label)
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

    def _reset_defaults(self):
        """Reset all controls to default values."""
        self.rl_hidden_spin.setValue(64)
        self.rl_steps_spin.setValue(20)
        self.rl_episodes_spin.setValue(200)
        self.rl_lr_spin.setValue(0.005)
        self.rl_gamma_spin.setValue(0.99)
        self.rl_env_combo.setCurrentIndex(0)
        # RL Algo combo and grad combo might depend on each other,
        # but resetting them to default (likely index 0) is a safe bet or specific index
        self.rl_algo_combo.setCurrentIndex(0)
        self.rl_grad_combo.setCurrentIndex(0)

    def _update_rl_controls(self, text):
        if "Backprop" in text:
            self.rl_grad_combo.setEnabled(False)
        else:
            self.rl_grad_combo.setEnabled(True)
            if "(EqProp)" in text:
                self.rl_grad_combo.setCurrentText("equilibrium")
            elif "(BPTT)" in text:
                self.rl_grad_combo.setCurrentText("bptt")

    def _update_model_desc(self, model_name):
        """Update model description label."""
        try:
            spec = get_model_spec(model_name)
            self.rl_desc_label.setText(spec.description)
        except Exception:
            self.rl_desc_label.setText("")

    def update_model_ref(self, model):
        """Store reference to trained model."""
        self.model_ref = model

    def _watch_agent(self):
        """Launch playback dialog."""
        if self.model_ref is None:
            # Try to load default if not trained? No, better warn.
            # But for UX, let's create a temporary one if user just wants to see it fail?
            # No, standard is to warn.
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Agent", "Train an agent first!")
            return

        env_name = self.rl_env_combo.currentText()
        dlg = RLPlaybackDialog(self.model_ref, env_name, self)
        dlg.exec()

    def update_theme(self, theme_colors, plot_colors):
        """Update plot colors based on theme."""
        if not HAS_PYQTGRAPH:
            return

        bg = theme_colors.get('background', '#0a0a0f')

        # Update Plots
        if hasattr(self, 'rl_reward_plot'):
            self.rl_reward_plot.setBackground(bg)
            self.rl_reward_curve.setPen(pg.mkPen(plot_colors.get('accuracy', 'y'), width=2)) # Reward ~= Accuracy color
            self.rl_avg_reward_curve.setPen(pg.mkPen(plot_colors.get('perplexity', 'y'), width=2))
            self.rl_reward_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.rl_reward_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

        if hasattr(self, 'rl_loss_plot'):
            self.rl_loss_plot.setBackground(bg)
            self.rl_loss_curve.setPen(pg.mkPen(plot_colors.get('loss', 'r'), width=2))
            self.rl_loss_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.rl_loss_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
