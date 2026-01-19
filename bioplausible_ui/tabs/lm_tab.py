
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel, QSlider, QTextEdit,
    QToolBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible_ui.dashboard_helpers import update_hyperparams_generic, get_current_hyperparams_generic
from bioplausible_ui.generation import count_parameters, format_parameter_count, UniversalGenerator
from bioplausible_ui.themes import PLOT_COLORS
from PyQt6.QtWidgets import QApplication

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class LMTab(QWidget):
    """Language Modeling Tab."""

    start_training_signal = pyqtSignal(str) # Mode ('lm')
    stop_training_signal = pyqtSignal()
    clear_plots_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.generator = None
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
        self.lm_preset_combo = QComboBox()
        self.lm_preset_combo.addItems(["Custom", "Small (Fast)", "Medium (Balanced)", "Large (Accurate)"])
        self.lm_preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.lm_preset_combo)
        left_panel.addLayout(preset_layout)

        # Toolbox
        self.toolbox = QToolBox()
        left_panel.addWidget(self.toolbox)

        # --- Section 1: Model ---
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)

        self.lm_model_combo = QComboBox()
        lm_items = []
        for spec in MODEL_REGISTRY:
            if spec.task_compat is None or "lm" in spec.task_compat:
                lm_items.append(f"{spec.name}")
        self.lm_model_combo.addItems(lm_items)
        self.lm_model_combo.currentTextChanged.connect(self._update_model_desc)

        self.lm_desc_label = QLabel("")
        self.lm_desc_label.setWordWrap(True)
        self.lm_desc_label.setStyleSheet("color: #a0a0b0; font-size: 11px; font-style: italic; margin-bottom: 5px;")

        self.lm_hidden_spin = QSpinBox()
        self.lm_hidden_spin.setRange(64, 1024)
        self.lm_hidden_spin.setValue(256)
        self.lm_hidden_spin.setSingleStep(64)
        self.lm_hidden_spin.setToolTip("Dimension of hidden state vectors")

        self.lm_layers_spin = QSpinBox()
        self.lm_layers_spin.setRange(1, 100)
        self.lm_layers_spin.setValue(4)
        self.lm_layers_spin.setToolTip("Number of Transformer/RNN layers")

        self.lm_steps_spin = QSpinBox()
        self.lm_steps_spin.setRange(5, 50)
        self.lm_steps_spin.setValue(15)
        self.lm_steps_spin.setToolTip("Number of equilibrium steps per forward pass")

        model_controls = [
            ("Architecture:", self.lm_model_combo),
            ("", self.lm_desc_label),
            ("Hidden Dim:", self.lm_hidden_spin),
            ("Layers:", self.lm_layers_spin),
            ("Eq Steps:", self.lm_steps_spin)
        ]

        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        for i, (label, widget) in enumerate(model_controls):
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(widget, i, 1)
        model_layout.addLayout(grid)

        # Dynamic Hyperparameters Group
        self.lm_hyperparam_group = QGroupBox("Dynamic Params")
        self.lm_hyperparam_layout = QGridLayout(self.lm_hyperparam_group)
        self.lm_hyperparam_widgets = {}
        model_layout.addWidget(self.lm_hyperparam_group)
        self.lm_hyperparam_group.setVisible(False)
        self.lm_model_combo.currentTextChanged.connect(self._update_lm_hyperparams)

        model_layout.addStretch()
        self.toolbox.addItem(model_widget, "🧠 Model Architecture")

        # --- Section 2: Dataset ---
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        self.lm_dataset_combo = QComboBox()
        self.lm_dataset_combo.addItems(["tiny_shakespeare", "wikitext-2", "ptb"])
        self.lm_dataset_combo.setToolTip("Source text dataset for training")

        self.lm_seqlen_spin = QSpinBox()
        self.lm_seqlen_spin.setRange(32, 512)
        self.lm_seqlen_spin.setValue(128)
        self.lm_seqlen_spin.setToolTip("Context length (tokens) for backpropagation")

        self.lm_batch_spin = QSpinBox()
        self.lm_batch_spin.setRange(8, 256)
        self.lm_batch_spin.setValue(64)
        self.lm_batch_spin.setToolTip("Number of sequences per training step")

        data_controls = [
            ("Dataset:", self.lm_dataset_combo),
            ("Seq Length:", self.lm_seqlen_spin),
            ("Batch Size:", self.lm_batch_spin)
        ]

        data_grid = QGridLayout()
        for i, (label, widget) in enumerate(data_controls):
            data_grid.addWidget(QLabel(label), i, 0)
            data_grid.addWidget(widget, i, 1)
        data_layout.addLayout(data_grid)
        data_layout.addStretch()

        self.toolbox.addItem(data_widget, "📚 Text Configuration")

        # --- Section 3: Training ---
        train_widget = QWidget()
        train_layout = QVBoxLayout(train_widget)

        self.lm_epochs_spin = QSpinBox()
        self.lm_epochs_spin.setRange(1, 500)
        self.lm_epochs_spin.setValue(50)
        self.lm_epochs_spin.setToolTip("Total number of passes over the dataset")

        self.lm_lr_spin = QDoubleSpinBox()
        self.lm_lr_spin.setRange(0.0001, 0.1)
        self.lm_lr_spin.setValue(0.001)
        self.lm_lr_spin.setSingleStep(0.0001)
        self.lm_lr_spin.setDecimals(4)
        self.lm_lr_spin.setToolTip("Step size for optimizer")

        self.lm_compile_check = QCheckBox("torch.compile (2x speedup)")
        self.lm_compile_check.setChecked(True)
        self.lm_compile_check.setToolTip("Use PyTorch 2.0 graph compilation")

        self.lm_kernel_check = QCheckBox("O(1) Kernel Mode (GPU)")
        self.lm_kernel_check.setToolTip("Use fused EqProp kernel for O(1) memory training")

        self.lm_micro_check = QCheckBox("Live Dynamics Analysis")
        self.lm_micro_check.setToolTip("Periodically analyze convergence dynamics during training")

        train_controls = [
            ("Epochs:", self.lm_epochs_spin),
            ("Learning Rate:", self.lm_lr_spin),
            ("", self.lm_compile_check),
            ("", self.lm_kernel_check),
            ("", self.lm_micro_check)
        ]

        train_grid = QGridLayout()
        for i, (label, widget) in enumerate(train_controls):
            train_grid.addWidget(QLabel(label), i, 0)
            train_grid.addWidget(widget, i, 1)
        train_layout.addLayout(train_grid)
        train_layout.addStretch()

        self.toolbox.addItem(train_widget, "⚙️ Optimization")

        # Trigger initial update
        self._update_model_desc(self.lm_model_combo.currentText())

        # Train/Stop Buttons
        btn_layout = QHBoxLayout()
        self.lm_train_btn = QPushButton("▶ Train")
        self.lm_train_btn.setObjectName("trainButton")
        self.lm_train_btn.clicked.connect(lambda: self.start_training_signal.emit('lm'))
        btn_layout.addWidget(self.lm_train_btn)

        self.lm_stop_btn = QPushButton("⏹ Stop")
        self.lm_stop_btn.setObjectName("stopButton")
        self.lm_stop_btn.setEnabled(False)
        self.lm_stop_btn.clicked.connect(self.stop_training_signal.emit)
        btn_layout.addWidget(self.lm_stop_btn)

        self.lm_pause_btn = QPushButton("⏸ Pause")
        self.lm_pause_btn.setObjectName("resetButton")
        self.lm_pause_btn.setCheckable(True)
        self.lm_pause_btn.setEnabled(False)
        btn_layout.addWidget(self.lm_pause_btn)

        self.lm_reset_btn = QPushButton("↺ Reset")
        self.lm_reset_btn.setObjectName("resetButton")
        self.lm_reset_btn.setToolTip("Reset all hyperparameters to default values")
        self.lm_reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.lm_reset_btn)

        self.lm_clear_btn = QPushButton("🗑️ Clear")
        self.lm_clear_btn.setObjectName("resetButton")
        self.lm_clear_btn.setToolTip("Clear plot history")
        self.lm_clear_btn.clicked.connect(self.clear_plots_signal.emit)
        btn_layout.addWidget(self.lm_clear_btn)
        left_panel.addLayout(btn_layout)

        # Progress bar
        self.lm_progress = QProgressBar()
        self.lm_progress.setTextVisible(True)
        self.lm_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.lm_progress)

        # ETA Label
        self.lm_eta_label = QLabel("ETA: --:-- | Speed: -- it/s")
        self.lm_eta_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.lm_eta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.lm_eta_label)

        # Parameter count
        self.lm_param_label = QLabel("Parameters: --")
        self.lm_param_label.setStyleSheet("color: #00d4ff; font-weight: bold; padding: 5px;")
        left_panel.addWidget(self.lm_param_label)
        left_panel.addStretch()

        # Right panel: Plots and Generation
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)

        if HAS_PYQTGRAPH:
            metrics_group = QGroupBox("📊 Training Metrics")
            metrics_layout = QVBoxLayout(metrics_group)
            self.lm_loss_plot, self.lm_loss_curve = self._create_plot_widget("Loss", "Loss")
            metrics_layout.addWidget(self.lm_loss_plot)
            self.lm_acc_plot, self.lm_acc_curve = self._create_plot_widget("Accuracy", "Accuracy", yrange=(0, 1.0))
            metrics_layout.addWidget(self.lm_acc_plot)
            self.lm_lip_plot, self.lm_lip_curve = self._create_plot_widget("Lipschitz L", "Lipschitz L")
            self.lm_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            metrics_layout.addWidget(self.lm_lip_plot)
            right_panel.addWidget(metrics_group, stretch=2)
        else:
            no_plot_label = QLabel("Install pyqtgraph for live plots: pip install pyqtgraph")
            no_plot_label.setStyleSheet("color: #808090; padding: 20px;")
            right_panel.addWidget(no_plot_label)

        # Generation panel
        gen_group = QGroupBox("✨ Text Generation")
        gen_layout = QVBoxLayout(gen_group)

        # Prompt Input
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.gen_prompt_input = QTextEdit()
        self.gen_prompt_input.setPlaceholderText("Enter prompt (e.g. ROMEO:)")
        self.gen_prompt_input.setMaximumHeight(40)
        prompt_layout.addWidget(self.gen_prompt_input)
        gen_layout.addLayout(prompt_layout)

        gen_controls = QHBoxLayout()
        gen_controls.addWidget(QLabel("Temp:"))
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)
        self.temp_slider.setValue(10)
        gen_controls.addWidget(self.temp_slider)
        self.temp_label = QLabel("1.0")
        self.temp_label.setFixedWidth(30)
        gen_controls.addWidget(self.temp_label)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/10:.1f}"))

        gen_btn = QPushButton("🎲 Generate")
        gen_btn.clicked.connect(self._generate_text)
        gen_controls.addWidget(gen_btn)
        gen_layout.addLayout(gen_controls)

        self.gen_output = QTextEdit()
        self.gen_output.setReadOnly(True)
        self.gen_output.setPlaceholderText("Generated text will appear here...")
        gen_layout.addWidget(self.gen_output)
        right_panel.addWidget(gen_group, stretch=1)

        # Weight Visualization
        if HAS_PYQTGRAPH:
            viz_group = QGroupBox("🎞️ Weight Matrices")
            viz_layout = QVBoxLayout(viz_group)
            self.lm_weight_widgets = []
            self.lm_weight_labels = []
            self.lm_weights_container = QWidget()
            self.lm_weights_layout = QVBoxLayout(self.lm_weights_container)
            viz_layout.addWidget(self.lm_weights_container)
            right_panel.addWidget(viz_group)

    def _apply_preset(self, preset_name):
        """Apply config preset."""
        if preset_name == "Custom": return

        if preset_name == "Small (Fast)":
            self.lm_hidden_spin.setValue(128)
            self.lm_layers_spin.setValue(2)
            self.lm_steps_spin.setValue(10)
            self.lm_batch_spin.setValue(128)

        elif preset_name == "Medium (Balanced)":
            self.lm_hidden_spin.setValue(256)
            self.lm_layers_spin.setValue(4)
            self.lm_steps_spin.setValue(20)
            self.lm_batch_spin.setValue(64)

        elif preset_name == "Large (Accurate)":
            self.lm_hidden_spin.setValue(512)
            self.lm_layers_spin.setValue(6)
            self.lm_steps_spin.setValue(30)
            self.lm_batch_spin.setValue(32)

    def _reset_defaults(self):
        """Reset all controls to default values."""
        self.lm_hidden_spin.setValue(256)
        self.lm_layers_spin.setValue(4)
        self.lm_steps_spin.setValue(15)
        self.lm_seqlen_spin.setValue(128)
        self.lm_batch_spin.setValue(64)
        self.lm_epochs_spin.setValue(50)
        self.lm_lr_spin.setValue(0.001)
        self.lm_compile_check.setChecked(True)
        self.lm_kernel_check.setChecked(False)
        self.lm_micro_check.setChecked(False)
        # Reset combo boxes if needed, though they usually default to index 0 or specific items
        self.lm_dataset_combo.setCurrentIndex(0)
        self.lm_preset_combo.setCurrentIndex(0)

    def _update_lm_hyperparams(self, model_name):
        update_hyperparams_generic(self, model_name, self.lm_hyperparam_layout, self.lm_hyperparam_widgets, self.lm_hyperparam_group)

    def _update_model_desc(self, model_name):
        """Update model description label."""
        try:
            spec = get_model_spec(model_name)
            self.lm_desc_label.setText(spec.description)
        except Exception:
            self.lm_desc_label.setText("")

    def _generate_text(self):
        """Generate text using the current model."""
        if self.model is None:
            self.gen_output.setText("⚠️ No model loaded. Start training to create a model.")
            return

        # Ensure generator exists
        if self.generator is None or self.generator.model is not self.model:
            try:
                # Determine vocab size
                vocab_size = 95
                if hasattr(self.model, 'vocab_size'):
                    vocab_size = self.model.vocab_size
                elif hasattr(self.model, 'lm_head'):
                    vocab_size = self.model.lm_head.out_features
                elif hasattr(self.model, 'output_dim'):
                    vocab_size = min(self.model.output_dim, 256)

                device = next(self.model.parameters()).device
                self.generator = UniversalGenerator(
                    self.model,
                    vocab_size=vocab_size,
                    device=str(device)
                )
            except Exception as e:
                self.gen_output.setText(f"❌ Failed to create generator: {e}")
                return

        temperature = self.temp_slider.value() / 10.0
        prompt = self.gen_prompt_input.toPlainText().strip()
        if not prompt: prompt = "ROMEO:"

        self.gen_output.setText(f"🎲 Generating from '{prompt}'...\n(May be gibberish if undertrained)")

        # Force UI update
        QApplication.processEvents()

        try:
            # Generate text
            text = self.generator.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=temperature
            )
            self.gen_output.setText(f"📝 Generated:\n\n{text}")
        except Exception as e:
            self.gen_output.setText(f"❌ Generation failed: {str(e)}\n\nTip: Train for a few epochs first!")

    def update_model_ref(self, model):
        self.model = model
        # Reset generator to ensure it uses new model
        self.generator = None

    def get_current_hyperparams(self):
        return get_current_hyperparams_generic(self.lm_hyperparam_widgets)

    def update_theme(self, theme_colors, plot_colors):
        """Update plot colors based on theme."""
        if not HAS_PYQTGRAPH:
            return

        bg = theme_colors.get('background', '#0a0a0f')

        # Update Plots
        if hasattr(self, 'lm_loss_plot'):
            self.lm_loss_plot.setBackground(bg)
            self.lm_loss_curve.setPen(pg.mkPen(plot_colors.get('loss', 'w'), width=2))
            self.lm_loss_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.lm_loss_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

        if hasattr(self, 'lm_acc_plot'):
            self.lm_acc_plot.setBackground(bg)
            self.lm_acc_curve.setPen(pg.mkPen(plot_colors.get('accuracy', 'w'), width=2))
            self.lm_acc_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.lm_acc_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

        if hasattr(self, 'lm_lip_plot'):
            self.lm_lip_plot.setBackground(bg)
            self.lm_lip_curve.setPen(pg.mkPen(plot_colors.get('lipschitz', 'w'), width=2))
            self.lm_lip_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
            self.lm_lip_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
