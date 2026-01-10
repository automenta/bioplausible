"""
EqProp Trainer Dashboard

Main window with tabbed interface for Language Modeling and Vision training.
Features stunning dark cyberpunk theme with live pyqtgraph plots.
"""

import sys
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QTextEdit, QProgressBar, QSlider,
    QSplitter, QFrame, QCheckBox, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

# Monkeypatch pyparsing for matplotlib compatibility (snake_case -> camelCase)
try:
    import pyparsing

    # helper to alias method if missing
    def alias_method(cls, snake, camel):
        if hasattr(cls, camel) and not hasattr(cls, snake):
            setattr(cls, snake, getattr(cls, camel))

    # Alias one_of -> oneOf, nested_expr -> nestedExpr
    alias_method(pyparsing, 'one_of', 'oneOf')
    alias_method(pyparsing, 'nested_expr', 'nestedExpr')

    # Generic alias for common snake_case -> camelCase mismatches in pyparsing
    for snake, camel in [
        ('delimited_list', 'delimitedList'),
        ('counted_array', 'countedArray'),
        ('infix_notation', 'infixNotation'),
        ('one_of', 'oneOf'),
    ]:
        alias_method(pyparsing, snake, camel)

    # Alias ParserElement methods (parse_string, reset_cache, enable_packrat)
    if hasattr(pyparsing, 'ParserElement'):
        PE = pyparsing.ParserElement
        alias_method(PE, 'parse_string', 'parseString')

        # fix for static methods (resetCache, enablePackrat) being called as instance methods
        if hasattr(PE, 'resetCache') and not hasattr(PE, 'reset_cache'):
            PE.reset_cache = lambda *args, **kwargs: PE.resetCache()

        if hasattr(PE, 'enablePackrat') and not hasattr(PE, 'enable_packrat'):
            PE.enable_packrat = lambda *args, **kwargs: PE.enablePackrat(*args, **kwargs)

    # Alias pyparsing_common attributes (convert_to_float -> convertToFloat, etc)
    if hasattr(pyparsing, 'pyparsing_common'):
        PC = pyparsing.pyparsing_common
        for snake, camel in [
            ('convert_to_float', 'convertToFloat'),
            ('convert_to_integer', 'convertToInteger'),
            ('html_entity', 'htmlEntity'),
            ('common_html_entity', 'commonHTMLEntity'),
            ('signed_integer', 'signedInteger'),
            ('grouped', 'grouped'),
        ]:
            alias_method(PC, snake, camel)

    # Alias ParseException
    if hasattr(pyparsing, 'ParseException'):
        alias_method(pyparsing.ParseException, 'mark_input_line', 'markInputLine')

except ImportError:
    pass

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

# Feature flags
ENABLE_WEIGHT_VIZ = False  # Disabled due to pyqtgraph/matplotlib/pyparsing version conflicts

from .themes import CYBERPUNK_DARK, PLOT_COLORS
from .worker import TrainingWorker
from .generation import UniversalGenerator, SimpleCharTokenizer, count_parameters, format_parameter_count
from .hyperparams import get_hyperparams_for_model, HyperparamSpec
from .viz_utils import extract_weights, format_weight_for_display, normalize_weights_for_display, get_layer_description
from .dashboard_helpers import (
    generate_text_universal,
    update_hyperparams_generic,
    get_current_hyperparams_generic,
    create_weight_viz_widgets_generic,
    update_weight_visualization_generic
)


class EqPropDashboard(QMainWindow):
    """Main dashboard window for EqProp training."""

    def __init__(self, initial_config: Optional[Dict] = None):
        super().__init__()
        self.initial_config = initial_config

        self.setWindowTitle("⚡ EqProp Trainer v0.1.0")
        self.setGeometry(100, 100, 1400, 900)

        # Apply theme
        self.setStyleSheet(CYBERPUNK_DARK)

        # Training state
        self.worker: Optional[TrainingWorker] = None
        self.model = None
        self.train_loader = None
        self.current_hyperparams: Dict = {}  # Model-specific hyperparameters
        self.generator: Optional[UniversalGenerator] = None

        # Plot data
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []
        self.lipschitz_history: List[float] = []

        # Initialize UI
        self._setup_ui()

        # Update timer for plots
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)

        # Apply initial configuration if provided
        if self.initial_config:
            QTimer.singleShot(100, lambda: self._apply_config(self.initial_config))

    def _apply_config(self, config: Dict):
        """Apply initial configuration to UI elements."""
        try:
            model_name = config.get('model_name', '')

            # Determine if it's Vision or LM based on model name or config
            is_vision = any(x in model_name for x in ['MLP', 'Conv', 'Vision']) or 'mnist' in str(self.initial_config).lower() or 'cifar' in str(self.initial_config).lower()

            if is_vision:
                self.tabs.setCurrentIndex(1)
                combo = self.vis_model_combo
                hidden_spin = self.vis_hidden_spin
                steps_spin = self.vis_steps_spin
                lr_spin = self.vis_lr_spin
                epochs_spin = self.vis_epochs_spin
            else:
                self.tabs.setCurrentIndex(0)
                combo = self.lm_model_combo
                hidden_spin = self.lm_hidden_spin
                steps_spin = self.lm_steps_spin
                lr_spin = self.lm_lr_spin
                epochs_spin = self.lm_epochs_spin

            # Select model in combo box
            index = combo.findText(model_name, Qt.MatchFlag.MatchContains)
            if index >= 0:
                combo.setCurrentIndex(index)

            # Set hyperparameters
            if 'hidden_dim' in config:
                hidden_spin.setValue(int(config['hidden_dim']))
            if 'steps' in config:
                steps_spin.setValue(int(config['steps']))
            if 'lr' in config:
                lr_spin.setValue(float(config['lr']))
            if 'epochs' in config:
                epochs_spin.setValue(int(config['epochs']))
            if 'num_layers' in config:
                if is_vision:
                    pass # Vision tab currently doesn't have layers spin (it's hardcoded or part of model)
                else:
                    self.lm_layers_spin.setValue(int(config['num_layers']))

            self.status_label.setText(f"Loaded configuration for {model_name}")

        except Exception as e:
            print(f"Error applying config: {e}")
            self.status_label.setText(f"Error loading config: {e}")

    def _setup_ui(self):
        """Set up the main user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("⚡ EqProp Trainer")
        header.setObjectName("headerLabel")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        layout.addWidget(header)

        # Main content area with tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Language Modeling
        lm_tab = self._create_lm_tab()
        self.tabs.addTab(lm_tab, "🔤 Language Model")

        # Tab 2: Vision
        vision_tab = self._create_vision_tab()
        self.tabs.addTab(vision_tab, "📷 Vision")

        # Status bar
        self.status_label = QLabel("Ready. Select a model and dataset to begin training.")
        self.status_label.setStyleSheet("color: #808090; padding: 5px;")
        layout.addWidget(self.status_label)

    def _create_control_group(self, title: str, controls: List[Tuple[str, object]]) -> QGroupBox:
        """Create a generic control group with common layout."""
        group = QGroupBox(title)
        layout = QGridLayout(group)

        for i, (label, widget) in enumerate(controls):
            layout.addWidget(QLabel(label), i, 0)
            layout.addWidget(widget, i, 1)

        return group

    def _create_plot_widget(self, title: str, ylabel: str, xlabel: str = 'Epoch',
                           yrange: Optional[Tuple[float, float]] = None) -> tuple:
        """Create a standardized plot widget."""
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('#0a0a0f')
        plot_widget.setLabel('left', ylabel, color=PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'))
        plot_widget.setLabel('bottom', xlabel)
        plot_widget.showGrid(x=True, y=True, alpha=0.2)

        if yrange:
            plot_widget.setYRange(*yrange)

        curve = plot_widget.plot(pen=pg.mkPen(PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'), width=2))
        return plot_widget, curve

    def _create_lm_tab(self) -> QWidget:
        """Create the Language Modeling tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(15)

        # Left panel: Controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)

        # Model Selection
        lm_items = [
            "FullEqProp Transformer",
            "Attention-Only EqProp",
            "Recurrent Core EqProp",
            "Hybrid EqProp",
            "LoopedMLP LM",
        ]

        # Add bioplausible research algorithms
        try:
            from eqprop_torch import HAS_BIOPLAUSIBLE, ALGORITHM_REGISTRY
            if HAS_BIOPLAUSIBLE:
                lm_items.append("--- Bio-Plausible Research Models ---")
                for key, desc in ALGORITHM_REGISTRY.items():
                    lm_items.append(f"{key} - {desc}")
        except:
            pass

        self.lm_model_combo = QComboBox()
        self.lm_model_combo.addItems(lm_items)

        self.lm_hidden_spin = QSpinBox()
        self.lm_hidden_spin.setRange(64, 1024)
        self.lm_hidden_spin.setValue(256)
        self.lm_hidden_spin.setSingleStep(64)

        self.lm_layers_spin = QSpinBox()
        self.lm_layers_spin.setRange(1, 100)  # Support deep architectures
        self.lm_layers_spin.setValue(4)

        self.lm_steps_spin = QSpinBox()
        self.lm_steps_spin.setRange(5, 50)
        self.lm_steps_spin.setValue(15)

        model_controls = [
            ("Architecture:", self.lm_model_combo),
            ("Hidden Dim:", self.lm_hidden_spin),
            ("Layers:", self.lm_layers_spin),
            ("Eq Steps:", self.lm_steps_spin)
        ]
        model_group = self._create_control_group("🧠 Model", model_controls)
        left_panel.addWidget(model_group)

        # Dynamic Hyperparameters Group
        self.lm_hyperparam_group = QGroupBox("⚙️ Model Hyperparameters")
        self.lm_hyperparam_layout = QGridLayout(self.lm_hyperparam_group)
        self.lm_hyperparam_widgets = {}  # Store widgets for cleanup
        left_panel.addWidget(self.lm_hyperparam_group)
        self.lm_hyperparam_group.setVisible(False)  # Hidden by default

        # Connect model selection to update hyperparameters
        self.lm_model_combo.currentTextChanged.connect(self._update_lm_hyperparams)

        # Dataset Selection
        self.lm_dataset_combo = QComboBox()
        self.lm_dataset_combo.addItems([
            "tiny_shakespeare",
            "wikitext-2",
            "ptb",
        ])

        self.lm_seqlen_spin = QSpinBox()
        self.lm_seqlen_spin.setRange(32, 512)
        self.lm_seqlen_spin.setValue(128)

        self.lm_batch_spin = QSpinBox()
        self.lm_batch_spin.setRange(8, 256)
        self.lm_batch_spin.setValue(64)

        data_controls = [
            ("Dataset:", self.lm_dataset_combo),
            ("Seq Length:", self.lm_seqlen_spin),
            ("Batch Size:", self.lm_batch_spin)
        ]
        data_group = self._create_control_group("📚 Dataset", data_controls)
        left_panel.addWidget(data_group)

        # Training Settings
        self.lm_epochs_spin = QSpinBox()
        self.lm_epochs_spin.setRange(1, 500)
        self.lm_epochs_spin.setValue(50)

        self.lm_lr_spin = QDoubleSpinBox()
        self.lm_lr_spin.setRange(0.0001, 0.1)
        self.lm_lr_spin.setValue(0.001)
        self.lm_lr_spin.setSingleStep(0.0001)
        self.lm_lr_spin.setDecimals(4)

        self.lm_compile_check = QCheckBox("torch.compile (2x speedup)")
        self.lm_compile_check.setChecked(True)

        train_controls = [
            ("Epochs:", self.lm_epochs_spin),
            ("Learning Rate:", self.lm_lr_spin),
            ("", self.lm_compile_check)
        ]
        train_group = self._create_control_group("⚙️ Training", train_controls)
        left_panel.addWidget(train_group)

        # Train/Stop Buttons
        btn_layout = QHBoxLayout()

        self.lm_train_btn = QPushButton("▶ Train")
        self.lm_train_btn.setObjectName("trainButton")
        self.lm_train_btn.clicked.connect(lambda: self._start_training('lm'))
        btn_layout.addWidget(self.lm_train_btn)

        self.lm_stop_btn = QPushButton("⏹ Stop")
        self.lm_stop_btn.setObjectName("stopButton")
        self.lm_stop_btn.setEnabled(False)
        self.lm_stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.lm_stop_btn)

        left_panel.addLayout(btn_layout)

        # Progress bar
        self.lm_progress = QProgressBar()
        self.lm_progress.setTextVisible(True)
        self.lm_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.lm_progress)

        # Parameter count display
        self.lm_param_label = QLabel("Parameters: --")
        self.lm_param_label.setStyleSheet("color: #00d4ff; font-weight: bold; padding: 5px;")
        left_panel.addWidget(self.lm_param_label)

        # Stretch to push everything up
        left_panel.addStretch()

        # Right panel: Plots and Generation
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)

        if HAS_PYQTGRAPH:
            # Configure pyqtgraph for dark theme
            pg.setConfigOptions(antialias=True)

            # Loss/Accuracy plot
            metrics_group = QGroupBox("📊 Training Metrics")
            metrics_layout = QVBoxLayout(metrics_group)

            self.lm_loss_plot, self.lm_loss_curve = self._create_plot_widget("Loss", "Loss")
            metrics_layout.addWidget(self.lm_loss_plot)

            # Accuracy plot
            self.lm_acc_plot, self.lm_acc_curve = self._create_plot_widget("Accuracy", "Accuracy", yrange=(0, 1.0))
            metrics_layout.addWidget(self.lm_acc_plot)

            # Lipschitz plot
            self.lm_lip_plot, self.lm_lip_curve = self._create_plot_widget("Lipschitz L", "Lipschitz L")
            self.lm_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            metrics_layout.addWidget(self.lm_lip_plot)

            right_panel.addWidget(metrics_group, stretch=2)
        else:
            # Fallback text display
            no_plot_label = QLabel("Install pyqtgraph for live plots: pip install pyqtgraph")
            no_plot_label.setStyleSheet("color: #808090; padding: 20px;")
            right_panel.addWidget(no_plot_label)

        # Generation panel
        gen_group = QGroupBox("✨ Text Generation")
        gen_layout = QVBoxLayout(gen_group)

        gen_controls = QHBoxLayout()
        gen_controls.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)  # 0.1 to 2.0
        self.temp_slider.setValue(10)
        gen_controls.addWidget(self.temp_slider)
        self.temp_label = QLabel("1.0")
        self.temp_label.setFixedWidth(40)
        gen_controls.addWidget(self.temp_label)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/10:.1f}"))

        gen_btn = QPushButton("Generate")
        gen_btn.clicked.connect(self._generate_text)
        gen_controls.addWidget(gen_btn)
        gen_layout.addLayout(gen_controls)

        self.gen_output = QTextEdit()
        self.gen_output.setReadOnly(True)
        self.gen_output.setPlaceholderText("Generated text will appear here...")
        gen_layout.addWidget(self.gen_output)

        right_panel.addWidget(gen_group, stretch=1)

        # Weight Visualization (if pyqtgraph available)
        if HAS_PYQTGRAPH and ENABLE_WEIGHT_VIZ:
            viz_group = QGroupBox("🎞️ Weight Matrices")
            viz_layout = QVBoxLayout(viz_group)

            self.lm_weight_widgets = []
            self.lm_weight_labels = []

            self.lm_weights_container = QWidget()
            self.lm_weights_layout = QVBoxLayout(self.lm_weights_container)
            viz_layout.addWidget(self.lm_weights_container)

            right_panel.addWidget(viz_group)

        return tab

    def _create_vision_tab(self) -> QWidget:
        """Create the Vision training tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(15)

        # Left panel: Controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)

        # Model Selection
        model_items = [
            "LoopedMLP",
            "ConvEqProp",
            "BackpropMLP (baseline)",
        ]

        # Try to add bio-plausible algorithms from research codebase
        try:
            from eqprop_torch import HAS_BIOPLAUSIBLE, ALGORITHM_REGISTRY
            if HAS_BIOPLAUSIBLE:
                model_items.append("--- Bio-Plausible Research Models ---")
                # Add algorithms with descriptions
                for key, desc in ALGORITHM_REGISTRY.items():
                    # Format: "algorithm_key - Description"
                    model_items.append(f"{key} - {desc}")
        except:
            pass  # Bio-plausible models not available

        self.vis_model_combo = QComboBox()
        self.vis_model_combo.addItems(model_items)

        self.vis_hidden_spin = QSpinBox()
        self.vis_hidden_spin.setRange(64, 1024)
        self.vis_hidden_spin.setValue(256)

        self.vis_steps_spin = QSpinBox()
        self.vis_steps_spin.setRange(5, 100)
        self.vis_steps_spin.setValue(30)

        model_controls = [
            ("Architecture:", self.vis_model_combo),
            ("Hidden Dim:", self.vis_hidden_spin),
            ("Max Steps:", self.vis_steps_spin)
        ]
        model_group = self._create_control_group("🧠 Model", model_controls)
        left_panel.addWidget(model_group)

        # Dynamic Hyperparameters Group
        self.vis_hyperparam_group = QGroupBox("⚙️ Model Hyperparameters")
        self.vis_hyperparam_layout = QGridLayout(self.vis_hyperparam_group)
        self.vis_hyperparam_widgets = {}  # Store widgets for cleanup
        left_panel.addWidget(self.vis_hyperparam_group)
        self.vis_hyperparam_group.setVisible(False)  # Hidden by default

        # Connect model selection to update hyperparameters
        self.vis_model_combo.currentTextChanged.connect(self._update_vis_hyperparams)

        # Dataset
        self.vis_dataset_combo = QComboBox()
        self.vis_dataset_combo.addItems([
            "MNIST",
            "Fashion-MNIST",
            "CIFAR-10",
            "KMNIST",
        ])

        self.vis_batch_spin = QSpinBox()
        self.vis_batch_spin.setRange(16, 512)
        self.vis_batch_spin.setValue(64)

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

        self.vis_lr_spin = QDoubleSpinBox()
        self.vis_lr_spin.setRange(0.0001, 0.1)
        self.vis_lr_spin.setValue(0.001)
        self.vis_lr_spin.setDecimals(4)

        self.vis_compile_check = QCheckBox("torch.compile")
        self.vis_compile_check.setChecked(True)

        train_controls = [
            ("Epochs:", self.vis_epochs_spin),
            ("Learning Rate:", self.vis_lr_spin),
            ("", self.vis_compile_check)
        ]
        train_group = self._create_control_group("⚙️ Training", train_controls)
        left_panel.addWidget(train_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.vis_train_btn = QPushButton("▶ Train")
        self.vis_train_btn.setObjectName("trainButton")
        self.vis_train_btn.clicked.connect(lambda: self._start_training('vision'))
        btn_layout.addWidget(self.vis_train_btn)

        self.vis_stop_btn = QPushButton("⏹ Stop")
        self.vis_stop_btn.setObjectName("stopButton")
        self.vis_stop_btn.setEnabled(False)
        self.vis_stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.vis_stop_btn)

        left_panel.addLayout(btn_layout)

        self.vis_progress = QProgressBar()
        self.vis_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.vis_progress)

        # Parameter count display
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

            # Accuracy plot
            self.vis_acc_plot, self.vis_acc_curve = self._create_plot_widget("Accuracy", "Accuracy", yrange=(0, 1.0))
            metrics_layout.addWidget(self.vis_acc_plot)

            self.vis_lip_plot, self.vis_lip_curve = self._create_plot_widget("Lipschitz L", "Lipschitz L")
            self.vis_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            metrics_layout.addWidget(self.vis_lip_plot)

            right_panel.addWidget(metrics_group)

        # Stats display
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

        # Weight Visualization (if pyqtgraph available)
        if HAS_PYQTGRAPH and ENABLE_WEIGHT_VIZ:
            viz_group = QGroupBox("🎞️ Weight Matrices")
            viz_layout = QVBoxLayout(viz_group)

            # Create container for weight heatmaps
            self.vis_weight_widgets = []  # Store ImageView widgets
            self.vis_weight_labels = []   # Store labels

            # We'll create these dynamically when model is available
            self.vis_weights_container = QWidget()
            self.vis_weights_layout = QVBoxLayout(self.vis_weights_container)
            viz_layout.addWidget(self.vis_weights_container)

            right_panel.addWidget(viz_group)

        right_panel.addStretch()

        return tab

    def _start_training(self, mode: str):
        """Start training in background thread."""
        try:
            import torch
            from eqprop_torch import LoopedMLP, ConvEqProp, BackpropMLP

            if mode == 'vision':
                self._start_vision_training()
            else:
                self._start_lm_training()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{e}")

    def _create_model_and_loader(self, mode: str):
        """Create model and data loader based on mode (vision or lm)."""
        import torch
        from eqprop_torch import LoopedMLP, ConvEqProp, BackpropMLP
        from torch.utils.data import DataLoader

        if mode == 'vision':
            return self._create_vision_model_and_loader()
        else:
            return self._create_lm_model_and_loader()

    def _create_vision_model_and_loader(self):
        """Create vision model and data loader."""
        from eqprop_torch.datasets import get_vision_dataset
        from torch.utils.data import DataLoader

        # Get dataset
        dataset_name = self.vis_dataset_combo.currentText().lower().replace('-', '_')
        use_flatten = 'MLP' in self.vis_model_combo.currentText()

        train_data = get_vision_dataset(dataset_name, train=True, flatten=use_flatten)
        self.train_loader = DataLoader(train_data, batch_size=self.vis_batch_spin.value(), shuffle=True)

        # Create model
        hidden = self.vis_hidden_spin.value()
        model_name = self.vis_model_combo.currentText()

        # Check if it's a bio-plausible research algorithm
        if ' - ' in model_name:  # Bio-plausible models are formatted as "key - description"
            algorithm_key = model_name.split(' - ')[0]
            try:
                from eqprop_torch import HAS_BIOPLAUSIBLE
                from algorithms import create_model, AlgorithmConfig

                if not HAS_BIOPLAUSIBLE:
                    raise ImportError("Research algorithms not available")

                # Determine input_dim based on dataset
                if 'MNIST' in self.vis_dataset_combo.currentText():
                    input_dim = 784
                else:  # CIFAR-10
                    input_dim = 3072

                # Create research algorithm model directly (first-class nn.Module)
                model = create_model(
                    algorithm_key,
                    input_dim,
                    [hidden],  # Single hidden layer
                    10  # 10 classes
                )
            except Exception as e:
                QMessageBox.warning(self, "Model Creation Failed",
                                   f"Could not create {algorithm_key}: {e}")
                return None, None
        elif 'LoopedMLP' in model_name:
            input_dim = 784 if 'MNIST' in self.vis_dataset_combo.currentText() else 3072
            model = LoopedMLP(input_dim, hidden, 10, max_steps=self.vis_steps_spin.value())
        elif 'ConvEqProp' in model_name:
            channels = 1 if 'MNIST' in self.vis_dataset_combo.currentText() else 3
            model = ConvEqProp(channels, hidden // 4, 10)
        else:  # BackpropMLP
            input_dim = 784 if 'MNIST' in self.vis_dataset_combo.currentText() else 3072
            model = BackpropMLP(input_dim, hidden, 10)

        return model, self.train_loader

    def _create_lm_model_and_loader(self):
        """Create language model and data loader."""
        from eqprop_torch import HAS_LM_VARIANTS, HAS_BIOPLAUSIBLE
        from eqprop_torch.datasets import get_lm_dataset
        from torch.utils.data import DataLoader

        model_name = self.lm_model_combo.currentText()

        # Check if it's a bioplausible algorithm
        if ' - ' in model_name:
            algorithm_key = model_name.split(' - ')[0]
            try:
                from algorithms import create_model

                # Get dataset to determine vocab size
                dataset_name = self.lm_dataset_combo.currentText()
                seq_len = self.lm_seqlen_spin.value()

                # Load dataset
                dataset = get_lm_dataset(dataset_name, seq_len=seq_len, split='train')
                vocab_size = dataset.vocab_size if hasattr(dataset, 'vocab_size') else 256

                # Create algorithm model
                hidden = self.lm_hidden_spin.value()
                model = create_model(
                    algorithm_key,
                    vocab_size,  # Input = vocab
                    [hidden],
                    vocab_size   # Output = vocab (next token prediction)
                )

                train_loader = DataLoader(dataset, batch_size=self.lm_batch_spin.value(), shuffle=True)
                return model, train_loader

            except Exception as e:
                QMessageBox.warning(self, "Model Creation Failed",
                                   f"Could not create {algorithm_key}: {e}")
                return None, None

        # LM variant models
        elif HAS_LM_VARIANTS:
            try:
                from eqprop_torch import get_eqprop_lm

                # Map UI names to variant keys
                variant_map = {
                    "FullEqProp Transformer": "full",
                    "Attention-Only EqProp": "attention_only",
                    "Recurrent Core EqProp": "recurrent_core",
                    "Hybrid EqProp": "hybrid",
                    "LoopedMLP LM": "looped_mlp"
                }

                variant = variant_map.get(model_name, "full")

                # Get dataset
                dataset_name = self.lm_dataset_combo.currentText()
                seq_len = self.lm_seqlen_spin.value()

                dataset = get_lm_dataset(dataset_name, seq_len=seq_len, split='train')
                vocab_size = dataset.vocab_size if hasattr(dataset, 'vocab_size') else 256

                # Create LM model
                model = get_eqprop_lm(
                    variant,
                    vocab_size=vocab_size,
                    hidden_dim=self.lm_hidden_spin.value(),
                    num_layers=self.lm_layers_spin.value(),
                    max_seq_len=seq_len,
                    eq_steps=self.lm_steps_spin.value()
                )

                train_loader = DataLoader(dataset, batch_size=self.lm_batch_spin.value(), shuffle=True)
                return model, train_loader

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create LM model:\n{e}")
                return None, None
        else:
            QMessageBox.warning(self, "Not Available", "LM variants not available")
            return None, None

    def _start_vision_training(self):
        """Start vision model training."""
        # Create model and data loader
        model, train_loader = self._create_vision_model_and_loader()

        if model is None or train_loader is None:
            return  # Error already shown to user

        self.model = model
        self.train_loader = train_loader

        # Clear history
        self.loss_history.clear()
        self.acc_history.clear()
        self.lipschitz_history.clear()

        # Get hyperparameters
        hyperparams = self._get_current_hyperparams(self.vis_hyperparam_widgets)

        # Update parameter count
        if hasattr(self, 'vis_param_label'):
            count = count_parameters(self.model)
            self.vis_param_label.setText(f"Parameters: {format_parameter_count(count)}")

        # Create and start worker
        self.worker = TrainingWorker(
            self.model,
            self.train_loader,
            epochs=self.vis_epochs_spin.value(),
            lr=self.vis_lr_spin.value(),
            use_compile=self.vis_compile_check.isChecked(),
            hyperparams=hyperparams,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.weights_updated.connect(self._update_weight_visualization)

        # Update UI
        self.vis_train_btn.setEnabled(False)
        self.vis_stop_btn.setEnabled(True)
        self.vis_progress.setMaximum(self.vis_epochs_spin.value())
        self.vis_progress.setValue(0)

        model_name = self.vis_model_combo.currentText()
        self.status_label.setText(f"Training {model_name}...")
        self.plot_timer.start(100)
        self.worker.start()

    def _start_lm_training(self):
        """Start language model training."""
        # Create model and data loader
        model, train_loader = self._create_lm_model_and_loader()

        if model is None or train_loader is None:
            return  # Error already shown to user

        self.model = model
        self.train_loader = train_loader

        # Clear history
        self.loss_history.clear()
        self.acc_history.clear()
        self.lipschitz_history.clear()

        # Get hyperparameters
        hyperparams = self._get_current_hyperparams(self.lm_hyperparam_widgets)

        # Update parameter count
        if hasattr(self, 'lm_param_label'):
            count = count_parameters(self.model)
            self.lm_param_label.setText(f"Parameters: {format_parameter_count(count)}")

        self.worker = TrainingWorker(
            self.model,
            self.train_loader,
            epochs=self.lm_epochs_spin.value(),
            lr=self.lm_lr_spin.value(),
            use_compile=self.lm_compile_check.isChecked(),
            hyperparams=hyperparams,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.weights_updated.connect(self._update_weight_visualization)

        # Update UI
        self.lm_train_btn.setEnabled(False)
        self.lm_stop_btn.setEnabled(True)
        self.lm_progress.setMaximum(self.lm_epochs_spin.value())
        self.lm_progress.setValue(0)

        model_name = self.lm_model_combo.currentText()
        dataset_name = self.lm_dataset_combo.currentText()
        self.status_label.setText(f"Training {model_name} on {dataset_name}...")
        self.plot_timer.start(100)
        self.worker.start()


    def _stop_training(self):
        """Stop training."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping training...")

    def _on_progress(self, metrics: dict):
        """Handle training progress update."""
        self.loss_history.append(metrics['loss'])
        self.acc_history.append(metrics['accuracy'])
        self.lipschitz_history.append(metrics['lipschitz'])

        # Update progress bar
        self.vis_progress.setValue(metrics['epoch'])
        self.lm_progress.setValue(metrics['epoch'])

        # Update labels
        self.vis_acc_label.setText(f"{metrics['accuracy']:.1%}")
        self.vis_loss_label.setText(f"{metrics['loss']:.4f}")
        self.vis_lip_label.setText(f"{metrics['lipschitz']:.4f}")

        self.status_label.setText(
            f"Epoch {metrics['epoch']}/{metrics['total_epochs']} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.1%} | "
            f"L: {metrics['lipschitz']:.4f}"
        )

    def _update_plots(self):
        """Update plot curves."""
        if not HAS_PYQTGRAPH:
            return

        if self.loss_history:
            epochs = list(range(1, len(self.loss_history) + 1))

            # Update vision plots
            if hasattr(self, 'vis_loss_curve'):
                self.vis_loss_curve.setData(epochs, self.loss_history)
                # Check for separate acc plot
                if hasattr(self, 'vis_acc_curve'):
                    self.vis_acc_curve.setData(epochs, self.acc_history)
                self.vis_lip_curve.setData(epochs, self.lipschitz_history)

            # Update LM plots
            if hasattr(self, 'lm_loss_curve'):
                self.lm_loss_curve.setData(epochs, self.loss_history)
                # Check for separate acc plot
                if hasattr(self, 'lm_acc_curve'):
                    self.lm_acc_curve.setData(epochs, self.acc_history)
                self.lm_lip_curve.setData(epochs, self.lipschitz_history)



    def _on_finished(self, result: dict):
        """Handle training completion."""
        self.plot_timer.stop()
        self.vis_train_btn.setEnabled(True)
        self.vis_stop_btn.setEnabled(False)
        self.lm_train_btn.setEnabled(True)
        self.lm_stop_btn.setEnabled(False)

        if result.get('success'):
            self.status_label.setText(f"✓ Training complete! ({result['epochs_completed']} epochs)")
        else:
            self.status_label.setText("Training stopped.")

    def _on_error(self, error: str):
        """Handle training error."""
        self.plot_timer.stop()
        self.vis_train_btn.setEnabled(True)
        self.vis_stop_btn.setEnabled(False)
        self.status_label.setText("Training error!")
        QMessageBox.critical(self, "Training Error", error)

    def _generate_text(self):
        """Generate text from the model (works even with untrained models)."""
        generate_text_universal(self)

    def _update_lm_hyperparams(self, model_name: str):
        """Update LM hyperparameter widgets based on selected model."""
        update_hyperparams_generic(self, model_name, self.lm_hyperparam_layout, self.lm_hyperparam_widgets, self.lm_hyperparam_group)

    def _update_vis_hyperparams(self, model_name: str):
        """Update Vision hyperparameter widgets based on selected model."""
        update_hyperparams_generic(self, model_name, self.vis_hyperparam_layout, self.vis_hyperparam_widgets, self.vis_hyperparam_group)

    def _get_current_hyperparams(self, widgets: dict) -> dict:
        """Extract current values from hyperparameter widgets."""
        return get_current_hyperparams_generic(widgets)

    def _update_weight_visualization(self, weights: dict):
        """Update weight visualization heatmaps."""
        if not HAS_PYQTGRAPH or not ENABLE_WEIGHT_VIZ:
            return

        # Determine which tab is active to update correct widgets
        active_idx = self.tabs.currentIndex()

        # 0 = LM Tab, 1 = Vision Tab
        if active_idx == 0:
            layout = self.lm_weights_layout
            widgets = self.lm_weight_widgets
            labels = self.lm_weight_labels
        else:
            layout = self.vis_weights_layout
            widgets = self.vis_weight_widgets
            labels = self.vis_weight_labels

        # If widgets list is empty, create them
        if not widgets:
            # Clear existing items in layout
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            widgets.clear()
            labels.clear()

            # Create widgets
            for name, W in list(weights.items())[:3]:
                # Label
                label = QLabel(get_layer_description(name))
                label.setStyleSheet("color: #00d4ff; font-weight: bold;")
                layout.addWidget(label)
                labels.append(label)

                # Image View
                img_view = pg.ImageView()
                img_view.setFixedHeight(150)
                img_view.ui.histogram.hide()
                img_view.ui.roiBtn.hide()
                img_view.ui.menuBtn.hide()
                layout.addWidget(img_view)
                widgets.append(img_view)

        # Update content
        for i, (name, W) in enumerate(weights.items()):
            if i >= len(widgets):
                break

            W_display = format_weight_for_display(W)
            W_norm = normalize_weights_for_display(W_display)

            try:
                widgets[i].setImage(W_norm.T, levels=(0, 1))
                labels[i].setText(get_layer_description(name))
            except Exception:
                pass