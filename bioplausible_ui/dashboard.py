"""
EqProp Trainer Dashboard

Main window with tabbed interface for Language Modeling and Vision training.
Features stunning dark cyberpunk theme with live pyqtgraph plots.
"""

import sys
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QTextEdit, QProgressBar, QSlider,
    QSplitter, QFrame, QCheckBox, QMessageBox, QApplication, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

# Feature flags
ENABLE_WEIGHT_VIZ = True

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec, ModelSpec
from bioplausible.models.factory import create_model

from .themes import CYBERPUNK_DARK, PLOT_COLORS
from .worker import TrainingWorker, RLWorker, BenchmarkWorker
from .generation import UniversalGenerator, SimpleCharTokenizer, count_parameters, format_parameter_count
from .hyperparams import get_hyperparams_for_model, HyperparamSpec
from .viz_utils import extract_weights, format_weight_for_display, normalize_weights_for_display, get_layer_description
from .dashboard_helpers import (
    update_hyperparams_generic,
    get_current_hyperparams_generic,
    create_weight_viz_widgets_generic,
    update_weight_visualization_generic
)

from bioplausible_ui.tabs.lm_tab import LMTab
from bioplausible_ui.tabs.vision_tab import VisionTab
from bioplausible_ui.tabs.rl_tab import RLTab
from bioplausible_ui.tabs.microscope_tab import MicroscopeTab
from bioplausible_ui.tabs.benchmarks_tab import BenchmarksTab
from bioplausible_ui.tabs.console_tab import ConsoleTab
from bioplausible_ui.tabs.p2p_tab import P2PTab
from bioplausible_ui.tabs.discovery_tab import DiscoveryTab


class QtLogHandler(logging.Handler, QObject):
    """Custom logging handler that emits a signal for each log record."""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


class EqPropDashboard(QMainWindow):
    """Main dashboard window for EqProp training."""

    def __init__(self, initial_config: Optional[Dict] = None):
        super().__init__()
        self.initial_config = initial_config

        self.setWindowTitle("⚡ EqProp Trainer v0.1.0")
        self.setGeometry(100, 100, 1400, 900)

        # Apply theme
        self.setStyleSheet(CYBERPUNK_DARK)

        # Setup system logger
        self._setup_logging()

        # Training state
        self.worker: Optional[TrainingWorker] = None
        self.model = None
        self.train_loader = None
        self.current_hyperparams: Dict = {}  # Model-specific hyperparameters
        self.generator: Optional[UniversalGenerator] = None
        self.start_time = None

        # Plot data
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []
        self.lipschitz_history: List[float] = []

        # RL History
        self.rl_reward_history: List[float] = []
        self.rl_loss_history: List[float] = []
        self.rl_avg_reward_history: List[float] = []

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
                combo = self.vis_tab.vis_model_combo
                hidden_spin = self.vis_tab.vis_hidden_spin
                steps_spin = self.vis_tab.vis_steps_spin
                lr_spin = self.vis_tab.vis_lr_spin
                epochs_spin = self.vis_tab.vis_epochs_spin
            else:
                self.tabs.setCurrentIndex(0)
                combo = self.lm_tab.lm_model_combo
                hidden_spin = self.lm_tab.lm_hidden_spin
                steps_spin = self.lm_tab.lm_steps_spin
                lr_spin = self.lm_tab.lm_lr_spin
                epochs_spin = self.lm_tab.lm_epochs_spin

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
                    pass
                else:
                    self.lm_tab.lm_layers_spin.setValue(int(config['num_layers']))

            self.status_label.setText(f"Loaded configuration for {model_name}")
            self.status_label.setStyleSheet("color: #00aacc; padding: 5px;")

        except Exception as e:
            print(f"Error applying config: {e}")
            self.status_label.setText(f"Error loading config: {e}")
            self.status_label.setStyleSheet("color: #ff5588; padding: 5px;")

    def _setup_ui(self):
        """Set up the main user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header Area
        header_layout = QHBoxLayout()

        header = QLabel("⚡ EqProp Trainer")
        header.setObjectName("headerLabel")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Save/Load Buttons
        self.save_btn = QPushButton("💾 Save Model")
        self.save_btn.clicked.connect(self._save_model)
        header_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("📂 Load Model")
        self.load_btn.clicked.connect(self._load_model)
        header_layout.addWidget(self.load_btn)

        layout.addLayout(header_layout)

        # Main content area with tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Language Modeling
        self.lm_tab = LMTab()
        self.lm_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
        self.lm_tab.stop_training_signal.connect(self._stop_training)
        self.lm_tab.clear_plots_signal.connect(self._clear_plots)
        self.tabs.addTab(self.lm_tab, "🔤 Language Model")

        # Tab 2: Vision
        self.vis_tab = VisionTab()
        self.vis_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
        self.vis_tab.stop_training_signal.connect(self._stop_training)
        self.vis_tab.clear_plots_signal.connect(self._clear_plots)
        self.tabs.addTab(self.vis_tab, "📷 Vision")

        # Tab 3: Reinforcement Learning
        self.rl_tab = RLTab()
        self.rl_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
        self.rl_tab.stop_training_signal.connect(self._stop_training)
        self.rl_tab.clear_plots_signal.connect(self._clear_plots)
        self.tabs.addTab(self.rl_tab, "🎮 Reinforcement Learning")

        # Tab 4: Microscope (Dynamics)
        self.micro_tab = MicroscopeTab()
        self.tabs.addTab(self.micro_tab, "🔬 Microscope")

        # Tab 5: Model Search (Hyperopt)
        search_tab = self._create_search_tab()
        self.tabs.addTab(search_tab, "🔍 Model Search")

        # Tab 6: Discovery (Viz)
        self.disc_tab = DiscoveryTab()
        self.tabs.addTab(self.disc_tab, "🗺️ Discovery")

        # Tab 7: Community Grid (P2P)
        self.p2p_tab = P2PTab()
        # Connect P2P log to main console
        self.p2p_tab.bridge_log_signal.connect(self._append_log)
        # Connect P2P status to Discovery Tab
        self.p2p_tab.bridge_status_signal.connect(lambda s, p, j: self.disc_tab.update_p2p_ref(self.p2p_tab.worker))
        self.tabs.addTab(self.p2p_tab, "🌐 Community Grid")

        # Tab 8: Benchmarks
        # Tab 8: Benchmarks
        self.bench_tab = BenchmarksTab()
        self.bench_tab.log_message.connect(self._append_log)
        self.tabs.addTab(self.bench_tab, "🏆 Benchmarks")

        # Tab 9: Console
        self.console_tab = ConsoleTab()
        self.tabs.addTab(self.console_tab, "💻 Console")

        # Status bar
        self.status_label = QLabel("Ready. Select a model and dataset to begin training.")
        self.status_label.setStyleSheet("color: #808090; padding: 5px;")
        layout.addWidget(self.status_label)

        # Keyboard Shortcuts
        self.train_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.train_shortcut.activated.connect(self._on_train_shortcut)

        self.stop_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.stop_shortcut.activated.connect(self._stop_training)

    def _clear_plots(self):
        """Clear all plot history and refresh plots."""
        self.loss_history.clear()
        self.acc_history.clear()
        self.lipschitz_history.clear()
        self.rl_reward_history.clear()
        self.rl_loss_history.clear()
        self.rl_avg_reward_history.clear()
        self._update_plots()
        self.status_label.setText("Plot history cleared.")
        self.status_label.setStyleSheet("color: #a0a0b0; padding: 5px;")

    def _on_train_shortcut(self):
        """Handle start training shortcut based on active tab."""
        current = self.tabs.currentWidget()
        if isinstance(current, LMTab):
            self._start_training('lm')
        elif isinstance(current, VisionTab):
            self._start_training('vision')
        elif isinstance(current, RLTab):
            self._start_training('rl')

    def _save_model(self):
        """Save the current model to a file, including current UI configuration."""
        if not self.model:
            QMessageBox.warning(self, "No Model", "No model to save.")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Model Checkpoint", "", "PyTorch Checkpoints (*.pt)")
        if fname:
            try:
                import torch

                # Capture current UI state
                current_config = {}

                # Active tab determines which controls to read
                if self.tabs.currentIndex() == 0: # LM
                    current_config.update({
                        'task': 'lm',
                        'model_name': self.lm_tab.lm_model_combo.currentText(),
                        'hidden_dim': self.lm_tab.lm_hidden_spin.value(),
                        'num_layers': self.lm_tab.lm_layers_spin.value(),
                        'steps': self.lm_tab.lm_steps_spin.value(),
                        'dataset': self.lm_tab.lm_dataset_combo.currentText(),
                        'seq_len': self.lm_tab.lm_seqlen_spin.value(),
                        'hyperparams': self._get_current_hyperparams(self.lm_tab.lm_hyperparam_widgets)
                    })
                elif self.tabs.currentIndex() == 1: # Vision
                    current_config.update({
                        'task': 'vision',
                        'model_name': self.vis_tab.vis_model_combo.currentText(),
                        'hidden_dim': self.vis_tab.vis_hidden_spin.value(),
                        'steps': self.vis_tab.vis_steps_spin.value(),
                        'dataset': self.vis_tab.vis_dataset_combo.currentText(),
                        'hyperparams': self._get_current_hyperparams(self.vis_tab.vis_hyperparam_widgets)
                    })

                state = {
                    'model_state_dict': self.model.state_dict(),
                    'config': current_config,
                    'model_name': current_config.get('model_name', 'Unknown')
                }
                torch.save(state, fname)
                self.status_label.setText(f"Model saved to {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _load_model(self):
        """Load a model from a file and restore UI state."""
        fname, _ = QFileDialog.getOpenFileName(self, "Load Model Checkpoint", "", "PyTorch Checkpoints (*.pt)")
        if fname:
            try:
                import torch
                checkpoint = torch.load(fname)
                config = checkpoint.get('config', {})

                # 1. Restore UI State from Config
                if config.get('task') == 'lm':
                    self.tabs.setCurrentIndex(0)
                    if 'model_name' in config:
                        idx = self.lm_tab.lm_model_combo.findText(config['model_name'])
                        if idx >= 0: self.lm_tab.lm_model_combo.setCurrentIndex(idx)

                    if 'hidden_dim' in config: self.lm_tab.lm_hidden_spin.setValue(config['hidden_dim'])
                    if 'num_layers' in config: self.lm_tab.lm_layers_spin.setValue(config['num_layers'])
                    if 'steps' in config: self.lm_tab.lm_steps_spin.setValue(config['steps'])
                    if 'dataset' in config:
                        idx = self.lm_tab.lm_dataset_combo.findText(config['dataset'])
                        if idx >= 0: self.lm_tab.lm_dataset_combo.setCurrentIndex(idx)

                elif config.get('task') == 'vision':
                    self.tabs.setCurrentIndex(1)
                    if 'model_name' in config:
                        idx = self.vis_tab.vis_model_combo.findText(config['model_name'])
                        if idx >= 0: self.vis_tab.vis_model_combo.setCurrentIndex(idx)

                    if 'hidden_dim' in config: self.vis_tab.vis_hidden_spin.setValue(config['hidden_dim'])
                    if 'steps' in config: self.vis_tab.vis_steps_spin.setValue(config['steps'])
                    if 'dataset' in config:
                        idx = self.vis_tab.vis_dataset_combo.findText(config['dataset'])
                        if idx >= 0: self.vis_tab.vis_dataset_combo.setCurrentIndex(idx)

                # Process pending events to ensure UI updates (like hyperparam widgets) are triggered
                QApplication.processEvents()

                # 2. Recreate Model Structure
                task = config.get('task', 'vision')

                if task == 'lm':
                    from bioplausible.datasets import get_lm_dataset
                    ds_name = config.get('dataset', 'tiny_shakespeare')
                    ds = get_lm_dataset(ds_name, seq_len=128, split='train')
                    vocab_size = ds.vocab_size

                    spec = get_model_spec(config['model_name'])
                    self.model = create_model(
                        spec=spec,
                        input_dim=None,
                        output_dim=vocab_size,
                        hidden_dim=config.get('hidden_dim', 256),
                        num_layers=config.get('num_layers', 4),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        task_type="lm"
                    )
                else:
                    from bioplausible.datasets import get_vision_dataset
                    ds_name = config.get('dataset', 'mnist').lower()
                    spec = get_model_spec(config['model_name'])
                    use_flatten = spec.model_type != "modern_conv_eqprop"

                    input_dim = 784
                    if 'cifar' in ds_name: input_dim = 3072
                    if 'svhn' in ds_name: input_dim = 3072
                    if not use_flatten:
                        input_dim = 3 if ('cifar' in ds_name or 'svhn' in ds_name) else 1

                    self.model = create_model(
                        spec=spec,
                        input_dim=input_dim,
                        output_dim=10,
                        hidden_dim=config.get('hidden_dim', 256),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        task_type="vision"
                    )

                # 3. Load Weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.status_label.setText(f"Model loaded from {fname}")
                self.status_label.setStyleSheet("color: #00ff88; padding: 5px;")

                # 4. Update Tab References
                if self.model:
                    self.lm_tab.update_model_ref(self.model)
                    self.micro_tab.update_model_ref(self.model)

            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load: {str(e)}")
                import traceback
                traceback.print_exc()

    def _run_microscope_analysis(self):
        """Delegate analysis to microscope tab."""
        # Need to wire up the model reference if not already done
        self.micro_tab.update_model_ref(self.model)
        self.micro_tab._run_microscope_analysis()

    def _create_search_tab(self) -> QWidget:
        """Create the Model Search tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title_label = QLabel("🚀 Automated Model Search")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "Use Evolutionary Algorithms to find the best hyperparameters and architectures.\n"
            "Compares Bio-Plausible algorithms (EqProp, DFA, Hebbian) against Backprop baselines."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #a0a0b0; font-size: 14px;")
        layout.addWidget(desc_label)

        # Task Selection
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("Task:"))
        self.search_task_combo = QComboBox()
        self.search_task_combo.addItems([
            "Language Modeling (TinyShakespeare)",
            "Vision (MNIST)",
            "Vision (CIFAR-10)",
            "RL (CartPole-v1)"
        ])
        self.search_task_combo.setMinimumWidth(200)
        form_layout.addWidget(self.search_task_combo)
        form_layout.addStretch()

        container = QWidget()
        container.setLayout(form_layout)
        layout.addWidget(container)

        # Launch Button
        launch_btn = QPushButton("Launch Search Tool")
        launch_btn.setMinimumHeight(60)
        launch_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        launch_btn.clicked.connect(self._launch_search_tool)
        layout.addWidget(launch_btn)

        layout.addStretch()
        return tab

    def _launch_search_tool(self):
        """Launch the Hyperopt Dashboard in a new window."""
        try:
            from bioplausible_ui.hyperopt_dashboard import HyperoptSearchDashboard

            # Determine task from combo box
            task_text = self.search_task_combo.currentText()
            if "Shakespeare" in task_text:
                task = "shakespeare"
            elif "MNIST" in task_text:
                task = "mnist"
            elif "CIFAR" in task_text:
                task = "cifar10"
            elif "CartPole" in task_text:
                task = "cartpole"
            else:
                task = "shakespeare"

            # Create as a new window
            self.search_window = HyperoptSearchDashboard(task=task, quick_mode=True)
            self.search_window.show()

            self.status_label.setText(f"Launched Model Search Tool for {task}")

        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch search tool:\n{e}")
            import traceback
            traceback.print_exc()

    def _start_training(self, mode: str):
        """Start training in background thread."""
        try:
            import torch
            from bioplausible import LoopedMLP, ConvEqProp, BackpropMLP

            if mode == 'vision':
                self._start_vision_training()
            elif mode == 'rl':
                self._start_rl_training()
            else:
                self._start_lm_training()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{e}")

    def _create_model_and_loader(self, mode: str):
        """Create model and data loader based on mode (vision or lm)."""
        import torch
        from bioplausible import LoopedMLP, ConvEqProp, BackpropMLP
        from torch.utils.data import DataLoader

        if mode == 'vision':
            return self._create_vision_model_and_loader()
        else:
            return self._create_lm_model_and_loader()

    def _create_vision_model_and_loader(self):
        """Create vision model and data loader."""
        from bioplausible.datasets import get_vision_dataset
        from torch.utils.data import DataLoader

        # Get dataset
        dataset_name = self.vis_tab.vis_dataset_combo.currentText().lower().replace('-', '_')

        # Determine flattening based on model type
        model_name = self.vis_tab.vis_model_combo.currentText()
        try:
            spec = get_model_spec(model_name)
            use_flatten = spec.model_type != "modern_conv_eqprop"
        except:
             use_flatten = True

        train_data = get_vision_dataset(dataset_name, train=True, flatten=use_flatten)
        self.train_loader = DataLoader(train_data, batch_size=self.vis_tab.vis_batch_spin.value(), shuffle=True)

        # Create model
        hidden = self.vis_tab.vis_hidden_spin.value()

        try:
            spec = get_model_spec(model_name)

            # Determine input_dim based on dataset
            if 'MNIST' in self.vis_tab.vis_dataset_combo.currentText():
                input_dim = 784 if use_flatten else 1
            else:  # CIFAR-10
                input_dim = 3072 if use_flatten else 3

            # Map combo text to internal string
            grad_text = self.vis_tab.vis_grad_combo.currentText()
            if "BPTT" in grad_text:
                grad_method = "bptt"
            elif "Equilibrium" in grad_text:
                grad_method = "equilibrium"
            elif "Contrastive" in grad_text:
                grad_method = "contrastive"
            else:
                grad_method = "bptt"

            model = create_model(
                spec=spec,
                input_dim=input_dim,
                output_dim=10,
                hidden_dim=hidden,
                device="cuda" if torch.cuda.is_available() else "cpu",
                task_type="vision",
                gradient_method=grad_method
            )

            # Update step if spin box is used
            if hasattr(model, 'max_steps'):
                model.max_steps = self.vis_tab.vis_steps_spin.value()
            elif hasattr(model, 'eq_steps'):
                model.eq_steps = self.vis_tab.vis_steps_spin.value()

        except Exception as e:
             QMessageBox.warning(self, "Model Creation Failed", f"Could not create {model_name}: {e}")
             return None, None

        return model, self.train_loader

    def _create_lm_model_and_loader(self):
        """Create language model and data loader."""
        from bioplausible.datasets import get_lm_dataset
        from torch.utils.data import DataLoader

        model_name = self.lm_tab.lm_model_combo.currentText()

        try:
            # Get dataset
            dataset_name = self.lm_tab.lm_dataset_combo.currentText()
            seq_len = self.lm_tab.lm_seqlen_spin.value()

            dataset = get_lm_dataset(dataset_name, seq_len=seq_len, split='train')
            vocab_size = dataset.vocab_size if hasattr(dataset, 'vocab_size') else 256

            spec = get_model_spec(model_name)

            model = create_model(
                spec=spec,
                input_dim=None, # Uses embedding
                output_dim=vocab_size,
                hidden_dim=self.lm_tab.lm_hidden_spin.value(),
                num_layers=self.lm_tab.lm_layers_spin.value(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                task_type="lm"
            )

            # Apply steps
            if hasattr(model, 'max_steps'):
                 model.max_steps = self.lm_tab.lm_steps_spin.value()
            elif hasattr(model, 'eq_steps'):
                 model.eq_steps = self.lm_tab.lm_steps_spin.value()

            train_loader = DataLoader(dataset, batch_size=self.lm_tab.lm_batch_spin.value(), shuffle=True)
            return model, train_loader

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create LM model:\n{e}")
            import traceback
            traceback.print_exc()
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
        hyperparams = self._get_current_hyperparams(self.vis_tab.vis_hyperparam_widgets)

        # Update parameter count
        if hasattr(self.vis_tab, 'vis_param_label'):
            count = count_parameters(self.model)
            self.vis_tab.vis_param_label.setText(f"Parameters: {format_parameter_count(count)}")

        # Create and start worker
        micro_interval = 1 if self.vis_tab.vis_micro_check.isChecked() else 0

        self.worker = TrainingWorker(
            self.model,
            self.train_loader,
            epochs=self.vis_tab.vis_epochs_spin.value(),
            lr=self.vis_tab.vis_lr_spin.value(),
            use_compile=self.vis_tab.vis_compile_check.isChecked(),
            use_kernel=self.vis_tab.vis_kernel_check.isChecked(),
            hyperparams=hyperparams,
            microscope_interval=micro_interval,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.weights_updated.connect(self._update_weight_visualization)
        self.worker.log.connect(self._append_log)
        self.worker.dynamics_update.connect(self._on_dynamics_update)

        # Update UI
        self.vis_tab.vis_train_btn.setEnabled(False)
        self.vis_tab.vis_stop_btn.setEnabled(True)
        self.vis_tab.vis_progress.setMaximum(self.vis_tab.vis_epochs_spin.value())
        self.vis_tab.vis_progress.setValue(0)

        import time
        self.start_time = time.time()

        model_name = self.vis_tab.vis_model_combo.currentText()
        self.status_label.setText(f"Training {model_name}...")
        self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
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

        # Keep reference for generation
        self.lm_tab.update_model_ref(self.model)

        # Clear history
        self.loss_history.clear()
        self.acc_history.clear()
        self.lipschitz_history.clear()

        # Get hyperparameters
        hyperparams = self._get_current_hyperparams(self.lm_tab.lm_hyperparam_widgets)

        # Update parameter count
        if hasattr(self.lm_tab, 'lm_param_label'):
            count = count_parameters(self.model)
            self.lm_tab.lm_param_label.setText(f"Parameters: {format_parameter_count(count)}")

        # Create and start worker
        micro_interval = 1 if self.lm_tab.lm_micro_check.isChecked() else 0

        self.worker = TrainingWorker(
            self.model,
            self.train_loader,
            epochs=self.lm_tab.lm_epochs_spin.value(),
            lr=self.lm_tab.lm_lr_spin.value(),
            use_compile=self.lm_tab.lm_compile_check.isChecked(),
            use_kernel=self.lm_tab.lm_kernel_check.isChecked(),
            hyperparams=hyperparams,
            microscope_interval=micro_interval,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.weights_updated.connect(self._update_weight_visualization)
        self.worker.log.connect(self._append_log)
        self.worker.dynamics_update.connect(self._on_dynamics_update)

        # Update UI
        self.lm_tab.lm_train_btn.setEnabled(False)
        self.lm_tab.lm_stop_btn.setEnabled(True)
        self.lm_tab.lm_progress.setMaximum(self.lm_tab.lm_epochs_spin.value())
        self.lm_tab.lm_progress.setValue(0)

        import time
        self.start_time = time.time()

        model_name = self.lm_tab.lm_model_combo.currentText()
        dataset_name = self.lm_tab.lm_dataset_combo.currentText()
        self.status_label.setText(f"Training {model_name} on {dataset_name}...")
        self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
        self.plot_timer.start(100)
        self.worker.start()

    def _start_rl_training(self):
        """Start RL training."""
        import torch
        from bioplausible.models.looped_mlp import LoopedMLP
        from bioplausible.models import BackpropMLP
        import gymnasium as gym

        env_name = self.rl_tab.rl_env_combo.currentText()
        algo_name = self.rl_tab.rl_algo_combo.currentText()
        grad_method = self.rl_tab.rl_grad_combo.currentText()
        hidden = self.rl_tab.rl_hidden_spin.value()
        steps = self.rl_tab.rl_steps_spin.value()
        episodes = self.rl_tab.rl_episodes_spin.value()
        lr = self.rl_tab.rl_lr_spin.value()

        # Determine Dimensions
        temp_env = gym.make(env_name)
        input_dim = temp_env.observation_space.shape[0]
        output_dim = temp_env.action_space.n
        temp_env.close()

        # Create Model
        if "Standard Backprop" in algo_name:
            model = BackpropMLP(input_dim, hidden, output_dim)
        else:
            # LoopedMLP
            model = LoopedMLP(
                input_dim, hidden, output_dim,
                max_steps=steps,
                gradient_method=grad_method, # 'equilibrium' or 'bptt'
                use_spectral_norm=True
            )

        # Clear history
        self.rl_reward_history.clear()
        self.rl_loss_history.clear()
        self.rl_avg_reward_history.clear()

        # Create Worker
        # Use CPU unless explicitly needed, RL is often CPU bound on small envs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gamma = self.rl_tab.rl_gamma_spin.value()

        self.worker = RLWorker(model, env_name, episodes=episodes, lr=lr, gamma=gamma, device=device)
        self.worker.progress.connect(self._on_rl_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._append_log)

        # Update UI
        self.rl_tab.rl_train_btn.setEnabled(False)
        self.rl_tab.rl_stop_btn.setEnabled(True)
        self.rl_tab.rl_progress.setMaximum(episodes)
        self.rl_tab.rl_progress.setValue(0)

        import time
        self.start_time = time.time()

        self.status_label.setText(f"Training {algo_name} on {env_name}...")
        self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
        self.plot_timer.start(100)
        self.worker.start()

    def _on_rl_progress(self, metrics: dict):
        """Handle RL progress."""
        self.rl_reward_history.append(metrics['reward'])
        self.rl_loss_history.append(metrics['loss'])
        self.rl_avg_reward_history.append(metrics['avg_reward'])

        self.rl_tab.rl_progress.setValue(metrics['episode'])
        self.rl_tab.rl_avg_label.setText(f"{metrics['avg_reward']:.1f}")

        # Calculate ETA
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            if metrics['episode'] > 0:
                speed = metrics['episode'] / elapsed
                remaining = metrics['total_episodes'] - metrics['episode']
                eta_seconds = remaining / speed if speed > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                self.rl_tab.rl_eta_label.setText(f"ETA: {eta_str} | Speed: {speed:.1f} ep/s")

        self.status_label.setText(
            f"Ep {metrics['episode']}/{metrics['total_episodes']} | "
            f"Reward: {metrics['reward']:.1f} | "
            f"Avg: {metrics['avg_reward']:.1f}"
        )

    def _stop_training(self):
        """Stop training."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping training...")
            self.status_label.setStyleSheet("color: #ffaa00; padding: 5px; font-weight: bold;")

    def _on_dynamics_update(self, dynamics: dict):
        """Handle live dynamics update from worker."""
        # Update microscope plots if they exist
        deltas = dynamics.get('deltas', [])
        traj = dynamics.get('trajectory', [])

        # Calculate activity
        activities = []
        if traj:
            try:
                activities = [h.abs().mean().item() for h in traj]
            except:
                pass
        else:
            activities = [0.0] * len(deltas)

        self.micro_tab.update_plots_from_data(deltas, activities)

    def _on_progress(self, metrics: dict):
        """Handle training progress update."""
        self.loss_history.append(metrics['loss'])
        self.acc_history.append(metrics['accuracy'])
        self.lipschitz_history.append(metrics['lipschitz'])

        # Update progress bar
        self.vis_tab.vis_progress.setValue(metrics['epoch'])
        self.lm_tab.lm_progress.setValue(metrics['epoch'])

        # Calculate ETA
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            if metrics['epoch'] > 0:
                speed = metrics['epoch'] / elapsed
                remaining = metrics['total_epochs'] - metrics['epoch']
                eta_seconds = remaining / speed if speed > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                # Update both tabs just in case (or determine active tab)
                eta_text = f"ETA: {eta_str} | Speed: {speed:.2f} ep/s"
                self.vis_tab.vis_eta_label.setText(eta_text)
                self.lm_tab.lm_eta_label.setText(eta_text)

        # Update labels
        self.vis_tab.vis_acc_label.setText(f"{metrics['accuracy']:.1%}")
        self.vis_tab.vis_loss_label.setText(f"{metrics['loss']:.4f}")
        self.vis_tab.vis_lip_label.setText(f"{metrics['lipschitz']:.4f}")

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
            if hasattr(self.vis_tab, 'vis_loss_curve'):
                self.vis_tab.vis_loss_curve.setData(epochs, self.loss_history)
                # Check for separate acc plot
                if hasattr(self.vis_tab, 'vis_acc_curve'):
                    self.vis_tab.vis_acc_curve.setData(epochs, self.acc_history)
                self.vis_tab.vis_lip_curve.setData(epochs, self.lipschitz_history)

            # Update LM plots
            if hasattr(self.lm_tab, 'lm_loss_curve'):
                self.lm_tab.lm_loss_curve.setData(epochs, self.loss_history)
                # Check for separate acc plot
                if hasattr(self.lm_tab, 'lm_acc_curve'):
                    self.lm_tab.lm_acc_curve.setData(epochs, self.acc_history)
                self.lm_tab.lm_lip_curve.setData(epochs, self.lipschitz_history)

        if self.rl_reward_history and hasattr(self.rl_tab, 'rl_reward_curve'):
            episodes = list(range(1, len(self.rl_reward_history) + 1))
            self.rl_tab.rl_reward_curve.setData(episodes, self.rl_reward_history)
            self.rl_tab.rl_avg_reward_curve.setData(episodes, self.rl_avg_reward_history)
            self.rl_tab.rl_loss_curve.setData(episodes, self.rl_loss_history)

    def _on_finished(self, result: dict):
        """Handle training completion."""
        self.plot_timer.stop()
        self.vis_tab.vis_train_btn.setEnabled(True)
        self.vis_tab.vis_stop_btn.setEnabled(False)
        self.lm_tab.lm_train_btn.setEnabled(True)
        self.lm_tab.lm_stop_btn.setEnabled(False)
        if hasattr(self.rl_tab, 'rl_train_btn'):
            self.rl_tab.rl_train_btn.setEnabled(True)
            self.rl_tab.rl_stop_btn.setEnabled(False)

        if result.get('success'):
            self.status_label.setText(f"✓ Training complete! ({result['epochs_completed']} epochs)")
            self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
        else:
            self.status_label.setText("Training stopped.")
            self.status_label.setStyleSheet("color: #ffaa00; padding: 5px;")

    def _on_error(self, error: str):
        """Handle training error."""
        self.plot_timer.stop()
        self.vis_tab.vis_train_btn.setEnabled(True)
        self.vis_tab.vis_stop_btn.setEnabled(False)
        self.lm_tab.lm_train_btn.setEnabled(True)
        self.lm_tab.lm_stop_btn.setEnabled(False)
        if hasattr(self.rl_tab, 'rl_train_btn'):
            self.rl_tab.rl_train_btn.setEnabled(True)
            self.rl_tab.rl_stop_btn.setEnabled(False)

        self.status_label.setText("Training error!")
        self.status_label.setStyleSheet("color: #ff5588; padding: 5px; font-weight: bold;")
        QMessageBox.critical(self, "Training Error", error)

    def _update_lm_hyperparams(self, model_name: str):
        """Update LM hyperparameter widgets based on selected model."""
        update_hyperparams_generic(self, model_name, self.lm_tab.lm_hyperparam_layout, self.lm_tab.lm_hyperparam_widgets, self.lm_tab.lm_hyperparam_group)

    def _update_vis_hyperparams(self, model_name: str):
        """Update Vision hyperparameter widgets based on selected model."""
        update_hyperparams_generic(self, model_name, self.vis_tab.vis_hyperparam_layout, self.vis_tab.vis_hyperparam_widgets, self.vis_tab.vis_hyperparam_group)

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
            layout = self.lm_tab.lm_weights_layout
            widgets = self.lm_tab.lm_weight_widgets
            labels = self.lm_tab.lm_weight_labels
        else:
            layout = self.vis_tab.vis_weights_layout
            widgets = self.vis_tab.vis_weight_widgets
            labels = self.vis_tab.vis_weight_labels

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

    def _setup_logging(self):
        """Setup logging to redirect to console tab."""
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.log_handler.log_signal.connect(self._append_log)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.log_handler)

    def _append_log(self, message: str):
        """Append a message to the console log."""
        if hasattr(self, 'console_log'):
             import time
             timestamp = time.strftime("%H:%M:%S")
             if not message.startswith("["): # Avoid double timestamping if already timestamped
                 self.console_log.append(f"[{timestamp}] {message}")
             else:
                 self.console_log.append(message)

        # Also append to the ConsoleTab if initialized
        if hasattr(self, 'console_tab'):
            self.console_tab.append_log(message)
