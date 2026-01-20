"""
Bioplausible Trainer Dashboard

Main window with tabbed interface for Language Modeling and Vision training.
Features stunning dark cyberpunk theme with live pyqtgraph plots.
"""

import sys
import logging
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QTextEdit, QProgressBar, QSlider,
    QSplitter, QFrame, QCheckBox, QMessageBox, QApplication, QFileDialog,
    QMenuBar, QMenu, QDialog, QFormLayout, QListWidget, QListWidgetItem, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QKeySequence, QShortcut, QAction

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

# Feature flags
ENABLE_WEIGHT_VIZ = True

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible.models.factory import create_model

from .themes import CYBERPUNK_DARK, LIGHT_THEME, PLOT_COLORS, LIGHT_PLOT_COLORS, DARK_THEME_COLORS, LIGHT_THEME_COLORS
from .worker import TrainingWorker, RLWorker, BenchmarkWorker
from .generation import UniversalGenerator, count_parameters, format_parameter_count
from .hyperparams import get_hyperparams_for_model
from .viz_utils import extract_weights, format_weight_for_display, normalize_weights_for_display, get_layer_description
from .dashboard_helpers import (
    update_hyperparams_generic,
    get_current_hyperparams_generic,
    create_weight_viz_widgets_generic,
    update_weight_visualization_generic
)
from .training_utils import (
    create_vision_model_and_loader,
    create_lm_model_and_loader,
    create_diffusion_model_and_loader,
    start_training_common
)
from .logging_utils import get_ui_logger, log_exception, safe_execute

from bioplausible_ui.tabs.lm_tab import LMTab
from bioplausible_ui.tabs.vision_tab import VisionTab
from bioplausible_ui.tabs.rl_tab import RLTab
from bioplausible_ui.tabs.microscope_tab import MicroscopeTab
from bioplausible_ui.tabs.benchmarks_tab import BenchmarksTab
from bioplausible_ui.tabs.console_tab import ConsoleTab
from bioplausible_ui.tabs.p2p_tab import P2PTab
from bioplausible_ui.tabs.discovery_tab import DiscoveryTab
from bioplausible_ui.tabs.deploy_tab import DeployTab
from bioplausible_ui.tabs.diffusion_tab import DiffusionTab


class QtLogHandler(logging.Handler, QObject):
    """Custom logging handler that emits a signal for each log record."""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


class AboutDialog(QDialog):
    """About dialog for the application."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Bioplausible Trainer")
        self.setFixedSize(500, 300)

        layout = QVBoxLayout(self)

        title = QLabel("⚡ Bioplausible Trainer v0.1.0")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel(
            "A bio-plausible neural network training platform.\n\n"
            "Features:\n"
            "• Equilibrium Propagation (EqProp) Training\n"
            "• Contrastive Hebbian Learning\n"
            "• Peer-to-Peer Neural Architecture Search\n"
            "• Live Training Dynamics Visualization"
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        layout.addStretch()

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class EqPropDashboard(QMainWindow):
    """Main dashboard window for Bioplausible training."""

    def __init__(self, initial_config: Optional[Dict] = None):
        super().__init__()
        self.initial_config = initial_config

        # Initialize logger
        self.logger = get_ui_logger('dashboard')
        self.logger.info("Initializing Bioplausible Trainer Dashboard")

        self.setWindowTitle("⚡ Bioplausible Trainer v0.1.0")
        self.setGeometry(100, 100, 1400, 900)

        # Theme state
        self.current_theme = 'dark'
        from .themes import THEME_REGISTRY
        self.setStyleSheet(THEME_REGISTRY[self.current_theme]['theme_css'])

        # Setup system logger
        self._setup_logging()

        # Initialize training state
        self._initialize_training_state()

        # Initialize plot data
        self._initialize_plot_data()

        # Initialize UI
        self._setup_ui()

        # Initialize timers
        self._setup_timers()

        # Apply initial configuration if provided
        if self.initial_config:
            QTimer.singleShot(100, lambda: self._apply_config(self.initial_config))

    def _initialize_training_state(self):
        """Initialize training-related state variables."""
        self.worker: Optional[TrainingWorker] = None
        self.model = None
        self.train_loader = None
        self.current_hyperparams: Dict = {}  # Model-specific hyperparameters
        self.generator: Optional[UniversalGenerator] = None
        self.start_time = None

    def _initialize_plot_data(self):
        """Initialize plot data storage."""
        # Training metrics
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []
        self.lipschitz_history: List[float] = []

        # RL metrics
        self.rl_reward_history: List[float] = []
        self.rl_loss_history: List[float] = []
        self.rl_avg_reward_history: List[float] = []

    def _setup_timers(self):
        """Setup QTimer objects."""
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply initial configuration to UI elements."""
        try:
            model_name = config.get('model_name', '')
            self.logger.info(f"Applying configuration for model: {model_name}")

            # Determine if it's Vision or LM based on model name or config
            is_vision = any(x in model_name for x in ['MLP', 'Conv', 'Vision']) or 'mnist' in str(self.initial_config).lower() or 'cifar' in str(self.initial_config).lower()

            if is_vision:
                self.content_stack.setCurrentIndex(1)
                self.nav_list.setCurrentRow(1)
                combo = self.vis_tab.vis_model_combo
                hidden_spin = self.vis_tab.vis_hidden_spin
                steps_spin = self.vis_tab.vis_steps_spin
                lr_spin = self.vis_tab.vis_lr_spin
                epochs_spin = self.vis_tab.vis_epochs_spin
            else:
                self.content_stack.setCurrentIndex(0)
                self.nav_list.setCurrentRow(0)
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

            # Apply dynamic hyperparameters if present
            if 'hyperparams' in config:
                # This needs to happen after model selection updates the widgets
                QApplication.processEvents()
                # Assuming widgets are updated, we'd need to set them here.
                # Simplification: we might need a delay or signal to handle this robustly.
                pass

            self.status_label.setText(f"Loaded configuration for {model_name}")
            self.status_label.setStyleSheet("color: #00aacc; padding: 5px;")
            self.logger.info(f"Configuration applied successfully for {model_name}")

        except Exception as e:
            self.logger.error(f"Error applying config: {e}")
            log_exception(self.logger, "_apply_config")
            self.status_label.setText(f"Error loading config: {e}")
            self.status_label.setStyleSheet("color: #ff5588; padding: 5px;")

    def _setup_ui(self):
        """Set up the main user interface."""
        # Menu Bar
        self._create_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main Splitter: Sidebar | Content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        # --- Sidebar ---
        sidebar_widget = self._create_sidebar()
        main_splitter.addWidget(sidebar_widget)

        # --- Content Area ---
        content_widget = self._create_content_area()
        main_splitter.addWidget(content_widget)
        main_splitter.setStretchFactor(1, 1) # Content stretches

        # Initialize Tabs (Pages) - with lazy loading
        self._initialize_tabs()

        # Status bar
        self._setup_status_bar()

        # Keyboard Shortcuts
        self._setup_keyboard_shortcuts()

    def _create_sidebar(self):
        """Create the sidebar widget."""
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(250)
        sidebar_widget.setStyleSheet("background-color: #15151a; border-right: 1px solid #333;")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(10)

        # Header
        header = QLabel("⚡ Bioplausible\nTrainer")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00d4ff; margin-bottom: 20px;")
        header.setToolTip("Bioplausible Neural Network Trainer - Bio-plausible learning algorithms")
        sidebar_layout.addWidget(header)

        # Navigation List
        self.nav_list = self._create_navigation_list()
        sidebar_layout.addWidget(self.nav_list)

        sidebar_layout.addStretch()

        # Save/Load Buttons in Sidebar
        self.save_btn = QPushButton("💾 Save Checkpoint")
        self.save_btn.clicked.connect(self._save_model)
        self.save_btn.setStyleSheet("text-align: left; padding: 10px;")
        self.save_btn.setToolTip("Save the current model and training configuration")
        sidebar_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("📂 Load Checkpoint")
        self.load_btn.clicked.connect(self._load_model)
        self.load_btn.setStyleSheet("text-align: left; padding: 10px;")
        self.load_btn.setToolTip("Load a previously saved model and training configuration")
        sidebar_layout.addWidget(self.load_btn)

        return sidebar_widget

    def _create_navigation_list(self):
        """Create the navigation list widget."""
        nav_list = QListWidget()
        nav_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QListWidget::item {
                padding: 12px;
                border-radius: 6px;
                color: #a0a0b0;
                margin-bottom: 4px;
            }
            QListWidget::item:selected {
                background-color: #2c3e50;
                color: #ffffff;
                border-left: 4px solid #00d4ff;
            }
            QListWidget::item:hover {
                background-color: #1e2530;
            }
        """)
        nav_list.setToolTip("Select a training task from the menu")

        items = [
            ("🔤 Language Model", 0),
            ("📷 Vision", 1),
            ("✨ Diffusion", 10),
            ("🎮 RL Agent", 2),
            ("🔬 Microscope", 3),
            ("🔍 Model Search", 4),
            ("🗺️ Discovery", 5),
            ("🌐 Community Grid", 6),
            ("🏆 Benchmarks", 7),
            ("🚀 Deploy", 8),
            ("💻 Console", 9)
        ]

        for name, idx in items:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            nav_list.addItem(item)

        nav_list.currentRowChanged.connect(self._on_nav_changed)
        return nav_list

    def _create_content_area(self):
        """Create the main content area."""
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)

        # We use a StackedWidget instead of Tabs
        self.content_stack = QStackedWidget()
        content_layout.addWidget(self.content_stack)

        return content_widget

    def _initialize_tabs(self):
        """Initialize tabs with lazy loading."""
        # Initialize Tabs (Pages) - with lazy loading
        self.tab_classes = {
            0: LMTab,
            1: VisionTab,
            2: RLTab,
            3: MicroscopeTab,
            4: None,  # Search tab is created dynamically
            5: DiscoveryTab,
            6: P2PTab,
            7: BenchmarksTab,
            8: DeployTab,
            9: ConsoleTab,
            10: DiffusionTab
        }

        # Pre-create only the first tab (active one)
        self.lm_tab = LMTab()
        self.lm_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
        self.lm_tab.stop_training_signal.connect(self._stop_training)
        self.lm_tab.clear_plots_signal.connect(self._clear_plots)
        self.content_stack.addWidget(self.lm_tab)

        # Create placeholder widgets for other tabs
        for i in range(1, 11):  # 1 to 10 (excluding 4 which is special)
            if i == 4:  # Search tab
                search_placeholder = QWidget()
                search_placeholder.setLayout(QVBoxLayout())
                search_placeholder.layout().addWidget(QLabel("Search tab - will load on demand"))
                self.content_stack.addWidget(search_placeholder)
            else:
                placeholder = QWidget()
                placeholder.setLayout(QVBoxLayout())
                placeholder.layout().addWidget(QLabel(f"Tab {i} - Loading..."))
                self.content_stack.addWidget(placeholder)

        # Note: Search tab (index 4) is handled specially in _init_tab_lazy

        # Track which tabs have been initialized
        self.tab_initialized = {0: True}  # First tab is already initialized
        for i in range(1, 11):
            self.tab_initialized[i] = False

        # Select first item
        self.nav_list.setCurrentRow(0)

    def _setup_status_bar(self):
        """Setup the status bar with indicators."""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready. Select a model and dataset to begin training.")
        self.status_label.setStyleSheet("color: #808090; padding: 5px;")
        self.status_label.setToolTip("Current status of the application")
        self.status_bar.addWidget(self.status_label, 1) # Stretch

        # Device Indicator
        import torch
        try:
            from bioplausible.kernel import HAS_TRITON_OPS
        except ImportError:
            HAS_TRITON_OPS = False

        device_name = "CPU"
        if torch.cuda.is_available():
            device_name = "CUDA"
            if HAS_TRITON_OPS:
                device_name += " (Triton Accel)"

        self.device_label = QLabel(f"Device: {device_name}")
        self.device_label.setStyleSheet("color: #00d4ff; font-weight: bold; padding: 0 10px;")
        self.device_label.setToolTip("Current compute device being used for training")
        self.status_bar.addPermanentWidget(self.device_label)

        # GPU and RAM monitoring
        try:
            import psutil
            self.gpu_monitor_label = QLabel("GPU: N/A | RAM: N/A")
            self.gpu_monitor_label.setStyleSheet("color: #00ff88; font-weight: bold; padding: 0 10px;")
            self.gpu_monitor_label.setToolTip("Real-time GPU and RAM usage statistics")
            self.status_bar.addPermanentWidget(self.gpu_monitor_label)

            # Timer for updating GPU/RAM stats
            self.system_monitor_timer = QTimer()
            self.system_monitor_timer.timeout.connect(self._update_system_stats)
            self.system_monitor_timer.start(2000)  # Update every 2 seconds
        except ImportError:
            # psutil not available, skip GPU/RAM monitoring
            pass

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the application."""
        # Training shortcuts
        self.train_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.train_shortcut.activated.connect(self._on_train_shortcut)
        self.train_shortcut.setToolTip("Start training (Ctrl+Return)")

        self.stop_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.stop_shortcut.activated.connect(self._stop_training)
        self.stop_shortcut.setToolTip("Stop training (Ctrl+Q)")

        # Additional global shortcuts
        self.ctrl_t_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self.ctrl_t_shortcut.activated.connect(self._on_train_shortcut)
        self.ctrl_t_shortcut.setToolTip("Start Training (Ctrl+T)")

        self.ctrl_s_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.ctrl_s_shortcut.activated.connect(self._save_model)
        self.ctrl_s_shortcut.setToolTip("Save Model (Ctrl+S)")

        self.ctrl_o_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.ctrl_o_shortcut.activated.connect(self._load_model)
        self.ctrl_o_shortcut.setToolTip("Load Model (Ctrl+O)")

        self.ctrl_r_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self.ctrl_r_shortcut.activated.connect(self._clear_plots)
        self.ctrl_r_shortcut.setToolTip("Reset Plots (Ctrl+R)")

        self.ctrl_e_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self.ctrl_e_shortcut.activated.connect(self._export_logs)
        self.ctrl_e_shortcut.setToolTip("Export Results (Ctrl+E)")

        self.ctrl_m_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        self.ctrl_m_shortcut.activated.connect(self._run_microscope_analysis)
        self.ctrl_m_shortcut.setToolTip("Microscope Capture (Ctrl+M)")

        self.ctrl_question_shortcut = QShortcut(QKeySequence("Ctrl+?"), self)
        self.ctrl_question_shortcut.activated.connect(self._show_shortcuts)
        self.ctrl_question_shortcut.setToolTip("Show Shortcuts (Ctrl+?)")

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # === File Menu ===
        file_menu = menubar.addMenu("&File")

        save_action = QAction("Save Model...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Save the current model and its training configuration")
        save_action.triggered.connect(self._save_model)
        file_menu.addAction(save_action)

        load_action = QAction("Load Model...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.setToolTip("Load a previously saved model and its configuration")
        load_action.triggered.connect(self._load_model)
        file_menu.addAction(load_action)

        # Recent Models submenu
        self.recent_models_menu = file_menu.addMenu("Recent Models")
        self.recent_models_menu.setToolTip("Access recently opened model files")
        self._load_recent_models()
        self._update_recent_models_menu()

        file_menu.addSeparator()

        save_config_action = QAction("Save Configuration...", self)
        save_config_action.setToolTip("Save only the hyperparameters (no weights)")
        save_config_action.triggered.connect(self._save_config_only)
        file_menu.addAction(save_config_action)

        load_config_action = QAction("Load Configuration...", self)
        load_config_action.setToolTip("Load a configuration file without model weights")
        load_config_action.triggered.connect(self._load_config_only)
        file_menu.addAction(load_config_action)

        file_menu.addSeparator()

        export_logs_action = QAction("Export Training Log...", self)
        export_logs_action.setToolTip("Export loss and accuracy history to CSV")
        export_logs_action.triggered.connect(self._export_logs)
        file_menu.addAction(export_logs_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+W")
        exit_action.setToolTip("Close the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # === View Menu ===
        view_menu = menubar.addMenu("&View")

        self.theme_action = QAction("Switch to Next Theme (now: Dark)", self)
        self.theme_action.setToolTip("Cycle through available UI themes")
        self.theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.theme_action)

        # === Help Menu ===
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setToolTip("Show available keyboard shortcuts")
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

        about_action = QAction("About", self)
        about_action.setToolTip("Show information about the Bioplausible Trainer")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _toggle_theme(self):
        """Cycle through available themes."""
        from .themes import THEME_REGISTRY

        # Get list of available themes
        available_themes = list(THEME_REGISTRY.keys())

        # Find next theme in cycle
        current_index = available_themes.index(self.current_theme)
        next_index = (current_index + 1) % len(available_themes)
        next_theme_name = available_themes[next_index]

        # Apply new theme
        next_theme = THEME_REGISTRY[next_theme_name]
        self.setStyleSheet(next_theme['theme_css'])
        self.current_theme = next_theme_name

        # Update theme action text
        self.theme_action.setText(f"Switch to Next Theme (now: {next_theme_name.title()})")

        # Update plots with new theme colors
        ui_colors = next_theme['ui']
        plot_colors = next_theme['plot_colors']

        if hasattr(self, 'lm_tab'):
            self.lm_tab.update_theme(ui_colors, plot_colors)
        if hasattr(self, 'vis_tab'):
            self.vis_tab.update_theme(ui_colors, plot_colors)
        if hasattr(self, 'rl_tab'):
            self.rl_tab.update_theme(ui_colors, plot_colors)
        if hasattr(self, 'micro_tab'):
            self.micro_tab.update_theme(ui_colors, plot_colors)
        if hasattr(self, 'bench_tab'):
            self.bench_tab.update_theme(ui_colors, plot_colors)
        if hasattr(self, 'deploy_tab'):
            self.deploy_tab.update_theme(ui_colors, plot_colors)
        if hasattr(self, 'diff_tab'):
            self.diff_tab.update_theme(ui_colors, plot_colors)

        self.status_label.setText(f"Switched to {next_theme_name.title()} theme")

    def _show_about(self):
        """Show About dialog."""
        dlg = AboutDialog(self)
        dlg.exec()

    def _show_shortcuts(self):
        """Show keyboard shortcuts."""
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "Ctrl+Return: Start Training\n"
            "Ctrl+T: Start Training\n"
            "Ctrl+Q: Stop Training\n"
            "Ctrl+S: Save Model\n"
            "Ctrl+O: Load Model\n"
            "Ctrl+R: Reset Plots\n"
            "Ctrl+E: Export Results\n"
            "Ctrl+M: Microscope Capture\n"
            "Ctrl+?: Show Shortcuts\n"
            "Ctrl+W: Exit"
        )

    def closeEvent(self, event):
        """Handle application close event."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 'Training in Progress',
                "Training is currently running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

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

    def _on_nav_changed(self, row: int) -> None:
        """Handle sidebar navigation with lazy loading."""
        # Check if tab needs to be initialized
        if not self.tab_initialized.get(row, False):
            self._init_tab_lazy(row)

        self.content_stack.setCurrentIndex(row)

    def _init_tab_lazy(self, tab_index: int) -> None:
        """Initialize a tab lazily when first accessed."""
        if tab_index == 4:  # Search tab - special handling
            search_tab = self._create_search_tab()
            # Replace the placeholder
            self.content_stack.removeWidget(self.content_stack.widget(tab_index))
            self.content_stack.insertWidget(tab_index, search_tab)
        elif self.tab_classes.get(tab_index):
            # Create the actual tab
            TabClass = self.tab_classes[tab_index]
            actual_tab = TabClass()

            # Set up connections based on tab type
            if tab_index == 1:  # Vision tab
                actual_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
                actual_tab.stop_training_signal.connect(self._stop_training)
                actual_tab.clear_plots_signal.connect(self._clear_plots)
                self.vis_tab = actual_tab
            elif tab_index == 2:  # RL tab
                actual_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
                actual_tab.stop_training_signal.connect(self._stop_training)
                actual_tab.clear_plots_signal.connect(self._clear_plots)
                self.rl_tab = actual_tab
            elif tab_index == 5:  # Discovery tab
                actual_tab.load_model_signal.connect(self._apply_config)
                self.disc_tab = actual_tab
            elif tab_index == 6:  # P2P tab
                actual_tab.bridge_log_signal.connect(self._append_log)
                actual_tab.bridge_status_signal.connect(lambda s, p, j: self.disc_tab.update_p2p_ref(self.p2p_tab.worker))
                actual_tab.load_model_signal.connect(self._apply_config)
                self.p2p_tab = actual_tab
            elif tab_index == 7:  # Benchmarks tab
                actual_tab.log_message.connect(self._append_log)
                actual_tab.load_model_signal.connect(self._apply_config)
                self.bench_tab = actual_tab
            elif tab_index == 10:  # Diffusion tab
                actual_tab.start_training_signal.connect(lambda mode: self._start_training(mode))
                actual_tab.stop_training_signal.connect(self._stop_training)
                actual_tab.clear_plots_signal.connect(self._clear_plots)
                self.diff_tab = actual_tab

            # Replace the placeholder widget with the actual tab
            placeholder = self.content_stack.widget(tab_index)
            self.content_stack.removeWidget(placeholder)
            self.content_stack.insertWidget(tab_index, actual_tab)

            # Clean up the placeholder
            placeholder.deleteLater()

        # Mark as initialized
        self.tab_initialized[tab_index] = True

    def _on_train_shortcut(self):
        """Handle start training shortcut based on active tab."""
        current = self.content_stack.currentWidget()
        if isinstance(current, LMTab):
            self._start_training('lm')
        elif isinstance(current, VisionTab):
            self._start_training('vision')
        elif isinstance(current, RLTab):
            self._start_training('rl')

    def _get_current_config_dict(self) -> Dict:
        """Helper to get current configuration dictionary from UI."""
        current_config = {}
        # Active tab determines which controls to read
        idx = self.content_stack.currentIndex()
        if idx == 0: # LM
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
        elif idx == 1: # Vision
            current_config.update({
                'task': 'vision',
                'model_name': self.vis_tab.vis_model_combo.currentText(),
                'hidden_dim': self.vis_tab.vis_hidden_spin.value(),
                'steps': self.vis_tab.vis_steps_spin.value(),
                'dataset': self.vis_tab.vis_dataset_combo.currentText(),
                'hyperparams': self._get_current_hyperparams(self.vis_tab.vis_hyperparam_widgets)
            })
        return current_config

    def _save_config_only(self):
        """Save only the configuration to a JSON file."""
        config = self._get_current_config_dict()
        fname, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Config (*.json)")
        if fname:
            try:
                with open(fname, 'w') as f:
                    json.dump(config, f, indent=4)
                self.status_label.setText(f"Configuration saved to {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _load_config_only(self):
        """Load configuration from a JSON file."""
        fname, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Config (*.json)")
        if fname:
            try:
                with open(fname, 'r') as f:
                    config = json.load(f)
                self._apply_config(config)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _export_logs(self):
        """Export training history to a CSV file."""
        if not self.loss_history:
             QMessageBox.warning(self, "No Data", "No training history to export.")
             return

        fname, _ = QFileDialog.getSaveFileName(self, "Export Training Logs", "", "CSV Files (*.csv)")
        if fname:
            try:
                import csv
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Lipschitz'])
                    for i in range(len(self.loss_history)):
                        acc = self.acc_history[i] if i < len(self.acc_history) else 0.0
                        lip = self.lipschitz_history[i] if i < len(self.lipschitz_history) else 0.0
                        writer.writerow([i + 1, self.loss_history[i], acc, lip])
                self.status_label.setText(f"Logs exported to {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _load_recent_models(self):
        """Load recent models from a file."""
        import os
        from pathlib import Path

        # Define the path for the recent models file
        recent_file = Path.home() / ".bioplausible" / "recent.json"

        try:
            if recent_file.exists():
                with open(recent_file, 'r') as f:
                    self.recent_models = json.load(f)
            else:
                self.recent_models = []
        except Exception:
            self.recent_models = []

    def _save_recent_models(self):
        """Save recent models to a file."""
        import os
        from pathlib import Path

        # Create directory if it doesn't exist
        recent_dir = Path.home() / ".bioplausible"
        recent_dir.mkdir(exist_ok=True)

        # Define the path for the recent models file
        recent_file = recent_dir / "recent.json"

        try:
            with open(recent_file, 'w') as f:
                json.dump(self.recent_models, f)
        except Exception as e:
            print(f"Error saving recent models: {e}")

    def _add_recent_model(self, filepath):
        """Add a model to the recent models list."""
        if not hasattr(self, 'recent_models'):
            self.recent_models = []

        # Remove the filepath if it's already in the list
        if filepath in self.recent_models:
            self.recent_models.remove(filepath)

        # Add to the beginning of the list
        self.recent_models.insert(0, filepath)

        # Keep only the last 5 models
        self.recent_models = self.recent_models[:5]

        # Save the updated list
        self._save_recent_models()

        # Update the menu
        self._update_recent_models_menu()

    def _update_recent_models_menu(self):
        """Update the recent models menu."""
        if not hasattr(self, 'recent_models'):
            self.recent_models = []

        # Clear the menu
        self.recent_models_menu.clear()

        if not self.recent_models:
            # Add a disabled "No recent models" entry
            no_recent_action = QAction("No recent models", self)
            no_recent_action.setEnabled(False)
            self.recent_models_menu.addAction(no_recent_action)
        else:
            # Add recent models to the menu
            for filepath in self.recent_models:
                action = QAction(filepath, self)
                action.triggered.connect(lambda checked, f=filepath: self._load_recent_model(f))
                self.recent_models_menu.addAction(action)

        # Add a separator and clear action
        self.recent_models_menu.addSeparator()
        clear_action = QAction("Clear Recent Models", self)
        clear_action.triggered.connect(self._clear_recent_models)
        self.recent_models_menu.addAction(clear_action)

    def _load_recent_model(self, filepath):
        """Load a recent model."""
        if os.path.exists(filepath):
            # Load the model
            self._load_model_from_path(filepath)
        else:
            # Remove the file from recent models if it doesn't exist
            if filepath in self.recent_models:
                self.recent_models.remove(filepath)
                self._save_recent_models()
                self._update_recent_models_menu()
            QMessageBox.warning(self, "File Not Found", f"The model file does not exist:\n{filepath}")

    def _load_model_from_path(self, filepath):
        """Load a model from a specific path."""
        try:
            import torch
            checkpoint = torch.load(filepath)
            config = checkpoint.get('config', {})

            # 1. Restore UI State from Config
            self._apply_config(config)

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
            self.status_label.setText(f"Model loaded from {filepath}")
            self.status_label.setStyleSheet("color: #00ff88; padding: 5px;")

            # 4. Update Tab References
            if self.model:
                self.lm_tab.update_model_ref(self.model)
                self.vis_tab.update_model_ref(self.model)
                self.micro_tab.update_model_ref(self.model)
                self.deploy_tab.update_model_ref(self.model)

            # Add to recent models
            self._add_recent_model(filepath)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load: {str(e)}")
            import traceback
            traceback.print_exc()

    def _clear_recent_models(self):
        """Clear the recent models list."""
        self.recent_models = []
        self._save_recent_models()
        self._update_recent_models_menu()

    def _save_model(self):
        """Save the current model to a file, including current UI configuration."""
        if not self.model:
            QMessageBox.warning(self, "No Model", "No model to save.")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Model Checkpoint", "", "PyTorch Checkpoints (*.pt)")
        if fname:
            try:
                import torch
                current_config = self._get_current_config_dict()
                state = {
                    'model_state_dict': self.model.state_dict(),
                    'config': current_config,
                    'model_name': current_config.get('model_name', 'Unknown')
                }
                torch.save(state, fname)
                self.status_label.setText(f"Model saved to {fname}")

                # Add to recent models
                self._add_recent_model(fname)
            except OSError as e:
                error_msg = f"Could not save to file {fname}: {e}"
                self._log_error(error_msg)
                QMessageBox.critical(self, "Save Error", error_msg)
            except Exception as e:
                error_msg = f"Failed to save model: {e}"
                self._log_error(error_msg)
                QMessageBox.critical(self, "Save Error", error_msg)

    def _load_model(self):
        """Load a model from a file and restore UI state."""
        fname, _ = QFileDialog.getOpenFileName(self, "Load Model Checkpoint", "", "PyTorch Checkpoints (*.pt)")
        if fname:
            try:
                self.logger.info(f"Loading model from: {fname}")
                import torch
                checkpoint = torch.load(fname)
                config = checkpoint.get('config', {})

                # 1. Restore UI State from Config
                self._apply_config(config)

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
                self.logger.info(f"Model loaded successfully from {fname}")

                # 4. Update Tab References
                if self.model:
                    self.lm_tab.update_model_ref(self.model)
                    self.vis_tab.update_model_ref(self.model)
                    self.micro_tab.update_model_ref(self.model)
                    self.deploy_tab.update_model_ref(self.model)

                # Add to recent models
                self._add_recent_model(fname)

            except FileNotFoundError:
                error_msg = f"Model file not found: {fname}"
                self._log_error(error_msg)
                self.logger.error(error_msg)
                QMessageBox.critical(self, "File Not Found", error_msg)
            except KeyError as e:
                error_msg = f"Missing key in checkpoint: {e}"
                self._log_error(error_msg)
                self.logger.error(error_msg)
                QMessageBox.critical(self, "Load Error", error_msg)
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                self._log_error(error_msg)
                self.logger.error(f"Failed to load model: {str(e)}")
                log_exception(self.logger, "_load_model")
                QMessageBox.critical(self, "Load Error", error_msg)

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
            elif mode == 'diffusion':
                self._start_diffusion_training()
            elif mode == 'rl':
                self._start_rl_training()
            else:
                self._start_lm_training()

        except ImportError as e:
            error_msg = f"Missing dependency: {e}"
            self._log_error(error_msg)
            QMessageBox.critical(self, "Import Error", error_msg)
        except Exception as e:
            error_msg = f"Failed to start training: {e}"
            self._log_error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _start_vision_training(self):
        """Start vision model training."""
        # Create model and data loader
        model, train_loader = create_vision_model_and_loader(self.vis_tab)

        if model is None or train_loader is None:
            return  # Error already shown to user

        self.model = model
        self.train_loader = train_loader

        # Keep reference for inference
        self.vis_tab.update_model_ref(self.model)
        self.deploy_tab.update_model_ref(self.model)

        # Create and start worker using common method
        self.worker = start_training_common(
            model=self.model,
            train_loader=train_loader,
            tab_instance=self.vis_tab,
            tab_name='vis',
            hyperparam_widgets=self.vis_tab.vis_hyperparam_widgets,
            microscope_check=self.vis_tab.vis_micro_check,
            compile_check=self.vis_tab.vis_compile_check,
            kernel_check=self.vis_tab.vis_kernel_check,
            epochs_spin=self.vis_tab.vis_epochs_spin,
            lr_spin=self.vis_tab.vis_lr_spin,
            progress_bar=self.vis_tab.vis_progress,
            status_label=self.status_label,
            worker_class=TrainingWorker
        )

        # Additional connections specific to vision training
        if self.worker:
            self.worker.weights_updated.connect(lambda w: self._update_weight_visualization(w, is_grad=False))
            self.worker.gradients_updated.connect(lambda g: self._update_weight_visualization(g, is_grad=True))
            self.worker.log.connect(self._append_log)
            self.worker.dynamics_update.connect(self._on_dynamics_update)

    def _start_lm_training(self):
        """Start language model training."""
        # Create model and data loader
        model, train_loader = create_lm_model_and_loader(self.lm_tab)

        if model is None or train_loader is None:
            return  # Error already shown to user

        self.model = model
        self.train_loader = train_loader

        # Keep reference for generation
        self.lm_tab.update_model_ref(self.model)
        self.deploy_tab.update_model_ref(self.model)

        # Create and start worker using common method
        self.worker = start_training_common(
            model=self.model,
            train_loader=train_loader,
            tab_instance=self.lm_tab,
            tab_name='lm',
            hyperparam_widgets=self.lm_tab.lm_hyperparam_widgets,
            microscope_check=self.lm_tab.lm_micro_check,
            compile_check=self.lm_tab.lm_compile_check,
            kernel_check=self.lm_tab.lm_kernel_check,
            epochs_spin=self.lm_tab.lm_epochs_spin,
            lr_spin=self.lm_tab.lm_lr_spin,
            progress_bar=self.lm_tab.lm_progress,
            status_label=self.status_label,
            worker_class=TrainingWorker
        )

        # Additional connections specific to LM training
        if self.worker:
            self.worker.weights_updated.connect(self._update_weight_visualization)
            self.worker.log.connect(self._append_log)
            self.worker.dynamics_update.connect(self._on_dynamics_update)

    def _start_diffusion_training(self):
        """Start diffusion training."""
        model, train_loader = create_diffusion_model_and_loader(self.diff_tab)

        if model is None or train_loader is None:
            return

        self.model = model
        self.train_loader = train_loader

        # Refs
        self.diff_tab.update_model_ref(self.model)
        self.deploy_tab.update_model_ref(self.model)

        # Create and start worker using common method
        self.worker = start_training_common(
            model=self.model,
            train_loader=train_loader,
            tab_instance=self.diff_tab,
            tab_name='diff',
            epochs_spin=self.diff_tab.epochs_spin,
            lr_spin=self.diff_tab.lr_spin,
            progress_bar=self.diff_tab.progress,
            status_label=self.status_label,
            worker_class=TrainingWorker
        )

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

        # Keep reference for playback
        self.rl_tab.update_model_ref(model)
        self.deploy_tab.update_model_ref(model)

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

    def _log_error(self, message: str):
        """Log an error message."""
        import logging
        logging.error(message)
        self._append_log(f"ERROR: {message}")

    def _log_warning(self, message: str):
        """Log a warning message."""
        import logging
        logging.warning(message)
        self._append_log(f"WARNING: {message}")

    def _log_info(self, message: str):
        """Log an info message."""
        import logging
        logging.info(message)
        self._append_log(f"INFO: {message}")

    def _safe_execute(self, func, *args, **kwargs):
        """Safely execute a function and handle exceptions."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            self._log_error(error_msg)
            import traceback
            self._append_log(f"Traceback: {traceback.format_exc()}")
            return None

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
        # This method is now replaced by the utility function
        # We keep it for backward compatibility if needed elsewhere
        try:
            return create_vision_model_and_loader(self.vis_tab)
        except Exception as e:
            self._log_error(f"Error creating vision model and loader: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _create_lm_model_and_loader(self):
        """Create language model and data loader."""
        # This method is now replaced by the utility function
        # We keep it for backward compatibility if needed elsewhere
        try:
            return create_lm_model_and_loader(self.lm_tab)
        except Exception as e:
            self._log_error(f"Error creating LM model and loader: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _create_diffusion_model_and_loader(self):
        """Create diffusion model and data loader."""
        # This method is now replaced by the utility function
        # We keep it for backward compatibility if needed elsewhere
        try:
            return create_diffusion_model_and_loader(self.diff_tab)
        except Exception as e:
            self._log_error(f"Error creating diffusion model and loader: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _reset_training_ui(self):
        """Reset UI state after training stops."""
        self.vis_tab.vis_train_btn.setEnabled(True)
        self.vis_tab.vis_stop_btn.setEnabled(False)
        self.vis_tab.vis_pause_btn.setEnabled(False)
        self.vis_tab.vis_pause_btn.setChecked(False)

        self.lm_tab.lm_train_btn.setEnabled(True)
        self.lm_tab.lm_stop_btn.setEnabled(False)
        self.lm_tab.lm_pause_btn.setEnabled(False)
        self.lm_tab.lm_pause_btn.setChecked(False)

        if hasattr(self.rl_tab, 'rl_train_btn'):
            self.rl_tab.rl_train_btn.setEnabled(True)
            self.rl_tab.rl_stop_btn.setEnabled(False)

    def _on_finished(self, result: dict):
        """Handle training completion."""
        self.plot_timer.stop()
        self._reset_training_ui()

        if result.get('success'):
            self.status_label.setText(f"✓ Training complete! ({result['epochs_completed']} epochs)")
            self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
        else:
            self.status_label.setText("Training stopped.")
            self.status_label.setStyleSheet("color: #ffaa00; padding: 5px;")

    def _on_error(self, error: str):
        """Handle training error."""
        self.plot_timer.stop()
        self._reset_training_ui()

        self.status_label.setText("Training error!")
        self.status_label.setStyleSheet("color: #ff5588; padding: 5px; font-weight: bold;")
        QMessageBox.critical(self, "Training Error", error)

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

        # Keep reference for playback
        self.rl_tab.update_model_ref(model)
        self.deploy_tab.update_model_ref(model)

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

    def _toggle_pause(self):
        """Pause/Resume training."""
        if not self.worker: return

        is_paused = self.vis_tab.vis_pause_btn.isChecked() or self.lm_tab.lm_pause_btn.isChecked()

        if is_paused:
            self.worker.pause()
            self.status_label.setText("Training paused.")
            self.status_label.setStyleSheet("color: #f1c40f; padding: 5px; font-weight: bold;")
        else:
            self.worker.resume()
            self.status_label.setText("Training resumed.")
            self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")

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

    def _reset_training_ui(self):
        """Reset UI state after training stops."""
        self.vis_tab.vis_train_btn.setEnabled(True)
        self.vis_tab.vis_stop_btn.setEnabled(False)
        self.vis_tab.vis_pause_btn.setEnabled(False)
        self.vis_tab.vis_pause_btn.setChecked(False)

        self.lm_tab.lm_train_btn.setEnabled(True)
        self.lm_tab.lm_stop_btn.setEnabled(False)
        self.lm_tab.lm_pause_btn.setEnabled(False)
        self.lm_tab.lm_pause_btn.setChecked(False)

        if hasattr(self.rl_tab, 'rl_train_btn'):
            self.rl_tab.rl_train_btn.setEnabled(True)
            self.rl_tab.rl_stop_btn.setEnabled(False)

    def _on_finished(self, result: dict):
        """Handle training completion."""
        self.plot_timer.stop()
        self._reset_training_ui()

        if result.get('success'):
            self.status_label.setText(f"✓ Training complete! ({result['epochs_completed']} epochs)")
            self.status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
        else:
            self.status_label.setText("Training stopped.")
            self.status_label.setStyleSheet("color: #ffaa00; padding: 5px;")

    def _on_error(self, error: str):
        """Handle training error."""
        self.plot_timer.stop()
        self._reset_training_ui()

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

    def _update_weight_visualization(self, weights: dict, is_grad: bool = False):
        """Update weight visualization heatmaps."""
        if not HAS_PYQTGRAPH or not ENABLE_WEIGHT_VIZ:
            return

        # Determine which tab is active to update correct widgets
        active_idx = self.content_stack.currentIndex()

        # 0 = LM Tab, 1 = Vision Tab
        if active_idx == 0:
            layout = self.lm_tab.lm_weights_layout
            widgets = self.lm_tab.lm_weight_widgets
            labels = self.lm_tab.lm_weight_labels
            # LM doesn't have toggle yet, assume Weights only
            if is_grad: return
        elif active_idx == 1:
            layout = self.vis_tab.vis_weights_layout
            widgets = self.vis_tab.vis_weight_widgets
            labels = self.vis_tab.vis_weight_labels
            # Check toggle
            if hasattr(self.vis_tab, 'viz_mode_combo'):
                mode = self.vis_tab.viz_mode_combo.currentText()
                if "Weights" in mode and is_grad: return
                if "Flow" in mode and not is_grad: return

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

    def _update_system_stats(self):
        """Update GPU and RAM usage statistics."""
        try:
            import psutil

            # Get RAM usage
            ram_usage = psutil.virtual_memory().percent

            # Get GPU usage if available
            gpu_usage = "N/A"
            gpu_mem = "N/A"

            try:
                import torch
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else "N/A"
                    gpu_mem = f"{torch.cuda.memory_allocated()/1024**3:.1f}GB/{torch.cuda.memory_reserved()/1024**3:.1f}GB"

                    # Format GPU usage string
                    if isinstance(gpu_usage, (int, float)):
                        gpu_usage = f"{gpu_usage}%"
                    else:
                        gpu_usage = "N/A"
            except:
                # If torch GPU operations fail, just show N/A
                gpu_usage = "N/A"
                gpu_mem = "N/A"

            # Update the label
            stats_text = f"RAM: {ram_usage}% | GPU: {gpu_usage} | Mem: {gpu_mem}"
            self.gpu_monitor_label.setText(stats_text)
        except Exception as e:
            # If anything goes wrong, silently fail to avoid disrupting the UI
            pass

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
