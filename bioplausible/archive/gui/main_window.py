
import sys
import torch
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QLabel, QGroupBox, QSplitter, QGridLayout,
    QSpinBox, QSlider, QTableWidgetItem, QScrollArea, QFrame, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor, QFont
import pyqtgraph as pg

from .widgets import StatsTable, AlgorithmCard
from .worker import TrainingWorker
from .algorithms import AlgorithmWrapper, MODEL_REGISTRY, get_model_spec
from .types import TrainingState
from .utils import load_shakespeare


class BioTrainerGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Trainer v6.0 - Model Comparison")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data
        self.data, self.c2i, self.i2c = load_shakespeare()
        self.vocab_size = len(self.c2i)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # State
        self.algorithms = {}  # name -> AlgorithmWrapper
        self.history = {}     # name -> list of states
        self.stats_cache = {} # name -> dict
        
        # Build color map from registry
        self.colors = {spec.name: spec.color for spec in MODEL_REGISTRY}
        
        # Worker
        self.worker = TrainingWorker({}, self.data, self.device)
        self.worker.update_signal.connect(self.on_worker_update)
        self.worker.start()
        self.worker.pause()

        # UI
        self.init_ui()
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(100)
        
        self.gen_timer = QTimer()
        self.gen_timer.timeout.connect(self.trigger_generation)
        self.gen_timer.start(2000)

    def init_ui(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Model Selection (Scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        
        # Title
        title = QLabel("<h3>Select Models to Compare</h3>")
        model_layout.addWidget(title)
        
        # Create checkboxes for each model
        self.model_checks = {}
        self.model_controls = {}  # name -> dict of widgets
        
        for spec in MODEL_REGISTRY:
            # Container for each model
            container = QGroupBox(spec.name)
            container.setStyleSheet(f"""
                QGroupBox {{ 
                    border: 2px solid {spec.color}; 
                    border-radius: 5px; 
                    margin-top: 10px; 
                    padding-top: 10px;
                }}
                QGroupBox::title {{ 
                    color: {spec.color}; 
                    font-weight: bold;
                }}
            """)
            c_layout = QVBoxLayout(container)
            
            # Checkbox to enable
            cb = QCheckBox("Enable")
            cb.setStyleSheet(f"color: {spec.color};")
            self.model_checks[spec.name] = cb
            c_layout.addWidget(cb)
            
            # Description
            desc = QLabel(f"<i>{spec.description}</i>")
            desc.setStyleSheet("color: #888; font-size: 10px;")
            c_layout.addWidget(desc)
            
            # Hyperparameter controls (hidden by default, shown when enabled)
            controls_widget = QWidget()
            controls_layout = QVBoxLayout(controls_widget)
            controls_layout.setContentsMargins(0, 5, 0, 0)
            
            ctrl_dict = {}
            
            # Learning Rate
            lr_layout = QHBoxLayout()
            lr_layout.addWidget(QLabel("LR:"))
            lr_spin = QDoubleSpinBox()
            lr_spin.setRange(0.00001, 0.1)
            lr_spin.setDecimals(5)
            lr_spin.setSingleStep(0.0001)
            lr_spin.setValue(spec.default_lr)
            ctrl_dict['lr'] = lr_spin
            lr_layout.addWidget(lr_spin)
            controls_layout.addLayout(lr_layout)
            
            # Beta (if applicable)
            if spec.has_beta:
                beta_layout = QHBoxLayout()
                beta_layout.addWidget(QLabel("Beta:"))
                beta_spin = QDoubleSpinBox()
                beta_spin.setRange(0.0, 1.0)
                beta_spin.setDecimals(2)
                beta_spin.setSingleStep(0.01)
                beta_spin.setValue(spec.default_beta)
                ctrl_dict['beta'] = beta_spin
                beta_layout.addWidget(beta_spin)
                controls_layout.addLayout(beta_layout)
            
            # Steps (if applicable)
            if spec.has_steps:
                steps_layout = QHBoxLayout()
                steps_layout.addWidget(QLabel("Steps:"))
                steps_spin = QSpinBox()
                steps_spin.setRange(1, 100)
                steps_spin.setValue(spec.default_steps)
                ctrl_dict['steps'] = steps_spin
                steps_layout.addWidget(steps_spin)
                controls_layout.addLayout(steps_layout)
            
            controls_widget.hide()
            self.model_controls[spec.name] = {'widget': controls_widget, 'ctrls': ctrl_dict}
            c_layout.addWidget(controls_widget)
            
            # Show/hide controls when checkbox toggled
            cb.stateChanged.connect(lambda state, w=controls_widget, n=spec.name: self.on_model_toggled(n, state, w))
            
            model_layout.addWidget(container)
        
        model_layout.addStretch()
        scroll.setWidget(model_widget)
        left_splitter.addWidget(scroll)
        
        # Action Buttons
        btn_group = QGroupBox("Training")
        btn_layout = QVBoxLayout(btn_group)
        
        self.btn = QPushButton("▶ Start Training")
        self.btn.clicked.connect(self.toggle_training)
        self.btn.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 10px; font-size: 14px;")
        btn_layout.addWidget(self.btn)
        
        self.reset_btn = QPushButton("↺ Reset All")
        self.reset_btn.clicked.connect(self.reset_model)
        self.reset_btn.setStyleSheet("background-color: #c0392b; color: white; padding: 8px;")
        btn_layout.addWidget(self.reset_btn)
        
        left_splitter.addWidget(btn_group)
        
        # Stats Table
        stats_group = QGroupBox("Real-time Statistics")
        sl = QVBoxLayout(stats_group)
        self.stats_table = StatsTable()
        sl.addWidget(self.stats_table)
        left_splitter.addWidget(stats_group)
        
        left_splitter.setStretchFactor(0, 5)
        left_splitter.setStretchFactor(1, 0)
        left_splitter.setStretchFactor(2, 2)
        
        left_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_panel)
        
        # --- Right Panel (Plots & Cards) ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Arena (Plots)
        self.arena = self.create_arena()
        right_splitter.addWidget(self.arena)
        
        # Cards
        self.mechanics = self.create_mechanics()
        right_splitter.addWidget(self.mechanics)
        
        right_splitter.setStretchFactor(0, 7)
        right_splitter.setStretchFactor(1, 3)
        
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 7)
        
        layout.addWidget(main_splitter)

    def on_model_toggled(self, name, state, widget):
        if state == Qt.CheckState.Checked.value:
            widget.show()
        else:
            widget.hide()
        self.rebuild_ui_for_active_models()

    def rebuild_ui_for_active_models(self):
        active_names = [n for n, cb in self.model_checks.items() if cb.isChecked()]
        self.rebuild_plots(active_names)
        self.rebuild_cards(active_names)

    def create_arena(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        self.loss_plot = pg.PlotWidget(title="Loss (Log)")
        self.loss_plot.setLogMode(y=True)
        self.loss_plot.addLegend()
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.loss_plot, 0, 0)
        
        self.acc_plot = pg.PlotWidget(title="Accuracy")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.acc_plot, 0, 1)
        
        self.ppl_plot = pg.PlotWidget(title="Perplexity (Log)")
        self.ppl_plot.setLogMode(y=True)
        self.ppl_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.ppl_plot, 1, 0)
        
        self.time_plot = pg.PlotWidget(title="Time/Iter (ms)")
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.time_plot, 1, 1)
        
        self.curves = {}
        return widget

    def create_mechanics(self):
        widget = QWidget()
        self.mech_layout = QGridLayout(widget)
        self.cards = {}
        
        for spec in MODEL_REGISTRY:
            card = AlgorithmCard(spec.name, spec.color)
            self.cards[spec.name] = card
        return widget

    def rebuild_plots(self, active_names):
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.ppl_plot.clear()
        self.time_plot.clear()
        self.curves = {}
        
        for name in active_names:
            color = self.colors.get(name, '#888888')
            pen = pg.mkPen(color=color, width=2)
            self.curves[('loss', name)] = self.loss_plot.plot(name=name, pen=pen)
            self.curves[('acc', name)] = self.acc_plot.plot(name=name, pen=pen)
            self.curves[('ppl', name)] = self.ppl_plot.plot(name=name, pen=pen)
            self.curves[('time', name)] = self.time_plot.plot(name=name, pen=pen)

    def rebuild_cards(self, active_names):
        for i in reversed(range(self.mech_layout.count())):
            self.mech_layout.itemAt(i).widget().setParent(None)
        
        if active_names:
            cols = min(4, len(active_names))
            for i, name in enumerate(active_names):
                if name in self.cards:
                    if not self.cards[name].state:
                        self.cards[name].update_state(TrainingState())
                    self.mech_layout.addWidget(self.cards[name], i // cols, i % cols)

    def reset_model(self):
        self.worker.mutex.lock()
        self.algorithms = {}
        self.history = {}
        self.stats_cache = {}
        self.worker.algorithms = {}
        self.worker.iteration = 0
        self.worker.mutex.unlock()
        
        for name in self.cards:
            self.cards[name].update_state(TrainingState())
        
        active_names = [n for n, cb in self.model_checks.items() if cb.isChecked()]
        self.rebuild_plots(active_names)
        print("Model Reset")

    def toggle_training(self):
        if not self.worker.paused:
            self.worker.pause()
            self.btn.setText("▶ Start Training")
            self.btn.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 10px; font-size: 14px;")
        else:
            active_names = [n for n, cb in self.model_checks.items() if cb.isChecked()]
            if not active_names:
                return
            
            if not self.algorithms:
                new_algos = {}
                for name in active_names:
                    spec = get_model_spec(name)
                    algo = AlgorithmWrapper(
                        spec, 
                        self.vocab_size,
                        hidden_dim=128,
                        num_layers=20,
                        device=self.device
                    )
                    
                    # Apply hyperparams from UI controls
                    ctrls = self.model_controls[name]['ctrls']
                    lr = ctrls['lr'].value() if 'lr' in ctrls else spec.default_lr
                    beta = ctrls['beta'].value() if 'beta' in ctrls else None
                    steps = ctrls['steps'].value() if 'steps' in ctrls else None
                    algo.update_hyperparams(lr=lr, beta=beta, steps=steps)
                    
                    new_algos[name] = algo
                    self.history[name] = []
                    self.stats_cache[name] = {
                        'acc': 0, 'ppl': 0, 'vram': 0, 'time': 0,
                        'params': algo.param_count,
                        'best_acc': 0
                    }
                
                self.worker.mutex.lock()
                self.algorithms = new_algos
                self.worker.algorithms = new_algos
                self.worker.mutex.unlock()
            
            self.worker.resume()
            self.btn.setText("⏸ Pause Training")
            self.btn.setStyleSheet("font-weight: bold; background-color: #f39c12; color: white; padding: 10px; font-size: 14px;")

    def on_worker_update(self, name, state):
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(state)
        
        best = self.stats_cache.get(name, {}).get('best_acc', 0)
        if state.accuracy > best:
            best = state.accuracy
        
        self.stats_cache[name] = {
            'acc': state.accuracy,
            'ppl': state.perplexity,
            'vram': state.vram_gb,
            'time': state.iter_time,
            'params': getattr(self.algorithms.get(name), 'param_count', 0),
            'best_acc': best
        }

    def update_plots(self):
        if not self.history:
            return
        
        global_best = 0
        for info in self.stats_cache.values():
            if info['acc'] > global_best:
                global_best = info['acc']
            
        for info in self.stats_cache.values():
            info['is_best_acc'] = (info['acc'] == global_best and global_best > 0)
            
        self.stats_table.update_stats(self.stats_cache)
        
        window = 10
        for name, states in self.history.items():
            if not states:
                continue
            view_states = states[-1000:]
            
            losses = [s.loss for s in view_states]
            accs = [s.accuracy for s in view_states]
            ppls = [s.perplexity for s in view_states]
            times = [s.iter_time * 1000 for s in view_states]
            
            def smooth(v):
                if len(v) < window:
                    return v
                return np.convolve(v, np.ones(window)/window, mode='valid')
            
            if ('loss', name) in self.curves:
                self.curves[('loss', name)].setData(smooth(losses))
            if ('acc', name) in self.curves:
                self.curves[('acc', name)].setData(smooth(accs))
            if ('ppl', name) in self.curves:
                self.curves[('ppl', name)].setData(smooth(ppls))
            if ('time', name) in self.curves:
                self.curves[('time', name)].setData(smooth(times))

    def trigger_generation(self):
        if self.worker.paused:
            return
        for name, algo in self.algorithms.items():
            seed = self.data[torch.randint(0, len(self.data)-64, (1,)).item():]
            try:
                txt = algo.generate(seed[:64], self.i2c, length=30)
                if self.history.get(name):
                    self.history[name][-1].sample = txt
                    if name in self.cards:
                        self.cards[name].update_state(self.history[name][-1])
            except:
                pass
