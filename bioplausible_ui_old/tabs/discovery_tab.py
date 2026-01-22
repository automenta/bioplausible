"""
Discovery Tab

Visualizes the architecture search space and P2P network activity.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QSplitter, QCheckBox, QMenu, QDialog,
    QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor, QPen, QAction

import numpy as np
import pyqtgraph as pg
import logging
import hashlib
import json

from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.hyperopt.analysis import encode_configs, reduce_dimensions

logger = logging.getLogger("DiscoveryTab")

def get_config_hash(config: dict) -> str:
    """Generate a hash for a configuration (same as evolution.py)."""

    def _default(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return str(obj)

    s = json.dumps(config, sort_keys=True, default=_default)
    return hashlib.md5(s.encode()).hexdigest()

class VizWorker(QObject):
    """Worker thread for heavy visualization calculations."""
    finished = pyqtSignal(list, list) # returns spots list, lines list

    def run(self, method='pca'):
        try:
            storage = HyperoptStorage()
            trials = storage.get_all_trials()
            storage.close()

            if not trials:
                self.finished.emit([], [])
                return

            configs = [t.config for t in trials]
            accuracies = [t.accuracy for t in trials]
            trial_hashes = [get_config_hash(c) for c in configs]
            hash_to_idx = {h: i for i, h in enumerate(trial_hashes)}

            features = encode_configs(configs)
            if features.size == 0:
                self.finished.emit([], [])
                return

            coords = reduce_dimensions(features, method=method)

            spots = []
            lines = [] # List of dicts describing lines

            for i in range(len(coords)):
                acc = accuracies[i]
                config = configs[i]

                # Color based on accuracy
                r = int(255 * (1 - acc))
                g = int(255 * acc)
                color = QColor(r, g, 0, 200)

                # Generation info
                gen = config.get('generation', 0)

                # Spot data
                spots.append({
                    'pos': (coords[i, 0], coords[i, 1]),
                    'data': trials[i],
                    'brush': pg.mkBrush(color),
                    'symbol': 'o',
                    'size': 10 + (gen * 2) if gen < 5 else 20 # visual cue for generation
                })

                # Lineage Lines
                parent_id = config.get('parent_id')
                if parent_id and parent_id in hash_to_idx:
                    p_idx = hash_to_idx[parent_id]
                    p_coord = coords[p_idx]
                    c_coord = coords[i]

                    lines.append({
                        'x': [p_coord[0], c_coord[0]],
                        'y': [p_coord[1], c_coord[1]],
                        'pen': pg.mkPen(color=QColor(255, 255, 255, 50), width=1)
                    })

            self.finished.emit(spots, lines)

        except Exception as e:
            logger.error(f"Viz calculation error: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit([], [])

class GenomeEditorDialog(QDialog):
    """Dialog for editing a JSON genome config."""
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Genome Editor")
        self.resize(600, 500)
        self.config = config
        self.result_config = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Modify architecture configuration:"))

        self.editor = QTextEdit()
        self.editor.setFont(QFont("Consolas", 10))

        # Helper to clean numpy types for display
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj

        clean_config = _clean(config)
        self.editor.setText(json.dumps(clean_config, indent=4))
        layout.addWidget(self.editor)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        inject_btn = QPushButton("💉 Inject Genome")
        inject_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        inject_btn.clicked.connect(self._validate_and_accept)
        btn_layout.addWidget(inject_btn)

        layout.addLayout(btn_layout)

    def _validate_and_accept(self):
        try:
            text = self.editor.toPlainText()
            self.result_config = json.loads(text)
            self.accept()
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Invalid JSON", f"Syntax error: {e}")

class DiscoveryTab(QWidget):
    """
    Visualization Dashboard for NAS and P2P.
    """

    load_model_signal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.p2p_node_ref = None # Holds DHTNode
        self.p2p_worker_ref = None # Holds P2PEvolution worker

        self._setup_ui()

        # Setup Thread for Viz
        self.thread = QThread()
        self.worker = VizWorker()
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self._on_viz_ready)
        self.start_calc_signal.connect(self.worker.run)
        self.thread.start()

        # Data timers
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self._request_viz_update)
        self.viz_timer.start(5000)

        self.net_timer = QTimer()
        self.net_timer.timeout.connect(self._refresh_network)
        self.net_timer.start(2000)

        # Store plot items
        self.line_items = []

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left Panel: Architecture Space ---
        arch_widget = QWidget()
        arch_layout = QVBoxLayout(arch_widget)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["PCA", "t-SNE"])
        controls.addWidget(self.algo_combo)

        self.refresh_btn = QPushButton("🔄 Refresh Map")
        self.refresh_btn.clicked.connect(self._refresh_viz)
        controls.addWidget(self.refresh_btn)

        self.auto_refresh_check = QCheckBox("Auto-Refresh")
        self.auto_refresh_check.setChecked(True)
        self.auto_refresh_check.toggled.connect(self._toggle_auto_refresh)
        controls.addWidget(self.auto_refresh_check)

        self.traj_check = QCheckBox("Show Trajectories")
        self.traj_check.setChecked(True)
        self.traj_check.toggled.connect(self._toggle_trajectories)
        controls.addWidget(self.traj_check)

        controls.addStretch()

        self.last_update_label = QLabel("Last Updated: Never")
        self.last_update_label.setStyleSheet("color: #666; font-size: 10px;")
        controls.addWidget(self.last_update_label)

        arch_layout.addLayout(controls)

        # Plot
        self.arch_plot = pg.PlotWidget(title="Evolutionary Architecture Space")
        self.arch_plot.setBackground('#0a0a0f')
        self.arch_plot.setLabel('bottom', "Dimension 1")
        self.arch_plot.setLabel('left', "Dimension 2")

        self.traj_curve = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 255, 255, 60), width=1))
        self.arch_plot.addItem(self.traj_curve)

        self.scatter_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.arch_plot.addItem(self.scatter_item)

        # Tooltip label
        self.info_label = QLabel("Hover over a point to see details. Right-click to edit.")
        self.info_label.setStyleSheet("color: #aaa; font-style: italic;")
        arch_layout.addWidget(self.arch_plot)
        arch_layout.addWidget(self.info_label)

        # Click interaction
        self.scatter_item.sigClicked.connect(self._on_point_clicked)

        # Context Menu (Right Click) logic is handled manually for ScatterPlotItem usually,
        # but easier to subclass or intercept events.
        # But ScatterPlotItem doesn't emit right-clicks easily.
        # We can set a custom ViewBox menu or check mouse buttons in clicked.
        # Actually sigClicked provides the plot and points.
        # But for right-click on background? Or specific point?
        # pg supports context menus on the ViewBox.
        # Let's augment `_on_point_clicked` to detect if it's a right click?
        # pg signals don't always pass the event button.
        # A trick: use scene().sigMouseClicked
        self.arch_plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

        splitter.addWidget(arch_widget)

        # --- Right Panel: Network Activity ---
        net_widget = QWidget()
        net_layout = QVBoxLayout(net_widget)

        net_group = QGroupBox("🌐 Network Galaxy")
        group_layout = QVBoxLayout(net_group)

        self.net_plot = pg.PlotWidget()
        self.net_plot.setBackground('#050510')
        self.net_plot.hideAxis('bottom')
        self.net_plot.hideAxis('left')
        self.net_plot.setRange(xRange=(-100, 100), yRange=(-100, 100))

        # Self node
        self.self_item = pg.ScatterPlotItem(spots=[{'pos': (0,0), 'size': 20, 'brush': pg.mkBrush('#00ff00')}])
        self.net_plot.addItem(self.self_item)

        # Peer nodes
        self.peers_item = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('#00ccff'))
        self.net_plot.addItem(self.peers_item)

        group_layout.addWidget(self.net_plot)

        self.net_status = QLabel("Status: Local Only")
        group_layout.addWidget(self.net_status)

        net_layout.addWidget(net_group)
        splitter.addWidget(net_widget)

        # Adjust splitter
        splitter.setSizes([700, 400])

    def update_p2p_ref(self, worker):
        """Update reference to the P2P worker/evolution object."""
        self.p2p_worker_ref = worker # Store full worker
        # Unpack if it's the evolution controller which holds the dht
        if hasattr(worker, 'dht'):
            self.p2p_node_ref = worker.dht
        else:
            self.p2p_node_ref = None # Standard worker doesn't expose DHT table

    def _toggle_auto_refresh(self, checked):
        if checked:
            self.viz_timer.start(5000)
        else:
            self.viz_timer.stop()

    def _toggle_trajectories(self, checked):
        if checked:
            self.traj_curve.setVisible(True)
        else:
            self.traj_curve.setVisible(False)

    def _refresh_viz(self):
        self._request_viz_update()

    def _request_viz_update(self):
        """Request update from worker thread."""
        method = self.algo_combo.currentText().lower()
        self.start_calc_signal.emit(method)

    start_calc_signal = pyqtSignal(str) # Define at class level

    def _on_viz_ready(self, spots, lines):
        """Update plot with calculated spots and lineage lines."""
        if spots:
            self.scatter_item.setData(spots=spots)

            # Construct single array for lines with NaNs
            if lines:
                x = []
                y = []
                for line in lines:
                    x.extend(line['x'])
                    x.append(np.nan) # Break line
                    y.extend(line['y'])
                    y.append(np.nan)

                self.traj_curve.setData(np.array(x), np.array(y))
            else:
                 self.traj_curve.setData([], [])

            import time
            self.last_update_label.setText(f"Last Updated: {time.strftime('%H:%M:%S')}")

    def _on_point_clicked(self, plot, points):
        """Handle left click on points."""
        if len(points) > 0:
            trial = points[0].data()
            gen = trial.config.get('generation', '?')
            pid = trial.config.get('parent_id', 'None')[:8] if trial.config.get('parent_id') else 'None'

            self.info_label.setText(
                f"Gen {gen} | Trial {trial.trial_id}: {trial.model_name} | Acc: {trial.accuracy:.4f} | Parent: {pid}"
            )

    def _on_scene_clicked(self, event):
        """Handle right click on scene to open context menu if over a point."""
        if event.button() == Qt.MouseButton.RightButton:
            # Check if we clicked on a point
            pos = event.scenePos()
            mouse_point = self.arch_plot.plotItem.vb.mapSceneToView(pos)

            # Find nearest point within tolerance
            # This is a bit manual, simpler if ScatterPlotItem handled it.
            # But let's check spots.
            # Ideally use pointsAt()
            points = self.scatter_item.pointsAt(pos)
            if points:
                point = points[0]
                self._show_context_menu(event.screenPos(), point)

    def _show_context_menu(self, screen_pos, point):
        """Show context menu for a point."""
        menu = QMenu(self)

        trial = point.data()

        fork_action = QAction("🧬 Fork & Edit Genome", self)
        fork_action.triggered.connect(lambda: self._open_genome_editor(trial))
        menu.addAction(fork_action)

        train_action = QAction("🏋️ Train Locally", self)
        train_action.triggered.connect(lambda: self._train_locally(trial))
        menu.addAction(train_action)

        menu.exec(screen_pos.toPoint())

    def _train_locally(self, trial):
        """Emit signal to load this config into the main trainer."""
        config = trial.config

        # Ensure we have enough info in config
        # Sometimes hyperopt configs are minimal, check if we need to infer
        if 'model_name' not in config:
            config['model_name'] = trial.model_name

        self.load_model_signal.emit(config)
        QMessageBox.information(self, "Model Loaded", f"Loaded configuration for trial {trial.trial_id}.\nSwitching to training tab...")

    def _open_genome_editor(self, trial):
        """Open the editor for the selected trial."""
        if not self.p2p_worker_ref:
            QMessageBox.warning(self, "Not Connected", "Connect to P2P network first to inject genomes.")
            return

        # Prepare config for forking
        config = trial.config.copy()
        config['parent_id'] = get_config_hash(trial.config) # Set parent to this trial
        config['generation'] = config.get('generation', 0) + 1

        dlg = GenomeEditorDialog(config, self)
        if dlg.exec():
            new_config = dlg.result_config
            if hasattr(self.p2p_worker_ref, 'inject_genome'):
                self.p2p_worker_ref.inject_genome(new_config)
                QMessageBox.information(self, "Injected", "Genome added to evaluation queue!")
            else:
                QMessageBox.warning(self, "Error", "Worker does not support injection.")

    def _refresh_network(self):
        """Update network visualization."""
        if not self.p2p_node_ref:
            self.net_status.setText("Status: Not Connected (DHT Mode Required)")
            self.peers_item.clear()
            return

        peers = self.p2p_node_ref.get_known_peers()
        self.net_status.setText(f"Status: Connected ({len(peers)} peers visible)")

        spots = []
        import hashlib

        for peer in peers:
            # Generate pseudo-random position based on ID
            # Distance based on XOR would be better but simple hash is fine for 'constellation'
            pid = peer['id']
            h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
            angle = (h % 360) * np.pi / 180.0
            dist = 50 + (h % 40)

            x = dist * np.cos(angle)
            y = dist * np.sin(angle)

            spots.append({
                'pos': (x, y),
                'brush': pg.mkBrush('#00ccff'),
                'symbol': 'h' # hexagon
            })

        self.peers_item.setData(spots=spots)
