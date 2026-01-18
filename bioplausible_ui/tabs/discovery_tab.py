"""
Discovery Tab

Visualizes the architecture search space and P2P network activity.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QSplitter, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor

import numpy as np
import pyqtgraph as pg
import logging

from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.hyperopt.analysis import encode_configs, reduce_dimensions

logger = logging.getLogger("DiscoveryTab")

class VizWorker(QObject):
    """Worker thread for heavy visualization calculations."""
    finished = pyqtSignal(list) # returns spots list

    def run(self, method='pca'):
        try:
            storage = HyperoptStorage()
            trials = storage.get_all_trials()
            storage.close()

            if not trials:
                self.finished.emit([])
                return

            configs = [t.config for t in trials]
            accuracies = [t.accuracy for t in trials]

            features = encode_configs(configs)
            if features.size == 0:
                self.finished.emit([])
                return

            coords = reduce_dimensions(features, method=method)

            spots = []
            for i in range(len(coords)):
                acc = accuracies[i]
                r = int(255 * (1 - acc))
                g = int(255 * acc)
                color = QColor(r, g, 0, 200)

                spots.append({
                    'pos': (coords[i, 0], coords[i, 1]),
                    'data': trials[i],
                    'brush': pg.mkBrush(color),
                    'symbol': 'o',
                    'size': 10
                })

            self.finished.emit(spots)

        except Exception as e:
            logger.error(f"Viz calculation error: {e}")
            self.finished.emit([])

class DiscoveryTab(QWidget):
    """
    Visualization Dashboard for NAS and P2P.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.p2p_node_ref = None
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

        controls.addStretch()

        self.last_update_label = QLabel("Last Updated: Never")
        self.last_update_label.setStyleSheet("color: #666; font-size: 10px;")
        controls.addWidget(self.last_update_label)

        arch_layout.addLayout(controls)

        # Plot
        self.arch_plot = pg.PlotWidget(title="Architecture Space")
        self.arch_plot.setBackground('#0a0a0f')
        self.arch_plot.setLabel('bottom', "Dimension 1")
        self.arch_plot.setLabel('left', "Dimension 2")
        self.scatter_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.arch_plot.addItem(self.scatter_item)

        # Tooltip label
        self.info_label = QLabel("Hover over a point to see details.")
        self.info_label.setStyleSheet("color: #aaa; font-style: italic;")
        arch_layout.addWidget(self.arch_plot)
        arch_layout.addWidget(self.info_label)

        # Click interaction
        self.scatter_item.sigClicked.connect(self._on_point_clicked)

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

    def _refresh_viz(self):
        self._request_viz_update()

    def _request_viz_update(self):
        """Request update from worker thread."""
        method = self.algo_combo.currentText().lower()
        # We can't call worker.run directly as it would run in main thread
        # We use QMetaObject.invokeMethod or just a signal if we set it up that way
        # But easier here is to just use a custom signal or lambda if worker was just a function
        # Since it's a QObject in a thread, best practice is signal-slot
        pass # We need to trigger the run.

        # Actually, let's just use a simple approach:
        # Re-instantiate worker logic? No.
        # Let's add a signal to this class to trigger worker
        self.start_calc_signal.emit(method)

    start_calc_signal = pyqtSignal(str) # Define at class level

    def _on_viz_ready(self, spots):
        """Update plot with calculated spots."""
        if spots:
            self.scatter_item.setData(spots=spots)
            import time
            self.last_update_label.setText(f"Last Updated: {time.strftime('%H:%M:%S')}")

    def _on_point_clicked(self, plot, points):
        if len(points) > 0:
            trial = points[0].data()
            self.info_label.setText(
                f"Trial {trial.trial_id}: {trial.model_name} | Acc: {trial.accuracy:.4f} | Loss: {trial.final_loss:.4f}"
            )

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
