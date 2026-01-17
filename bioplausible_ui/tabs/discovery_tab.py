"""
Discovery Tab

Visualizes the architecture search space and P2P network activity.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import numpy as np
import pyqtgraph as pg
import logging

from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.hyperopt.analysis import encode_configs, reduce_dimensions

logger = logging.getLogger("DiscoveryTab")

class DiscoveryTab(QWidget):
    """
    Visualization Dashboard for NAS and P2P.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.p2p_node_ref = None # Reference to active DHT node/worker
        self._setup_ui()

        # Data timers
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self._refresh_viz)
        self.viz_timer.start(5000) # Every 5s

        self.net_timer = QTimer()
        self.net_timer.timeout.connect(self._refresh_network)
        self.net_timer.start(2000) # Every 2s

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

        controls.addStretch()
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
        self.self_item = pg.ScatterPlotItem(pos=[{'pos': (0,0), 'size': 20, 'brush': pg.mkBrush('#00ff00')}])
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

    def _refresh_viz(self):
        """Fetch results and update architecture plot."""
        try:
            storage = HyperoptStorage() # Opens default db
            trials = storage.get_all_trials()
            storage.close()

            if not trials: return

            configs = [t.config for t in trials]
            accuracies = [t.accuracy for t in trials]

            # Encode
            features = encode_configs(configs)
            if features.size == 0: return

            # Reduce
            method = self.algo_combo.currentText().lower()
            coords = reduce_dimensions(features, method=method)

            # Update Plot
            spots = []
            for i in range(len(coords)):
                # Color map: Red (0.0) -> Yellow (0.5) -> Green (1.0)
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

            self.scatter_item.setData(spots=spots)

        except Exception as e:
            logger.error(f"Viz refresh error: {e}")

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
