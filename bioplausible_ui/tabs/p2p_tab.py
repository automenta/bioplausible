"""
P2P Network Tab

Allows users to contribute to the Bio-Plausible Research Network.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QTextEdit, QProgressBar, QScrollArea, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt6.QtGui import QFont, QDesktopServices, QColor
from PyQt6.QtCore import QUrl

from bioplausible.p2p import Worker, CLOUD_PROVIDERS, DEPLOYMENT_TIPS
from bioplausible.p2p.evolution import P2PEvolution

class P2PWorkerBridge(QObject):
    """Bridges P2P Worker/Evolution callbacks to Qt Signals."""
    status_changed = pyqtSignal(str, int, int) # status, points, jobs
    log_received = pyqtSignal(str)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.on_status_change = self.emit_status
        self.worker.on_log = self.emit_log

    def emit_status(self, status, points, jobs):
        self.status_changed.emit(status, points, jobs)

    def emit_log(self, msg):
        self.log_received.emit(msg)

class P2PTab(QWidget):
    """Community Grid / P2P Tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None # Can be Worker or P2PEvolution
        self.bridge = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(20)

        # Left Column: Connection & Controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)

        # Status Group
        status_group = QGroupBox("📡 Connection Status")
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel("DISCONNECTED")
        self.status_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #ff5555; border: 2px solid #ff5555; border-radius: 5px; padding: 10px;")
        status_layout.addWidget(self.status_label)

        stats_layout = QHBoxLayout()

        # Points
        self.points_label = QLabel("0")
        self.points_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.points_label.setStyleSheet("color: #f39c12;")
        points_container = QVBoxLayout()
        points_container.addWidget(QLabel("Contribution Points (CP)"))
        points_container.addWidget(self.points_label)
        stats_layout.addLayout(points_container)

        # Jobs
        self.jobs_label = QLabel("0")
        self.jobs_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.jobs_label.setStyleSheet("color: #00d4ff;")
        jobs_container = QVBoxLayout()
        jobs_container.addWidget(QLabel("Jobs Completed"))
        jobs_container.addWidget(self.jobs_label)
        stats_layout.addLayout(jobs_container)

        status_layout.addLayout(stats_layout)
        left_panel.addWidget(status_group)

        # Connection Controls
        conn_group = QGroupBox("🔌 Network Settings")
        conn_layout = QVBoxLayout(conn_group)

        # Mode Selection
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)

        self.radio_coord = QRadioButton("Standard (Coordinator)")
        self.radio_coord.setChecked(True)
        self.mode_group.addButton(self.radio_coord)
        mode_layout.addWidget(self.radio_coord)

        self.radio_dht = QRadioButton("True P2P (DHT Mesh)")
        self.mode_group.addButton(self.radio_dht)
        mode_layout.addWidget(self.radio_dht)

        conn_layout.addLayout(mode_layout)

        self.radio_coord.toggled.connect(self._update_input_label)

        self.input_label = QLabel("Coordinator URL:")
        conn_layout.addWidget(self.input_label)

        # Bootstrap Nodes Dropdown (for DHT mode)
        self.bootstrap_combo = QComboBox()
        self.bootstrap_combo.setEditable(True)
        self.bootstrap_combo.addItems([
            "bootstrap1.bioplausible.org",
            "bootstrap2.bioplausible.org",
            "127.0.0.1:8468 (Local Test)"
        ])
        self.bootstrap_combo.setVisible(False)
        conn_layout.addWidget(self.bootstrap_combo)

        self.url_input = QLineEdit("http://localhost:8000") # Default for local testing
        self.url_input.setPlaceholderText("http://grid.bioplausible.org")
        conn_layout.addWidget(self.url_input)

        self.connect_btn = QPushButton("🚀 Join Network")
        self.connect_btn.setMinimumHeight(50)
        self.connect_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #27ae60;")
        self.connect_btn.clicked.connect(self._toggle_connection)
        conn_layout.addWidget(self.connect_btn)

        # Auto Mode info
        info_label = QLabel("Auto Mode uses your idle compute to help discover new neural architectures.\nWe respect your time: process runs at low priority.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #808090; font-style: italic; margin-top: 10px;")
        conn_layout.addWidget(info_label)

        left_panel.addWidget(conn_group)

        # Strategy & Constraints
        strat_group = QGroupBox("🎯 Discovery Strategy")
        strat_layout = QVBoxLayout(strat_group)

        # Strategy Radio
        strat_radio_layout = QHBoxLayout()
        self.strat_group_btn = QButtonGroup(self)
        self.radio_quick = QRadioButton("🐇 Quick Exploration")
        self.radio_quick.setToolTip("Fast, shallow searches (1 epoch). Good for broad discovery.")
        self.radio_quick.setChecked(True)
        self.strat_group_btn.addButton(self.radio_quick)

        self.radio_deep = QRadioButton("🐢 Deep Focus")
        self.radio_deep.setToolTip("Longer training (5+ epochs). Good for refining high performers.")
        self.strat_group_btn.addButton(self.radio_deep)

        strat_radio_layout.addWidget(self.radio_quick)
        strat_radio_layout.addWidget(self.radio_deep)
        strat_layout.addLayout(strat_radio_layout)

        # Constraints Checkboxes
        self.check_low_mem = QCheckBox("Low Memory (Mobile/Laptop)")
        self.check_low_mem.setToolTip("Limits model size (hidden dimensions, layers).")
        strat_layout.addWidget(self.check_low_mem)

        self.check_fast = QCheckBox("Time Constrained (Max Steps)")
        self.check_fast.setToolTip("Limits equilibrium steps for faster iterations.")
        strat_layout.addWidget(self.check_fast)

        left_panel.addWidget(strat_group)

        # Log
        log_group = QGroupBox("📜 Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: Consolas; font-size: 10px; background-color: #1a1a1e;")
        log_layout.addWidget(self.log_output)
        left_panel.addWidget(log_group)

        # Right Column: Cloud Guide
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)

        cloud_group = QGroupBox("☁️ Cloud Compute Resources")
        cloud_layout = QVBoxLayout(cloud_group)

        intro = QLabel("Don't have a GPU? Rent one easily! (Click links to open)")
        intro.setStyleSheet("font-weight: bold; color: #a0a0b0;")
        cloud_layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for provider in CLOUD_PROVIDERS:
            p_group = QGroupBox(provider['name'])
            p_layout = QVBoxLayout(p_group)

            # Link
            link_btn = QPushButton(f"🌐 Open {provider['name']}")
            link_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            link_btn.clicked.connect(lambda checked, url=provider['url']: QDesktopServices.openUrl(QUrl(url)))
            p_layout.addWidget(link_btn)

            # Desc
            p_layout.addWidget(QLabel(provider['description']))

            # Tiers
            tier_text = "<b>Pricing Examples:</b><br>"
            for tier in provider['tiers']:
                tier_text += f"• {tier['gpu']} ({tier['vram']}): <span style='color: #00ff88'>{tier['price']}</span><br>"

            tier_label = QLabel(tier_text)
            tier_label.setStyleSheet("background-color: #222; padding: 5px; border-radius: 4px;")
            p_layout.addWidget(tier_label)

            # Setup Command
            p_layout.addWidget(QLabel("<b>Setup Command:</b>"))
            cmd_edit = QLineEdit(provider['setup_cmd'])
            cmd_edit.setReadOnly(True)
            cmd_edit.setStyleSheet("font-family: Consolas; color: #f1c40f;")
            p_layout.addWidget(cmd_edit)

            scroll_layout.addWidget(p_group)

        # Deployment Tips
        tips_label = QLabel(DEPLOYMENT_TIPS.replace("\n", "<br>"))
        tips_label.setWordWrap(True)
        tips_label.setStyleSheet("background-color: #1e1e24; padding: 10px; margin-top: 10px; border: 1px solid #444;")
        scroll_layout.addWidget(tips_label)

        scroll.setWidget(scroll_content)
        cloud_layout.addWidget(scroll)
        right_panel.addWidget(cloud_group)

    def _update_input_label(self):
        if self.radio_coord.isChecked():
            self.input_label.setText("Coordinator URL:")
            self.url_input.setVisible(True)
            self.bootstrap_combo.setVisible(False)
            self.url_input.setPlaceholderText("http://grid.bioplausible.org")
            self.url_input.setText("http://localhost:8000")
        else:
            self.input_label.setText("Bootstrap Node (IP):")
            self.url_input.setVisible(False)
            self.bootstrap_combo.setVisible(True)
            self.bootstrap_combo.setCurrentIndex(2) # Default to local for safety

    # Helper signals to expose inner bridge signals to dashboard
    bridge_log_signal = pyqtSignal(str)
    bridge_status_signal = pyqtSignal(str, int, int)

    def _toggle_connection(self):
        if self.worker and self.worker.running:
            # Stop
            self.worker.stop()
            self.connect_btn.setText("🚀 Join Network")
            self.connect_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #27ae60;")
            self.status_label.setText("DISCONNECTED")
            self.status_label.setStyleSheet("color: #ff5555; border: 2px solid #ff5555; border-radius: 5px; padding: 10px;")
            self._log("Worker stopped.")
            self.radio_coord.setEnabled(True)
            self.radio_dht.setEnabled(True)
        else:
            # Start
            self.radio_coord.setEnabled(False)
            self.radio_dht.setEnabled(False)

            if self.radio_coord.isChecked():
                # Client Mode
                target = self.url_input.text()
                if not target: target = "http://localhost:8000"
                self.worker = Worker(target)
                self.worker.start_loop()
            else:
                # DHT Mode
                target = self.bootstrap_combo.currentText()
                # Logic: if target is empty, we are a bootstrap node or solo
                # if target has IP, we bootstrap from it
                # target format: IP or IP:PORT
                ip = None
                port = 8468
                if target and "bootstrap" not in target: # Skip placeholder domains if not resolved
                     parts = target.split(':')
                     ip = parts[0]
                     if len(parts) > 1: port = int(parts[1])

                # Gather settings
                mode = 'quick' if self.radio_quick.isChecked() else 'deep'
                constraints = {}
                if self.check_low_mem.isChecked():
                    constraints['max_hidden'] = 64
                    constraints['max_layers'] = 3
                if self.check_fast.isChecked():
                    constraints['max_steps'] = 12

                self.worker = P2PEvolution(
                    bootstrap_ip=ip,
                    bootstrap_port=port,
                    discovery_mode=mode,
                    constraints=constraints
                )
                self.worker.start(auto_nice=True)

            # Setup Bridge
            self.bridge = P2PWorkerBridge(self.worker)
            self.bridge.status_changed.connect(self._on_status_changed)
            self.bridge.log_received.connect(self._log)

            # Forward signals to external world
            self.bridge.status_changed.connect(self.bridge_status_signal.emit)
            self.bridge.log_received.connect(self.bridge_log_signal.emit)

            self.connect_btn.setText("⏹ Stop Contributing")
            self.connect_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #c0392b;")
            self.status_label.setText("CONNECTING...")
            self.status_label.setStyleSheet("color: #f39c12; border: 2px solid #f39c12; border-radius: 5px; padding: 10px;")

    def _on_status_changed(self, status, points, jobs):
        self.status_label.setText(status.upper())
        if "Running" in status or "Mesh" in status or "Evaluating" in status:
             self.status_label.setStyleSheet("color: #00ff88; border: 2px solid #00ff88; border-radius: 5px; padding: 10px;")
        elif "Idle" in status or "Resting" in status:
             self.status_label.setStyleSheet("color: #3498db; border: 2px solid #3498db; border-radius: 5px; padding: 10px;")

        self.points_label.setText(str(points))
        self.jobs_label.setText(str(jobs))

    def _log(self, msg):
        self.log_output.append(msg)
