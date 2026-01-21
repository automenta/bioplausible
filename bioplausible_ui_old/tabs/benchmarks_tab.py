from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QCheckBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox,
    QDialog, QLabel, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from bioplausible_ui_old.worker import BenchmarkWorker

class ComparisonDialog(QDialog):
    """Dialog to compare benchmark results."""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Benchmark Comparison")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        label = QLabel("Performance Comparison")
        label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Bar Chart
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('#0a0a0f')
        plot_widget.showGrid(y=True, alpha=0.3)
        layout.addWidget(plot_widget)

        # Prepare Data
        names = [d['name'] for d in data]
        scores = [d['score'] for d in data]

        x = range(len(data))

        # Create Bar Graph Item
        bargraph = pg.BarGraphItem(x=x, height=scores, width=0.6, brush='b')
        plot_widget.addItem(bargraph)

        # Custom Axis for Strings
        # This is tricky in pure pyqtgraph without a custom AxisItem subclass
        # Simple workaround: just show names in legend or tooltip?
        # Or standard text items.

        # Adding text labels below bars
        # This is a bit manual
        pass
        # For MVP, we just show the chart. Tooltips would be nice but complex here.

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class BenchmarksTab(QWidget):
    """A tab for running validation tracks (benchmarks)."""

    log_message = pyqtSignal(str) # Signal to log to console
    load_model_signal = pyqtSignal(dict) # Signal to load a config for training

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.bench_worker = None

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left Panel: Control and List
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=2)

        # Controls
        controls_group = QGroupBox("Benchmark Controls")
        controls_layout = QHBoxLayout(controls_group)

        self.bench_quick_check = QCheckBox("Quick Mode (Smoke Tests)")
        self.bench_quick_check.setChecked(True)
        controls_layout.addWidget(self.bench_quick_check)

        self.bench_parallel_check = QCheckBox("Parallel Execution")
        self.bench_parallel_check.setToolTip("Run tracks in parallel processes (faster, but harder to debug)")
        controls_layout.addWidget(self.bench_parallel_check)

        run_sel_btn = QPushButton("▶ Run Selected")
        run_sel_btn.clicked.connect(self._run_selected_benchmarks)
        controls_layout.addWidget(run_sel_btn)

        run_all_btn = QPushButton("⏩ Run All")
        run_all_btn.clicked.connect(self._run_all_benchmarks)
        controls_layout.addWidget(run_all_btn)

        compare_btn = QPushButton("📊 Compare Selected")
        compare_btn.clicked.connect(self._compare_selected)
        controls_layout.addWidget(compare_btn)

        left_panel.addWidget(controls_group)

        # Search Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search tracks...")
        self.search_input.textChanged.connect(self._filter_tracks)
        filter_layout.addWidget(self.search_input)
        left_panel.addLayout(filter_layout)

        # Track List Table
        self.bench_table = QTableWidget()
        self.bench_table.setColumnCount(4)
        self.bench_table.setHorizontalHeaderLabels(["ID", "Track Name", "Status", "Score"])
        self.bench_table.horizontalHeader().setStretchLastSection(True)
        self.bench_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.bench_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.bench_table.customContextMenuRequested.connect(self._show_context_menu)

        # Populate Tracks
        try:
            from bioplausible.validation.core import Verifier
            # Create a dummy verifier to get list
            v = Verifier(quick_mode=True)
            self.bench_table.setRowCount(len(v.tracks))

            for i, (tid, (name, _)) in enumerate(v.tracks.items()):
                self.bench_table.setItem(i, 0, QTableWidgetItem(str(tid)))
                self.bench_table.setItem(i, 1, QTableWidgetItem(name))
                self.bench_table.setItem(i, 2, QTableWidgetItem("Pending"))
                self.bench_table.setItem(i, 3, QTableWidgetItem("--"))
        except Exception as e:
            self.bench_table.setRowCount(1)
            self.bench_table.setItem(0, 1, QTableWidgetItem(f"Error loading tracks: {e}"))

        left_panel.addWidget(self.bench_table)

        # Right Panel: Details/Output
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)

        details_group = QGroupBox("Benchmark Details")
        details_layout = QVBoxLayout(details_group)

        self.bench_output = QTextEdit()
        self.bench_output.setReadOnly(True)
        self.bench_output.setPlaceholderText("Select a track to see details or run benchmarks...")
        details_layout.addWidget(self.bench_output)

        right_panel.addWidget(details_group)

    def _filter_tracks(self, text):
        """Filter rows based on text."""
        for r in range(self.bench_table.rowCount()):
            name_item = self.bench_table.item(r, 1)
            id_item = self.bench_table.item(r, 0)
            if name_item and id_item:
                match = (text.lower() in name_item.text().lower()) or (text.lower() in id_item.text().lower())
                self.bench_table.setRowHidden(r, not match)

    def _run_selected_benchmarks(self):
        """Run selected benchmarks."""
        selected_rows = self.bench_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one track to run.")
            return

        track_ids = []
        for row in selected_rows:
            # Skip hidden rows
            if self.bench_table.isRowHidden(row.row()):
                continue

            tid_item = self.bench_table.item(row.row(), 0)
            if tid_item:
                track_ids.append(int(tid_item.text()))

        self._start_benchmark_worker(track_ids)

    def _run_all_benchmarks(self):
        """Run all benchmarks."""
        rows = self.bench_table.rowCount()
        track_ids = []
        for r in range(rows):
            # Skip hidden rows? Usually run all means all visible?
            # Or truly all? Let's assume all visible if filter is active.
            if self.bench_table.isRowHidden(r):
                continue

            tid_item = self.bench_table.item(r, 0)
            if tid_item:
                track_ids.append(int(tid_item.text()))

        self._start_benchmark_worker(track_ids)

    def _start_benchmark_worker(self, track_ids):
        """Start the benchmark worker."""
        quick = self.bench_quick_check.isChecked()
        parallel = self.bench_parallel_check.isChecked()

        self.bench_output.clear()
        self.bench_output.append(f"Starting {len(track_ids)} tracks (Quick={quick}, Parallel={parallel})...\n")

        # Reset status in table
        for r in range(self.bench_table.rowCount()):
            tid_item = self.bench_table.item(r, 0)
            if tid_item and int(tid_item.text()) in track_ids:
                self.bench_table.setItem(r, 2, QTableWidgetItem("Running..."))
                self.bench_table.item(r, 2).setForeground(Qt.GlobalColor.yellow)

        self.bench_worker = BenchmarkWorker(track_ids, quick_mode=quick, parallel=parallel)
        self.bench_worker.progress.connect(self._on_bench_progress)
        self.bench_worker.finished.connect(self._on_bench_finished)
        self.bench_worker.error.connect(self._on_bench_error)
        self.bench_worker.start()

    def _on_bench_progress(self, msg):
        self.bench_output.append(msg)
        self.log_message.emit(msg) # Also log to console

    def _on_bench_finished(self, results):
        self.bench_output.append("\nBenchmarking Complete!")

        # Update table
        for tid, res in results.items():
            # Find row
            for r in range(self.bench_table.rowCount()):
                if int(self.bench_table.item(r, 0).text()) == tid:
                    status = res['status']
                    score = res['score']

                    status_item = QTableWidgetItem(status.upper())
                    if status == 'pass':
                        status_item.setForeground(Qt.GlobalColor.green)
                    else:
                        status_item.setForeground(Qt.GlobalColor.red)

                    self.bench_table.setItem(r, 2, status_item)
                    self.bench_table.setItem(r, 3, QTableWidgetItem(f"{score:.1f}"))
                    break

    def _on_bench_error(self, err):
        self.bench_output.append(f"\nERROR: {err}")
        QMessageBox.critical(self, "Benchmark Error", err)

    def _show_context_menu(self, pos):
        """Show context menu for tracks."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        index = self.bench_table.indexAt(pos)
        if not index.isValid():
            return

        row = index.row()
        tid_item = self.bench_table.item(row, 0)
        status_item = self.bench_table.item(row, 2)

        if not tid_item or not status_item:
            return

        menu = QMenu(self)

        # Action: Load Configuration (Transfer Learning / Exploit)
        # Only if passed? No, maybe debug failed ones too.
        load_action = QAction("⚡ Load Configuration (Exploit)", self)
        load_action.triggered.connect(lambda: self._load_track_config(int(tid_item.text())))
        menu.addAction(load_action)

        menu.exec(self.bench_table.viewport().mapToGlobal(pos))

    def _load_track_config(self, track_id):
        """Load configuration from a benchmark track."""
        try:
            # We need to peek into the track definition
            from bioplausible.validation.core import Verifier
            # This is a bit inefficient to instantiate again but safe
            v = Verifier(quick_mode=True)
            track_func = v.tracks.get(track_id, (None, None))[1]

            if not track_func:
                return

            # Infer config from track name or logic?
            # Ideally tracks would expose their config.
            # Currently tracks are functions.
            # We can use a heuristic mapping for major tracks.

            config = {}
            name = v.tracks[track_id][0].lower()

            if "vision" in name or "mnist" in name or "cifar" in name:
                config['task'] = 'vision'
                config['model_name'] = 'ConvEqProp (CIFAR)' if 'cifar' in name else 'EqProp MLP (Standard)'
                config['dataset'] = 'CIFAR-10' if 'cifar' in name else 'MNIST'
                config['hidden_dim'] = 256
                config['steps'] = 30
            elif "language" in name or "lm" in name:
                config['task'] = 'lm'
                config['model_name'] = 'EqProp Recurrent LM'
                config['dataset'] = 'tiny_shakespeare'
                config['hidden_dim'] = 64
                config['steps'] = 20
            else:
                config['task'] = 'vision'
                config['model_name'] = 'EqProp MLP (Standard)'

            self.load_model_signal.emit(config)
            QMessageBox.information(self, "Config Loaded", f"Loaded heuristic configuration for Track {track_id}.\nSwitching tabs...")

        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Could not load config: {e}")

    def _compare_selected(self):
        """Compare scores of selected benchmarks."""
        selected_rows = self.bench_table.selectionModel().selectedRows()
        if len(selected_rows) < 2:
            QMessageBox.warning(self, "Select More", "Please select at least two tracks to compare.")
            return

        data = []
        for row in selected_rows:
            r = row.row()
            # Handle filtered rows correctly?
            # selectedRows returns model indices, which map correctly even if filtered visually?
            # QTableWidget items are persistent.
            # But let's be safe and check if hidden.
            if self.bench_table.isRowHidden(r):
                continue

            name = self.bench_table.item(r, 1).text()
            score_text = self.bench_table.item(r, 3).text()

            try:
                score = float(score_text)
            except ValueError:
                score = 0.0 # Pending or Error

            data.append({'name': name, 'score': score})

        dlg = ComparisonDialog(data, self)
        dlg.exec()
