from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QCheckBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from bioplausible_ui.worker import BenchmarkWorker

class BenchmarksTab(QWidget):
    """A tab for running validation tracks (benchmarks)."""

    log_message = pyqtSignal(str) # Signal to log to console

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

        run_sel_btn = QPushButton("▶ Run Selected")
        run_sel_btn.clicked.connect(self._run_selected_benchmarks)
        controls_layout.addWidget(run_sel_btn)

        run_all_btn = QPushButton("⏩ Run All")
        run_all_btn.clicked.connect(self._run_all_benchmarks)
        controls_layout.addWidget(run_all_btn)

        left_panel.addWidget(controls_group)

        # Track List Table
        self.bench_table = QTableWidget()
        self.bench_table.setColumnCount(4)
        self.bench_table.setHorizontalHeaderLabels(["ID", "Track Name", "Status", "Score"])
        self.bench_table.horizontalHeader().setStretchLastSection(True)
        self.bench_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

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

    def _run_selected_benchmarks(self):
        """Run selected benchmarks."""
        selected_rows = self.bench_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one track to run.")
            return

        track_ids = []
        for row in selected_rows:
            tid_item = self.bench_table.item(row.row(), 0)
            if tid_item:
                track_ids.append(int(tid_item.text()))

        self._start_benchmark_worker(track_ids)

    def _run_all_benchmarks(self):
        """Run all benchmarks."""
        rows = self.bench_table.rowCount()
        track_ids = []
        for r in range(rows):
            tid_item = self.bench_table.item(r, 0)
            if tid_item:
                track_ids.append(int(tid_item.text()))

        self._start_benchmark_worker(track_ids)

    def _start_benchmark_worker(self, track_ids):
        """Start the benchmark worker."""
        quick = self.bench_quick_check.isChecked()
        self.bench_output.clear()
        self.bench_output.append(f"Starting {len(track_ids)} tracks (Quick={quick})...\n")

        # Reset status in table
        for r in range(self.bench_table.rowCount()):
            tid_item = self.bench_table.item(r, 0)
            if tid_item and int(tid_item.text()) in track_ids:
                self.bench_table.setItem(r, 2, QTableWidgetItem("Running..."))
                self.bench_table.item(r, 2).setForeground(Qt.GlobalColor.yellow)

        self.bench_worker = BenchmarkWorker(track_ids, quick_mode=quick)
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
