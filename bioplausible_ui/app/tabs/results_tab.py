from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.results import RESULTS_TAB_SCHEMA
from PyQt6.QtWidgets import QMessageBox

class ResultsTab(BaseTab):
    """Results tab - UI auto-generated from schema."""

    SCHEMA = RESULTS_TAB_SCHEMA

    def _refresh_results(self):
        # Mock data for now
        self.results_table.table.setRowCount(0)
        self.results_table.add_run("run_001", "vision", "LeNet", 0.98)
        self.results_table.add_run("run_002", "lm", "Transformer", 0.85)

    def _delete_run(self):
        current_row = self.results_table.table.currentRow()
        if current_row >= 0:
            self.results_table.table.removeRow(current_row)
        else:
            QMessageBox.warning(self, "Warning", "Please select a run to delete.")
