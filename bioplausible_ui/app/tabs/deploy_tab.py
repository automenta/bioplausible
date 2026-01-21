from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.deploy import DEPLOY_TAB_SCHEMA
from PyQt6.QtWidgets import QMessageBox

class DeployTab(BaseTab):
    """Deploy tab - UI auto-generated from schema."""

    SCHEMA = DEPLOY_TAB_SCHEMA

    def _export_model(self):
        model = self.model_selector.get_selected_model()
        fmt = self.format_selector.get_format()
        QMessageBox.information(self, "Export", f"Exporting {model} to {fmt}...")

    def _serve_model(self):
        model = self.model_selector.get_selected_model()
        QMessageBox.information(self, "Serve", f"Starting inference server for {model}...")
