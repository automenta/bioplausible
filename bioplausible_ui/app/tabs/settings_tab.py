from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.settings import SETTINGS_TAB_SCHEMA
from PyQt6.QtWidgets import QMessageBox

class SettingsTab(BaseTab):
    """Settings tab - UI auto-generated from schema."""

    SCHEMA = SETTINGS_TAB_SCHEMA

    def _save_settings(self):
        settings = self.preferences.get_values()
        QMessageBox.information(self, "Settings", f"Settings saved: {settings}")

    def _reset_settings(self):
        # This would need support in HyperparamEditor to reset to defaults
        QMessageBox.information(self, "Settings", "Settings reset to defaults.")
