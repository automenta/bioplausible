from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.console import CONSOLE_TAB_SCHEMA
import sys

class ConsoleTab(BaseTab):
    """Console tab - UI auto-generated from schema."""

    SCHEMA = CONSOLE_TAB_SCHEMA

    def _run_diagnostics(self):
        self.log_output.log("Running system diagnostics...")
        self.log_output.log(f"Python version: {sys.version}")
        self.log_output.log("Diagnostics complete.")

    def _clear_logs(self):
        self.log_output.text_edit.clear()
