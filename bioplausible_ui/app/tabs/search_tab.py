from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.search import SEARCH_TAB_SCHEMA
from PyQt6.QtWidgets import QMessageBox

class SearchTab(BaseTab):
    """Search tab - UI auto-generated from schema."""

    SCHEMA = SEARCH_TAB_SCHEMA

    def _start_search(self):
        task = self.task_selector.get_task()
        dataset = self.dataset_picker.get_dataset()
        model = self.model_selector.get_selected_model()
        space = self.search_space.get_values()

        QMessageBox.information(self, "Search", f"Starting search for {model} on {dataset} ({task})")
        # TODO: Implement search logic

    def _stop_search(self):
        QMessageBox.information(self, "Search", "Stopping search...")
