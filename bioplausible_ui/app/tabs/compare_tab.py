from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.compare import COMPARE_TAB_SCHEMA
from PyQt6.QtWidgets import QMessageBox

class CompareTab(BaseTab):
    """Comparison tab - UI auto-generated from schema."""

    SCHEMA = COMPARE_TAB_SCHEMA

    def _start_comparison(self):
        task = self.task_selector.get_task()
        dataset = self.dataset_picker.get_dataset()
        model1 = self.model_selector_1.get_selected_model()
        model2 = self.model_selector_2.get_selected_model()

        QMessageBox.information(self, "Compare", f"Comparing {model1} vs {model2} on {dataset} ({task})")
        # TODO: Implement actual comparison logic
