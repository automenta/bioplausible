import threading
from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.search import SEARCH_TAB_SCHEMA
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.hyperopt.search_space import GridSearch, RandomSearch
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject, QThread

class SearchWorker(QThread):
    trial_finished = pyqtSignal(dict) # {params, accuracy}
    finished = pyqtSignal()

    def __init__(self, task, model, space, strategy="grid", max_trials=10, parent=None):
        super().__init__(parent)
        self.task = task
        self.model = model
        self.space = space # Dict of params
        self.strategy = strategy
        self.max_trials = max_trials
        self.running = True

    def run(self):
        # Generate configurations
        configs = []
        if self.strategy == "grid":
            configs = list(GridSearch(self.space))
        else:
            configs = list(RandomSearch(self.space, n_iter=self.max_trials))

        for i, config in enumerate(configs):
            if not self.running: break

            # Default training params
            config['epochs'] = 5

            # Run trial
            metrics = run_single_trial_task(
                task=self.task,
                model_name=self.model,
                config=config,
                quick_mode=True
            )

            if metrics:
                result = {
                    "params": config,
                    "accuracy": metrics.get("accuracy", 0.0),
                    "loss": metrics.get("loss", 0.0)
                }
                self.trial_finished.emit(result)

        self.finished.emit()

    def stop(self):
        self.running = False

class SearchTab(BaseTab):
    """Search tab - UI auto-generated from schema."""

    SCHEMA = SEARCH_TAB_SCHEMA
    transfer_config = pyqtSignal(dict)

    def _post_init(self):
        self.worker = None
        # Connect RadarView signal
        if hasattr(self, 'radar_view'):
            self.radar_view.pointClicked.connect(self._on_radar_click)

    def _start_search(self):
        task = self.task_selector.get_task()
        dataset = self.dataset_picker.get_dataset()
        model = self.model_selector.get_selected_model()
        space_config = self.search_space.get_values()

        self.log_output.append(f"Starting {space_config.get('strategy', 'grid')} search for {model}...")
        self.results_table.table.setRowCount(0)
        self.radar_view.clear()

        # Extract strategy
        strategy = space_config.pop('strategy', 'grid')
        max_trials = space_config.pop('max_trials', 10)

        # Clean config for runner
        search_space = {k: v for k,v in space_config.items() if k not in ['strategy', 'max_trials']}

        self.worker = SearchWorker(dataset, model, search_space, strategy, max_trials)
        self.worker.trial_finished.connect(self._on_trial_finished)
        self.worker.finished.connect(self._on_search_finished)
        self.worker.start()

        self._actions['start'].setEnabled(False)
        self._actions['stop'].setEnabled(True)

    def _stop_search(self):
        if self.worker:
            self.worker.stop()
            self.worker.quit()
        self.log_output.append("Search stopped.")
        self._actions['start'].setEnabled(True)
        self._actions['stop'].setEnabled(False)

    def _on_trial_finished(self, result):
        params_str = ", ".join([f"{k}={v}" for k,v in result['params'].items() if k != 'epochs'])
        self.log_output.append(f"Trial: {params_str} -> Acc: {result['accuracy']:.4f}")

        # Add to Radar
        self.radar_view.add_result(result)

        # Add to table
        row = self.results_table.table.rowCount()
        self.results_table.table.insertRow(row)
        from PyQt6.QtWidgets import QTableWidgetItem
        self.results_table.table.setItem(row, 0, QTableWidgetItem("")) # Timestamp
        self.results_table.table.setItem(row, 1, QTableWidgetItem(params_str))
        self.results_table.table.setItem(row, 2, QTableWidgetItem(self.task_selector.get_task()))
        self.results_table.table.setItem(row, 3, QTableWidgetItem(self.model_selector.get_selected_model()))
        self.results_table.table.setItem(row, 4, QTableWidgetItem(f"{result['accuracy']:.4f}"))

    def _on_search_finished(self):
        self.log_output.append("Search finished.")
        self._actions['start'].setEnabled(True)
        self._actions['stop'].setEnabled(False)

    def _on_radar_click(self, result):
        params = result.get('params', {})
        acc = result.get('accuracy', 0.0)

        msg = QMessageBox(self)
        msg.setWindowTitle("Trial Details")
        msg.setText(f"Configuration:\n{params}\n\nAccuracy: {acc:.4f}")
        train_btn = msg.addButton("Train This Config", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Close)

        msg.exec()

        if msg.clickedButton() == train_btn:
            config = {
                "task": self.task_selector.get_task(),
                "dataset": self.dataset_picker.get_dataset(),
                "model": self.model_selector.get_selected_model(),
                "hyperparams": params
            }
            self.transfer_config.emit(config)
