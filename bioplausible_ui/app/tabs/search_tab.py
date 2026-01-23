"""
Search Tab - Optuna-powered hyperparameter search
"""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from bioplausible.hyperopt import create_optuna_space, create_study
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible_ui.app.schemas.search import SEARCH_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab


class OptunaSearchWorker(QThread):
    """Worker thread for Optuna-based hyperparameter search."""

    trial_finished = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, task, model, max_trials=10, parent=None):
        super().__init__(parent)
        self.task = task
        self.model = model
        self.max_trials = max_trials
        self.running = True

    def run(self):
        # Create Optuna study
        study = create_study(
            model_names=[self.model],
            n_objectives=2,  # accuracy, loss
            storage=None,  # in-memory for UI
            study_name=f"{self.model}_{self.task}",
            use_pruning=True,
            sampler_name="tpe",
        )

        def objective(trial):
            if not self.running:
                import optuna
                raise optuna.TrialPruned()

            # Sample hyperparameters
            config = create_optuna_space(trial, self.model)
            config["epochs"] = 5  # Quick mode

            # Run trial
            metrics = run_single_trial_task(
                task=self.task, model_name=self.model, config=config, quick_mode=True
            )

            if metrics:
                accuracy = metrics.get("accuracy", 0.0)
                loss = metrics.get("loss", float("inf"))

                # Report to UI
                result = {
                    "params": config,
                    "accuracy": accuracy,
                    "loss": loss,
                    "trial_number": trial.number,
                }
                self.trial_finished.emit(result)

                return accuracy, loss
            else:
                import optuna
                raise optuna.TrialPruned()

        try:
            study.optimize(objective, n_trials=self.max_trials, show_progress_bar=False)
        except Exception as e:
            print(f"Optimization error: {e}")

        self.finished.emit()

    def stop(self):
        self.running = False


class SearchTab(BaseTab):
    """Search tab - Optuna-powered hyperparameter optimization."""

    SCHEMA = SEARCH_TAB_SCHEMA
    transfer_config = pyqtSignal(dict)

    def _post_init(self):
        self.worker = None
        # Connect RadarView signal
        if hasattr(self, "radar_view"):
            self.radar_view.pointClicked.connect(self._on_radar_click)

    def _start_search(self):
        task = self.task_selector.get_task()
        dataset = self.dataset_picker.get_dataset()
        model = self.model_selector.get_selected_model()

        self.log_output.append(f"Starting Optuna TPE search for {model}...")
        self.results_table.table.setRowCount(0)
        self.radar_view.clear()

        # Create Optuna worker
        self.worker = OptunaSearchWorker(dataset, model, max_trials=10)
        self.worker.trial_finished.connect(self._on_trial_finished)
        self.worker.finished.connect(self._on_search_finished)
        self.worker.start()

        self._actions["start"].setEnabled(False)
        self._actions["stop"].setEnabled(True)

    def _stop_search(self):
        if self.worker:
            self.worker.stop()
            self.worker.quit()
        self.log_output.append("Search stopped.")
        self._actions["start"].setEnabled(True)
        self._actions["stop"].setEnabled(False)

    def _on_trial_finished(self, result):
        params_str = ", ".join(
            [f"{k}={v}" for k, v in result["params"].items() if k != "epochs"]
        )
        trial_num = result.get("trial_number", "")
        self.log_output.append(
            f"Trial {trial_num}: {params_str} -> Acc: {result['accuracy']:.4f}"
        )

        # Add to Radar
        self.radar_view.add_result(result)

        # Add to table
        row = self.results_table.table.rowCount()
        self.results_table.table.insertRow(row)
        from PyQt6.QtWidgets import QTableWidgetItem

        self.results_table.table.setItem(row, 0, QTableWidgetItem(str(trial_num)))
        self.results_table.table.setItem(row, 1, QTableWidgetItem(params_str))
        self.results_table.table.setItem(
            row, 2, QTableWidgetItem(self.task_selector.get_task())
        )
        self.results_table.table.setItem(
            row, 3, QTableWidgetItem(self.model_selector.get_selected_model())
        )
        self.results_table.table.setItem(
            row, 4, QTableWidgetItem(f"{result['accuracy']:.4f}")
        )

    def _on_search_finished(self):
        self.log_output.append("Search finished.")
        self._actions["start"].setEnabled(True)
        self._actions["stop"].setEnabled(False)

    def _on_radar_click(self, result):
        params = result.get("params", {})
        acc = result.get("accuracy", 0.0)

        msg = QMessageBox(self)
        msg.setWindowTitle("Trial Details")
        msg.setText(f"Configuration:\n{params}\n\nAccuracy: {acc:.4f}")
        train_btn = msg.addButton(
            "Train This Config", QMessageBox.ButtonRole.AcceptRole
        )
        msg.addButton(QMessageBox.StandardButton.Close)

        msg.exec()

        if msg.clickedButton() == train_btn:
            config = {
                "task": self.task_selector.get_task(),
                "dataset": self.dataset_picker.get_dataset(),
                "model": self.model_selector.get_selected_model(),
                "hyperparams": params,
            }
            self.transfer_config.emit(config)
