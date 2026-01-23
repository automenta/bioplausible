"""
Search Tab - Optuna-powered hyperparameter search
"""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from bioplausible.hyperopt import create_optuna_space, create_study
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.hyperopt.eval_tiers import PatientLevel, get_evaluation_config
from bioplausible_ui.app.schemas.search import SEARCH_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab


class OptunaSearchWorker(QThread):
    """Worker thread for Optuna-based hyperparameter search."""

    trial_finished = pyqtSignal(dict)
    finished = pyqtSignal()
    search_started = pyqtSignal(str)  # Emits model name

    def __init__(self, task, models, tier=PatientLevel.SHALLOW, parent=None):
        super().__init__(parent)
        self.task = task
        self.models = models if isinstance(models, list) else [models]
        self.tier = tier
        
        # Get config for this tier
        self.eval_config = get_evaluation_config(tier)
        self.max_trials = self.eval_config.n_trials
        self.running = True

    def run(self):
        for model in self.models:
            if not self.running:
                break
                
            self.search_started.emit(model)
            
            # Create Optuna study per model
            # Include tier in study name to separate results!
            study_name = f"{model}_{self.task}_{self.tier.value}"
            
            study = create_study(
                model_names=[model],
                n_objectives=2,  # accuracy, loss
                storage=f"sqlite:///bioplausible.db",  # Persist to DB now
                study_name=study_name,
                use_pruning=self.eval_config.use_pruning,
                sampler_name="tpe",
                load_if_exists=True, 
            )

            current_model = model # closure capture

            def objective(trial):
                if not self.running:
                    import optuna
                    raise optuna.TrialPruned()

                # Sample hyperparameters
                config = create_optuna_space(trial, current_model)
                
                # Apply Tier constraints
                config["epochs"] = self.eval_config.epochs
                config["batch_size"] = self.eval_config.batch_size
                
                # Tag the trial with tier
                trial.set_user_attr("tier", self.tier.value)
                trial.set_user_attr("model_family", current_model) # Simplify downstream analysis

                # Run trial
                # quick_mode logic might override epochs, so we pass explicit config
                # and assume runner respects it if quick_mode is False or handled
                metrics = run_single_trial_task(
                    task=self.task, 
                    model_name=current_model, 
                    config=config, 
                    quick_mode=False, # We control epochs manually via config
                    train_samples=self.eval_config.train_samples,
                    val_samples=self.eval_config.val_samples
                )

                if metrics:
                    accuracy = metrics.get("accuracy", 0.0)
                    loss = metrics.get("loss", float("inf"))

                    # Report to UI
                    result = {
                        "model": current_model,
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
                print(f"Optimization error for {model}: {e}")

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
        models = self.model_selector.get_selected_models()
        
        if not models:
            QMessageBox.warning(self, "No Models Selected", "Please select at least one algorithm to search.")
            return

        # Get Tier
        tier_str = self.ui_elements["tier_selector"].currentText()
        if "Smoke" in tier_str: tier = PatientLevel.SMOKE
        elif "Shallow" in tier_str: tier = PatientLevel.SHALLOW
        elif "Standard" in tier_str: tier = PatientLevel.STANDARD
        elif "Deep" in tier_str: tier = PatientLevel.DEEP
        else: tier = PatientLevel.SHALLOW

        self.log_output.append(f"Starting Multi-Algorithm Search (Tier: {tier.name})...")
        self.results_table.table.setRowCount(0)
        self.radar_view.clear()

        # Create Optuna worker
        self.worker = OptunaSearchWorker(dataset, models, tier=tier)
        self.worker.trial_finished.connect(self._on_trial_finished)
        self.worker.search_started.connect(lambda m: self.log_output.append(f"🔍 Optimizing {m}..."))
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
        model_name = result.get("model", "Unknown")
        self.log_output.append(
            f"[{model_name}] Trial {trial_num}: {params_str} -> Acc: {result['accuracy']:.4f}"
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
            row, 3, QTableWidgetItem(model_name)
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
                "model": result.get("model", "Unknown"),
                "hyperparams": params,
            }
            self.transfer_config.emit(config)
