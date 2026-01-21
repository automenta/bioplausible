from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.train import TRAIN_TAB_SCHEMA
from bioplausible_ui.core.bridge import SessionBridge
from bioplausible.pipeline.config import TrainingConfig
from PyQt6.QtWidgets import QMessageBox

class TrainTab(BaseTab):
    """Training tab - UI auto-generated from schema."""

    SCHEMA = TRAIN_TAB_SCHEMA

    def _post_init(self):
        if 'stop' in self._actions:
            self._actions['stop'].setEnabled(False)

    def _start_training(self):
        try:
            train_config = self.training_config.get_values()
            config = TrainingConfig(
                task=self.task_selector.get_task(),
                dataset=self.dataset_picker.get_dataset(),
                model=self.model_selector.get_selected_model(),
                epochs=train_config.get('epochs', 10),
                batch_size=train_config.get('batch_size', 64),
                gradient_method=train_config.get('gradient_method', "BPTT (Standard)"),
                use_compile=train_config.get('use_compile', True),
                use_kernel=train_config.get('use_kernel', False),
                monitor_dynamics=train_config.get('monitor_dynamics', False),
                hyperparams=self.hyperparam_editor.get_values()
            )
            self.bridge = SessionBridge(config)
            self.bridge.progress_updated.connect(self._on_progress)
            self.bridge.training_completed.connect(self._on_completed)
            self.bridge.start()

            # Disable start, enable stop
            self._actions['start'].setEnabled(False)
            self._actions['stop'].setEnabled(True)

            # Clear plots
            self.plot_loss.clear()
            self.plot_accuracy.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            import traceback
            traceback.print_exc()

    def _stop_training(self):
        if hasattr(self, 'bridge'):
            self.bridge.stop()
            self._actions['start'].setEnabled(True)
            self._actions['stop'].setEnabled(False)

    def _on_progress(self, epoch, metrics):
        self.plot_loss.add_point(epoch, metrics.get('loss', 0))
        self.plot_accuracy.add_point(epoch, metrics.get('accuracy', 0))

    def _on_completed(self, final_metrics):
        QMessageBox.information(self, "Training Complete", f"Final Accuracy: {final_metrics.get('accuracy', 0):.4f}")
        self._actions['start'].setEnabled(True)
        self._actions['stop'].setEnabled(False)

    def _test_model(self):
        # Placeholder for inference logic
        # In a real implementation, this would load a test sample and run prediction
        # mimicking _test_random_sample from old vision_tab.py
        QMessageBox.information(self, "Test", "Running inference on random sample... (Not implemented in this demo)")
