from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.compare import COMPARE_TAB_SCHEMA
from bioplausible_ui.core.bridge import SessionBridge
from bioplausible.pipeline.config import TrainingConfig
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal, QThread

class ComparisonWorker(QThread):
    progress = pyqtSignal(int, dict, dict) # epoch, metrics1, metrics2
    completed = pyqtSignal()

    def __init__(self, config1, config2, parent=None):
        super().__init__(parent)
        self.config1 = config1
        self.config2 = config2
        self.bridge1 = SessionBridge(config1)
        self.bridge2 = SessionBridge(config2)
        self.running = True

    def run(self):
        # We need to interleave generators manually
        gen1 = self.bridge1.session.start()
        gen2 = self.bridge2.session.start()

        from bioplausible.pipeline.events import ProgressEvent, CompletedEvent

        active1 = True
        active2 = True
        epoch = 0

        metrics1 = {}
        metrics2 = {}

        while (active1 or active2) and self.running:
            # Step 1
            if active1:
                try:
                    event = next(gen1)
                    if isinstance(event, ProgressEvent):
                        metrics1 = event.metrics
                    elif isinstance(event, CompletedEvent):
                        metrics1 = event.final_metrics
                        active1 = False
                except StopIteration:
                    active1 = False
                except Exception as e:
                    print(f"Error Model 1: {e}")
                    active1 = False

            # Step 2
            if active2:
                try:
                    event = next(gen2)
                    if isinstance(event, ProgressEvent):
                        metrics2 = event.metrics
                    elif isinstance(event, CompletedEvent):
                        metrics2 = event.final_metrics
                        active2 = False
                except StopIteration:
                    active2 = False
                except Exception as e:
                    print(f"Error Model 2: {e}")
                    active2 = False

            if metrics1 or metrics2:
                self.progress.emit(epoch, metrics1, metrics2)

            epoch += 1

        self.completed.emit()

    def stop(self):
        self.running = False
        self.bridge1.stop()
        self.bridge2.stop()

class CompareTab(BaseTab):
    """Comparison tab - UI auto-generated from schema."""

    SCHEMA = COMPARE_TAB_SCHEMA

    def _post_init(self):
        self.worker = None

    def _start_comparison(self):
        task = self.task_selector.get_task()
        dataset = self.dataset_picker.get_dataset()
        model1 = self.model_selector_1.get_selected_model()
        model2 = self.model_selector_2.get_selected_model()

        # We use defaults for now, or need hyperparam editor for both?
        # Schema only has selectors. We assume default hyperparams.

        conf1 = TrainingConfig(task=task, dataset=dataset, model=model1, epochs=10)
        conf2 = TrainingConfig(task=task, dataset=dataset, model=model2, epochs=10)

        self.plot_loss.clear()
        self.plot_accuracy.clear()

        # Legend
        self.plot_loss.add_legend(["Model 1", "Model 2"]) # BaseTab helper might need enhancement for legend
        # Actually `create_plot_widget` returns plot and curve.
        # BaseTab stores specific plots by name.
        # We need 2 curves per plot.
        # BaseTab logic creates one curve by default?
        # Let's inspect BaseTab later or just use direct pyqtgraph calls if accessible.
        # BaseTab `self.plot_loss` is a `PlotWidget` wrapper or `pg.PlotWidget`.
        # Assuming `PlotWidget` wrapper from `bioplausible_ui.core.widgets.plot_widget`.

        self.worker = ComparisonWorker(conf1, conf2)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_finished)
        self.worker.start()

        self._actions['compare'].setEnabled(False)

    def _on_progress(self, epoch, m1, m2):
        # We need to add point to TWO curves.
        # `PlotWidget` (our wrapper) usually manages one curve via `add_point`.
        # We might need to access underlying plot or add second curve.

        # Hacky: access internal plot item
        if hasattr(self.plot_loss, 'plot_item'): # if it's our wrapper
            if not hasattr(self, 'curve1_loss'):
                import pyqtgraph as pg
                self.curve1_loss = self.plot_loss.plot_item.plot(pen='r', name="Model 1")
                self.curve2_loss = self.plot_loss.plot_item.plot(pen='b', name="Model 2")
                self.curve1_acc = self.plot_accuracy.plot_item.plot(pen='r', name="Model 1")
                self.curve2_acc = self.plot_accuracy.plot_item.plot(pen='b', name="Model 2")

                self.x_data = []
                self.y1_loss = []
                self.y2_loss = []
                self.y1_acc = []
                self.y2_acc = []

            self.x_data.append(epoch)
            self.y1_loss.append(m1.get('loss', 0))
            self.y2_loss.append(m2.get('loss', 0))
            self.y1_acc.append(m1.get('accuracy', 0))
            self.y2_acc.append(m2.get('accuracy', 0))

            self.curve1_loss.setData(self.x_data, self.y1_loss)
            self.curve2_loss.setData(self.x_data, self.y2_loss)
            self.curve1_acc.setData(self.x_data, self.y1_acc)
            self.curve2_acc.setData(self.x_data, self.y2_acc)

    def _on_finished(self):
        self._actions['compare'].setEnabled(True)
        QMessageBox.information(self, "Done", "Comparison Completed")
