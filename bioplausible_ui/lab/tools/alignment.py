from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QComboBox, QMessageBox
from bioplausible_ui_old.tabs.vision_specialized_components import AlignmentWorker, AlignmentDialog

@ToolRegistry.register("alignment", requires=["alignment"])
class AlignmentTool(BaseTool):
    ICON = "📐"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Alignment Tool"))
        self.layout.addWidget(QLabel("Measure gradient alignment with backpropagation."))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"])
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_combo)

        self.check_btn = QPushButton("Check Alignment")
        self.check_btn.clicked.connect(self._check_alignment)
        self.layout.addWidget(self.check_btn)

        self.layout.addStretch()

    def _check_alignment(self):
        self.check_btn.setEnabled(False)
        self.check_btn.setText("Checking...")

        dataset = self.dataset_combo.currentText()
        self.worker = AlignmentWorker(self.model, dataset)
        self.worker.finished.connect(self._show_results)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _show_results(self, results):
        self.check_btn.setEnabled(True)
        self.check_btn.setText("Check Alignment")
        dlg = AlignmentDialog(results, self)
        dlg.exec()

    def _on_error(self, msg):
        self.check_btn.setEnabled(True)
        self.check_btn.setText("Check Alignment")
        QMessageBox.critical(self, "Error", msg)
