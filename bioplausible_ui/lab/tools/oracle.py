from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QComboBox, QMessageBox
from bioplausible_ui_old.tabs.vision_specialized_components import OracleWorker, OracleDialog

@ToolRegistry.register("oracle", requires=["oracle"])
class OracleTool(BaseTool):
    ICON = "🔮"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Oracle Tool"))
        self.layout.addWidget(QLabel("Analyze uncertainty by correlating input noise with settling time."))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"])
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_combo)

        self.run_btn = QPushButton("Run Oracle Analysis")
        self.run_btn.clicked.connect(self._run_oracle)
        self.layout.addWidget(self.run_btn)

        self.layout.addStretch()

    def _run_oracle(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Analyzing...")

        dataset = self.dataset_combo.currentText()
        self.worker = OracleWorker(self.model, dataset)
        self.worker.finished.connect(self._show_results)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _show_results(self, results):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Oracle Analysis")
        dlg = OracleDialog(results, self)
        dlg.exec()

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Oracle Analysis")
        QMessageBox.critical(self, "Error", msg)
