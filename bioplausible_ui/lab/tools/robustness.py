from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout

@ToolRegistry.register("robustness", requires=["robustness"])
class RobustnessTool(BaseTool):
    ICON = "🛡️"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Robustness Tool"))
        self.layout.addWidget(QLabel("Test model robustness against noise."))

        self.test_btn = QPushButton("Run Robustness Test")
        self.test_btn.clicked.connect(self._run_test)
        self.layout.addWidget(self.test_btn)

        self.layout.addStretch()

    def _run_test(self):
        # TODO: Implement robustness test logic
        print("Running robustness test...")
