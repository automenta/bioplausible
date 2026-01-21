from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout

@ToolRegistry.register("alignment", requires=["alignment"])
class AlignmentTool(BaseTool):
    ICON = "📐"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Alignment Tool"))
        self.layout.addWidget(QLabel("Measure gradient alignment with backpropagation."))

        self.check_btn = QPushButton("Check Alignment")
        self.check_btn.clicked.connect(self._check_alignment)
        self.layout.addWidget(self.check_btn)

        self.layout.addStretch()

    def _check_alignment(self):
        # TODO: Implement alignment check logic
        print("Checking alignment...")
