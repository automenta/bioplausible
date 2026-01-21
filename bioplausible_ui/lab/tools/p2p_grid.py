from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout

@ToolRegistry.register("p2p_grid", requires=["p2p"])
class P2PGridTool(BaseTool):
    ICON = "🕸️"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("P2P Grid"))
        self.layout.addWidget(QLabel("Visualize distributed training network."))

        self.connect_btn = QPushButton("Connect to Grid")
        self.connect_btn.clicked.connect(self._connect)
        self.layout.addWidget(self.connect_btn)

        self.layout.addStretch()

    def _connect(self):
        # TODO: Implement P2P grid logic
        print("Connecting to P2P Grid...")
