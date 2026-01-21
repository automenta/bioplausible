from PyQt6.QtWidgets import QLabel
from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
import pyqtgraph as pg

@ToolRegistry.register("microscope", requires=["dynamics"])
class MicroscopeTool(BaseTool):
    ICON = "🔬"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Live Network Dynamics"))
        self.plot = pg.PlotWidget(title="Activity Dynamics")
        self.layout.addWidget(self.plot)

    def refresh(self):
        if self.model:
            # logic to inspect model
            pass
