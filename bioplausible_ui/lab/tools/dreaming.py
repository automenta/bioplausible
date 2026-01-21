from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QSlider
from PyQt6.QtCore import Qt

@ToolRegistry.register("dreaming", requires=["dreaming"])
class DreamingTool(BaseTool):
    ICON = "💤"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Dreaming Tool"))
        self.layout.addWidget(QLabel("Optimize input to maximize target class activation."))

        self.layout.addWidget(QLabel("Target Class:"))
        self.class_slider = QSlider(Qt.Orientation.Horizontal)
        self.class_slider.setRange(0, 9)
        self.class_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.layout.addWidget(self.class_slider)

        self.start_btn = QPushButton("Start Dreaming")
        self.start_btn.clicked.connect(self._start_dreaming)
        self.layout.addWidget(self.start_btn)

        self.layout.addStretch()

    def _start_dreaming(self):
        target_class = self.class_slider.value()
        # TODO: Implement dreaming logic
        print(f"Dreaming of class {target_class}...")
