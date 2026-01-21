from PyQt6.QtWidgets import QMainWindow, QTabWidget, QFileDialog, QMessageBox
from bioplausible_ui.lab.registry import ToolRegistry
import bioplausible_ui.lab.tools  # Register all tools
import torch
from bioplausible.models.registry import get_model_spec

class LabMainWindow(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.setWindowTitle("Bioplausible Lab (biopl-lab)")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Menu
        menu = self.menuBar().addMenu("File")
        menu.addAction("Load Model", self.load_model_dialog)

        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Model Checkpoint", "", "PyTorch Checkpoints (*.pt)")
        if fname:
            self.load_model(fname)

    def load_model(self, path):
        try:
            checkpoint = torch.load(path)
            config = checkpoint.get('config', {})
            model_name = config.get('model_name')

            if not model_name:
                raise ValueError("Model name not found in checkpoint")

            spec = get_model_spec(model_name)

            # TODO: Real model loading
            self.model = object()

            self.tabs.clear()
            tools = ToolRegistry.get_compatible_tools(spec)

            if not tools:
                QMessageBox.information(self, "Info", "No compatible tools found for this model.")

            for tool_name in tools:
                ToolClass = ToolRegistry.get_tool_class(tool_name)
                tool = ToolClass(self.model)
                self.tabs.addTab(tool, f"{tool.ICON} {tool_name.title()}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
