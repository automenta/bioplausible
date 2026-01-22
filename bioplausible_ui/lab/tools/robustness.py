from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.registry import ToolRegistry
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QComboBox, QMessageBox
from bioplausible_ui_old.tabs.vision_specialized_components import RobustnessDialog
import torch
import numpy as np
from bioplausible.datasets import get_vision_dataset
from torch.utils.data import DataLoader

@ToolRegistry.register("robustness", requires=["robustness"])
class RobustnessTool(BaseTool):
    ICON = "🛡️"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Robustness Tool"))
        self.layout.addWidget(QLabel("Test model robustness against noise."))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"])
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_combo)

        self.test_btn = QPushButton("Run Robustness Test")
        self.test_btn.clicked.connect(self._run_test)
        self.layout.addWidget(self.test_btn)

        self.layout.addStretch()

    def _run_test(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return

        try:
            self.test_btn.setEnabled(False)
            self.test_btn.setText("Running...")

            # Logic adapted from VisionTab._run_robustness_check
            ds_name = self.dataset_combo.currentText().lower().replace('-', '_')
            use_flatten = True # Simplified assumption for now

            dataset = get_vision_dataset(ds_name, train=False, flatten=use_flatten)
            subset_indices = np.random.choice(len(dataset), 200, replace=False)
            subset = torch.utils.data.Subset(dataset, subset_indices)
            loader = DataLoader(subset, batch_size=50, shuffle=False)

            device = next(self.model.parameters()).device
            self.model.eval()

            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            results = []

            with torch.no_grad():
                for noise_sigma in noise_levels:
                    correct = 0
                    total = 0
                    for x, y in loader:
                        x = x.to(device)
                        y = y.to(device)
                        if noise_sigma > 0:
                            x = x + torch.randn_like(x) * noise_sigma
                        try:
                            out = self.model(x)
                        except TypeError:
                            out = self.model(x, steps=20)
                        pred = out.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                    results.append((noise_sigma, correct / total))

            dlg = RobustnessDialog(results, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("Run Robustness Test")
