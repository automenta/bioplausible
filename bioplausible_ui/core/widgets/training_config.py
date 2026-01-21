from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QFormLayout, QComboBox, QCheckBox

class TrainingConfigWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(64)
        layout.addRow("Batch Size:", self.batch_spin)

        self.grad_combo = QComboBox()
        self.grad_combo.addItems(["BPTT (Standard)", "Equilibrium (Implicit Diff)", "Contrastive (Hebbian)"])
        layout.addRow("Gradient:", self.grad_combo)

        self.compile_check = QCheckBox("torch.compile")
        self.compile_check.setChecked(True)
        layout.addRow("", self.compile_check)

        self.kernel_check = QCheckBox("O(1) Kernel Mode (GPU)")
        layout.addRow("", self.kernel_check)

        self.micro_check = QCheckBox("Live Dynamics Analysis")
        layout.addRow("", self.micro_check)

    def get_values(self):
        return {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "gradient_method": self.grad_combo.currentText(),
            "use_compile": self.compile_check.isChecked(),
            "use_kernel": self.kernel_check.isChecked(),
            "monitor_dynamics": self.micro_check.isChecked()
        }
