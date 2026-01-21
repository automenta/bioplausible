from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QFormLayout
from bioplausible.models.registry import get_model_spec

class HyperparamEditor(QWidget):
    def __init__(self, model=None, defaults=None, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.values = {}
        if model:
            self.update_for_model(model)
        elif defaults:
            self.update_from_dict(defaults)

    def set_model(self, model_name):
        self.update_for_model(model_name)

    def update_from_dict(self, defaults):
        # Clear layout
        self._clear_layout()
        self.values = {}

        for key, value in defaults.items():
            if isinstance(value, bool):
                # Checkbox (not implemented yet, but for now generic spinner or something?)
                # Actually QCheckBox
                from PyQt6.QtWidgets import QCheckBox
                widget = QCheckBox()
                widget.setChecked(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-10000.0, 10000.0)
                widget.setValue(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-10000, 10000)
                widget.setValue(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, str):
                from PyQt6.QtWidgets import QLineEdit
                widget = QLineEdit(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget

    def _clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def update_for_model(self, model_name):
        # Clear layout
        self._clear_layout()

        self.values = {}
        try:
            spec = get_model_spec(model_name)

            # LR
            self.lr_spin = QDoubleSpinBox()
            self.lr_spin.setRange(0.00001, 1.0)
            self.lr_spin.setSingleStep(0.0001)
            self.lr_spin.setDecimals(5)
            self.lr_spin.setValue(spec.default_lr)
            self.layout.addRow("Learning Rate:", self.lr_spin)
            self.values['learning_rate'] = self.lr_spin

            if spec.has_beta:
                self.beta_spin = QDoubleSpinBox()
                self.beta_spin.setRange(0.0, 10.0)
                self.beta_spin.setSingleStep(0.01)
                self.beta_spin.setValue(spec.default_beta)
                self.layout.addRow("Beta:", self.beta_spin)
                self.values['beta'] = self.beta_spin

            if spec.has_steps:
                self.steps_spin = QSpinBox()
                self.steps_spin.setRange(1, 100)
                self.steps_spin.setValue(spec.default_steps)
                self.layout.addRow("Steps:", self.steps_spin)
                self.values['steps'] = self.steps_spin

        except ValueError:
            pass

    def get_values(self):
        res = {}
        for k, v in self.values.items():
            if hasattr(v, 'value'):
                res[k] = v.value()
            elif hasattr(v, 'isChecked'):
                res[k] = v.isChecked()
            elif hasattr(v, 'text'):
                res[k] = v.text()
        return res
