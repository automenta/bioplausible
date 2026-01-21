import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout

class BasePlotWidget(QWidget):
    def __init__(self, title="", xlabel="", ylabel="", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(title=title)
        self.plot_widget.setLabel('bottom', xlabel)
        self.plot_widget.setLabel('left', ylabel)
        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen='y')
        self.data_x = []
        self.data_y = []

    def add_point(self, x, y):
        self.data_x.append(x)
        self.data_y.append(y)
        self.curve.setData(self.data_x, self.data_y)

    def clear(self):
        self.data_x = []
        self.data_y = []
        self.curve.setData([], [])
