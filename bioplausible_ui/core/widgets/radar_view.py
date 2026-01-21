import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtCore import pyqtSignal

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

class RadarView(QWidget):
    """
    Visualizes Architecture Space as a 2D scatter plot.

    Mapping strategy (simple heuristic for now):
    X-axis: Model Complexity (e.g. Layers * Hidden Dim)
    Y-axis: Optimization Dynamics (e.g. log(Learning Rate) or Beta)
    Color: Accuracy (Heatmap)
    """

    pointClicked = pyqtSignal(dict) # Emits result dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_PYQTGRAPH:
            self.layout.addWidget(QLabel("PyQtGraph required for Radar View"))
            return

        self.plot_widget = pg.PlotWidget(title="Architecture Radar")
        self.plot_widget.setLabel('bottom', "Model Complexity (Param Count Proxy)")
        self.plot_widget.setLabel('left', "Optimization Dynamics (log LR)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('#1e1e2e') # Dark theme

        # Scatter plot item
        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter.sigClicked.connect(self._on_point_clicked)
        self.plot_widget.addItem(self.scatter)

        self.layout.addWidget(self.plot_widget)

        self.data_points = [] # List of {x, y, result}

    def add_result(self, result):
        """
        Add a search result to the radar.

        Args:
            result: dict containing 'params', 'accuracy', etc.
        """
        params = result.get('params', {})

        # 1. Calculate X (Complexity)
        # Proxy: layers * hidden_dim
        layers = params.get('num_layers', 1)
        hidden = params.get('hidden_dim', 64)
        complexity = layers * hidden

        # 2. Calculate Y (Dynamics)
        # Proxy: log(lr)
        lr = params.get('lr', 0.001)
        dynamics = np.log10(max(lr, 1e-6))

        # 3. Color by Accuracy
        acc = result.get('accuracy', 0.0)
        # Map 0.0-1.0 to Blue-Red
        # Simple: (R, G, B)
        # Low acc: Blue (0, 0, 255)
        # High acc: Red (255, 0, 0)
        r = int(acc * 255)
        b = int((1-acc) * 255)
        color = (r, 0, b, 200) # alpha 200

        # Add point
        self.scatter.addPoints(
            x=[complexity],
            y=[dynamics],
            brush=pg.mkBrush(*color),
            data=result # Store full result in spot
        )

        self.data_points.append(result)

    def clear(self):
        self.scatter.clear()
        self.data_points = []

    def _on_point_clicked(self, plot, points):
        if len(points) > 0:
            point = points[0]
            result = point.data()
            self.pointClicked.emit(result)
