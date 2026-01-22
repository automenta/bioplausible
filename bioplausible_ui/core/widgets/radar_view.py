import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class RadarView(QWidget):
    """
    Visualizes Architecture Space as a 2D scatter plot.

    Uses PCA dimensionality reduction to project hyperparameter space
    onto 2D coordinates if sklearn is available and enough points exist.
    Otherwise falls back to a heuristic complexity vs dynamics mapping.
    """

    pointClicked = pyqtSignal(dict)  # Emits result dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_PYQTGRAPH:
            self.layout.addWidget(QLabel("PyQtGraph required for Radar View"))
            return

        self.plot_widget = pg.PlotWidget(title="Architecture Radar")
        self.plot_widget.setLabel("bottom", "PC1 (Architecture Space)")
        self.plot_widget.setLabel("left", "PC2 (Architecture Space)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground("#1e1e2e")  # Dark theme

        # Scatter plot item
        self.scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen("w", width=1), brush=pg.mkBrush(255, 255, 255, 120)
        )
        self.scatter.sigClicked.connect(self._on_point_clicked)
        self.plot_widget.addItem(self.scatter)

        self.layout.addWidget(self.plot_widget)

        self.data_points = []  # List of result dicts
        self.vectors = []  # List of feature vectors
        self.param_keys = None  # To ensure consistent ordering

    def _vectorize_params(self, params):
        """Convert param dict to vector."""
        if self.param_keys is None:
            self.param_keys = sorted(params.keys())

        vec = []
        for k in self.param_keys:
            val = params.get(k, 0)
            if isinstance(val, (int, float)):
                vec.append(float(val))
            elif isinstance(val, str):
                # Simple hash for categorical
                vec.append(float(hash(val) % 100))
            else:
                vec.append(0.0)
        return vec

    def _update_plot(self):
        """Re-calculate coordinates and update scatter plot."""
        if not self.data_points:
            return

        N = len(self.data_points)

        # Determine coordinates
        if HAS_SKLEARN and N >= 3:
            try:
                X = np.array(self.vectors)
                # Standardize
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)

                # PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_std)

                xs = coords[:, 0]
                ys = coords[:, 1]

                self.plot_widget.setTitle(f"Architecture Radar (PCA, N={N})")
                self.plot_widget.setLabel("bottom", "Principal Component 1")
                self.plot_widget.setLabel("left", "Principal Component 2")

            except Exception as e:
                print(f"PCA failed: {e}, falling back to heuristic")
                xs, ys = self._get_heuristic_coords()
        else:
            xs, ys = self._get_heuristic_coords()
            self.plot_widget.setTitle(f"Architecture Radar (Heuristic, N={N})")
            self.plot_widget.setLabel("bottom", "Complexity (Layers × Width)")
            self.plot_widget.setLabel("left", "Dynamics (log LR)")

        # Determine brushes (colors) based on accuracy
        brushes = []
        for res in self.data_points:
            acc = res.get("accuracy", 0.0)
            # Heatmap: Blue (0.0) -> Red (1.0)
            r = int(np.clip(acc * 255, 0, 255))
            b = int(np.clip((1 - acc) * 255, 0, 255))
            brushes.append(pg.mkBrush(r, 0, b, 200))

        # Update scatter
        # We must rebuild spots
        spots = []
        for i in range(N):
            spots.append(
                {
                    "pos": (xs[i], ys[i]),
                    "brush": brushes[i],
                    "data": self.data_points[i],
                }
            )

        self.scatter.setData(spots=spots)

    def _get_heuristic_coords(self):
        """Calculate coordinates using heuristic mapping."""
        xs = []
        ys = []
        for res in self.data_points:
            params = res.get("params", {})
            # X: Complexity
            layers = params.get("num_layers", 1)
            hidden = params.get("hidden_dim", 64)
            xs.append(layers * hidden)

            # Y: Dynamics
            lr = params.get("lr", 0.001)
            ys.append(np.log10(max(lr, 1e-9)))
        return xs, ys

    def add_result(self, result):
        """
        Add a search result to the radar.
        """
        params = result.get("params", {})
        vec = self._vectorize_params(params)

        self.data_points.append(result)
        self.vectors.append(vec)

        self._update_plot()

    def clear(self):
        self.scatter.clear()
        self.data_points = []
        self.vectors = []
        self.param_keys = None

    def _on_point_clicked(self, plot, points):
        if len(points) > 0:
            point = points[0]
            result = point.data()
            self.pointClicked.emit(result)
