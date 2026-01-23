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
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    
from bioplausible_ui.core.themes import Theme
from PyQt6.QtGui import QFont


class RadarView(QWidget):
    """
    Visualizes Architecture Space as a 2D scatter plot.

    Uses UMAP, t-SNE, or PCA dimensionality reduction to project hyperparameter space
    onto 2D coordinates. Colors points by algorithm family.
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
        self.plot_widget.setLabel("bottom", "Dimension 1")
        self.plot_widget.setLabel("left", "Dimension 2")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground("#1e1e2e")  # Dark theme

        # Scatter plot item
        self.scatter = pg.ScatterPlotItem(
            size=12, 
            pen=pg.mkPen("w", width=0.5), 
            brush=pg.mkBrush(255, 255, 255, 120),
            hoverable=True,
            hoverPen=pg.mkPen("w", width=2),
            hoverBrush=pg.mkBrush(255, 255, 255, 200),
            hoverSize=16
        )
        self.scatter.sigClicked.connect(self._on_point_clicked)
        self.scatter.sigHovered.connect(self._on_point_hovered)
        self.plot_widget.addItem(self.scatter)
        
        # Tooltip label
        self.tooltip = pg.TextItem(text="", color="#e2e8f0", fill=pg.mkBrush(0, 0, 0, 200), anchor=(0, 1))
        self.tooltip.setFont(QFont("Arial", 11))
        self.tooltip.setVisible(False)
        self.plot_widget.addItem(self.tooltip)
        
        self.layout.addWidget(self.plot_widget)
        
        # Legend (simulated via label for now, or use pg.LegendItem)
        self.legend = pg.LegendItem((80, 60), offset=(70, 20))
        self.legend.setParentItem(self.plot_widget.graphicsItem())
        
        self.data_points = []  # List of result dicts
        self.vectors = []  # List of feature vectors
        self.param_keys = None  # To ensure consistent ordering
        self.families = set()

        # Color palette for families
        self.palette = [
            (59, 130, 246),   # Blue
            (16, 185, 129),   # Green
            (249, 115, 22),   # Orange
            (168, 85, 247),   # Purple
            (236, 72, 153),   # Pink
            (239, 68, 68),    # Red
            (14, 165, 233),   # Sky
            (250, 204, 21),   # Yellow
        ]
        
    def _get_family_color(self, model_name):
        """Get color for model family."""
        family = model_name.split()[0].lower()
        if 'eqprop' in family: color_idx = 0
        elif 'backprop' in family or 'baseline' in family: color_idx = 1
        elif 'dfa' in family: color_idx = 2
        elif 'lcod' in family or 'predictive' in family: color_idx = 3
        else: color_idx = hash(family) % len(self.palette)
        
        return self.palette[color_idx]

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
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)

                # Prioritize UMAP > t-SNE > PCA
                if HAS_UMAP and N > 5:
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    title_method = "UMAP"
                elif N > 10:
                    reducer = TSNE(n_components=2, perplexity=min(30, N-1), random_state=42)
                    title_method = "t-SNE"
                else:
                    reducer = PCA(n_components=2)
                    title_method = "PCA"
                
                coords = reducer.fit_transform(X_std)
                xs = coords[:, 0]
                ys = coords[:, 1]

                self.plot_widget.setTitle(f"Architecture Radar ({title_method}, N={N})")

            except Exception as e:
                print(f"Reduction failed: {e}, falling back to heuristic")
                xs, ys = self._get_heuristic_coords()
        else:
            xs, ys = self._get_heuristic_coords()
            self.plot_widget.setTitle(f"Architecture Radar (Heuristic, N={N})")
            self.plot_widget.setLabel("bottom", "Complexity (Layers × Width)")
            self.plot_widget.setLabel("left", "Dynamics (log LR)")

        # Determine colors based on Model Family
        brushes = []
        
        # Legend tracking
        seen_families = set()
        self.legend.clear()
        
        spots = []
        for i in range(N):
            res = self.data_points[i]
            model = res.get("model", "Unknown")
            family = model.split()[0]
            
            color = self._get_family_color(model)
            brush = pg.mkBrush(*color, 200)
            
            spots.append({
                "pos": (xs[i], ys[i]),
                "brush": brush,
                "data": res,
                "size": 14,
                "pen": pg.mkPen("w", width=0.5),
            })
            
            if family not in seen_families:
                self.legend.addItem(pg.ScatterPlotItem(pen=brush, brush=brush, size=10), family)
                seen_families.add(family)

        self.scatter.setData(spots=spots)

    def _get_heuristic_coords(self):
        """Calculate coordinates using heuristic mapping."""
        xs = []
        ys = []
        for res in self.data_points:
            params = res.get("params", {})
            layers = params.get("num_layers", 1)
            hidden = params.get("hidden_dim", 64)
            xs.append(layers * hidden)

            lr = params.get("lr", 0.001)
            ys.append(np.log10(max(lr, 1e-9)))
        return xs, ys

    def add_result(self, result):
        """Add a search result to the radar."""
        params = result.get("params", {})
        vec = self._vectorize_params(params)

        self.data_points.append(result)
        self.vectors.append(vec)
        self._update_plot()

    def clear(self):
        self.scatter.clear()
        self.legend.clear()
        self.data_points = []
        self.vectors = []
        self.param_keys = None

    def _on_point_clicked(self, plot, points):
        if len(points) > 0:
            point = points[0]
            result = point.data()
            self.pointClicked.emit(result)

    def _on_point_hovered(self, plot, points):
        """Handle hover event to show tooltip."""
        if len(points) > 0:
            point = points[0]
            result = point.data()
            
            # Create tooltip text
            model = result.get("model", "Unknown")
            acc = result.get("accuracy", 0.0)
            text = f"{model}\nAcc: {acc:.2%}"
            
            # Position tooltip near the point
            pos = point.pos()
            self.tooltip.setText(text)
            self.tooltip.setPos(pos.x(), pos.y())
            self.tooltip.setVisible(True)
        else:
            self.tooltip.setVisible(False)
