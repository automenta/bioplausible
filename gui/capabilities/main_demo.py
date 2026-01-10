"""
Main Demo Window - Tab-based interface for capability demonstrations
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .core_panel import CorePanel
from .biological_panel import BiologicalPanel
from .neuromorphic_panel import NeuromorphicPanel
from .depth_panel import DepthPanel


class CapabilitiesDemoWindow(QMainWindow):
    """
    Main window for EqProp capabilities demonstrations.
    
    Features 4 capability tabs:
    1. Core EqProp - Energy landscape & basic principles
    2. Biological Plausibility - Self-healing, Hebbian learning
    3. Neuromorphic/Hardware - Memory, noise resilience, quantization
    4. Deep Architecture - 500-layer stability, 3D topology
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EqProp Capabilities Demo - Visual Explainer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Build the main UI with tabs."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create panels
        self.core_panel = CorePanel()
        self.bio_panel = BiologicalPanel()
        self.neuro_panel = NeuromorphicPanel()
        self.depth_panel = DepthPanel()
        
        # Add tabs
        self.tabs.addTab(self.core_panel, "🔬 Core EqProp")
        self.tabs.addTab(self.bio_panel, "🧠 Biological Plausibility")
        self.tabs.addTab(self.neuro_panel, "⚡ Neuromorphic/Hardware")
        self.tabs.addTab(self.depth_panel, "📊 Deep Architecture")
        
        layout.addWidget(self.tabs)
        
        # Footer controls
        footer = self.create_footer()
        layout.addWidget(footer)
    
    def create_header(self):
        """Create header with title and description."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 10)
        
        # Title
        title = QLabel("Equilibrium Propagation: Beyond Backpropagation")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel(
            "Interactive demonstrations of biological plausibility, "
            "hardware efficiency, and deep architecture stability"
        )
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888;")
        layout.addWidget(subtitle)
        
        return widget
    
    def create_footer(self):
        """Create footer with global controls."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Info label
        info = QLabel("Use tabs above to explore different capabilities")
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info)
        
        layout.addStretch()
        
        # Reset all button
        reset_btn = QPushButton("↺ Reset All Demos")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        reset_btn.clicked.connect(self.reset_all)
        layout.addWidget(reset_btn)
        
        return widget
    
    def reset_all(self):
        """Reset all panels to initial state."""
        self.core_panel.reset()
        self.bio_panel.reset()
        self.neuro_panel.reset()
        self.depth_panel.reset()
