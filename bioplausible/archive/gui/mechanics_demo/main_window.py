"""
Main Window for Mechanics Demo
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .equilibrium_demo import EquilibriumDemo
from .healing_demo import HealingDemo
from .cube_demo import CubeDemo
from .training_demo import TrainingDemo
from .live_demo import LiveMechanicsDemo


class MechanicsDemoWindow(QMainWindow):
    """
    Main window for EqProp mechanics demonstrations.
    
    Features 5 real demos with MNIST:
    0. Live Mechanics - Hands-on training with parameter controls (MAIN DEMO)
    1. Equilibrium Convergence - Hidden state animation
    2. Self-Healing - Noise damping comparison
    3. 3D Neural Cube - Training + activation flow
    4. EqProp vs Backprop - Side-by-side training comparison
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EqProp Mechanics - Real MNIST Demonstrations")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.init_ui()
    
    def init_ui(self):
        """Build the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel(
            "<h1>EqProp Mechanics: Real-World Demonstrations</h1>"
            "<p style='color: #888;'>All demos use actual MNIST training and computation. "
            "No simulated curves - only real evidence.</p>"
        )
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create demos
        self.live_demo = LiveMechanicsDemo()
        self.equilibrium_demo = EquilibriumDemo()
        self.healing_demo = HealingDemo()
        self.cube_demo = CubeDemo()
        self.training_demo = TrainingDemo()
        
        # Add tabs - Live demo FIRST as main experience
        self.tabs.addTab(self.live_demo, "🔬 Live Training (Main)")
        self.tabs.addTab(self.equilibrium_demo, "1️⃣ Equilibrium Convergence")
        self.tabs.addTab(self.healing_demo, "2️⃣ Self-Healing")
        self.tabs.addTab(self.cube_demo, "3️⃣ 3D Neural Cube")
        self.tabs.addTab(self.training_demo, "4️⃣ EqProp vs Backprop")
        
        layout.addWidget(self.tabs)
