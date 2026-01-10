#!/usr/bin/env python3
"""
EqProp Mechanics Demo - Real MNIST Demonstrations

Interactive demonstrations using actual computation:
1. Equilibrium Convergence - Hidden state heatmap animation
2. Self-Healing - Noise damping comparison
3. Deep Signal Survival - 500-layer waterfall
4. 3D Neural Cube - Rotating activation visualization  
5. Live Training - EqProp vs Backprop side-by-side
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

from gui.mechanics_demo import MechanicsDemoWindow


def main():
    """Launch the mechanics demo."""
    app = QApplication(sys.argv)
    
    # Dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    # Create and show window
    window = MechanicsDemoWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
