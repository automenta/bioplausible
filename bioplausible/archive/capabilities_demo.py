#!/usr/bin/env python3
"""
EqProp Capabilities Demo - Visual Explainer

Interactive demonstrations of:
1. Core EqProp principles (energy minimization, free/nudged phases)
2. Biological plausibility (self-healing, Hebbian learning, local updates)
3. Neuromorphic/Hardware efficiency (O(1) memory, noise resilience, INT8)
4. Deep architecture stability (500-layer survival, 3D topology)
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtCore import Qt

from gui.capabilities import CapabilitiesDemoWindow


def main():
    """Launch the capabilities demo."""
    app = QApplication(sys.argv)
    
    # Dark theme (same as bio_trainer_gui)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    # Global font
    app.setFont(QFont("Segoe UI", 9))
    
    # Create and show window
    window = CapabilitiesDemoWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
