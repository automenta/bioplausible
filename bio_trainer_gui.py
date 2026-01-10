#!/usr/bin/env python3
"""
Bio-Trainer GUI Entry Point
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtCore import Qt

# Import from package
from gui.main_window import BioTrainerGUI

def main():
    app = QApplication(sys.argv)
    
    # Dark theme
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(40, 40, 40))
    p.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    p.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    p.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(p)
    
    # Global Font
    app.setFont(QFont("Segoe UI", 9))
    
    gui = BioTrainerGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
