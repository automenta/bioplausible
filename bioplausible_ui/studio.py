"""
Bioplausible Studio - Unified Application

Main entry point integrating Experiment Runner, Validation Lab, Leaderboard, and Radar View.
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QLabel
from PyQt6.QtCore import Qt

from bioplausible_ui.studio_sidebar import StudioSidebar
from bioplausible_ui.core.themes import Theme

# Import sub-applications
# Note: We import the widgets/contents, not the MainWindows if possible, 
# or adapt them. Existing windows usually inherit QMainWindow. 
# We'll treat them as central widgets or wrap them.

from bioplausible_ui.app.window import AppMainWindow
from bioplausible_ui.lab.window import LabMainWindow
from bioplausible_ui.leaderboard_window import LeaderboardWindow
from bioplausible_ui.core.widgets.radar_view import RadarView
from bioplausible_ui.leaderboard_data import load_trials

class BioplausibleStudio(QMainWindow):
    """Unified application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bioplausible Studio")
        self.resize(1600, 1000)
        
        # Apply global theme
        self.setStyleSheet(Theme.get_stylesheet() + """
            QMainWindow { background-color: #0f172a; }
        """)
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = StudioSidebar()
        self.sidebar.mode_changed.connect(self.switch_mode)
        main_layout.addWidget(self.sidebar)
        
        # Content Area (Stacked Widget)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)
        
        # Initialize modes
        self.init_modes()
        
    def init_modes(self):
        """Initialize and add all sub-application modes."""
        
        # 1. Experiment Runner (App)
        # Assuming AppMainWindow is a QMainWindow. QMainWindow inside QWidget wrapper works but is odd.
        # Ideally we refactor to extract central widgets, but for now we wrap.
        self.app_window = AppMainWindow()
        self.stack.addWidget(self.wrap_window(self.app_window))
        
        # 2. Validation Lab
        self.lab_window = LabMainWindow()
        self.stack.addWidget(self.wrap_window(self.lab_window))
        
        # 3. Leaderboard
        self.leaderboard_window = LeaderboardWindow()
        self.leaderboard_window.request_training.connect(self.on_request_training)
        self.stack.addWidget(self.wrap_window(self.leaderboard_window))
        
        # 4. Radar View
        self.radar_view = RadarView()
        self.radar_view.pointClicked.connect(self.on_radar_click)
        self.stack.addWidget(self.radar_view)
        
    def wrap_window(self, window):
        """Wrap a QMainWindow to be used as a widget."""
        # If the sub-app is QMainWindow, we usually take its central widget + toolbars + statusbar
        # But simpler hack: just use it as is, QMainWindow inherits QWidget.
        # But setWindowFlags to Widget to rely on parent layout
        window.setWindowFlags(Qt.WindowType.Widget)
        return window

    def switch_mode(self, mode):
        """Switch the displayed content stack."""
        if mode == "experiment":
            self.stack.setCurrentIndex(0)
        elif mode == "lab":
            self.stack.setCurrentIndex(1)
        elif mode == "leaderboard":
            self.stack.setCurrentIndex(2)
            # Auto-refresh leaderboard when switched to
            if hasattr(self.leaderboard_window, 'refresh_data'):
                self.leaderboard_window.refresh_data()
        elif mode == "radar":
            self.stack.setCurrentIndex(3)
            self.refresh_radar()

    def refresh_radar(self):
        """Load data into global Radar View."""
        # Use same DB as leaderboard (default for now)
        db_path = "examples/shallow_benchmark.db"
        try:
            trials = load_trials(db_path)
            self.radar_view.clear()
            for trial in trials:
                # convert to radar format
                result = {
                    'params': trial.get('config', {}),
                    'accuracy': trial.get('accuracy', 0.0),
                    'model': trial.get('model_name', 'Unknown')
                }
                self.radar_view.add_result(result)
        except Exception as e:
            print(f"Failed to refresh radar: {e}")

    def on_request_training(self, config):
        """Handle request to train a specific config."""
        # Switch sidebar to experiment
        # We need to manually update sidebar state if possible
        for btn in self.sidebar.btn_group.buttons():
            if btn.property("mode") == "experiment":
                btn.setChecked(True)
                break
        
        self.switch_mode("experiment")
        
        # Access train tab
        if hasattr(self.app_window, 'train_tab'):
            self.app_window.train_tab.set_config(config)
            # Switch App's internal tab to Train (Index 1)
            self.app_window.tabs.setCurrentIndex(1)

    def on_radar_click(self, result):
        """Handle click on radar point."""
        # Convert radar result to training config
        from PyQt6.QtWidgets import QMessageBox
        
        params = result.get('params', {})
        acc = result.get('accuracy', 0.0)
        model = result.get('model', 'Unknown')
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Trial Details")
        msg.setText(f"Model: {model}\nAccuracy: {acc:.4f}\n\nConfiguration:\n{params}")
        train_btn = msg.addButton("Train This Config", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Close)
        
        msg.exec()
        
        if msg.clickedButton() == train_btn:
            config = {
                'model': model,
                'hyperparams': params,
                # task/dataset likely inside params or we default
                'task': params.get('task', 'vision'), 
                'dataset': params.get('dataset', 'mnist')
            }
            self.on_request_training(config)


def main():
    app = QApplication(sys.argv)
    window = BioplausibleStudio()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
