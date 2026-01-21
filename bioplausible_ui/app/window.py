from PyQt6.QtWidgets import QMainWindow, QTabWidget
from bioplausible_ui.app.tabs.train_tab import TrainTab
from bioplausible_ui.app.tabs.benchmarks_tab import BenchmarksTab

class AppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bioplausible Trainer (biopl)")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(TrainTab(), "Train")
        self.tabs.addTab(BenchmarksTab(), "Benchmarks")
