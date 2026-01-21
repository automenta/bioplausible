from PyQt6.QtWidgets import QMainWindow, QTabWidget
from bioplausible_ui.app.tabs.train_tab import TrainTab
from bioplausible_ui.app.tabs.compare_tab import CompareTab
from bioplausible_ui.app.tabs.search_tab import SearchTab
from bioplausible_ui.app.tabs.results_tab import ResultsTab
from bioplausible_ui.app.tabs.benchmarks_tab import BenchmarksTab
from bioplausible_ui.app.tabs.deploy_tab import DeployTab
from bioplausible_ui.app.tabs.console_tab import ConsoleTab
from bioplausible_ui.app.tabs.settings_tab import SettingsTab

class AppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bioplausible Trainer (biopl)")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(TrainTab(), "Train")
        self.tabs.addTab(CompareTab(), "Compare")
        self.tabs.addTab(SearchTab(), "Search")
        self.tabs.addTab(ResultsTab(), "Results")
        self.tabs.addTab(BenchmarksTab(), "Benchmarks")
        self.tabs.addTab(DeployTab(), "Deploy")
        self.tabs.addTab(ConsoleTab(), "Console")
        self.tabs.addTab(SettingsTab(), "Settings")
