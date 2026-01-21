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

        self.train_tab = TrainTab()
        self.search_tab = SearchTab()
        self.results_tab = ResultsTab()
        self.deploy_tab = DeployTab()

        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(CompareTab(), "Compare")
        self.tabs.addTab(self.search_tab, "Search")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(BenchmarksTab(), "Benchmarks")
        self.tabs.addTab(self.deploy_tab, "Deploy")
        self.tabs.addTab(ConsoleTab(), "Console")
        self.tabs.addTab(SettingsTab(), "Settings")

        # Connect Search -> Train
        self.search_tab.transfer_config.connect(self._on_transfer_config)

    def _on_transfer_config(self, config):
        self.train_tab.set_config(config)
        self.tabs.setCurrentWidget(self.train_tab)
