import pytest
from bioplausible_ui.app.window import AppMainWindow
from bioplausible_ui.app.tabs.train_tab import TrainTab
from bioplausible_ui.app.tabs.benchmarks_tab import BenchmarksTab

def test_app_window(qtbot):
    window = AppMainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Bioplausible Trainer (biopl)"
    assert window.tabs.count() == 2
    assert isinstance(window.tabs.widget(0), TrainTab)
    assert isinstance(window.tabs.widget(1), BenchmarksTab)

def test_train_tab(qtbot):
    tab = TrainTab()
    qtbot.addWidget(tab)

    assert hasattr(tab, 'task_selector')
    assert hasattr(tab, 'dataset_picker')
    assert tab._actions['start'].isEnabled()
    assert not tab._actions['stop'].isEnabled()
