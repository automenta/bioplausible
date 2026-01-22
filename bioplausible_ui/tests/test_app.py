import pytest
from bioplausible_ui.app.window import AppMainWindow
from bioplausible_ui.app.tabs.train_tab import TrainTab
from bioplausible_ui.app.tabs.benchmarks_tab import BenchmarksTab

def test_app_window(qtbot):
    window = AppMainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Bioplausible Trainer (biopl)"
    # We added Home tab, so 9 tabs total.
    # Total: Home, Train, Compare, Search, Results, Benchmarks, Deploy, Console, Settings = 9
    assert window.tabs.count() == 9
    # Tab 0 is now HomeTab, Tab 1 is TrainTab
    from bioplausible_ui.app.tabs.home_tab import HomeTab
    assert isinstance(window.tabs.widget(0), HomeTab)
    assert isinstance(window.tabs.widget(1), TrainTab)
    # Check other tabs if necessary, but count is a good indicator

def test_train_tab(qtbot):
    tab = TrainTab()
    qtbot.addWidget(tab)

    assert hasattr(tab, 'task_selector')
    assert hasattr(tab, 'dataset_picker')
    assert tab._actions['start'].isEnabled()
    assert not tab._actions['stop'].isEnabled()
