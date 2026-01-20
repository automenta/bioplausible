
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class TestTabs(unittest.TestCase):

    def test_microscope_tab_init(self):
        from bioplausible_ui.tabs.microscope_tab import MicroscopeTab
        tab = MicroscopeTab()
        self.assertIsNotNone(tab)

        # Check if heatmap widget exists
        if hasattr(tab, 'micro_heat_view'):
            self.assertIsNotNone(tab.micro_heat_view)

    def test_discovery_tab_init(self):
        from bioplausible_ui.tabs.discovery_tab import DiscoveryTab
        tab = DiscoveryTab()
        self.assertIsNotNone(tab)

        # Check signal exists
        self.assertTrue(hasattr(tab, 'load_model_signal'))

        # Cleanup thread
        if hasattr(tab, 'thread'):
            tab.thread.quit()
            tab.thread.wait()

        # Cleanup timers
        if hasattr(tab, 'viz_timer'):
            tab.viz_timer.stop()
        if hasattr(tab, 'net_timer'):
            tab.net_timer.stop()

    def test_vision_tab_inference_dialog(self):
        from bioplausible_ui.tabs.vision_tab import VisionInferenceDialog
        import torch

        # Mock data
        img = torch.zeros(1, 28, 28)
        pred = 5
        gt = 5

        dlg = VisionInferenceDialog(img, pred, gt)
        self.assertIsNotNone(dlg)

    def test_benchmarks_tab_init(self):
        from bioplausible_ui.tabs.benchmarks_tab import BenchmarksTab
        tab = BenchmarksTab()
        self.assertIsNotNone(tab)
        self.assertTrue(hasattr(tab, 'load_model_signal'))

    def test_benchmarks_comparison_dialog(self):
        from bioplausible_ui.tabs.benchmarks_tab import ComparisonDialog
        data = [{'name': 'A', 'score': 10}, {'name': 'B', 'score': 20}]
        dlg = ComparisonDialog(data)
        self.assertIsNotNone(dlg)

    def test_lm_tab_generation(self):
        from bioplausible_ui.tabs.lm_tab import LMTab
        tab = LMTab()
        self.assertIsNotNone(tab)

        # Check if generation UI exists
        self.assertTrue(hasattr(tab, 'gen_output'))
        self.assertTrue(hasattr(tab, 'gen_prompt_input'))

    def test_rl_tab_playback(self):
        from bioplausible_ui.tabs.rl_tab import RLTab
        tab = RLTab()
        self.assertIsNotNone(tab)

        # Check if playback method exists
        self.assertTrue(hasattr(tab, '_watch_agent'))

    def test_console_tab_commands(self):
        from bioplausible_ui.tabs.console_tab import ConsoleTab
        tab = ConsoleTab()
        self.assertIsNotNone(tab)

        # Test basic command
        tab.cmd_input.setText("!help")
        tab.cmd_input.returnPressed.emit()

        # Check if text was cleared (command processed)
        self.assertEqual(tab.cmd_input.text(), "")

    def test_deploy_tab_init(self):
        from bioplausible_ui.tabs.deploy_tab import DeployTab
        tab = DeployTab()
        self.assertIsNotNone(tab)
        # Check model ref setter
        self.assertTrue(hasattr(tab, 'update_model_ref'))

if __name__ == '__main__':
    unittest.main()
