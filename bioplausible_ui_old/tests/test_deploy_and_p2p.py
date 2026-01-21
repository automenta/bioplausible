
import unittest
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
import torch

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class MockModel(torch.nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.linear = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

class TestNewFeatures(unittest.TestCase):

    def test_deploy_tab_new_features(self):
        from bioplausible_ui_old.tabs.deploy_tab import DeployTab, ServerWorker
        tab = DeployTab()
        self.assertIsNotNone(tab)

        # Test input shape guessing
        tab.model = MockModel(input_dim=10)
        shape_info = tab._guess_input_shape()
        self.assertEqual(shape_info, (1, 10))

        # Test ServerWorker logic (just init)
        worker = ServerWorker(tab.model)
        self.assertIsNotNone(worker)
        # We won't start it as it blocks port 8000 and needs uvicorn

    def test_p2p_tab_new_features(self):
        from bioplausible_ui_old.tabs.p2p_tab import P2PTab
        tab = P2PTab()
        self.assertIsNotNone(tab)

        # Check for new signal
        self.assertTrue(hasattr(tab, 'load_model_signal'))

        # Check button exists and is initially disabled
        self.assertTrue(hasattr(tab, 'load_best_btn'))
        self.assertFalse(tab.load_best_btn.isEnabled())

        # Mock a worker update
        mock_worker = type('obj', (object,), {'running': True, 'global_best_config': {'accuracy': 0.95}})
        tab.worker = mock_worker

        # Trigger check
        tab._check_best_model()

        # Button should be enabled
        self.assertTrue(tab.load_best_btn.isEnabled())
        self.assertIn("95.0%", tab.load_best_btn.text())

        # Cleanup timer
        if hasattr(tab, 'update_timer'):
            tab.update_timer.stop()

    def test_dashboard_init(self):
        from bioplausible_ui_old.dashboard import EqPropDashboard
        dash = EqPropDashboard()
        self.assertIsNotNone(dash)
        # Check device label text
        self.assertTrue("Device:" in dash.device_label.text())

        # Cleanup dashboard components to prevent thread leaks
        if hasattr(dash, 'disc_tab'):
            if hasattr(dash.disc_tab, 'viz_timer'):
                dash.disc_tab.viz_timer.stop()
            if hasattr(dash.disc_tab, 'net_timer'):
                dash.disc_tab.net_timer.stop()
            if hasattr(dash.disc_tab, 'thread'):
                dash.disc_tab.thread.quit()
                dash.disc_tab.thread.wait()

        if hasattr(dash, 'p2p_tab'):
             if hasattr(dash.p2p_tab, 'update_timer'):
                dash.p2p_tab.update_timer.stop()

        if hasattr(dash, 'plot_timer'):
             dash.plot_timer.stop()

if __name__ == '__main__':
    unittest.main()
