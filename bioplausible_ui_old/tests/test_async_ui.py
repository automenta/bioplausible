
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

class TestAsyncFeatures(unittest.TestCase):

    def test_lm_async_gen(self):
        from bioplausible_ui_old.tabs.lm_tab import GenerationWorker

        # Mock generator
        class MockGen:
            def generate(self, prompt, max_new_tokens, temperature):
                return "Mock generated text"

        worker = GenerationWorker(MockGen(), "test", 1.0)
        self.assertIsNotNone(worker)

        # Test signals
        received = []
        worker.finished.connect(lambda t: received.append(t))
        worker.run() # Run synchronously for test
        self.assertEqual(received[0], "Mock generated text")

    def test_rl_async_playback(self):
        from bioplausible_ui_old.tabs.rl_tab import PlaybackWorker

        # Mock gym environment is harder, but we can test init
        class MockModel(torch.nn.Module):
            def forward(self, x): return torch.tensor([[0.5, 0.5]])

        worker = PlaybackWorker(MockModel(), "CartPole-v1")
        self.assertIsNotNone(worker)

if __name__ == '__main__':
    unittest.main()
