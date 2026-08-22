"""Legacy CoreTrainer tests - skipped as CoreTrainer/TrainerConfig removed in Sprint 7."""

import pytest

pytestmark = pytest.mark.skip(
    reason="CoreTrainer/TrainerConfig removed in Sprint 7; uses new SystemTrainer/ExperimentConfig"
)
        b = torch.randn(5)

        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
