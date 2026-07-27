import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# torch is a hard dependency (pyproject.toml) — no mock needed

try:
    import torchvision
except ImportError:
    import sys
    import types

    torchvision = types.ModuleType("torchvision")
    sys.modules["torchvision"] = torchvision
    torchvision.transforms = types.ModuleType("torchvision.transforms")
    torchvision.datasets = types.ModuleType("torchvision.datasets")
    torchvision.utils = types.ModuleType("torchvision.utils")
    sys.modules["torchvision.transforms"] = torchvision.transforms
    sys.modules["torchvision.datasets"] = torchvision.datasets
    sys.modules["torchvision.utils"] = torchvision.utils

try:
    import gymnasium
except ImportError:
    import sys
    import types

    gymnasium = types.ModuleType("gymnasium")
    sys.modules["gymnasium"] = gymnasium
    gymnasium.spaces = types.ModuleType("gymnasium.spaces")
    sys.modules["gymnasium.spaces"] = gymnasium.spaces

# bioplausible.acceleration checks for cupy
import sys
from unittest.mock import MagicMock

sys.modules["cupy"] = MagicMock()
