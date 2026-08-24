"""
Bioplausible PyTorch Lightning Integration

Provides LightningModule, HPO, callbacks, and strategies
for biologically plausible learning algorithms.
"""

from computronium.lightning_.callbacks import (
    BioPrecisionCallback,
    BioPredictionWriter,
    EnergyConvergenceCallback,
)
from computronium.lightning_.experiment import run_pl_trial, run_pl_trial_with_wandb
from computronium.lightning_.hpo import BioOptunaPruner, BioRayTuneSearch
from computronium.lightning_.module import BioLightningModule
from computronium.lightning_.nas import run_nas_search
from computronium.lightning_.strategies import BioPrecisionMixin, build_trainer

__all__ = [
    "BioLightningModule",
    "BioOptunaPruner",
    "BioPrecisionCallback",
    "BioPrecisionMixin",
    "BioPredictionWriter",
    "BioRayTuneSearch",
    "EnergyConvergenceCallback",
    "build_trainer",
    "run_nas_search",
    "run_pl_trial",
    "run_pl_trial_with_wandb",
]
