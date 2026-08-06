"""The thin experiment layer (architecture §6).

A YAML-driven survivor-cascade verdict layer over the existing
``cli/run.py`` + ``hyperopt`` + ``cli/parity.py`` surface. The only genuinely
new code in the experiment system lives here.
"""

from bioplausible.experiment.param_estimator import (
    InstantiateEstimator,
    ParamEstimateError,
    estimate_param_count,
)
from bioplausible.experiment.probe import ProbeDriver, ProbeResult, run_probe
from bioplausible.experiment.producer import ConfigProducer, HyperoptGridProducer
from bioplausible.experiment.report import Report
from bioplausible.experiment.schema import Campaign, Stage, load_campaign
from bioplausible.experiment.staircase import Outcome, StaircaseRunner, Verdict

__all__ = [
    "Campaign",
    "ConfigProducer",
    "HyperoptGridProducer",
    "InstantiateEstimator",
    "Outcome",
    "ParamEstimateError",
    "ProbeDriver",
    "ProbeResult",
    "Report",
    "Stage",
    "StaircaseRunner",
    "Verdict",
    "estimate_param_count",
    "load_campaign",
    "run_probe",
]
