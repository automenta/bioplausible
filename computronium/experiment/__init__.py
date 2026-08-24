"""The thin experiment layer (architecture §6).

A YAML-driven survivor-cascade verdict layer over the existing
``cli/run.py`` + ``hyperopt`` + ``cli/parity.py`` surface. The only genuinely
new code in the experiment system lives here.
"""

from computronium.experiment.param_estimator import (
    InstantiateEstimator,
    ParamEstimateError,
    estimate_param_count,
)
from computronium.experiment.probe import ProbeDriver, ProbeResult, run_probe
from computronium.experiment.producer import ConfigProducer, HyperoptGridProducer
from computronium.experiment.report import Report
from computronium.experiment.schema import Campaign, Stage, load_campaign
from computronium.experiment.staircase import Outcome, StaircaseRunner, Verdict

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
