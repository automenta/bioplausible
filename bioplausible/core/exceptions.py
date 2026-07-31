"""Domain exception hierarchy for bioplausible.

All library errors derive from :class:`BioplausibleError`. Subdomains
define their own leaf types so callers can catch precisely (e.g. the
AutoScientist catches :class:`TrialExecutionError` instead of bare
``Exception``). Every internal ``raise`` should chain from the original
exception: ``raise DomainError("msg") from original_exception``.
"""

__all__ = [
    "BioplausibleError",
    "ConfigError",
    "RegistryError",
    "IncompatibilityError",
    "CheckpointError",
    "LoadStateError",
    "KnowledgeBaseError",
    "TrialExecutionError",
    "PropagatorError",
    "TileGraphError",
]


class BioplausibleError(Exception):
    """Base class for all bioplausible domain errors."""


class ConfigError(BioplausibleError):
    """Invalid or unsupported configuration."""


class RegistryError(BioplausibleError):
    """Component registration or lookup failure."""


class IncompatibilityError(RegistryError):
    """A component cannot be composed with the requested configuration."""


class CheckpointError(BioplausibleError):
    """Model or training state could not be saved or restored."""


class LoadStateError(CheckpointError):
    """A checkpoint's state dict could not be loaded into a model."""


class KnowledgeBaseError(BioplausibleError):
    """Knowledge base storage or analysis failure."""


class TrialExecutionError(BioplausibleError):
    """An experiment trial failed to execute."""


class PropagatorError(BioplausibleError):
    """A learning-rule propagator failed during forward/backward."""


class TileGraphError(BioplausibleError):
    """Invalid or inconsistent tile-graph topology."""
