"""Domain exception hierarchy for bioplausible.

All library errors derive from :class:`BioplausibleError`. Subdomains
define their own leaf types so callers can catch precisely (e.g. the
AutoScientist catches :class:`TrialExecutionError` instead of bare
``Exception``). Every internal ``raise`` should chain from the original
exception: ``raise DomainError("msg") from original_exception``.
"""

__all__ = [
    "BioplausibleError",
    "CheckpointError",
    "ConditionalQueryError",
    "ConfigError",
    "IncompatibilityError",
    "KnowledgeBaseError",
    "LoadStateError",
    "NumericalInstabilityError",
    "PropagatorError",
    "RegistryError",
    "SpaceSignatureMismatchError",
    "TileGraphError",
    "TrialExecutionError",
]


class BioplausibleError(Exception):
    """Base class for all bioplausible domain errors."""


class ConfigError(BioplausibleError):
    """Invalid or unsupported configuration."""


class SpaceSignatureMismatchError(ConfigError):
    """A rule's ``RULE_SPACES`` entry advertises knobs its model constructor drops.

    Raised by the P0a integrity gate when an advertised search-space dimension is
    neither accepted by the model constructor nor absorbed via ``**kwargs`` — the
    ``build_model_kwargs`` silent-drop drift that wastes probe budget.

    Attributes:
        rule: The offending rule key.
        phantoms: The advertised keys with no consumer on the model.
    """

    def __init__(self, rule: str, phantoms: frozenset[str]) -> None:
        self.rule = rule
        self.phantoms = phantoms
        reason = (
            "none"
            if not phantoms
            else "".join(
                f"\n  - {p!r} (dropped by build_model_kwargs)" for p in sorted(phantoms)
            )
        )
        super().__init__(f"RULE_SPACES[{rule!r}] advertises phantom knobs:{reason}")


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


class ConditionalQueryError(KnowledgeBaseError):
    """A conditional query against the KnowledgeBase is malformed or unanswerable."""


class TrialExecutionError(BioplausibleError):
    """An experiment trial failed to execute."""


class PropagatorError(BioplausibleError):
    """A learning-rule propagator failed during forward/backward."""


class NumericalInstabilityError(BioplausibleError):
    """A training step produced a non-finite loss or parameter value.

    Raised by the trainer's run-wide numerical-health guard so a diverging probe
    (e.g. an eqprop model with too high a learning rate) aborts *fast* instead
    of wasting the rest of its epoch budget, and is recorded as a definite
    divergence — never silently counted as a healthy run.
    """


class TileGraphError(BioplausibleError):
    """Invalid or inconsistent tile-graph topology."""
