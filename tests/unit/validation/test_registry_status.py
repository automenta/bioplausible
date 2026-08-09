"""Registry status metadata tests (Plan 8 Track D1).

Every registered model must carry a ``status:<x>`` tag so sweeps can exclude
known-broken probes by default and the parity runner can flag experimental
results appropriately.
"""

import pytest

import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)
from bioplausible.core.model_status import STATUS_TAG_PREFIX, ModelStatus, status_tag
from bioplausible.core.registry import ComponentCategory, Registry

VALID_STATUSES = frozenset(s.value for s in ModelStatus)


def _all_model_tags() -> dict[str, list[str]]:
    """Map every registered model name → its metadata tags."""
    return {
        rec["name"]: rec["metadata"].tags
        for rec in Registry.query(category=ComponentCategory.MODEL)
    }


def test_status_tag_renders_prefix() -> None:
    assert status_tag(ModelStatus.STABLE) == "status:stable"
    assert status_tag("broken") == "status:broken"


def test_status_tag_prefix_constant() -> None:
    assert STATUS_TAG_PREFIX == "status:"


def test_every_registered_model_has_status_tag() -> None:
    """Every model must carry exactly one valid ``status:<x>`` tag."""
    for name, tags in _all_model_tags().items():
        statuses = [t for t in tags if t.startswith(STATUS_TAG_PREFIX)]
        assert len(statuses) == 1, (
            f"{name}: expected exactly one status tag, got {statuses} (tags={tags})"
        )
        value = statuses[0].split(":", 1)[1]
        assert value in VALID_STATUSES, (
            f"{name}: invalid status {value!r}; valid={sorted(VALID_STATUSES)}"
        )


def test_broken_models_are_quarantined_models() -> None:
    """The status:broken population matches the phantom-knob audit, roughly.

    These are the models quarantined in ``docs/phantom_knob_audit.md``. If the
    audit changes, update that doc and this test together.
    """
    broken = sorted(
        n for n, tags in _all_model_tags().items() if "status:broken" in tags
    )
    # At minimum the phantom-num_layers families are quarantined.
    for expected in (
        "graph_eqprop",
        "conv_eqprop",
        "modern_conv_eqprop",
        "equilibrium_alignment",
        "hebbian_chain",
        "hebbian_3d",
    ):
        assert expected in broken, f"{expected} must be tagged status:broken"
    
    # These were previously broken but are now fixed (depth-cap fix)
    for fixed in ("direct_feedback_alignment_eqprop", "dfa_deep"):
        assert fixed not in broken, f"{fixed} should no longer be status:broken"


def test_sweep_filters_broken_by_default() -> None:
    """Default sweep family queries exclude status:broken models."""
    import scripts.broad_sweep as sweep

    for family in ("eqprop", "fa", "hebbian"):
        models = sweep._models_in_family(family)
        broken_present = [m for m in models if sweep._model_status(m) == "broken"]
        assert not broken_present, (
            f"default sweep for {family} must not include broken models: "
            f"{broken_present}"
        )


def test_sweep_include_broken_restores_them() -> None:
    """--include-broken restores status:broken models to the sweep."""
    import scripts.broad_sweep as sweep

    include = sweep._models_in_family("eqprop", include_broken=True)
    assert "conv_eqprop" in include


@pytest.mark.parametrize(
    "model, expected",
    [
        ("backprop_mlp", ModelStatus.STABLE),
        ("eqprop_mlp", ModelStatus.STABLE),
        ("directed_ep", ModelStatus.EXPERIMENTAL),
        ("conv_eqprop", ModelStatus.BROKEN),
        ("direct_feedback_alignment_eqprop", ModelStatus.EXPERIMENTAL),
        ("dfa_deep", ModelStatus.EXPERIMENTAL),
    ],
)
def test_key_models_have_expected_status(model: str, expected: ModelStatus) -> None:
    """Spot-check the statuses the parity/documents rely on."""
    tags = _all_model_tags()[model]
    statuses = [t for t in tags if t.startswith(STATUS_TAG_PREFIX)]
    assert statuses == [status_tag(expected)], (
        f"{model}: expected {expected}, got {statuses}"
    )
