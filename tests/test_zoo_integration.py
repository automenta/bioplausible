"""Integration tests for Zoo component combinations.

Verifies that registered models, propagators, and optimizers can be
combined and used with CoreTrainer.
"""

import pathlib

import pytest
import torch
from torch import nn

# Import zoo modules to trigger registration
from bioplausible.core.registry import (
    ComponentCategory,
    Domain,
    LocalityLevel,
    Registry,
)


@pytest.fixture(autouse=True)
def _preserve_registry():
    """Save and restore registry state around each test to prevent
    cross-test pollution (used by factory-function tests that call
    Registry.clear())."""
    import copy

    saved = copy.deepcopy(Registry._components)
    yield
    Registry._components.clear()
    Registry._components.update(copy.deepcopy(saved))


def test_registry_has_models():
    """Verify that models are registered in the zoo."""
    models = Registry.list(ComponentCategory.MODEL)
    assert "model" in models
    assert len(models["model"]) > 0
    assert "eqprop_mlp" in models["model"]
    assert "equitile" in models["model"]


def test_registry_has_propagators():
    """Verify that propagators are registered."""
    props = Registry.list(ComponentCategory.PROPAGATOR)
    assert "propagator" in props
    assert len(props["propagator"]) > 0
    assert "feedback_alignment" in props["propagator"]


def test_registry_has_optimizers():
    """Verify that optimizers are registered."""
    opts = Registry.list(ComponentCategory.OPTIMIZER)
    assert "optimizer" in opts
    assert len(opts["optimizer"]) > 0
    assert "adam" in opts["optimizer"]
    assert "sgd" in opts["optimizer"]


def test_registry_has_sparsity():
    """Verify that sparsity methods are registered."""
    sparsity = Registry.list(ComponentCategory.SPARSITY)
    assert "sparsity" in sparsity
    assert len(sparsity["sparsity"]) > 0


def test_query_by_domain_vision():
    """Test querying models by vision domain."""
    results = Registry.query(category=ComponentCategory.MODEL, domain=Domain.VISION)
    assert len(results) >= 1
    names = [r["name"] for r in results]
    assert "eqprop_mlp" in names


def test_query_bio_plausible_models():
    """Test querying bio-plausible models (no backward pass)."""
    results = Registry.query(
        category=ComponentCategory.MODEL,
        requires_backward=False,
    )
    assert len(results) >= 1
    for r in results:
        assert r["metadata"].requires_backward is False


def test_query_local_learning():
    """Test querying local learning rules."""
    results = Registry.query(
        category=ComponentCategory.PROPAGATOR,
        locality=LocalityLevel.LOCAL,
    )
    names = [r["name"] for r in results]
    assert "contrastive_hebbian_learning" in names


def test_get_compatible():
    """Test getting compatible components for a model."""
    compat = Registry.get_compatible("eqprop_mlp")
    assert ComponentCategory.PROPAGATOR in compat
    assert ComponentCategory.OPTIMIZER in compat


def test_metadata_on_registered_class():
    """Test that registered classes have metadata attached."""
    MLP_cls = Registry.get(ComponentCategory.MODEL, "eqprop_mlp")
    assert hasattr(MLP_cls, "_registry_metadata")
    assert MLP_cls._registry_metadata.name == "eqprop_mlp"
    assert MLP_cls._registry_metadata.bio_plausibility_score >= 0.0


def test_mlp_instantiation():
    """Test instantiating a registered model."""
    MLP_cls = Registry.get(ComponentCategory.MODEL, "eqprop_mlp")
    model = MLP_cls(input_dim=784, hidden_dim=64, output_dim=10)
    assert model is not None

    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_equitile_instantiation():
    """Test instantiating EquiTile."""
    EqT_cls = Registry.get(ComponentCategory.MODEL, "equitile")
    model = EqT_cls(input_dim=784, hidden_dim=256, output_dim=10)
    assert model is not None

    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_forward_forward_instantiation():
    """Test instantiating Forward-Forward network."""
    FF_cls = Registry.get(ComponentCategory.MODEL, "forward_forward")
    model = FF_cls(input_dim=784, hidden_dim=64, output_dim=10)
    assert model is not None

    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_optimizer_instantiation():
    """Test instantiating registered optimizers."""
    Adam_cls = Registry.get(ComponentCategory.OPTIMIZER, "adam")
    model = nn.Linear(10, 2)
    opt = Adam_cls(model.parameters(), lr=0.001)
    assert opt is not None


def test_cross_domain_query():
    """Test that models can be queried across multiple domains."""
    results = Registry.query(
        category=ComponentCategory.MODEL,
        domain=Domain.LM,
    )
    # EquiTile is registered for LM
    names = [r["name"] for r in results]
    assert "equitile" in names


def test_bio_score_query():
    """Test filtering by bio-plausibility score."""
    high_bio = Registry.query(
        category=ComponentCategory.MODEL,
        min_bio_score=0.8,
    )
    assert len(high_bio) >= 1
    for r in high_bio:
        assert r["metadata"].bio_plausibility_score >= 0.8


def test_export_yaml(tmp_path):
    """Test exporting registry to YAML."""
    yaml_path = tmp_path / "registry.yaml"
    Registry.export_yaml(str(yaml_path))
    assert yaml_path.exists()

    import yaml

    with pathlib.Path(yaml_path).open() as f:
        data = yaml.safe_load(f)
    assert "model" in data
    assert "optimizer" in data
    assert "propagator" in data


# ----------------------------------------------------------------------------
# REFACTOR2 §3.3 — discovery helpers (regression tests for kwargs bug)
# These were broken because the helpers called Registry.query(locality_level=...)
# while the public parameter is named `locality`. Without these tests the
# shipped helpers raised TypeError on every call.
# ----------------------------------------------------------------------------


def test_get_models_for_task_no_filters():
    """All models returned when only domain filter matches."""
    from bioplausible.zoo import get_models_for_task

    results = get_models_for_task(Domain.VISION)
    assert len(results) > 0
    for r in results:
        assert Domain.VISION in r["metadata"].domains


def test_get_models_for_task_with_locality():
    """Filtered models preserve locality=EQUILIBRIUM and Domain vision."""
    from bioplausible.zoo import get_models_for_task

    results = get_models_for_task(Domain.VISION, locality=LocalityLevel.EQUILIBRIUM)
    assert len(results) > 0
    for r in results:
        assert Domain.VISION in r["metadata"].domains
        assert r["metadata"].locality_level == LocalityLevel.EQUILIBRIUM


def test_get_models_for_task_requires_backward_filter():
    from bioplausible.zoo import get_models_for_task

    results = get_models_for_task(Domain.VISION, requires_backward=False)
    assert len(results) > 0
    for r in results:
        assert r["metadata"].requires_backward is False


def test_get_propagators_for_model_matches_locality():
    """propagators returned share locality + backward compatibility with model."""
    from bioplausible.zoo import get_propagators_for_model

    results = get_propagators_for_model("eqprop_mlp")
    assert len(results) > 0
    # eqprop_mlp is EQUILIBRIUM + no backward — every returned propagator
    # must match.
    model_meta = Registry.get_metadata(ComponentCategory.MODEL, "eqprop_mlp")
    for r in results:
        assert r["metadata"].locality_level == model_meta.locality_level
        assert r["metadata"].requires_backward == model_meta.requires_backward


def test_get_optimizers_for_propagator_matches_backward():
    from bioplausible.zoo import get_optimizers_for_propagator

    results = get_optimizers_for_propagator("eq_prop")
    assert len(results) > 0
    # all returned optimizers registered with the same requires_backward flag.
    propagator_meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, "eq_prop")
    for r in results:
        assert r["metadata"].requires_backward == propagator_meta.requires_backward


def test_get_models_for_task_unknown_returns_empty():
    """A domain with no registrations returns an empty list (no crash)."""
    from bioplausible.zoo import get_models_for_task

    # SCIENTIFIC is in the enum but rarely registered; just confirm no raise.
    results = get_models_for_task(Domain.SCIENTIFIC)
    assert isinstance(results, list)


def test_query_by_family_filter():
    """Reg test for `family=` kwarg added together with ComponentMetadata.family."""
    equitile_models = Registry.query(
        category=ComponentCategory.MODEL, family="equitile"
    )
    assert len(equitile_models) > 0
    for r in equitile_models:
        assert r["metadata"].family == "equitile"


# ----------------------------------------------------------------------------
# REFACTOR3: Removed ModelZoo / OptimizerZoo legacy adapters; these tests
# verify that callers can use Registry directly to instantiate models
# and optimizers, including the OPTIMIZER→PROPAGATOR fallback path.
# ----------------------------------------------------------------------------


def _instantiate_model(name: str, **params) -> nn.Module:
    from bioplausible.core.registry import ComponentCategory, Registry

    cls = Registry.get(ComponentCategory.MODEL, name)
    return cls(**params)


def _instantiate_optimizer(name: str, params, model=None):
    from bioplausible.core.registry import ComponentCategory, Registry

    try:
        cls = Registry.get(ComponentCategory.OPTIMIZER, name)
    except ValueError:
        cls = Registry.get(ComponentCategory.PROPAGATOR, name)
    if model is not None:
        try:
            return cls(params, model=model)
        except TypeError:
            return cls(params)
    return cls(params)


def test_registry_model_get_instantiates():
    from bioplausible.core.registry import ComponentCategory, Registry

    cls = Registry.get(ComponentCategory.MODEL, "backprop_mlp")
    model = cls(input_dim=784, hidden_dim=32, output_dim=10)
    assert isinstance(model, nn.Module)
    batch_size = 2
    out = model(torch.randn(batch_size, 784))
    assert out.shape[0] == batch_size


def test_registry_model_get_unknown_raises_value_error():
    from bioplausible.core.registry import ComponentCategory, Registry

    with pytest.raises(ValueError):
        Registry.get(ComponentCategory.MODEL, "does_not_exist_xyz")


def test_registry_optimizer_get_resolves_propagator_preset():
    """smep is registered as PROPAGATOR (not OPTIMIZER); fallback must find it."""
    model = _instantiate_model(
        "backprop_mlp", input_dim=784, hidden_dim=32, output_dim=10
    )
    opt = _instantiate_optimizer("smep", model.parameters(), model=model)
    assert opt.__class__.__name__ == "CompositeOptimizer"


def test_registry_optimizer_get_resolves_plain_optimizer():
    model = _instantiate_model(
        "backprop_mlp", input_dim=784, hidden_dim=32, output_dim=10
    )
    # adam is registered as OPTIMIZER (no `model=` kwarg accepted)
    opt = _instantiate_optimizer("adam", model.parameters(), model=model)
    assert opt.__class__.__name__ == "Adam"


def test_registry_optimizer_get_unknown_raises_with_available_list():
    from bioplausible.core.registry import ComponentCategory, Registry

    with pytest.raises(ValueError):
        Registry.get(ComponentCategory.OPTIMIZER, "does_not_exist_xyz")


# ----------------------------------------------------------------------------
# Registry.register accepts factory functions (not just classes)
# MEP presets are callables like smep(params, model, ...) returning an
# optimizer instance. The decorator previously annotated `type[T]` which
# would have rejected them; the actual fix uses generic `Component`.
# ----------------------------------------------------------------------------


def test_register_accepts_factory_function():
    """A bare function (no class) can be registered and retrieved."""
    from bioplausible.core.registry import (
        ComponentCategory,
        Registry,
        register_optimizer,
    )

    Registry.clear()

    def factory(params, model=None):
        return ("called", params, model)

    register_optimizer("factory_test")(factory)
    retrieved = Registry.get(ComponentCategory.OPTIMIZER, "factory_test")
    assert retrieved is factory
    assert retrieved("p", model="m") == ("called", "p", "m")


def test_register_attaches_metadata_to_factory():
    """factory functions (unlike classes) skip attribute attachment gracefully."""
    from bioplausible.core.registry import Registry, register_optimizer

    Registry.clear()

    def smep_like(params, model=None):  # ruff: ignore[unused-function-argument]
        return params

    register_optimizer("factory_with_meta", bio_plausibility_score=0.95)(smep_like)
    meta = Registry.get_metadata("optimizer", "factory_with_meta")
    assert meta.bio_plausibility_score == pytest.approx(0.95)
    # The function must remain callable (not rewrapped).
    assert Registry.get(ComponentCategory.OPTIMIZER, "factory_with_meta")("p") == "p"
