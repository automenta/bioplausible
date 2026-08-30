"""Tests for the Registry system."""

import copy

import pytest
from torch import nn

from computronium.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    LocalityLevel,
    Registry,
    register_model,
    register_optimizer,
)


@pytest.fixture(autouse=True)
def _preserve_registry():
    """Save and restore registry state around each test to prevent cross-test pollution."""
    saved_components = copy.deepcopy(Registry._components)
    yield
    Registry._components.clear()
    Registry._components.update(copy.deepcopy(saved_components))


def test_registry_clear():
    """Test clearing the registry."""
    Registry.clear()
    assert Registry.list() == {}


def test_register_and_get():
    """Test registering and getting a component."""
    Registry.clear()

    @register_model(name="TestModel", family="test", description="A test model")
    class TestModel:
        pass

    # Get by category and name
    cls = Registry.get(ComponentCategory.MODEL, "TestModel")
    assert cls == TestModel

    # Get metadata
    meta = Registry.get_metadata(ComponentCategory.MODEL, "TestModel")
    assert meta.name == "TestModel"
    assert meta.family == "test"
    assert meta.description == "A test model"
    assert meta.category == ComponentCategory.MODEL


def test_register_duplicate_warning(caplog):
    """Test warning on duplicate registration."""
    Registry.clear()

    @register_model(name="DupModel")
    class ModelA:
        pass

    @register_model(name="DupModel")
    class ModelB:
        pass

    assert "Overwriting" in caplog.text


def test_get_unknown():
    """Test error on getting unknown component."""
    Registry.clear()

    # Register a model so category exists
    @register_model(name="DummyModel")
    class DummyModel:
        pass

    with pytest.raises(ValueError, match="Unknown model"):
        Registry.get(ComponentCategory.MODEL, "NonExistent")


def test_list_empty():
    """Test listing when registry is empty."""
    Registry.clear()
    assert Registry.list() == {}


def test_list_with_entries():
    """Test listing registered components."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    @register_optimizer(name="OptA")
    class OptA:
        pass

    result = Registry.list()
    assert "model" in result
    assert "param_update" in result
    assert result["model"] == ["ModelA"]
    assert result["param_update"] == ["OptA"]


def test_list_by_category():
    """Test listing by category."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    result = Registry.list(ComponentCategory.MODEL)
    assert "model" in result
    assert result["model"] == ["ModelA"]
    assert "param_update" not in result


def test_query_no_filters():
    """Test query without filters returns everything."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    @register_model(name="ModelB")
    class ModelB:
        pass

    results = Registry.query()
    assert len(results) == 2
    assert {r["name"] for r in results} == {"ModelA", "ModelB"}


def test_query_by_family():
    """Test query by family."""
    Registry.clear()

    @register_model(name="EqpropModel", family="eqprop")
    class EqpropModel:
        pass

    @register_model(name="BackpropModel", family="backprop")
    class BackpropModel:
        pass

    eqprop_results = Registry.query(family="eqprop")
    assert len(eqprop_results) == 1
    assert eqprop_results[0]["name"] == "EqpropModel"

    backprop_results = Registry.query(family="backprop")
    assert len(backprop_results) == 1
    assert backprop_results[0]["name"] == "BackpropModel"


def test_query_by_locality():
    """Test query by locality level."""
    Registry.clear()

    @register_model(name="GlobalModel", locality_level=LocalityLevel.GLOBAL)
    class GlobalModel:
        pass

    @register_model(name="LocalModel", locality_level=LocalityLevel.LOCAL)
    class LocalModel:
        pass

    results = Registry.query(locality=LocalityLevel.LOCAL)
    assert len(results) == 1
    assert results[0]["name"] == "LocalModel"


def test_query_by_backward():
    """Test query by requires_backward."""
    Registry.clear()

    @register_model(name="GradModel", requires_backward=True)
    class GradModel:
        pass

    @register_model(name="BioModel", requires_backward=False)
    class BioModel:
        pass

    results = Registry.query(requires_backward=False)
    assert len(results) == 1
    assert results[0]["name"] == "BioModel"


def test_query_by_bio_score():
    """Test query by bio-plausibility score range."""
    Registry.clear()

    @register_model(name="LowBio", bio_plausibility_score=0.1)
    class LowBio:
        pass

    @register_model(name="HighBio", bio_plausibility_score=0.9)
    class HighBio:
        pass

    results = Registry.query(min_bio_score=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "HighBio"

    results = Registry.query(max_bio_score=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "LowBio"


def test_query_tags():
    """Test query by tags."""
    Registry.clear()

    @register_model(name="TaggedModel", tags=["foo", "bar"])
    class TaggedModel:
        pass

    @register_model(name="OtherModel", tags=["baz"])
    class OtherModel:
        pass

    results = Registry.query(tags=["foo"])
    assert len(results) == 1
    assert results[0]["name"] == "TaggedModel"

    results = Registry.query(tags=["foo", "bar"])
    assert len(results) == 1

    results = Registry.query(tags=["foo", "nonexistent"])
    assert len(results) == 0


def test_query_category():
    """Test query by category."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    @register_optimizer(name="OptA")
    class OptA:
        pass

    results = Registry.query(category=ComponentCategory.MODEL)
    assert len(results) == 1
    assert results[0]["category"] == ComponentCategory.MODEL

    results = Registry.query(category=ComponentCategory.PARAM_UPDATE)
    assert len(results) == 1
    assert results[0]["category"] == ComponentCategory.PARAM_UPDATE


def test_component_metadata_defaults():
    """Test ComponentMetadata default values."""
    meta = ComponentMetadata(name="Test", category=ComponentCategory.MODEL)
    assert meta.bio_plausibility_score == pytest.approx(0.5)
    assert meta.requires_backward is True
    assert meta.locality_level == LocalityLevel.GLOBAL
    assert meta.memory_complexity == "O(N)"


def test_registry_metadata_on_class():
    """Test that metadata is attached to the registered class."""
    Registry.clear()

    @register_model(name="TestModel")
    class TestModel:
        pass

    assert hasattr(TestModel, "_registry_metadata")
    assert TestModel._registry_metadata.name == "TestModel"
    assert TestModel._registry_name == "TestModel"
    assert TestModel._registry_category == ComponentCategory.MODEL


def test_infer_metadata_default_factory():
    """_infer_metadata reads class-level provides/requires with default_factory."""
    Registry.clear()

    class ModelWithProvides:
        provides = ["transition_graph", "standard_autograd"]

    meta = ComponentMetadata(name="test", category=ComponentCategory.MODEL)
    Registry._infer_metadata(ModelWithProvides, meta)
    assert meta.provides == ["transition_graph", "standard_autograd"]


def test_infer_metadata_preserves_explicit():
    """Explicit decorator kwargs are NOT overwritten by _infer_metadata."""
    Registry.clear()

    class ModelWithProvides:
        provides = ["transition_graph"]

    meta = ComponentMetadata(
        name="test", category=ComponentCategory.MODEL, provides=["explicit"]
    )
    Registry._infer_metadata(ModelWithProvides, meta)
    assert meta.provides == ["explicit"]


def test_infer_metadata_regular_field():
    """Regular fields (non-default_factory) are still inferred."""
    Registry.clear()

    class ModelWithFamily:
        family = "eqprop"

    meta = ComponentMetadata(name="test2", category=ComponentCategory.MODEL)
    Registry._infer_metadata(ModelWithFamily, meta)
    assert meta.family == "eqprop"


def test_runtime_checkable_transition_graph():
    """All registered EqProp models expose transition_modules() via geometry."""
    from computronium.models.native.eqprop_native import native_eqprop_mlp

    system = native_eqprop_mlp(
        input_dim=4, hidden_dim=8, output_dim=4, num_layers=1, beta=0.5, settle_steps=2
    )
    # Check for transition_modules method on geometry (replaces TransitionGraph protocol)
    assert hasattr(system.geometry, "transition_modules"), (
        f"{system.geometry.__class__.__name__} has no transition_modules()"
    )
    modules = system.geometry.transition_modules()
    assert len(modules) >= 1
    for m in modules:
        assert isinstance(m, nn.Module)


def test_all_models_have_transition_modules_or_override():
    """Verify all registered BioModel subclasses expose transition_modules()."""
    from computronium.core.model import BioModel
    from computronium.core.registry import ComponentCategory, Registry

    models = Registry.list(ComponentCategory.MODEL).get("model", [])
    # Models that use non-nn.Module internal dynamics (graph, kernel, plain nn.Module)
    skip = {
        "pepita",
        "forward_forward",
        "diff_target_prop",
        "contrastive_hebbian",
        "three_factor_hebbian",
        "predictive_coding_hybrid",
        "fabricpc_graph_pcn",  # graph-based, explicit error
    }
    for name in models:
        if name in skip:
            continue
        component = Registry.get(ComponentCategory.MODEL, name)
        # Skip factory functions (not classes)
        if not isinstance(component, type):
            continue
        if not issubclass(component, BioModel):
            continue  # plain nn.Module, no requirement
        assert hasattr(component, "transition_modules"), (
            f"Model {name!r} ({component.__name__}) inherits BioModel "
            f"but has no transition_modules()"
        )
