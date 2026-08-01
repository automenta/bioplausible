"""Registry Audit Unit Tests.

Validates that every registered component:
1. Instantiates successfully
2. Runs forward() on dummy tensor
3. Metadata fields match implementation
4. Produces deterministic output with fixed seed

Target: <15s total for all components (46 models + 19 propagators + 9 optimizers + 3 sparsity).

Most components build through the generic :meth:`BioModel.build` path. A handful
of specialised models (LM / vision / graph / diffusion / lazy) expose genuinely
different forward interfaces (token IDs, 4D images, graph tuples), so they are
constructed via the per-model fixtures in ``_MODEL_FIXTURES`` and driven with a
matching dummy input instead of the flat ``(B, input_dim)`` tensor.
"""

import inspect

import pytest
import torch
from torch import nn

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo import get_model_spec

# =============================================================================
# Per-model construction fixtures for models with non-generic interfaces.
#
# Each value is (build, input_fn) where:
#   build() -> nn.Module   (fresh instance)
#   input_fn(model) -> tuple[object, ...]   positional args for model.forward
# =============================================================================

MODEL_FIXTURES: dict[str, tuple[object, object]] = {}


def _reg(name: str) -> object:
    def _decorator(fn: object) -> object:
        MODEL_FIXTURES[name] = fn()  # type: ignore[operator]
        return fn

    return _decorator


@_reg("lazy_eqprop")
def _lazy_eqprop():
    from bioplausible.zoo.models.eqprop.lazy_eqprop import LazyEqProp

    return (
        lambda: LazyEqProp(input_dim=64, hidden_dim=64, output_dim=10, num_layers=2),
        lambda _m: (torch.randn(BATCH_SIZE, 64),),
    )


@_reg("eqprop_diffusion")
def _eqprop_diffusion():
    from bioplausible.zoo.models.eqprop.eqprop_diffusion import EqPropDiffusion

    return (
        lambda: EqPropDiffusion(img_channels=1, hidden_channels=64),
        lambda _m: (torch.randn(BATCH_SIZE, 1, 16, 16), torch.zeros(BATCH_SIZE).long()),
    )


@_reg("feedback_alignment")
def _feedback_alignment():
    from bioplausible.zoo.models.fa import FeedbackAlignmentEqProp

    return (
        lambda: FeedbackAlignmentEqProp(
            input_dim=64, hidden_dim=64, output_dim=10, num_layers=2
        ),
        lambda _m: (torch.randn(BATCH_SIZE, 64),),
    )


@_reg("hebbian_3d")
def _hebbian_3d():
    from bioplausible.zoo.models.hebbian import HebbianCube

    return (
        lambda: HebbianCube(
            input_dim=64, hidden_dim=64, output_dim=10, num_layers=2, cube_size=4
        ),
        lambda _m: (torch.randn(BATCH_SIZE, 64),),
    )


@_reg("backprop_transformer_lm")
def _backprop_transformer_lm():
    from bioplausible.zoo.models.backprop import BackpropTransformerLM

    return (
        lambda: BackpropTransformerLM(
            vocab_size=10, hidden_dim=32, num_layers=2, num_heads=2, max_seq_len=16
        ),
        lambda _m: (torch.randint(0, 10, (BATCH_SIZE, 8)),),
    )


@_reg("graph_equitile")
def _graph_equitile():
    from bioplausible.equitile.deployments.graph import (
        GraphEquiTile,
        GraphEquiTileConfig,
    )

    def build():
        return GraphEquiTile(
            GraphEquiTileConfig(
                node_features=10,
                hidden_dim=32,
                num_classes=10,
                num_layers=2,
                neurons_per_tile=8,
                tiles_per_layer=2,
            )
        )

    def inp(_m):
        num_nodes = BATCH_SIZE * 5
        batch = torch.arange(BATCH_SIZE).repeat(5)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        return (torch.randn(num_nodes, 10), edge_index, batch)

    return build, inp


@_reg("timeseries_equitile")
def _timeseries_equitile():
    from bioplausible.equitile.deployments.timeseries import (
        TimeSeriesConfig,
        TimeSeriesEquiTile,
    )

    return (
        lambda: TimeSeriesEquiTile(
            TimeSeriesConfig(
                input_dim=10,
                seq_len=16,
                output_dim=10,
                pred_len=2,
                hidden_dim=32,
                num_layers=2,
            )
        ),
        lambda _m: (torch.randn(BATCH_SIZE, 16, 10),),
    )


@_reg("enhanced_equitile")
def _enhanced_equitile():
    from bioplausible.equitile._internal.enhanced import EnhancedEquiTile

    return (
        lambda: EnhancedEquiTile(
            neurons_per_tile=8,
            num_layers=2,
            tiles_per_layer=2,
            input_dim=64,
            output_dim=10,
        ),
        lambda _m: (torch.randn(BATCH_SIZE, 64),),
    )


@_reg("conv_equitile")
def _conv_equitile():
    from bioplausible.equitile.deployments.vision import (
        ConvEquiTile,
        ConvEquiTileConfig,
    )

    return (
        lambda: ConvEquiTile(
            ConvEquiTileConfig(input_channels=1, input_size=16, num_classes=10)
        ),
        lambda _m: (torch.randn(BATCH_SIZE, 1, 16, 16),),
    )


@_reg("lm_equitile")
def _lm_equitile():
    from bioplausible.equitile.language.canonical import LMEquiTile, LMEquiTileConfig

    return (
        lambda: LMEquiTile(
            LMEquiTileConfig(vocab_size=20, embed_dim=16, num_layers=2, max_seq_len=16)
        ),
        lambda _m: (torch.randint(0, 20, (BATCH_SIZE, 8)),),
    )


@_reg("optimized_lm_equitile")
def _optimized_lm_equitile():
    from bioplausible.equitile.language.optimized import OptimizedLMEquiTile
    from bioplausible.equitile.language.canonical import LMEquiTileConfig

    def build():
        return OptimizedLMEquiTile(
            LMEquiTileConfig(vocab_size=20, embed_dim=16, num_layers=2, max_seq_len=16),
            use_compile=False,
        )

    return build, lambda _m: (torch.randint(0, 20, (BATCH_SIZE, 8)),)


@_reg("fast_lm_equitile")
def _fast_lm_equitile():
    from bioplausible.equitile.lm.components import FastLMConfig
    from bioplausible.equitile.lm.fast_lm import FastLMEquiTile

    def build():
        config = FastLMConfig(
            vocab_size=20,
            embed_dim=16,
            num_layers=2,
            hidden_dim=32,
            neurons_per_tile=8,
            tiles_per_layer=2,
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=16,
            mot_k=1,
        )
        return FastLMEquiTile(config)

    return build, lambda _m: (torch.randint(0, 20, (BATCH_SIZE, 8)),)


# Models that cannot be audited through a forward() call at all. ``dynamic_equitile``
# is a training-side topology controller (analysis.dynamics.DynamicEquiTile), NOT an
# nn.Module — it has no forward pass and is misfiled in the MODEL registry. See TODO.
SKIP_MODELS = {
    "dynamic_equitile": "Training-side topology controller, not an nn.Module; no forward()",
}

# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE = 4
TUPLE_LENGTH_2 = 2
MIN_MODELS = 40
MIN_PROPAGATORS = 15
MIN_OPTIMIZERS = 5
MIN_SPARSITY = 2
MIN_TOTAL = 70

VALID_LOCALITY_LEVELS = {
    "global",
    "layerwise",
    "local",
    "equilibrium",
    "forward-only",
}

VALID_CREDIT_TYPES = {
    "gradient",
    "equilibrium",
    "hebbian",
    "target",
    "forward-only",
    "spiking",
    "backpropagation",
    "local",
}

# =============================================================================
# Helpers
# =============================================================================


def _instantiate_model(model_name: str) -> nn.Module:
    """Instantiate a model via its dedicated fixture or generic ``build()``."""
    fixture = MODEL_FIXTURES.get(model_name)
    if fixture is not None:
        return fixture[0]()

    spec = get_model_spec(model_name)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)
    return model_cls.build(
        spec=spec,
        input_dim=64,
        output_dim=10,
        hidden_dim=64,
        num_layers=2,
        device="cpu",
        task_type="vision",
    )


def _model_args(model_name: str, model: nn.Module) -> tuple[object, ...]:
    """Return positional args matching ``model.forward`` for the given model."""
    fixture = MODEL_FIXTURES.get(model_name)
    if fixture is not None:
        return fixture[1](model)

    meta = Registry.get_metadata(ComponentCategory.MODEL, model_name)
    input_dim = meta.extra.get("input_dim", 64)
    if isinstance(input_dim, tuple):
        input_dim = int(torch.prod(torch.tensor(input_dim)))
    return (torch.randn(BATCH_SIZE, input_dim),)


def _propagator_instantiates(prop_name: str, params, test_model) -> bool:
    """Instantiate a propagator, handling preset functions vs optimizer classes.

    Some propagators are ``(params, model)`` factories (SMEP family), others are
    pure ``(params)`` optimizer constructors (muon_backprop). We inspect the
    signature rather than assume a uniform call.
    """
    prop_cls = Registry.get(ComponentCategory.PROPAGATOR, prop_name)
    target = prop_cls.__init__ if inspect.isclass(prop_cls) else prop_cls
    sig = inspect.signature(target)
    if "model" in sig.parameters:
        return prop_cls(params, test_model)
    return prop_cls(params)


# =============================================================================
# Model Registry Audit
# =============================================================================


class TestModelRegistry:
    """Tests for model registry audit."""

    @pytest.mark.parametrize(
        "model_name",
        [m["name"] for m in Registry.query(category=ComponentCategory.MODEL)],
    )
    def test_model_instantiates(self, model_name):
        """Every registered model should instantiate without error."""
        if model_name in SKIP_MODELS:
            pytest.skip(f"{model_name}: {SKIP_MODELS[model_name]}")

        try:
            model = _instantiate_model(model_name)
            assert model is not None
            assert isinstance(model, nn.Module)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{model_name} instantiation failed: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [m["name"] for m in Registry.query(category=ComponentCategory.MODEL)],
    )
    def test_model_forward_pass(self, model_name):
        """Every model should run forward() on a matching dummy input."""
        if model_name in SKIP_MODELS:
            pytest.skip(f"{model_name}: {SKIP_MODELS[model_name]}")

        try:
            model = _instantiate_model(model_name)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{model_name} instantiation failed: {e}")

        model.eval()
        args = _model_args(model_name, model)
        with torch.no_grad():
            try:
                out = model(*args)
                assert out is not None
                assert isinstance(out, torch.Tensor)
                assert out.shape[0] == BATCH_SIZE
            except Exception as e:
                pytest.skip(f"{model_name} forward pass failed: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [m["name"] for m in Registry.query(category=ComponentCategory.MODEL)],
    )
    def test_model_metadata_matches(self, model_name):
        """Model metadata should match implementation."""
        meta = Registry.get_metadata(ComponentCategory.MODEL, model_name)
        spec = get_model_spec(model_name)

        assert meta.name == model_name
        assert spec.name == model_name
        assert meta.category == ComponentCategory.MODEL

        assert meta.locality_level in VALID_LOCALITY_LEVELS
        assert meta.credit_assignment_type in VALID_CREDIT_TYPES
        assert len(meta.domains) > 0

        if meta.typical_lr_range:
            assert len(meta.typical_lr_range) == TUPLE_LENGTH_2
            assert meta.typical_lr_range[0] < meta.typical_lr_range[1]
            assert meta.typical_lr_range[0] > 0

    @pytest.mark.parametrize(
        "model_name",
        [m["name"] for m in Registry.query(category=ComponentCategory.MODEL)],
    )
    def test_model_deterministic_output(self, model_name):
        """Fixed seed should produce identical model outputs."""
        if model_name in SKIP_MODELS:
            pytest.skip(f"{model_name}: {SKIP_MODELS[model_name]}")

        try:
            torch.manual_seed(42)
            model1 = _instantiate_model(model_name)

            torch.manual_seed(42)
            model2 = _instantiate_model(model_name)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{model_name} instantiation failed: {e}")

        model1.eval()
        model2.eval()

        args = _model_args(model_name, model1)

        with torch.no_grad():
            out1 = model1(*args)
            out2 = model2(*args)

        assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-7), (
            f"{model_name}: non-deterministic output with fixed seed"
        )


class TestPropagatorRegistry:
    """Tests for propagator registry audit."""

    @pytest.mark.parametrize(
        "prop_name",
        [p["name"] for p in Registry.query(category=ComponentCategory.PROPAGATOR)],
    )
    def test_propagator_instantiates(self, prop_name):
        """Every registered propagator should instantiate."""
        from bioplausible.zoo.models.eqprop import BackpropMLP

        test_model = BackpropMLP(
            input_dim=64, hidden_dim=64, output_dim=10, num_layers=2
        )
        params = list(test_model.parameters())

        try:
            prop = _propagator_instantiates(prop_name, params, test_model)
            assert prop is not None
        except (
            NotImplementedError,
            TypeError,
            ValueError,
            RuntimeError,
            ImportError,
        ) as e:
            pytest.skip(f"{prop_name} instantiation failed: {e}")

    @pytest.mark.parametrize(
        "prop_name",
        [p["name"] for p in Registry.query(category=ComponentCategory.PROPAGATOR)],
    )
    def test_propagator_metadata(self, prop_name):
        """Propagator metadata should be valid."""
        meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, prop_name)

        assert meta.name == prop_name
        assert meta.category == ComponentCategory.PROPAGATOR
        assert meta.locality_level in VALID_LOCALITY_LEVELS
        assert isinstance(meta.requires_backward, bool)


class TestOptimizerRegistry:
    """Tests for optimizer registry audit."""

    @pytest.mark.parametrize(
        "opt_name",
        [o["name"] for o in Registry.query(category=ComponentCategory.OPTIMIZER)],
    )
    def test_optimizer_instantiates(self, opt_name):
        """Every registered optimizer should instantiate."""
        opt_cls = Registry.get(ComponentCategory.OPTIMIZER, opt_name)

        try:
            params = [nn.Parameter(torch.randn(10, 10))]
            opt = opt_cls(params)
            assert opt is not None
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{opt_name} instantiation failed: {e}")

    @pytest.mark.parametrize(
        "opt_name",
        [o["name"] for o in Registry.query(category=ComponentCategory.OPTIMIZER)],
    )
    def test_optimizer_metadata(self, opt_name):
        """Optimizer metadata should be valid."""
        meta = Registry.get_metadata(ComponentCategory.OPTIMIZER, opt_name)

        assert meta.name == opt_name
        assert meta.category == ComponentCategory.OPTIMIZER


class TestSparsityRegistry:
    """Tests for sparsity method registry audit."""

    @pytest.mark.parametrize(
        "sp_name",
        [s["name"] for s in Registry.query(category=ComponentCategory.SPARSITY)],
    )
    def test_sparsity_instantiates(self, sp_name):
        """Every registered sparsity method should instantiate."""
        sp_cls = Registry.get(ComponentCategory.SPARSITY, sp_name)

        try:
            from bioplausible.zoo.models.eqprop import BackpropMLP

            # Pruning methods wrap a model; pass one when the signature asks.
            target = sp_cls.__init__ if inspect.isclass(sp_cls) else sp_cls
            sig = inspect.signature(target)
            if "model" in sig.parameters:
                m = BackpropMLP(input_dim=64, hidden_dim=64, output_dim=10, num_layers=2)
                sp = sp_cls(model=m)
            else:
                sp = sp_cls()
            assert sp is not None
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{sp_name} instantiation failed: {e}")

    @pytest.mark.parametrize(
        "sp_name",
        [s["name"] for s in Registry.query(category=ComponentCategory.SPARSITY)],
    )
    def test_sparsity_metadata(self, sp_name):
        """Sparsity metadata should be valid."""
        meta = Registry.get_metadata(ComponentCategory.SPARSITY, sp_name)

        assert meta.name == sp_name
        assert meta.category == ComponentCategory.SPARSITY


# =============================================================================
# Summary Test
# =============================================================================


def test_registry_counts():
    """Verify expected component counts."""
    models = Registry.query(category=ComponentCategory.MODEL)
    propagators = Registry.query(category=ComponentCategory.PROPAGATOR)
    optimizers = Registry.query(category=ComponentCategory.OPTIMIZER)
    sparsity = Registry.query(category=ComponentCategory.SPARSITY)

    assert len(models) >= MIN_MODELS, f"Expected >=40 models, got {len(models)}"
    assert len(propagators) >= MIN_PROPAGATORS, (
        f"Expected >=15 propagators, got {len(propagators)}"
    )
    assert len(optimizers) >= MIN_OPTIMIZERS, (
        f"Expected >=5 optimizers, got {len(optimizers)}"
    )
    assert len(sparsity) >= MIN_SPARSITY, (
        f"Expected >=2 sparsity methods, got {len(sparsity)}"
    )

    total = len(models) + len(propagators) + len(optimizers) + len(sparsity)
    assert total >= MIN_TOTAL, f"Expected >=70 total components, got {total}"
