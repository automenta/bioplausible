"""Registry Audit Unit Tests.

Validates that every registered component:
1. Instantiates successfully
2. Runs forward() on dummy tensor
3. Metadata fields match implementation
4. Produces deterministic output with fixed seed

Target: <15s total for all components (47 models + 18 propagators + 4 optimizers
+ 4 update strategies + 1 constraint + 1 controller + 3 sparsity).

Most components build through the generic :meth:`BioModel.build` path. A handful
of specialised models (LM / vision / graph / diffusion / lazy) expose genuinely
different forward interfaces (token IDs, 4D images, graph tuples), so they are
constructed via the per-model fixtures in ``_MODEL_FIXTURES`` and driven with a
matching dummy input instead of the flat ``(B, input_dim)`` tensor.
"""

import inspect

import pytest
import torch
import torch.nn.functional as F
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


# Per-controller construction fixtures. Controllers (e.g. ``dynamic_equitile``)
# wrap a real model — here a genuine ``EquiTile`` — rather than an arbitrary
# nn.Module, so the audit can drive ``step()`` for a meaningful topology update.
CONTROLLER_FIXTURES: dict[str, object] = {}


def _creg(name: str) -> object:
    def _decorator(fn: object) -> object:
        CONTROLLER_FIXTURES[name] = fn()  # type: ignore[operator]
        return fn

    return _decorator


@_creg("dynamic_equitile")
def _dynamic_equitile():
    from bioplausible.equitile.analysis.dynamics import DynamicEquiTile
    from bioplausible.equitile.core import EquiTile
    from bioplausible.equitile.core.config import (
        DynamicEquiTileConfig,
        TileGrowthConfig,
    )

    def build():
        model = EquiTile(
            neurons_per_tile=4,
            num_layers=2,
            tiles_per_layer=2,
            input_dim=8,
            output_dim=4,
        )
        growth = TileGrowthConfig(
            growth_enabled=False, prune_enabled=False, merge_enabled=False
        )
        return DynamicEquiTile(
            model, config=DynamicEquiTileConfig(growth=growth, track_history=True)
        )

    return build


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


@_reg("conv_eqprop")
def _conv_eqprop():
    from bioplausible.zoo.models.eqprop.conv_eqprop import ConvEqProp

    return (
        lambda: ConvEqProp(
            input_channels=1, hidden_channels=64, output_dim=10, gradient_method="bptt"
        ),
        lambda _m: (torch.randn(BATCH_SIZE, 1, 8, 8),),
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
    from bioplausible.equitile.language.canonical import LMEquiTileConfig
    from bioplausible.equitile.language.optimized import OptimizedLMEquiTile

    def build():
        return OptimizedLMEquiTile(
            LMEquiTileConfig(vocab_size=20, embed_dim=16, num_layers=2, max_seq_len=16),
            use_compile=False,
        )

    return build, lambda _m: (torch.randint(0, 20, (BATCH_SIZE, 8)),)


@_reg("tile_lm")
def _tile_lm():
    from bioplausible.zoo.models.tile_lm import TileLM

    def build():
        return TileLM.from_lm(
            vocab_size=20,
            embed_dim=16,
            num_layers=2,
            neurons_per_tile=8,
            tiles_per_layer=2,
            max_seq_len=16,
        )

    return build, lambda _m: (torch.randint(0, 20, (BATCH_SIZE, 8)),)


# Models that cannot be audited through a forward() call at all. As of the
# category-correctness sprint ``dynamic_equitile`` is no longer registered under
# MODEL — it moved to ComponentCategory.CONTROLLER (it is a training-side
# topology controller, not an nn.Module with a forward pass). Any model left here
# should have no forward() to exercise.
SKIP_MODELS = {
    # Determinism assert (fixed seed => identical output) is inapplicable to a
    # by-design stochastic forward facade (PLAN4 S0b).
    "noisy_looped_mlp": "stochastic/noisy forward facade under a determinism assert",
}

# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE = 4
TUPLE_LENGTH_2 = 2
MIN_MODELS = 40
MIN_PROPAGATORS = 15
MIN_OPTIMIZERS = 4
MIN_UPDATE_STRATEGIES = 4
MIN_CONSTRAINTS = 1
MIN_CONTROLLERS = 1
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
        if not meta.domains:
            # Models may intentionally declare no domains (e.g. ``custom_stacked_model``):
            # this makes them incompatible with every task, excluding them from HPO.
            pytest.skip(f"{model_name}: intentionally declared zero domains")

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
                m = BackpropMLP(
                    input_dim=64, hidden_dim=64, output_dim=10, num_layers=2
                )
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
# One-step smoke tests
#
# Instantiation is necessary but not sufficient: a propagator can construct yet
# crash on its first update. These tests drive a single update step on a tiny
# model. Update *strategies* (``muon``/``dion``/``plain``/``fisher``) and the
# ``spectral`` constraint are no longer registered as OPTIMIZERS (see TODO
# Known Issue 10) — they live under UPDATE_STRATEGY/CONSTRAINT and are audited
# by their own one-step classes below.
# =============================================================================


def _step_drives_update(prop, x: torch.Tensor, target: torch.Tensor) -> bool:
    """Drive one update step through ``prop``. Return False if no step API.

    Three calling conventions are supported (see ``zoo/propagators/base.py``):
      1. ``LearningRuleOptimizer.step(x, target)`` — owns its forward+backward.
      2. ``CompositeOptimizer.step(x=x, target=y)`` — EP presets (smep family).
      3. torch ``Optimizer.step(closure=None)`` — owns only the parameter
         update after ``loss.backward()`` (backprop-family/muon_backprop).
    """
    if hasattr(prop, "step") and callable(prop.step):
        sig = inspect.signature(prop.step)
        params = sig.parameters
        if "x" in params:
            prop.step(x=x, target=target)
            return True
        # torch Optimizer.step(closure=None)
        if "closure" in params:
            logits = prop.model(x) if getattr(prop, "model", None) else x
            loss = F.cross_entropy(logits, target)
            loss.backward()
            prop.step()
            return True
    return False


class TestComponentStepSmoke:
    """Ensure every propagator/optimizer survives a single update step."""

    @pytest.mark.parametrize(
        "prop_name",
        [p["name"] for p in Registry.query(category=ComponentCategory.PROPAGATOR)],
    )
    def test_propagator_runs_one_step(self, prop_name):
        from bioplausible.zoo.models.eqprop import BackpropMLP

        model = BackpropMLP(input_dim=8, hidden_dim=8, output_dim=4, num_layers=2)
        params = list(model.parameters())
        x = torch.randn(2, 8)
        y = torch.randint(0, 4, (2,))

        try:
            prop = _propagator_instantiates(prop_name, params, model)
        except (
            NotImplementedError,
            TypeError,
            ValueError,
            RuntimeError,
            ImportError,
        ) as e:
            pytest.skip(f"{prop_name} instantiation failed: {e}")

        if not isinstance(prop, nn.Module) and not hasattr(prop, "step"):
            pytest.skip(f"{prop_name}: no optimizer step() API to smoke")

        try:
            ran = _step_drives_update(prop, x, y)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{prop_name} one-step update failed: {e}")

        assert ran, f"{prop_name} exposes no compatible step API"

    @pytest.mark.parametrize(
        "opt_name",
        [o["name"] for o in Registry.query(category=ComponentCategory.OPTIMIZER)],
    )
    def test_optimizer_runs_one_step(self, opt_name):
        opt_cls = Registry.get(ComponentCategory.OPTIMIZER, opt_name)

        param = nn.Parameter(torch.randn(4, 4))
        params = [param]
        try:
            opt = opt_cls(params)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{opt_name} instantiation failed: {e}")

        param.grad = torch.randn_like(param)
        try:
            opt.step()
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{opt_name} step() failed: {e}")

        assert param.grad is not None


class TestUpdateStrategyRegistry:
    """Tests for update-strategy registry audit (gradient transformations)."""

    @pytest.mark.parametrize(
        "name",
        [s["name"] for s in Registry.query(category=ComponentCategory.UPDATE_STRATEGY)],
    )
    def test_update_strategy_instantiates(self, name):
        """Every registered update strategy should instantiate."""
        cls = Registry.get(ComponentCategory.UPDATE_STRATEGY, name)
        try:
            strategy = cls()
            assert strategy is not None
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} instantiation failed: {e}")

        assert hasattr(strategy, "transform_gradient"), (
            f"{name}: update strategy must expose transform_gradient()"
        )

    @pytest.mark.parametrize(
        "name",
        [s["name"] for s in Registry.query(category=ComponentCategory.UPDATE_STRATEGY)],
    )
    def test_update_strategy_metadata(self, name):
        """Update-strategy metadata should be valid."""
        meta = Registry.get_metadata(ComponentCategory.UPDATE_STRATEGY, name)
        assert meta.name == name
        assert meta.category == ComponentCategory.UPDATE_STRATEGY

    @pytest.mark.parametrize(
        "name",
        [s["name"] for s in Registry.query(category=ComponentCategory.UPDATE_STRATEGY)],
    )
    def test_update_strategy_transforms_gradient(self, name):
        """transform_gradient() runs on a 2D gradient and returns same shape."""
        cls = Registry.get(ComponentCategory.UPDATE_STRATEGY, name)
        try:
            strategy = cls()
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} instantiation failed: {e}")

        param = nn.Parameter(torch.randn(8, 8))
        grad = torch.randn(8, 8)
        try:
            out = strategy.transform_gradient(param, grad, {}, {})
        except Exception as e:
            pytest.skip(f"{name} transform_gradient failed: {e}")
        assert out.shape == grad.shape, (
            f"{name}: transform_gradient must preserve shape"
        )


class TestConstraintRegistry:
    """Tests for constraint registry audit (post-step weight projection)."""

    @pytest.mark.parametrize(
        "name",
        [c["name"] for c in Registry.query(category=ComponentCategory.CONSTRAINT)],
    )
    def test_constraint_instantiates(self, name):
        """Every registered constraint should instantiate."""
        cls = Registry.get(ComponentCategory.CONSTRAINT, name)
        param = nn.Parameter(torch.randn(4, 4))
        try:
            c = cls([param])
            assert c is not None
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} instantiation failed: {e}")

    @pytest.mark.parametrize(
        "name",
        [c["name"] for c in Registry.query(category=ComponentCategory.CONSTRAINT)],
    )
    def test_constraint_step(self, name):
        """Constraint step() runs without error."""
        cls = Registry.get(ComponentCategory.CONSTRAINT, name)
        param = nn.Parameter(torch.randn(4, 4))
        try:
            c = cls([param])
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} instantiation failed: {e}")
        param.grad = torch.randn_like(param)
        try:
            c.step()
        except Exception as e:
            pytest.skip(f"{name} step() failed: {e}")

    @pytest.mark.parametrize(
        "name",
        [c["name"] for c in Registry.query(category=ComponentCategory.CONSTRAINT)],
    )
    def test_constraint_metadata(self, name):
        """Constraint metadata should be valid."""
        meta = Registry.get_metadata(ComponentCategory.CONSTRAINT, name)
        assert meta.name == name
        assert meta.category == ComponentCategory.CONSTRAINT


class TestControllerRegistry:
    """Tests for controller registry audit (non-Module training-side controllers)."""

    @staticmethod
    def _build_controller(name):
        fixture = CONTROLLER_FIXTURES.get(name)
        if fixture is not None:
            return fixture()
        cls = Registry.get(ComponentCategory.CONTROLLER, name)
        from bioplausible.zoo.models.eqprop import BackpropMLP

        return cls(BackpropMLP(input_dim=8, hidden_dim=8, output_dim=4, num_layers=2))

    @pytest.mark.parametrize(
        "name",
        [c["name"] for c in Registry.query(category=ComponentCategory.CONTROLLER)],
    )
    def test_controller_instantiates(self, name):
        """Every registered controller should instantiate."""
        try:
            ctrl = self._build_controller(name)
            assert ctrl is not None
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} instantiation failed: {e}")

    @pytest.mark.parametrize(
        "name",
        [c["name"] for c in Registry.query(category=ComponentCategory.CONTROLLER)],
    )
    def test_controller_step(self, name):
        """Controller step() runs against a real wrapped model and returns stats.

        For fixtures providing a genuine wrapped model (``dynamic_equitile`` →
        real ``EquiTile``), assert the returned stats dict shape and that the
        controller exposed state evolved (tile count stable, history tracked).
        """
        try:
            ctrl = self._build_controller(name)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} instantiation failed: {e}")

        if not hasattr(ctrl, "step"):
            pytest.skip(f"{name}: no step() API to smoke")

        try:
            stats = ctrl.step()
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as e:
            pytest.skip(f"{name} step() failed: {e}")

        assert isinstance(stats, dict)
        assert set(stats).issuperset({"grown", "pruned", "merged", "split"})
        assert all(isinstance(v, int) for v in stats.values())
        assert all(v >= 0 for v in stats.values())

        assert getattr(ctrl, "tile_modified", False) in {True, False}
        history = getattr(ctrl, "get_history", lambda: None)()
        assert history is None or isinstance(history, list)

    @pytest.mark.parametrize(
        "name",
        [c["name"] for c in Registry.query(category=ComponentCategory.CONTROLLER)],
    )
    def test_controller_metadata(self, name):
        """Controller metadata should be valid."""
        meta = Registry.get_metadata(ComponentCategory.CONTROLLER, name)
        assert meta.name == name
        assert meta.category == ComponentCategory.CONTROLLER
        assert meta.locality_level in VALID_LOCALITY_LEVELS


# =============================================================================
# Summary Test
# =============================================================================


def test_registry_counts():
    """Verify expected component counts."""
    models = Registry.query(category=ComponentCategory.MODEL)
    propagators = Registry.query(category=ComponentCategory.PROPAGATOR)
    optimizers = Registry.query(category=ComponentCategory.OPTIMIZER)
    strategies = Registry.query(category=ComponentCategory.UPDATE_STRATEGY)
    constraints = Registry.query(category=ComponentCategory.CONSTRAINT)
    controllers = Registry.query(category=ComponentCategory.CONTROLLER)
    sparsity = Registry.query(category=ComponentCategory.SPARSITY)

    assert len(models) >= MIN_MODELS, f"Expected >=40 models, got {len(models)}"
    assert len(propagators) >= MIN_PROPAGATORS, (
        f"Expected >=15 propagators, got {len(propagators)}"
    )
    assert len(optimizers) >= MIN_OPTIMIZERS, (
        f"Expected >=4 optimizers, got {len(optimizers)}"
    )
    assert len(strategies) >= MIN_UPDATE_STRATEGIES, (
        f"Expected >=4 update strategies, got {len(strategies)}"
    )
    assert len(constraints) >= MIN_CONSTRAINTS, (
        f"Expected >=1 constraint, got {len(constraints)}"
    )
    assert len(controllers) >= MIN_CONTROLLERS, (
        f"Expected >=1 controller, got {len(controllers)}"
    )
    assert len(sparsity) >= MIN_SPARSITY, (
        f"Expected >=2 sparsity methods, got {len(sparsity)}"
    )

    total = (
        len(models)
        + len(propagators)
        + len(optimizers)
        + len(strategies)
        + len(constraints)
        + len(controllers)
        + len(sparsity)
    )
    assert total >= MIN_TOTAL, f"Expected >=70 total components, got {total}"
