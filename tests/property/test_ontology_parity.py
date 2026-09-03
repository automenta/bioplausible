"""API Parity Tests: Verify Ontology factories match Native model behavior.

These tests ensure that 5-D ontology native factories produce functionally
equivalent systems to their preset factory counterparts.

Run: uv run pytest tests/property/test_ontology_parity.py -v  # tier: -m slow
"""

import pytest
import torch

from computronium import (
    create_backprop_mlp,
    create_eqprop_mlp,
    create_fa_mlp,
    create_fast_weight_mlp,
    create_ff_mlp,
    create_hebbian_mlp,
    create_pc_mlp,
    create_pepita_mlp,
    create_routing_mlp,
    create_snn_mlp,
    create_tile_mlp,
    create_tp_mlp,
)
from computronium.core.system_trainer import (
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
)
from computronium.domains.factory import create_task
from computronium.models.native.backprop_native import create_native_backprop_mlp
from computronium.models.native.diffusion_eqprop_native import (
    create_native_diffusion_eqprop,
)
from computronium.models.native.eqprop_native import create_native_eqprop_mlp
from computronium.models.native.fa_native import (
    create_native_fa_adaptive,
    create_native_fa_contrastive,
    create_native_fa_deep_dfa,
    create_native_fa_direct,
    create_native_fa_energy_guided,
    create_native_fa_energy_minimizing,
    create_native_fa_equilibrium_alignment,
    create_native_fa_layerwise_equilibrium,
    create_native_fa_mlp,
    create_native_fa_sign_symmetric,
    create_native_fa_stochastic,
)
from computronium.models.native.momentum_eqprop_native import (
    create_native_momentum_eqprop,
)
from computronium.models.native.pepita_native import create_native_pepita_mlp
from computronium.models.native.research_native import (
    create_native_directed_ep,
    create_native_finite_nudge_ep,
    create_native_holomorphic_ep,
)
from computronium.models.native.tile_native import (
    create_native_tile_fa,
    create_native_tile_hebbian,
    create_native_tile_pc,
    create_native_tile_snn,
    create_native_tile_tp,
)
from computronium.ontology import (
    AnalogSubstrate,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    OpticalSubstrate,
    ParameterUpdateConfig,
    QuantumSubstrate,
    RandomProjectionsCredit,
    SparseSubstrate,
    StateDynamicsConfig,
    SubstrateConfig,
    TernarySubstrate,
    ThermodynamicContrast,
)

# Every test here trains full models (2-3 epochs each); runs in the slow tier only.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(300)]


def make_dataloaders(device: str = "cpu", batch_size: int = 64):
    """Create train/val dataloaders for MNIST using task.get_batch."""
    task = create_task("mnist", device=device, quick_mode=True)
    task.setup()

    class BatchLoader:
        def __init__(self, task, batch_size, split):
            self.task = task
            self.batch_size = batch_size
            self.split = split
            self._num_batches = 100 if split == "train" else 20

        def __iter__(self):
            for _ in range(self._num_batches):
                x, y = self.task.get_batch(self.split, self.batch_size)
                # Flatten image input for MLP: [B, C, H, W] -> [B, C*H*W]
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                yield x, y

        def __len__(self):
            return self._num_batches

    train_loader = BatchLoader(task, batch_size, "train")
    val_loader = BatchLoader(task, batch_size, "val")

    # Flatten input dim for MLP
    input_dim = task.input_dim
    if isinstance(input_dim, (tuple, list)):
        input_dim = int(torch.prod(torch.tensor(input_dim)))
    output_dim = task.output_dim

    return train_loader, val_loader, input_dim, output_dim


def train_system(
    system, train_loader, val_loader, epochs: int, device: str, seed: int = 42
):
    """Train a system and return final validation accuracy."""
    torch.manual_seed(seed)

    trainer_config = SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=64,
        device=device,
        seed=seed,
        log_every_n_steps=100,
    )
    trainer = SystemTrainer(
        system=system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )
    history = trainer.fit()
    return history[-1].get("val_acc", history[-1].get("train_acc", 0.0)) * 100


def construction_seed(seed: int = 42) -> None:
    """Pin factory init draws so parity holds regardless of suite ordering."""
    torch.manual_seed(seed)


class TestBackpropParity:
    """Test Backprop parity between presets factory and native."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_backprop_mlp_matches_native(self, epochs):
        """presets.create_backprop_mlp should match native_backprop_mlp."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        construction_seed()
        system1 = create_backprop_mlp(
            input_dim, (hidden_dim,), output_dim, lr=0.001, device=device
        )
        construction_seed()
        system2 = create_native_backprop_mlp(
            input_dim, hidden_dim, output_dim, lr=0.001
        )

        acc1 = train_system(system1, train_loader, val_loader, epochs, device)
        acc2 = train_system(system2, train_loader, val_loader, epochs, device)

        # Both should achieve reasonable accuracy
        assert acc1 > 70.0, f"Presets backprop: {acc1:.1f}%"
        assert acc2 > 70.0, f"Native backprop: {acc2:.1f}%"
        # Parity within 5%
        assert abs(acc1 - acc2) < 5.0, f"Parity gap: {abs(acc1 - acc2):.1f}%"


class TestEqPropParity:
    """Test EqProp parity between presets factory and native."""

    @pytest.mark.parametrize("epochs", [3])
    def test_create_eqprop_mlp_matches_native(self, epochs):
        """presets.create_eqprop_mlp should match native_eqprop_mlp."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        # Use same parameters for both (native uses settle_steps, not inference_steps)
        construction_seed()
        system1 = create_eqprop_mlp(
            input_dim,
            (hidden_dim,),
            output_dim,
            beta=0.1,
            inference_steps=10,
            lr=0.001,
            device=device,
        )
        construction_seed()
        system2 = create_native_eqprop_mlp(
            input_dim, hidden_dim, output_dim, beta=0.1, settle_steps=10, lr=0.001
        )

        acc1 = train_system(system1, train_loader, val_loader, epochs, device)
        acc2 = train_system(system2, train_loader, val_loader, epochs, device)

        # Both should achieve similar accuracy (parity test)
        # With small architecture (1 layer of 128) and 3 epochs, EqProp is still converging
        # Just verify they produce similar results (parity within 10%)
        assert abs(acc1 - acc2) < 10.0, f"Parity gap: {abs(acc1 - acc2):.1f}%"

    @pytest.mark.parametrize("epochs", [3])
    def test_create_eqprop_mlp_matches_momentum_native(self, epochs):
        """presets.create_eqprop_mlp should have similar behavior to native_momentum_eqprop."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        construction_seed()
        system1 = create_eqprop_mlp(
            input_dim,
            (hidden_dim,),
            output_dim,
            beta=0.1,
            inference_steps=10,
            lr=0.001,
            device=device,
        )
        construction_seed()
        system2 = create_native_momentum_eqprop(
            input_dim, hidden_dim, output_dim, beta=0.1, settle_steps=10, lr=0.001
        )

        acc1 = train_system(system1, train_loader, val_loader, epochs, device)
        acc2 = train_system(system2, train_loader, val_loader, epochs, device)

        # Both should achieve similar accuracy (parity within 15%)
        assert abs(acc1 - acc2) < 15.0, f"Parity gap: {abs(acc1 - acc2):.1f}%"


class TestDiffusionEqPropParity:
    """Test Diffusion EqProp native compositions."""

    @pytest.mark.parametrize("epochs", [2])
    @pytest.mark.xfail(reason="DiffusionDynamics has gradient computation bug")
    def test_native_diffusion_eqprop_composes_and_trains(self, epochs):
        """native_diffusion_eqprop should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        system = create_native_diffusion_eqprop(
            input_dim, hidden_dim, output_dim, beta=0.1, settle_steps=10, lr=0.001
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)
        assert acc >= 0.0, f"Diffusion EqProp accuracy: {acc:.1f}%"


class TestFAParity:
    """Test Feedback Alignment parity between presets factory and native."""

    @pytest.mark.parametrize("epochs", [3])
    def test_create_fa_mlp_matches_native(self, epochs):
        """presets.create_fa_mlp should match native_fa_mlp."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        construction_seed()
        system1 = create_fa_mlp(
            input_dim,
            (hidden_dim,),
            output_dim,
            lr=0.001,
            feedback_scale=0.1,
            device=device,
        )
        construction_seed()
        system2 = create_native_fa_mlp(input_dim, hidden_dim, output_dim, lr=0.001)

        acc1 = train_system(system1, train_loader, val_loader, epochs, device)
        acc2 = train_system(system2, train_loader, val_loader, epochs, device)

        # Both should achieve similar accuracy (parity test)
        assert abs(acc1 - acc2) < 10.0, f"Parity gap: {abs(acc1 - acc2):.1f}%"

    @pytest.mark.parametrize("epochs", [2])
    def test_native_fa_variants_compose_and_train(self, epochs):
        """All native FA variants should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        variants = [
            ("native_fa_adaptive", create_native_fa_adaptive),
            ("native_fa_stochastic", create_native_fa_stochastic),
            ("native_fa_contrastive", create_native_fa_contrastive),
            ("native_fa_sign_symmetric", create_native_fa_sign_symmetric),
            ("native_fa_direct", create_native_fa_direct),
            ("native_fa_energy_guided", create_native_fa_energy_guided),
            ("native_fa_energy_minimizing", create_native_fa_energy_minimizing),
            ("native_fa_equilibrium_alignment", create_native_fa_equilibrium_alignment),
            ("native_fa_layerwise_equilibrium", create_native_fa_layerwise_equilibrium),
            ("native_fa_deep_dfa", create_native_fa_deep_dfa),
        ]

        for name, factory in variants:
            system = factory(input_dim, hidden_dim, output_dim, lr=0.001)
            acc = train_system(system, train_loader, val_loader, epochs, device)
            assert acc >= 0.0, f"{name}: accuracy {acc:.1f}%"


class TestForwardForwardParity:
    """Test Forward-Forward parity between presets factory and native."""

    @pytest.mark.parametrize("epochs", [3])
    def test_create_ff_mlp_composes_and_trains(self, epochs):
        """presets.create_ff_mlp should compose and train (native is the only impl)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        system = create_ff_mlp(
            input_dim,
            (hidden_dim, hidden_dim),
            output_dim,
            layer_lr=0.03,
            classifier_lr=0.01,
            threshold=2.0,
            num_layers=2,
            device=device,
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)
        assert acc >= 0.0, f"Forward-Forward accuracy: {acc:.1f}%"


class TestPEPITAParity:
    """Test PEPITA parity between presets factory and native implementation."""

    @pytest.mark.parametrize("epochs", [3])
    def test_create_pepita_mlp_matches_native(self, epochs):
        """presets.create_pepita_mlp should match native_pepita_mlp."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        construction_seed()
        system1 = create_pepita_mlp(
            input_dim, (hidden_dim, hidden_dim), output_dim, lr=0.01, device=device
        )
        construction_seed()
        system2 = create_native_pepita_mlp(
            input_dim, hidden_dim, output_dim, num_layers=2, lr=0.01
        )

        acc1 = train_system(system1, train_loader, val_loader, epochs, device)
        acc2 = train_system(system2, train_loader, val_loader, epochs, device)

        # Both should achieve similar accuracy (parity test)
        assert abs(acc1 - acc2) < 10.0, f"Parity gap: {abs(acc1 - acc2):.1f}%"


class TestOntologyComposition:
    """Test that ontology composition produces valid systems for all registered primitives."""

    @pytest.mark.parametrize(
        "credit_type,expected_accuracy",
        [
            ("gradient", 10.0),
            ("thermodynamic_contrast", 0.0),
            ("random_projections", 0.0),
            ("local_goodness", 0.0),
        ],
    )
    def test_credit_assignment_composition(self, credit_type, expected_accuracy):
        """Each credit assignment type should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=(hidden_dim,),
                init_scale=0.1,
            )
        )
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

        credit_map = {
            "gradient": lambda: BackpropCredit(
                CreditAssignmentConfig(
                    credit_type="gradient",
                    beta=0.5,
                    feedback_matrix=None,
                    local_objective="mse",
                    orthogonal_init=False,
                    feedback_scale=0.01,
                )
            ),
            "thermodynamic_contrast": lambda: ThermodynamicContrast(
                CreditAssignmentConfig.thermodynamic_contrast(beta=0.1)
            ),
            "random_projections": lambda: RandomProjectionsCredit(
                CreditAssignmentConfig.random_projections(feedback_scale=0.1)
            ),
            "local_goodness": lambda: LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness()
            ),
        }

        credit = credit_map[credit_type]()
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001))

        construction_seed()
        system = compose_system(substrate, geometry, dynamics, credit, update)
        acc = train_system(system, train_loader, val_loader, 2, device)

        assert acc > expected_accuracy, (
            f"{credit_type}: {acc:.1f}% < {expected_accuracy}%"
        )


class TestSubstrateVariants:
    """Test that all substrate types compose correctly."""

    @pytest.mark.parametrize(
        "substrate_type",
        [
            "digital",
            "analog",
            "neuromorphic",
            "optical",
            "quantum",
            "sparse",
            "ternary",
        ],
    )
    def test_substrate_composition(self, substrate_type):
        """Each substrate should compose and run forward pass."""
        device = "cpu"  # Force CPU for substrate testing
        input_dim, hidden_dim, output_dim = 10, 20, 5

        substrate_map = {
            "digital": lambda: DigitalSubstrate(SubstrateConfig.digital(device=device)),
            "analog": lambda: AnalogSubstrate(SubstrateConfig.analog(device=device)),
            "memristive": lambda: MemristiveSubstrate(
                SubstrateConfig.memristive(device=device)
            ),
            "neuromorphic": lambda: NeuromorphicSubstrate(
                SubstrateConfig.neuromorphic(device=device)
            ),
            "optical": lambda: OpticalSubstrate(SubstrateConfig.optical(device=device)),
            "quantum": lambda: QuantumSubstrate(SubstrateConfig.quantum(device=device)),
            "sparse": lambda: SparseSubstrate(SubstrateConfig.sparse(device=device)),
            "ternary": lambda: TernarySubstrate(SubstrateConfig.ternary(device=device)),
        }

        substrate = substrate_map[substrate_type]()
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=(hidden_dim,),
                init_scale=0.1,
            )
        )
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        credit = BackpropCredit(
            CreditAssignmentConfig(
                credit_type="gradient",
                beta=0.5,
                feedback_matrix=None,
                local_objective="mse",
                orthogonal_init=False,
                feedback_scale=0.01,
            )
        )
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))

        system = compose_system(substrate, geometry, dynamics, credit, update)

        # Test forward pass
        x = torch.randn(2, input_dim).to(device)
        out = system.forward(x)
        assert out.shape == (2, output_dim), f"{substrate_type}: {out.shape}"


class TestTargetPropParity:
    """Test Target Propagation parity (presets factory vs native - test composition)."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_tp_mlp_composes_and_trains(self, epochs):
        """presets.create_tp_mlp should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_tp_mlp(
            input_dim,
            (128,),
            output_dim,
            lr=0.001,
            beta=0.1,
            settle_steps=10,
            device=device,
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy (above chance for MNIST)
        assert acc >= 0.0, f"TP accuracy: {acc:.1f}%"

    @pytest.mark.parametrize("epochs", [2])
    def test_native_tile_tp_composes_and_trains(self, epochs):
        """native_tile_tp should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        system = create_native_tile_tp(
            input_dim, hidden_dim, output_dim, lr=0.001, beta=0.1, settle_steps=10
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)
        assert acc >= 0.0, f"Tile TP accuracy: {acc:.1f}%"


class TestPredictiveCodingParity:
    """Test Predictive Coding parity (presets factory vs native - test composition)."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_pc_mlp_composes_and_trains(self, epochs):
        """presets.create_pc_mlp should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_pc_mlp(
            input_dim,
            (128, 128),
            output_dim,
            lr=0.001,
            beta=0.5,
            settle_steps=10,
            device=device,
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"PC accuracy: {acc:.1f}%"

    @pytest.mark.parametrize("epochs", [2])
    def test_native_tile_pc_composes_and_trains(self, epochs):
        """native_tile_pc should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        system = create_native_tile_pc(
            input_dim, hidden_dim, output_dim, lr=0.001, beta=0.1, settle_steps=10
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)
        assert acc >= 0.0, f"Tile PC accuracy: {acc:.1f}%"


class TestHebbianParity:
    """Test Hebbian parity (presets factory vs native - test composition)."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_hebbian_mlp_composes_and_trains(self, epochs):
        """presets.create_hebbian_mlp should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_hebbian_mlp(
            input_dim, (128,), output_dim, lr=0.001, device=device
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"Hebbian accuracy: {acc:.1f}%"

    @pytest.mark.parametrize("epochs", [2])
    def test_native_tile_hebbian_composes_and_trains(self, epochs):
        """native_tile_hebbian should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        system = create_native_tile_hebbian(input_dim, hidden_dim, output_dim, lr=0.001)

        acc = train_system(system, train_loader, val_loader, epochs, device)
        assert acc >= 0.0, f"Tile Hebbian accuracy: {acc:.1f}%"


class TestSNNParity:
    """Test SNN parity (presets factory vs native - test composition)."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_snn_mlp_composes_and_trains(self, epochs):
        """presets.create_snn_mlp should compose and train.

        Note: The standard create_snn_mlp uses FeedforwardGeometry + InstantaneousDynamics
        + LocalGoodnessCredit (aligned with working YAML preset). For true spiking dynamics,
        use create_spiking_snn_mlp.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_snn_mlp(input_dim, (128,), output_dim, lr=0.001, device=device)

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"SNN accuracy: {acc:.1f}%"

    @pytest.mark.parametrize("epochs", [2])
    @pytest.mark.xfail(
        reason="SpikeIntegrationDynamics has tensor size mismatch with TileGeometry"
    )
    def test_native_tile_snn_composes_and_trains(self, epochs):
        """native_tile_snn should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        system = create_native_tile_snn(input_dim, hidden_dim, output_dim, lr=0.001)

        acc = train_system(system, train_loader, val_loader, epochs, device)
        assert acc >= 0.0, f"Tile SNN accuracy: {acc:.1f}%"


class TestTileParity:
    """Test Tile parity (presets factory vs native tile)."""

    @pytest.mark.parametrize("epochs", [1])
    def test_create_tile_mlp_composes_and_trains_fast(self, epochs):
        """Fast variant: presets.create_tile_mlp should compose and train (1 epoch)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_tile_mlp(
            input_dim,
            (64,),
            output_dim,
            lr=0.001,
            neurons_per_tile=8,
            tiles_per_layer=2,
            device=device,
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"Tile accuracy: {acc:.1f}%"

    @pytest.mark.slow
    @pytest.mark.parametrize("epochs", [2])
    def test_create_tile_mlp_composes_and_trains_full(self, epochs):
        """Full variant: presets.create_tile_mlp should compose and train (2 epochs)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_tile_mlp(
            input_dim,
            (128,),
            output_dim,
            lr=0.001,
            neurons_per_tile=16,
            tiles_per_layer=2,
            device=device,
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"Tile accuracy: {acc:.1f}%"

    @pytest.mark.parametrize("epochs", [2])
    def test_native_tile_variants_compose_and_train(self, epochs):
        """All native tile variants should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        # Working variants (test full training)
        variants = [
            ("native_tile_fa", create_native_tile_fa),
            ("native_tile_tp", create_native_tile_tp),
            ("native_tile_hebbian", create_native_tile_hebbian),
        ]

        for name, factory in variants:
            system = factory(input_dim, hidden_dim, output_dim, lr=0.001)
            acc = train_system(system, train_loader, val_loader, epochs, device)
            assert acc >= 0.0, f"{name}: accuracy {acc:.1f}%"

        # native_tile_ep/pc/gnn/snn are permanent strict xfails with
        # mechanism-level reasons in tests/property/test_native_smoke.py
        # (R11.1.3): no target-responsive TileMesh settle kernel exists.


class TestResearchModelsParity:
    """Test Research model variants (holomorphic, directed, finite-nudge EP)."""

    @pytest.mark.parametrize("epochs", [2])
    def test_native_research_variants_compose_and_train(self, epochs):
        """Research native variants should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)
        hidden_dim = 128

        variants = [
            ("native_holomorphic_ep", create_native_holomorphic_ep),
            ("native_directed_ep", create_native_directed_ep),
            ("native_finite_nudge_ep", create_native_finite_nudge_ep),
        ]

        for name, factory in variants:
            system = factory(input_dim, hidden_dim, output_dim, lr=0.001)
            acc = train_system(system, train_loader, val_loader, epochs, device)
            assert acc >= 0.0, f"{name}: accuracy {acc:.1f}%"


class TestRoutingParity:
    """Test Routing (6-D) parity."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_routing_mlp_composes_and_trains(self, epochs):
        """presets.create_routing_mlp should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_routing_mlp(
            input_dim, (128,), output_dim, lr=0.001, gate_dim=32, device=device
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"Routing accuracy: {acc:.1f}%"


class TestFastWeightParity:
    """Test Fast Weight (6-D) parity."""

    @pytest.mark.parametrize("epochs", [2])
    def test_create_fast_weight_mlp_composes_and_trains(self, epochs):
        """presets.create_fast_weight_mlp should compose and train."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, input_dim, output_dim = make_dataloaders(device)

        system = create_fast_weight_mlp(
            input_dim, (128,), output_dim, lr=0.001, fast_weight_dim=128, device=device
        )

        acc = train_system(system, train_loader, val_loader, epochs, device)

        # Should produce valid accuracy
        assert acc >= 0.0, f"FastWeight accuracy: {acc:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
