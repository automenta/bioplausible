"""Native model registration with explicit 5-D ontology axis metadata.

This module registers all native models with the Registry, providing
deterministic 5-D ontology layer assignments that bypass ModelAdapter heuristics.
"""

from computronium.core.registry import (
    ComponentCategory,
    ComputeProfile,
    LocalityLevel,
    register_model,
)

# Import the factory functions to register them
from computronium.models.native.backprop_native import native_backprop_mlp
from computronium.models.native.diffusion_eqprop_native import native_diffusion_eqprop
from computronium.models.native.eqprop_native import native_eqprop_mlp
from computronium.models.native.fa_native import native_fa_mlp
from computronium.models.native.momentum_eqprop_native import native_momentum_eqprop
from computronium.models.native.pepita_native import native_pepita_mlp
from computronium.models.native.research_native import (
    native_directed_ep,
    native_finite_nudge_ep,
    native_holomorphic_ep,
)
from computronium.models.native.sparse_eqprop_native import native_sparse_eqprop
from computronium.models.native.ternary_eqprop_native import native_ternary_eqprop
from computronium.models.native.tile_native import (
    native_tile_ep,
    native_tile_fa,
    native_tile_snn,
    native_tile_tp,
)

# Register native Equilibrium Propagation models
register_model(
    "native_eqprop_mlp",
    family="eqprop",
    domain="general",
    tags=["native", "equilibrium", "energy-based"],
    bio_plausibility_score=0.9,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_eqprop_mlp)

register_model(
    "native_diffusion_eqprop",
    family="eqprop",
    domain="general",
    tags=["native", "equilibrium", "diffusion", "stochastic"],
    bio_plausibility_score=0.95,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="DiffusionDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_diffusion_eqprop)

register_model(
    "native_momentum_eqprop",
    family="eqprop",
    domain="general",
    tags=["native", "equilibrium", "momentum", "accelerated"],
    bio_plausibility_score=0.9,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_momentum_eqprop)

register_model(
    "native_sparse_eqprop",
    family="eqprop",
    domain="general",
    tags=["native", "equilibrium", "sparse", "n:m", "structured"],
    bio_plausibility_score=0.9,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="SparseSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_sparse_eqprop)

register_model(
    "native_ternary_eqprop",
    family="eqprop",
    domain="general",
    tags=["native", "equilibrium", "ternary", "quantized", "ste"],
    bio_plausibility_score=0.85,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="TernarySubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_ternary_eqprop)

# Register native Feedback Alignment models
register_model(
    "native_fa_mlp",
    family="fa",
    domain="general",
    tags=["native", "feedback-alignment", "random-projections"],
    bio_plausibility_score=0.7,
    locality_level=LocalityLevel.GLOBAL,
    credit_assignment_type="feedback_alignment",
    compute_profile=ComputeProfile.GPU,
    requires_backward=True,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="FeedforwardGeometry",
    ontology_dynamics="InstantaneousDynamics",
    ontology_credit="RandomProjectionsCredit",
    ontology_update="EuclideanUpdate",
)(native_fa_mlp)

# Register native Backprop model
register_model(
    "native_backprop_mlp",
    family="backprop",
    domain="general",
    tags=["native", "backprop", "gradient"],
    bio_plausibility_score=0.1,
    locality_level=LocalityLevel.GLOBAL,
    credit_assignment_type="gradient",
    compute_profile=ComputeProfile.GPU,
    requires_backward=True,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="FeedforwardGeometry",
    ontology_dynamics="InstantaneousDynamics",
    ontology_credit="BackpropCredit",
    ontology_update="EuclideanUpdate",
)(native_backprop_mlp)

# Register native PEPITA model
register_model(
    "native_pepita_mlp",
    family="forward_only",
    domain="general",
    tags=["native", "forward-only", "local", "pepita"],
    bio_plausibility_score=0.85,
    locality_level=LocalityLevel.FORWARD_ONLY,
    credit_assignment_type="forward-only",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(1)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="FeedforwardGeometry",
    ontology_dynamics="InstantaneousDynamics",
    ontology_credit="LocalGoodnessCredit",
    ontology_update="EuclideanUpdate",
)(native_pepita_mlp)

# Register native Tile models
register_model(
    "native_tile_ep",
    family="equitile",
    domain="general",
    tags=["native", "tile", "equilibrium", "energy-based"],
    bio_plausibility_score=0.9,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="TileGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_tile_ep)

register_model(
    "native_tile_fa",
    family="equitile",
    domain="general",
    tags=["native", "tile", "feedback-alignment", "random-projections"],
    bio_plausibility_score=0.75,
    locality_level=LocalityLevel.GLOBAL,
    credit_assignment_type="feedback_alignment",
    compute_profile=ComputeProfile.GPU,
    requires_backward=True,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="TileGeometry",
    ontology_dynamics="InstantaneousDynamics",
    ontology_credit="RandomProjectionsCredit",
    ontology_update="EuclideanUpdate",
)(native_tile_fa)

register_model(
    "native_tile_tp",
    family="equitile",
    domain="general",
    tags=["native", "tile", "target-prop", "predictive"],
    bio_plausibility_score=0.8,
    locality_level=LocalityLevel.LAYERWISE,
    credit_assignment_type="target",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="TileGeometry",
    ontology_dynamics="PredictiveSettlingDynamics",
    ontology_credit="TargetInversionCredit",
    ontology_update="EuclideanUpdate",
)(native_tile_tp)

register_model(
    "native_tile_snn",
    family="equitile",
    domain="general",
    tags=["native", "tile", "spiking", "neuromorphic"],
    bio_plausibility_score=0.95,
    locality_level=LocalityLevel.LOCAL,
    credit_assignment_type="spiking",
    compute_profile=ComputeProfile.NEUROMORPHIC,
    requires_backward=False,
    memory_complexity="O(1)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="TileGeometry",
    ontology_dynamics="SpikeIntegrationDynamics",
    ontology_credit="LocalGoodnessCredit",
    ontology_update="EuclideanUpdate",
)(native_tile_snn)

# Register native Research models
register_model(
    "native_holomorphic_ep",
    family="eqprop",
    domain="research",
    tags=["native", "equilibrium", "complex", "holomorphic", "quantum"],
    bio_plausibility_score=0.9,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="QuantumSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_holomorphic_ep)

register_model(
    "native_directed_ep",
    family="eqprop",
    domain="research",
    tags=["native", "equilibrium", "directed", "feedback-alignment"],
    bio_plausibility_score=0.85,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="feedback_alignment",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="RandomProjectionsCredit",
    ontology_update="EuclideanUpdate",
)(native_directed_ep)

register_model(
    "native_finite_nudge_ep",
    family="eqprop",
    domain="research",
    tags=["native", "equilibrium", "finite-nudge", "large-beta"],
    bio_plausibility_score=0.85,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    compute_profile=ComputeProfile.GPU,
    requires_backward=False,
    memory_complexity="O(N)",
    ontology_substrate="DigitalSubstrate",
    ontology_geometry="RecurrentGeometry",
    ontology_dynamics="EnergyMinimizationDynamics",
    ontology_credit="ThermodynamicContrast",
    ontology_update="EuclideanUpdate",
)(native_finite_nudge_ep)