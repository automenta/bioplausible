"""Registration of MEP presets and strategies with the unified Registry."""

from bioplausible.core.registry import (
    ComponentCategory,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
    register_optimizer,
)

from .optimizers import DionUpdate, FisherUpdate, MuonUpdate, PlainUpdate
from .presets import local_ep, muon_backprop, natural_ep, sdmep, smep, smep_fast

# Register MEP presets as propagators (credit assignment + update combined)
Registry.register(
    ComponentCategory.PROPAGATOR,
    name="smep",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(smep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="smep_fast",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(smep_fast)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="sdmep",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.93,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(sdmep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="local_ep",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.97,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(local_ep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="natural_ep",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.90,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(N^2)",
    family="mep",
)(natural_ep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="muon_backprop",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM, Domain.RL],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.3,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(muon_backprop)

# Pure update strategies as optimizers (complement the propagator presets).
#
# As of the category-correctness sprint these are registered under
# ComponentCategory.UPDATE_STRATEGY, NOT OPTIMIZER: they are gradient
# transformation strategies (no torch.optim parameter/state ownership) rather
# than optimizers. Consumers resolve them via the presets (smep/muon_backprop).
Registry.register(
    ComponentCategory.UPDATE_STRATEGY,
    name="muon",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM, Domain.RL],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(MuonUpdate)

Registry.register(
    ComponentCategory.UPDATE_STRATEGY,
    name="dion",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(DionUpdate)

Registry.register(
    ComponentCategory.UPDATE_STRATEGY,
    name="plain",
    domains=[
        Domain.VISION,
        Domain.TABULAR,
        Domain.LM,
        Domain.RL,
        Domain.GRAPH,
        Domain.TIMESERIES,
    ],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(PlainUpdate)

Registry.register(
    ComponentCategory.UPDATE_STRATEGY,
    name="fisher",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N^2)",
    family="mep",
)(FisherUpdate)


# OPTIMIZER-category registrations: expose the EP presets to ``CoreTrainer`` so
# ``optimizer="smep"`` (etc.) drives the learning-rule path. The wrapper forces
# ``mode="ep"`` by default so ``dispatch_train_step``'s ``step(x, target)`` call
# computes gradients — the raw preset defaults to backprop mode, which would be a
# silent no-op under the learning-rule calling convention. An explicit ``mode``
# passed via ``optimizer_kwargs`` still wins (setdefault).
def _ep_optimizer_factory(preset):
    def factory(params, model=None, **kwargs):
        kwargs.setdefault("mode", "ep")
        return preset(params, model=model, **kwargs)

    return factory


for _name, _preset in (
    ("smep", smep),
    ("smep_fast", smep_fast),
    ("sdmep", sdmep),
    ("local_ep", local_ep),
    ("natural_ep", natural_ep),
):
    register_optimizer(
        _name,
        domains=[Domain.VISION, Domain.TABULAR],
        locality_level=LocalityLevel.EQUILIBRIUM,
        compute_profile=ComputeProfile.GPU,
        bio_plausibility_score=0.9,
        credit_assignment_type="equilibrium",
        requires_backward=False,
        memory_complexity="O(1)",
        family="mep",
    )(_ep_optimizer_factory(_preset))
