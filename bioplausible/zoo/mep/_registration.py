"""Registration of MEP presets and strategies with the unified Registry."""

from bioplausible.core.registry import (
    ComponentCategory,
    ComputeProfile,
    LocalityLevel,
    Registry,
    register_param_update,
    register_credit_assignment,
)

from .optimizers import DionUpdate, FisherUpdate, MuonUpdate, PlainUpdate
from .presets import local_ep, muon_backprop, natural_ep, sdmep, smep, smep_fast

# Register MEP presets as credit assignments (credit assignment + update combined)
Registry.register(
    ComponentCategory.CREDIT_ASSIGNMENT,
    name="smep",
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(smep)

Registry.register(
    ComponentCategory.CREDIT_ASSIGNMENT,
    name="smep_fast",
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(smep_fast)

Registry.register(
    ComponentCategory.CREDIT_ASSIGNMENT,
    name="sdmep",
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.93,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(sdmep)

Registry.register(
    ComponentCategory.CREDIT_ASSIGNMENT,
    name="local_ep",
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.97,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(local_ep)

Registry.register(
    ComponentCategory.CREDIT_ASSIGNMENT,
    name="natural_ep",
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.90,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(N^2)",
    family="mep",
)(natural_ep)

Registry.register(
    ComponentCategory.CREDIT_ASSIGNMENT,
    name="muon_backprop",
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.3,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(muon_backprop)

# Pure update strategies as param_update (complement the credit assignment presets).
#
# These are registered under ComponentCategory.PARAM_UPDATE: they are gradient
# transformation strategies (no torch.optim parameter/state ownership) rather
# than optimizers. Consumers resolve them via the presets (smep/muon_backprop).
Registry.register(
    ComponentCategory.PARAM_UPDATE,
    name="muon",
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(MuonUpdate)

Registry.register(
    ComponentCategory.PARAM_UPDATE,
    name="dion",
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(DionUpdate)

Registry.register(
    ComponentCategory.PARAM_UPDATE,
    name="plain",
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(PlainUpdate)

Registry.register(
    ComponentCategory.PARAM_UPDATE,
    name="fisher",
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N^2)",
    family="mep",
)(FisherUpdate)


# PARAM_UPDATE-category registrations: expose the EP presets to ``CoreTrainer`` so
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
    register_param_update(
        _name,
        locality_level=LocalityLevel.EQUILIBRIUM,
        compute_profile=ComputeProfile.GPU,
        bio_plausibility_score=0.9,
        credit_assignment_type="equilibrium",
        requires_backward=False,
        memory_complexity="O(1)",
        family="mep",
    )(_ep_optimizer_factory(_preset))
