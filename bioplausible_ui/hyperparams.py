"""
Hyperparameter Schemas for Bioplausible Trainer

Defines model-specific hyperparameters that appear dynamically in the UI.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec, ModelSpec


@dataclass
class HyperparamSpec:
    """Specification for a single hyperparameter."""
    name: str
    label: str
    type: str  # 'int', 'float', 'bool'
    default: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    description: str = ""


class HyperparamRegistry:
    """Registry for hyperparameter schemas organized by model type."""

    def __init__(self):
        self.schemas: Dict[str, List[HyperparamSpec]] = {}
        self._initialize_schemas()

    def _initialize_schemas(self):
        """Initialize the hyperparameter schemas."""
        # Define all schemas in a structured way
        self._register_eqprop_schemas()
        self._register_feedback_alignment_schemas()
        self._register_standard_model_schemas()
        self._register_lm_variant_schemas()
        self._register_hebbian_schemas()
        self._register_hybrid_schemas()

    def _register_eqprop_schemas(self):
        """Register EqProp-related hyperparameter schemas."""
        # Standard EqProp models
        self.schemas['eqprop'] = [
            HyperparamSpec(
                name='beta',
                label='Beta (Nudge Strength)',
                type='float',
                default=0.2,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                description='Nudging strength for contrastive phase (0 = no nudging)'
            ),
            HyperparamSpec(
                name='eq_steps',
                label='Equilibrium Steps',
                type='int',
                default=30,
                min_val=5,
                max_val=100,
                step=5,
                description='Number of equilibrium settling iterations'
            ),
            HyperparamSpec(
                name='alpha',
                label='Alpha (Damping)',
                type='float',
                default=0.5,
                min_val=0.0,
                max_val=1.0,
                step=0.05,
                description='Damping factor for equilibrium updates'
            ),
        ]

        # Momentum-based EqProp
        self.schemas['mom_eq'] = [
            HyperparamSpec(
                name='beta',
                label='Beta',
                type='float',
                default=0.2,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
            ),
            HyperparamSpec(
                name='momentum',
                label='Momentum',
                type='float',
                default=0.9,
                min_val=0.0,
                max_val=0.99,
                step=0.01,
                description='Momentum coefficient for equilibrium updates'
            ),
        ]

        # Sparse EqProp
        self.schemas['sparse_eq'] = [
            HyperparamSpec(
                name='beta',
                label='Beta',
                type='float',
                default=0.2,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
            ),
            HyperparamSpec(
                name='sparsity',
                label='Sparsity Target',
                type='float',
                default=0.1,
                min_val=0.01,
                max_val=0.5,
                step=0.01,
                description='Target sparsity level (fraction of active neurons)'
            ),
        ]
        # Holomorphic EP
        self.schemas['holomorphic_ep'] = [
            HyperparamSpec(
                name='beta', label='Beta', type='float', default=0.1, min_val=0.0, max_val=1.0, step=0.01
            ),
            HyperparamSpec(
                name='steps', label='Steps', type='int', default=30, min_val=5, max_val=100, step=5
            ),
        ]
        # Directed EP
        self.schemas['directed_ep'] = [
            HyperparamSpec(
                name='beta', label='Beta', type='float', default=0.2, min_val=0.0, max_val=1.0, step=0.01
            ),
            HyperparamSpec(
                name='steps', label='Steps', type='int', default=30, min_val=5, max_val=100, step=5
            ),
        ]
        # Finite Nudge EP
        self.schemas['finite_nudge_ep'] = [
            HyperparamSpec(
                name='beta', label='Beta (Finite)', type='float', default=1.0, min_val=0.1, max_val=5.0, step=0.1
            ),
            HyperparamSpec(
                name='steps', label='Steps', type='int', default=30, min_val=5, max_val=100, step=5
            ),
        ]

    def _register_feedback_alignment_schemas(self):
        """Register feedback alignment hyperparameter schemas."""
        # Feedback Alignment variants
        self.schemas['feedback_align'] = [
            HyperparamSpec(
                name='fa_scale',
                label='FA Feedback Scale',
                type='float',
                default=1.0,
                min_val=0.1,
                max_val=2.0,
                step=0.1,
                description='Scale of random feedback alignment matrix'
            ),
        ]

        # Adaptive FA
        self.schemas['ada_fa'] = [
            HyperparamSpec(
                name='fa_scale',
                label='FA Feedback Scale',
                type='float',
                default=1.0,
                min_val=0.1,
                max_val=2.0,
                step=0.1,
                description='Initial scale of feedback alignment matrix'
            ),
            HyperparamSpec(
                name='adapt_rate',
                label='Adaptation Rate',
                type='float',
                default=0.01,
                min_val=0.001,
                max_val=0.1,
                step=0.001,
                description='Rate of feedback matrix adaptation'
            ),
        ]
        self.schemas['stochastic_fa'] = [
            HyperparamSpec(name='noise_scale', label='Noise Scale', type='float', default=0.1, min_val=0.0, max_val=1.0, step=0.05)
        ]
        self.schemas['energy_guided_fa'] = [
            HyperparamSpec(name='energy_weight', label='Energy Weight', type='float', default=0.1, min_val=0.0, max_val=1.0, step=0.05)
        ]
        self.schemas['energy_minimizing_fa'] = [
             HyperparamSpec(name='lr_energy', label='Energy LR', type='float', default=0.01, min_val=0.001, max_val=0.1, step=0.001)
        ]
        self.schemas['dfa'] = [
             HyperparamSpec(name='feedback_scale', label='Feedback Scale', type='float', default=1.0, min_val=0.1, max_val=2.0, step=0.1)
        ]

        # Equilibrium + Alignment hybrid
        self.schemas['eq_align'] = [
            HyperparamSpec(
                name='beta',
                label='Beta (Nudge)',
                type='float',
                default=0.2,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
            ),
            HyperparamSpec(
                name='eq_steps',
                label='Eq Steps',
                type='int',
                default=20,
                min_val=5,
                max_val=50,
                step=5,
            ),
            HyperparamSpec(
                name='align_weight',
                label='Alignment Weight',
                type='float',
                default=0.5,
                min_val=0.0,
                max_val=1.0,
                step=0.05,
                description='Weight for gradient alignment loss'
            ),
        ]

    def _register_standard_model_schemas(self):
        """Register standard model hyperparameter schemas."""
        # Standard models (LoopedMLP, ConvEqProp, etc.)
        self.schemas['looped_mlp'] = [
            HyperparamSpec(
                name='max_steps',
                label='Max Steps',
                type='int',
                default=30,
                min_val=5,
                max_val=100,
                step=5,
                description='Maximum equilibrium iterations'
            ),
        ]

        self.schemas['conv_eqprop'] = [
            HyperparamSpec(
                name='gamma',
                label='Gamma (Damping)',
                type='float',
                default=0.5,
                min_val=0.1,
                max_val=1.0,
                step=0.05,
                description='Damping factor for convolutional layers'
            ),
        ]

    def _register_lm_variant_schemas(self):
        """Register language model variant hyperparameter schemas."""
        # Transformer LM variants
        self.schemas['transformer_lm'] = [
            HyperparamSpec(
                name='eq_steps',
                label='Eq Steps',
                type='int',
                default=15,
                min_val=5,
                max_val=50,
                step=5,
            ),
            HyperparamSpec(
                name='alpha',
                label='Alpha',
                type='float',
                default=0.5,
                min_val=0.0,
                max_val=1.0,
                step=0.05,
            ),
        ]

    def _register_hebbian_schemas(self):
        """Register Hebbian learning schemas."""
        self.schemas['chl'] = [
            HyperparamSpec(name='beta', label='Beta', type='float', default=0.1, min_val=0.0, max_val=1.0, step=0.01),
            HyperparamSpec(name='steps', label='Steps', type='int', default=20, min_val=5, max_val=50, step=5)
        ]
        self.schemas['deep_hebbian'] = [
             HyperparamSpec(name='hebbian_lr', label='Hebbian LR', type='float', default=0.01, min_val=0.0001, max_val=0.1, step=0.0001)
        ]

    def _register_hybrid_schemas(self):
        """Register hybrid algorithm schemas."""
        self.schemas['predictive_coding_hybrid'] = [
             HyperparamSpec(name='pred_steps', label='Pred Steps', type='int', default=20, min_val=5, max_val=50, step=5),
             HyperparamSpec(name='pred_lr', label='Pred LR', type='float', default=0.1, min_val=0.01, max_val=1.0, step=0.01)
        ]
        self.schemas['layerwise_equilibrium_fa'] = [
             HyperparamSpec(name='steps', label='Steps', type='int', default=20, min_val=5, max_val=50, step=5)
        ]

    def get_schema(self, model_type: str) -> List[HyperparamSpec]:
        """
        Get hyperparameter schema for a model type.

        Args:
            model_type: Type of model to get hyperparameters for

        Returns:
            List of HyperparamSpec objects
        """
        return self.schemas.get(model_type, [])

    def register_schema(self, model_type: str, specs: List[HyperparamSpec]):
        """
        Register a new hyperparameter schema.

        Args:
            model_type: Name of the model type
            specs: List of hyperparameter specifications
        """
        self.schemas[model_type] = specs


# Global registry instance
HYPERPARAM_REGISTRY = HyperparamRegistry()


def get_hyperparams_for_model(model_name: str) -> List[HyperparamSpec]:
    """
    Get hyperparameter specs for a given model name.

    Dynamically resolves schema based on MODEL_REGISTRY.

    Args:
        model_name: Model or algorithm name (can include description)

    Returns:
        List of HyperparamSpec objects
    """
    # 1. Try to find exact spec in registry
    spec: Optional[ModelSpec] = None

    # Check if exact name
    for s in MODEL_REGISTRY:
        if s.name == model_name:
            spec = s
            break

    # Check "Key - Description" format
    if spec is None and ' - ' in model_name:
        key = model_name.split(' - ')[0]
        for s in MODEL_REGISTRY:
            if s.name == key: # Some registry entries might just be the key
                spec = s
                break

    # Check legacy/UI mapping if still not found
    if spec is None:
        # Map legacy names to model_types if possible, or fallback to key logic
        # Legacy: "LoopedMLP" -> model_type "eqprop_mlp"
        legacy_map = {
            "LoopedMLP": "eqprop_mlp",
            "ConvEqProp": "modern_conv_eqprop",
            "FullEqProp Transformer": "eqprop_transformer",
            "Attention-Only EqProp": "eqprop_transformer",
            "BackpropMLP (baseline)": "backprop",
            "Backprop Baseline": "backprop"
        }

        target_type = legacy_map.get(model_name)
        if target_type:
             # Find a spec with this type
             for s in MODEL_REGISTRY:
                 if s.model_type == target_type:
                     spec = s
                     break

    # 2. Get detailed schema from HYPERPARAM_REGISTRY using model_type
    if spec:
        # Try specific model type first
        schema = HYPERPARAM_REGISTRY.get_schema(spec.model_type)
        if schema:
            return schema

        # Fallback: Construct default schema based on flags
        return _create_default_schema(spec)

    # 3. Fallback for completely unknown models (legacy behavior)
    # Extract algorithm key from formatted names like "eqprop - Description"
    if ' - ' in model_name:
        key = model_name.split(' - ')[0].lower()
    else:
        key = model_name.lower().replace(' ', '_')

    # Map UI names to schema keys
    key_mappings = {
        'loopedmlp': 'looped_mlp',
        'conveqprop': 'conv_eqprop',
        'fulleqprop_transformer': 'transformer_lm',
        'attention-only_eqprop': 'transformer_lm',
        'recurrent_core_eqprop': 'transformer_lm',
        'hybrid_eqprop': 'transformer_lm',
        'loopedmlp_lm': 'looped_mlp',
        'backpropmlp_(baseline)': 'standard',
        'standardeqprop': 'eqprop',
    }

    key = key_mappings.get(key, key)
    return HYPERPARAM_REGISTRY.get_schema(key)


def _create_default_schema(spec: ModelSpec) -> List[HyperparamSpec]:
    """Create a default schema based on ModelSpec flags."""
    schema = []

    if spec.has_beta:
        schema.append(HyperparamSpec(
            name='beta',
            label='Beta (Nudge)',
            type='float',
            default=spec.default_beta,
            min_val=0.0,
            max_val=1.0,
            step=0.05,
            description='Nudging strength'
        ))

    if spec.has_steps:
        schema.append(HyperparamSpec(
            name='steps', # Note: Using 'steps' vs 'eq_steps' requires care. Dashboard/Experiment uses 'steps'.
            label='Equilibrium Steps',
            type='int',
            default=spec.default_steps,
            min_val=5,
            max_val=100,
            step=5,
            description='Number of equilibrium iterations'
        ))

    return schema


def hyperparams_to_dict(specs: List[HyperparamSpec]) -> Dict[str, Any]:
    """
    Convert list of specs to dict of default values.

    Args:
        specs: List of hyperparameter specifications

    Returns:
        Dictionary mapping parameter names to their default values
    """
    return {spec.name: spec.default for spec in specs}