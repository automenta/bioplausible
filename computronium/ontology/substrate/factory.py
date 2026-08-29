"""Substrate factory for creating substrates from configuration."""

from computronium.ontology._substrate import (
    AnalogSubstrate,
    DigitalSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    NoisySubstrate,
    OpticalSubstrate,
    QuantizedSubstrate,
    QuantumSubstrate,
    SparseSubstrate,
    Substrate,
    SubstrateConfig,
    TernarySubstrate,
)


def substrate_from_config(config: SubstrateConfig) -> Substrate:
    """Create a Substrate instance from its configuration.

    Args:
        config: SubstrateConfig specifying the substrate type and parameters.

    Returns:
        A Substrate instance matching the configuration.
    """
    match config.substrate_type:
        case "digital":
            return DigitalSubstrate(config)
        case "analog":
            return AnalogSubstrate(config)
        case "memristive":
            return MemristiveSubstrate(config)
        case "neuromorphic":
            return NeuromorphicSubstrate(config)
        case "optical":
            return OpticalSubstrate(config)
        case "quantum":
            return QuantumSubstrate(config)
        case "sparse":
            return SparseSubstrate(config)
        case "ternary":
            return TernarySubstrate(config)
        case "quantized":
            return QuantizedSubstrate(config)
        case "noisy":
            return NoisySubstrate(config)
        case _:
            raise ValueError(f"Unknown substrate type: {config.substrate_type}")


__all__ = ["substrate_from_config"]
