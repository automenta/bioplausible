#!/usr/bin/env python3
"""Inventory & Parity Baseline for EqProp model family migration.

Identifies all eqprop-family models in the registry and validates
their 5-D ontology projection against legacy behavior.
"""

from __future__ import annotations

import torch
from computronium.core.registry import Registry, ComponentCategory
from computronium.core.ontology import ModelAdapter

# Import all model modules to populate registry
import computronium.zoo.models.eqprop  # noqa: F401
import computronium.zoo.models  # noqa: F401


def _instantiate_model(model_cls, name: str):
    """Try to instantiate a model with appropriate arguments."""
    # Try with input_dim, hidden_dim, output_dim
    try:
        return model_cls(input_dim=32, hidden_dim=32, output_dim=10)
    except TypeError:
        pass

    # Try with config
    try:
        if hasattr(model_cls, "config_class"):
            config = model_cls.config_class()
            return model_cls(config)
    except Exception:
        pass

    # Try with ModelConfig if available
    try:
        from computronium.zoo.models.base import ModelConfig
        config = ModelConfig(input_dim=32, hidden_dim=32, output_dim=10)
        return model_cls(config)
    except Exception:
        pass

    # Try no args
    try:
        return model_cls()
    except Exception:
        pass

    raise TypeError(f"Could not instantiate {name}")


def main() -> None:
    """Run inventory and parity check for eqprop models."""
    print("=" * 80)
    print("EqProp Family Migration Inventory & Parity Baseline")
    print("=" * 80)

    # Get all registered models
    models_dict = Registry.list(ComponentCategory.MODEL)
    models = models_dict.get("model", [])
    eqprop_models = [m for m in models if m.startswith("eqprop") or "ep" in m.lower()]

    print(f"\nTotal registered models: {len(models)}")
    print(f"EqProp family models: {len(eqprop_models)}")
    print("-" * 80)

    for name in sorted(eqprop_models):
        print(f"\nChecking: {name}")
        try:
            model_cls = Registry.get(ComponentCategory.MODEL, name)
            metadata = Registry.get_metadata(ComponentCategory.MODEL, name)

            # Instantiate model
            model = _instantiate_model(model_cls, name)

            # Create adapter and validate
            adapter = ModelAdapter(model, metadata)
            result = adapter.validate(rtol=0.1, atol=1e-2)

            status = "PASS" if result["passed"] else "FAIL"
            diffs = len(result.get("differences", {}))
            print(f"  {status}: passed={result['passed']}, diffs={diffs}")

            if not result["passed"]:
                for key, diff in result.get("differences", {}).items():
                    if isinstance(diff, dict) and "rel_diff" in diff:
                        print(f"    {key}: rel_diff={diff['rel_diff']:.4f}, abs_diff={diff['abs_diff']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print("Inventory complete")
    print("=" * 80)


if __name__ == "__main__":
    main()