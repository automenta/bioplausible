#!/usr/bin/env python
"""Generate machine-readable API schema from type hints.

Usage:
    uv run scripts/generate_api_schema.py --output api_schema.json

Outputs a JSON schema describing the public API of computronium,
including classes, functions, protocols, and their type signatures.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, get_type_hints

# Modules to scan for public API
PUBLIC_MODULES = [
    "computronium.core.ontology",
    "computronium.core.system_trainer",
    "computronium.core.joint.transition",
    "computronium.core.joint.state",
    "computronium.core.joint.context",
    "computronium.core.substrates.digital_substrate",
    "computronium.core.substrates.analog_substrate",
    "computronium.core.substrates.memristive_substrate",
    "computronium.core.substrates.neuromorphic_substrate",
    "computronium.core.substrates.optical_substrate",
    "computronium.core.substrates.quantum_substrate",
    "computronium.core.substrates.sparse_substrate",
    "computronium.core.substrates.ternary_substrate",
    "computronium.core.plasticity.routing",
    "computronium.core.plasticity.fast_weights",
    "computronium.core.plasticity.rule_state",
    "computronium.core.plasticity.substrate_coupled",
    "computronium",
]


def _get_type_repr(annotation: Any) -> str:
    """Get string representation of a type annotation."""
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return "Any"

    if hasattr(annotation, "__origin__"):
        # Generic types like list[int], dict[str, int], etc.
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if origin is None:
            return str(annotation)

        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            arg_strs = [_get_type_repr(a) for a in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return origin_name

    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation)


def _extract_dataclass_fields(cls: type) -> list[dict]:
    """Extract field information from a dataclass."""
    from dataclasses import MISSING

    result = []
    for field in fields(cls):
        default_val = field.default
        if default_val is MISSING and field.default_factory is MISSING:
            default_val = None
        elif default_val is MISSING:
            default_val = "<factory>"
        field_info = {
            "name": field.name,
            "type": _get_type_repr(field.type),
            "default": default_val,
            "kw_only": getattr(field, "kw_only", False),
        }
        result.append(field_info)
    return result


def _extract_function_signature(obj: Any) -> dict:
    """Extract function/method signature information."""
    try:
        sig = inspect.signature(obj)
        try:
            type_hints = get_type_hints(obj)
        except NameError, AttributeError:
            # Handle forward references that can't be resolved
            type_hints = {}
    except ValueError, TypeError:
        return {"error": "Could not inspect signature"}

    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        param_info = {
            "name": name,
            "kind": param.kind.name,
            "type": _get_type_repr(type_hints.get(name, param.annotation)),
        }
        if param.default is not param.empty:
            param_info["default"] = param.default
        params.append(param_info)

    return_type = _get_type_repr(type_hints.get("return", sig.return_annotation))

    return {
        "params": params,
        "return_type": return_type,
    }


def _extract_class_info(cls: type) -> dict:
    """Extract information about a class."""
    info = {
        "name": cls.__name__,
        "module": cls.__module__,
        "docstring": inspect.getdoc(cls) or "",
        "is_dataclass": hasattr(cls, "__dataclass_fields__"),
        "is_protocol": hasattr(cls, "__protocol_attrs__"),
        "bases": [b.__name__ for b in cls.__bases__ if b != object],
        "methods": {},
        "fields": [],
    }

    # Dataclass fields
    if info["is_dataclass"]:
        info["fields"] = _extract_dataclass_fields(cls)

    # Methods
    for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        info["methods"][name] = _extract_function_signature(obj)

    # Class methods
    for name, obj in inspect.getmembers(
        cls, predicate=lambda x: isinstance(x, classmethod)
    ):
        if name.startswith("_"):
            continue
        info["methods"][name] = _extract_function_signature(obj.__func__)
        info["methods"][name]["is_classmethod"] = True

    return info


def _extract_protocol_info(proto: type) -> dict:
    """Extract information about a Protocol."""
    info = {
        "name": proto.__name__,
        "module": proto.__module__,
        "docstring": inspect.getdoc(proto) or "",
        "methods": {},
    }

    for name in getattr(proto, "__protocol_attrs__", []):
        if name.startswith("_"):
            continue
        method = getattr(proto, name, None)
        if method:
            info["methods"][name] = _extract_function_signature(method)

    return info


def _scan_module(module_name: str) -> dict:
    """Scan a module for public API elements."""
    try:
        module = __import__(module_name, fromlist=["*"])
    except ImportError as e:
        return {"error": str(e)}

    result = {
        "module": module_name,
        "classes": {},
        "functions": {},
        "protocols": {},
        "type_aliases": {},
        "constants": {},
    }

    # Get __all__ if defined
    all_names = getattr(module, "__all__", None)

    for name in dir(module):
        if name.startswith("_"):
            continue
        if all_names is not None and name not in all_names:
            continue

        obj = getattr(module, name)

        # Classes
        if inspect.isclass(obj):
            if obj.__module__ != module_name:
                continue  # Skip imported classes

            # Check if it's a Protocol
            if hasattr(obj, "__protocol_attrs__"):
                result["protocols"][name] = _extract_protocol_info(obj)
            else:
                result["classes"][name] = _extract_class_info(obj)

        # Functions
        elif inspect.isfunction(obj):
            if obj.__module__ != module_name:
                continue
            result["functions"][name] = _extract_function_signature(obj)

        # Type aliases (variables with type annotations)
        elif isinstance(obj, type) and hasattr(obj, "__origin__"):
            result["type_aliases"][name] = _get_type_repr(obj)

    return result


def generate_schema(output_path: Path, modules: list[str] | None = None):
    """Generate the complete API schema."""
    modules = modules or PUBLIC_MODULES

    schema = {
        "version": "1.0",
        "generated_from": "computronium",
        "modules": {},
    }

    for module_name in modules:
        print(f"Scanning {module_name}...")
        module_data = _scan_module(module_name)
        if "error" not in module_data:
            schema["modules"][module_name] = module_data
        else:
            print(f"  Warning: {module_data['error']}")

    # Write output
    with output_path.open("w") as f:
        json.dump(schema, f, indent=2)

    print(f"\nSchema written to {output_path}")
    print(f"Modules scanned: {len(schema['modules'])}")

    # Print summary
    total_classes = sum(len(m.get("classes", {})) for m in schema["modules"].values())
    total_functions = sum(
        len(m.get("functions", {})) for m in schema["modules"].values()
    )
    total_protocols = sum(
        len(m.get("protocols", {})) for m in schema["modules"].values()
    )
    print(
        f"Classes: {total_classes}, Functions: {total_functions}, Protocols: {total_protocols}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate API schema from type hints")
    parser.add_argument(
        "--output",
        default="api_schema.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific modules to scan (default: all public modules)",
    )

    args = parser.parse_args()

    generate_schema(Path(args.output), args.modules)


if __name__ == "__main__":
    main()
