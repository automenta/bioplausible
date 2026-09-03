"""
Model Exporter - High-level export interface.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.deployment.serialization import ModelExporter, ModelInfo

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Configuration for model export."""

    output_dir: str = "./exports"
    formats: list[str] | None = None
    input_shape: tuple[int, ...] = (1, 784)
    verbose: bool = True


def export_model(
    model: nn.Module,
    model_name: str,
    model_params: dict[str, object],
    config: ExportConfig | None = None,
    optimizer: object | None = None,
    training_metrics: dict[str, float] | None = None,
) -> ModelInfo:
    """
    Convenience function to export a model.

    Args:
        model: Model to export.
        model_name: Name of the model.
        model_params: Model parameters.
        config: Export configuration.
        optimizer: Optional optimizer.
        training_metrics: Training metrics.

    Returns:
        ModelInfo with export details.
    """
    config = config or ExportConfig()
    exporter = ModelExporter()
    return exporter.export(
        model=model,
        model_name=model_name,
        model_params=model_params,
        output_dir=config.output_dir,
        formats=config.formats,
        optimizer=optimizer,
        training_metrics=training_metrics,
        input_shape=config.input_shape,
        verbose=config.verbose,
    )


def load_model(
    export_dir: str,
    device: str = "auto",
) -> tuple[nn.Module, dict[str, object]]:
    """
    Convenience function to load a model.

    Args:
        export_dir: Export directory with config.json.
        device: Device for model.

    Returns:
        Tuple of (model, config).
    """
    from computronium.deployment.serialization import ModelLoader

    loader = ModelLoader(device="cpu")
    config_path = str(Path(export_dir) / "config.json")
    return loader.load_from_config(config_path)


__all__ = [
    "ExportConfig",
    "ModelExporter",
    "ModelInfo",
    "export_model",
    "load_model",
]
