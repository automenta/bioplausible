"""SystemModule: drop-in ``nn.Module`` facade over a composed 5-D System."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from computronium.ontology import System


class SystemModule(nn.Module):
    """Wrap a composed :class:`~computronium.ontology.System` as an nn.Module.

    Inference is plain PyTorch: ``forward`` runs the system's free phase, so
    the wrapper composes with ``torch.no_grad``/``eval`` exactly like any
    other module. Training is *not* optimizer-driven — credit assignment is
    internal to the system — so step via :meth:`fit_step` instead of
    ``loss.backward()``.

    Example:
        model = SystemModule(compose_system(...))
        for x, y in loader:
            metrics = model.fit_step(x, y)
        with torch.no_grad():
            logits = model(x)
    """

    def __init__(self, system: System):
        super().__init__()
        self.system = system

    def forward(self, x: Tensor) -> Tensor:
        return self.system.forward(x)

    def fit_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """One pipeline training step (settle → credit → update)."""
        return self.system.train_step(x, y)

    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        yield from self.system.geometry.parameters()

    def train(self, mode: bool = True) -> Self:  # type: ignore[override]
        super().train(mode)
        if hasattr(self.system.geometry, "train"):
            self.system.geometry.train(mode)
        return self
