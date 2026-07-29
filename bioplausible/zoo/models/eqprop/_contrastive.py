"""Shared contrastive (free/nudged) train_step helper for EqProp models.

StandardEqProp, DirectedEP, HolomorphicEP all follow the same pattern:
1. Create one-hot target
2. Free phase forward (beta=0)
3. Nudged phase forward (beta=self.beta, target=target)
4. Compute weight/bias updates from contrastive (nudged - free) difference
5. Apply gradients and step optimizer
"""

import torch
import torch.nn.functional as F
from torch import nn


def _make_onehot_target(
    y: torch.Tensor,
    output_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a one-hot target tensor from class indices."""
    target = torch.zeros(y.size(0), output_dim, device=device or y.device)
    target.scatter_(1, y.unsqueeze(1), 1.0)
    if dtype is not None:
        target = target.to(dtype)
    return target


def _run_free_nudged(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Run free (beta=0) and nudged phase forward passes.

    Returns ``(free_activations, nudged_activations, free_output)``.
    """
    with torch.no_grad():
        model.forward(x, beta=0.0)
        free_acts: list[torch.Tensor] = model._last_activations  # type: ignore[attr-defined]
        free_out = free_acts[-1]

    with torch.no_grad():
        model.forward(x, beta=beta, target=target)
        nudged_acts: list[torch.Tensor] = model._last_activations  # type: ignore[attr-defined]

    return free_acts, nudged_acts, free_out


def _apply_layer_update(
    layer: nn.Module,
    dW: torch.Tensor,
    db: torch.Tensor | None,
    *,
    use_conj: bool = False,
) -> None:
    """Apply contrastive gradient to a single layer's weight and bias."""
    # Handle spectral norm parameterization
    param_container = layer
    weight_name: str = "weight"
    if (
        not use_conj
        and hasattr(layer, "parametrizations")
        and hasattr(layer.parametrizations, "weight")
    ):
        param_container = layer.parametrizations.weight  # type: ignore[attr-defined]
        weight_name = "original"

    w_param = getattr(param_container, weight_name)
    if w_param.grad is None:
        w_param.grad = -dW
    else:
        w_param.grad += -dW

    if layer.bias is not None and db is not None:
        if layer.bias.grad is None:
            layer.bias.grad = -db
        else:
            layer.bias.grad += -db


def _contrastive_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    layer_list: list[nn.Module],
    beta: float,
    use_conj: bool = False,
    feedback_layer_list: list[nn.Module] | None = None,
) -> dict[str, float]:
    """Full contrastive train_step for EqProp-style models.

    Parameters
    ----------
    model:
        The model instance (provides ``forward`` and ``_last_activations``).
    x, y:
        Input and target tensors.
    layer_list:
        Forward layers to update (``self.layers`` or ``self.forward_layers``).
    beta:
        Nudging strength.
    use_conj:
        Whether to conjugate the pre-synaptic activation (HolomorphicEP).
    feedback_layer_list:
        Optional backward/feedback layers (DirectedEP).

    Returns
    -------
    dict with ``loss`` and ``accuracy`` keys.
    """
    target = _make_onehot_target(
        y,
        model.config.output_dim,  # type: ignore[attr-defined]
        dtype=torch.complex64 if use_conj else None,
    )
    free_acts, nudged_acts, free_out = _run_free_nudged(model, x, target, beta)
    batch_size = x.size(0)

    model.optimizer.zero_grad()  # type: ignore[attr-defined]

    with torch.no_grad():
        for i, layer in enumerate(layer_list):
            h_prev_free = free_acts[i]
            h_post_free = free_acts[i + 1]
            h_prev_nudge = nudged_acts[i]
            h_post_nudge = nudged_acts[i + 1]

            # Contrastive weight update
            matmul_left = h_post_nudge.T
            matmul_right = h_prev_nudge
            if use_conj:
                matmul_right = h_prev_nudge.conj()

            prod_nudge = torch.matmul(matmul_left, matmul_right)
            prod_free = torch.matmul(
                h_post_free.T, h_prev_free if not use_conj else h_prev_free.conj()
            )

            dW = (prod_nudge - prod_free) / beta
            dW = dW / batch_size

            db = (h_post_nudge - h_post_free).sum(0) / beta / batch_size

            _apply_layer_update(layer, dW, db, use_conj=use_conj)

            # Optional feedback layer update (DirectedEP)
            if feedback_layer_list is not None and i < len(feedback_layer_list):
                bprod_nudge = torch.matmul(h_prev_nudge.T, h_post_nudge)
                bprod_free = torch.matmul(h_prev_free.T, h_post_free)
                dB = (bprod_nudge - bprod_free) / beta / batch_size
                _apply_layer_update(feedback_layer_list[i], dB, None, use_conj=use_conj)

    model.optimizer.step()  # type: ignore[attr-defined]

    # Loss/accuracy on free-phase output
    ce_input = free_out.real if use_conj else free_out
    loss = F.cross_entropy(ce_input, y).item()
    acc = (ce_input.argmax(dim=1) == y).float().mean().item()

    return {"loss": loss, "accuracy": acc}
