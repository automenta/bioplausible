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

from bioplausible.core.losses import compute_accuracy


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


def _compute_layer_diagnostics(
    layer_index: int,
    states: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    grad_w: torch.Tensor,
    grad_b: torch.Tensor | None,
    scale: float,
) -> dict[str, float]:
    """Compute per-layer contrastive diagnostics.

    ``states`` is ``((h_pf, h_qf), (h_pn, h_qn))`` — free/nudged pre/post
    activations for the layer. Bundling the four tensors keeps the signature
    under the PLR0913 argument cap while staying explicit at the call site.
    """
    (h_prev_free, h_post_free), (h_prev_nudge, h_post_nudge) = states
    return {
        "layer": layer_index,
        "pre_state_delta_norm": (h_prev_nudge - h_prev_free).norm().item(),
        "post_state_delta_norm": (h_post_nudge - h_post_free).norm().item(),
        "weight_grad_norm": grad_w.norm().item(),
        "bias_grad_norm": grad_b.norm().item() if grad_b is not None else 0.0,
        "update_scale": scale,
    }


def _run_free_nudged(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    *,
    track_settle: bool = False,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Run free (beta=0) and nudged phase forward passes.

    Returns ``(free_activations, nudged_activations, free_output,
    free_settle, nudged_settle)``. ``free_settle``/``nudged_settle`` are the
    per-phase settle dynamics dicts (``final_delta``, ``steps_taken``,
    ``converged``, ``settle_time_s``) when ``track_settle`` is True, else None.

    Uses the model's *explicit* settle (``_explicit_forward`` when present)
    because the contrastive rule needs the per-layer activations list — a
    single-hidden ``equilibrium`` model would otherwise route ``forward``
    through the O(1) implicit backward, which returns only the output and
    leaves ``_last_activations`` unset.
    """
    _forward = getattr(model, "_explicit_forward", model.forward)

    def _run(beta_phase: float) -> tuple[list[torch.Tensor], dict[str, object] | None]:
        if _forward is model.forward:
            model.forward(x, beta=beta_phase, target=target if beta_phase else None)
            return model._last_activations, None  # type: ignore[attr-defined]
        out = _forward(
            x,
            beta=beta_phase,
            target=target if beta_phase else None,
            steps=None,
            return_trajectory=False,
            return_dynamics=track_settle,
        )
        if track_settle:
            # ``_explicit_forward`` returns ``(out, dynamics)`` when
            # ``return_dynamics=True``.
            dynamics = out[1] if isinstance(out, tuple) else None
            return model._last_activations, dynamics  # type: ignore[attr-defined]
        return model._last_activations, None  # type: ignore[attr-defined]

    with torch.no_grad():
        free_acts, free_settle = _run(0.0)
        free_out = free_acts[-1]
        # Reset momentum velocity between phases (if applicable)
        if hasattr(model, "_velocity") and model._velocity is not None:
            for v in model._velocity:
                v.zero_()

    with torch.no_grad():
        nudged_acts, nudged_settle = _run(beta)

    return free_acts, nudged_acts, free_out, free_settle, nudged_settle


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
    update_scales: list[float] | None = None,
    diagnostics: bool = False,
    use_conj: bool = False,
    feedback_layer_list: list[nn.Module] | None = None,
    recurrent_layer_list: list[nn.Module] | None = None,
) -> dict[str, object]:
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
    update_scales:
        Optional per-layer multipliers on the computed contrastive update.
        Applied after the EqProp gradient is computed.
    diagnostics:
        If True, collect and return per-layer diagnostic information.
    use_conj:
        Whether to conjugate the pre-synaptic activation (HolomorphicEP).
    feedback_layer_list:
        Optional backward/feedback layers (DirectedEP).
    recurrent_layer_list:
        Optional per-hidden self-recurrent layers (``W_rec[i]``). Each is
        updated by the contrastive difference of its own hidden state — the
        "pre" and "post" sides are both ``h_i``, so ``dW = (h_iᵀh_i
        _{nudge} − h_iᵀh_i_{free})/β`` (Scellier-Bengio self-loop rule).

    Returns
    -------
    dict with ``loss`` and ``accuracy`` keys. If ``diagnostics=True``, also
    includes ``layer_diagnostics`` (list of per-layer dicts) and
    ``global_diagnostics`` (dict with output delta, beta, loss, accuracy).
    """
    target = _make_onehot_target(
        y,
        model.config.output_dim,  # type: ignore[attr-defined]
        dtype=torch.complex64 if use_conj else None,
    )
    free_acts, nudged_acts, free_out, free_settle, nudged_settle = _run_free_nudged(
        model, x, target, beta, track_settle=diagnostics
    )
    batch_size = x.size(0)

    model.optimizer.zero_grad()  # type: ignore[attr-defined]

    layer_diagnostics: list[dict[str, float]] = []

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

            scale = update_scales[i] if update_scales is not None else 1.0
            dW = dW * scale / batch_size

            db = (h_post_nudge - h_post_free).sum(0) / beta
            db = db * scale / batch_size

            _apply_layer_update(layer, dW, db, use_conj=use_conj)

            if diagnostics:
                layer_diagnostics.append(
                    _compute_layer_diagnostics(
                        i,
                        ((h_prev_free, h_post_free), (h_prev_nudge, h_post_nudge)),
                        dW,
                        db,
                        scale,
                    )
                )

            # Optional feedback layer update (DirectedEP)
            if feedback_layer_list is not None and i < len(feedback_layer_list):
                # Feedback layer i maps from output -> hidden_i
                # pre = output (last activation), post = hidden_i (i+1)
                # Weight shape: [hidden_i, output_dim], so gradient needs [hidden_i, output_dim]
                h_out_free = free_acts[-1]
                h_out_nudge = nudged_acts[-1]
                h_hidden_free = free_acts[i + 1]
                h_hidden_nudge = nudged_acts[i + 1]
                bprod_nudge = torch.matmul(h_hidden_nudge.T, h_out_nudge)
                bprod_free = torch.matmul(h_hidden_free.T, h_out_free)
                dB = (bprod_nudge - bprod_free) / beta / batch_size
                _apply_layer_update(feedback_layer_list[i], dB, None, use_conj=use_conj)

        # Self-recurrent layers: pre/post are the *same* hidden state, so the
        # contrastive difference collapses to h_iᵀh_i.
        if recurrent_layer_list is not None:
            for i, rec in enumerate(recurrent_layer_list):
                h_free = free_acts[i + 1]
                h_nudged = nudged_acts[i + 1]
                prod_nudge = h_nudged.T @ (h_nudged.conj() if use_conj else h_nudged)
                prod_free = h_free.T @ (h_free.conj() if use_conj else h_free)
                dW = (prod_nudge - prod_free) / beta / batch_size
                db = (h_nudged - h_free).sum(0) / beta / batch_size
                _apply_layer_update(rec, dW, db, use_conj=use_conj)

    model.optimizer.step()  # type: ignore[attr-defined]

    # Loss/accuracy on free-phase output
    ce_input = free_out.real if use_conj else free_out
    loss = F.cross_entropy(ce_input, y).item()
    acc = compute_accuracy(ce_input, y)

    result: dict[str, object] = {"loss": loss, "accuracy": acc}

    if diagnostics:
        result["layer_diagnostics"] = layer_diagnostics
        result["global_diagnostics"] = {
            "output_state_delta_norm": (nudged_acts[-1] - free_acts[-1]).norm().item(),
            "beta": beta,
            "loss": loss,
            "accuracy": acc,
            "free_converged": bool(free_settle and free_settle.get("converged")),
            "nudged_converged": bool(nudged_settle and nudged_settle.get("converged")),
            "free_settle_residual": float(
                (free_settle or {}).get("final_delta", float("nan"))
            ),
            "nudged_settle_residual": float(
                (nudged_settle or {}).get("final_delta", float("nan"))
            ),
            "free_steps_taken": int((free_settle or {}).get("steps_taken", 0)),
            "nudged_steps_taken": int((nudged_settle or {}).get("steps_taken", 0)),
        }

    return result
