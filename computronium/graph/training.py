# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.

"""Training procedures for GraphStructure — backprop and predictive coding.

Both train_backprop and train_pcn accept the same GraphStructure and params,
enabling fair comparison on identical architectures.
"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from computronium.core.losses import compute_accuracy
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer
from computronium.graph.topology import GraphStructure

__all__ = [
    "train_backprop",
    "train_pcn",
]


def _feedforward(
    structure: GraphStructure,
    params: dict[str, dict[str, torch.Tensor]],
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run a pure feedforward pass through the graph.

    Uses topological order. Runs each node's forward with predecessor
    activities as slot inputs and the node's own parameters.

    Returns dict of {node_name: activation_tensor}.
    """
    activities: dict[str, torch.Tensor] = {}
    topo_order = structure.topological_order()

    for node in topo_order:
        preds = structure.get_predecessors(node)
        slot_inputs: dict[str, torch.Tensor] = {}
        if not preds:
            if node.name == structure.task_map.x.name:
                slot_inputs["input"] = x
            else:
                slot_inputs["input"] = torch.zeros(x.shape[0], 1, device=x.device)
        else:
            for src_node, target_slot in preds:
                val = activities[src_node.name]
                if target_slot.name in slot_inputs:
                    slot_inputs[target_slot.name] = slot_inputs[target_slot.name] + val
                else:
                    slot_inputs[target_slot.name] = val
        slot_inputs.update(params.get(node.name, {}))
        activities[node.name] = node.forward(**slot_inputs)

    return activities


def _train_loop(
    optimizer,
    structure,
    params,
    train_loader,
    test_loader,
    epochs,
    device,
    step_fn,
    eval_fn,
) -> dict[str, float]:
    """Run ``epochs`` of training plus optional test-set evaluation.

    ``step_fn`` must perform one batch's forward + backward + optimizer update
    and return ``(output, loss)`` for logging.  ``eval_fn`` returns the model
    output for a batch (no grad).
    """
    total_time = 0.0
    final_train_acc = 0.0
    final_train_loss = 0.0

    for _ in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output, loss = step_fn(optimizer, structure, params, batch_x, batch_y)

            epoch_loss += loss.item()
            epoch_acc += compute_accuracy(output, batch_y)
            n_batches += 1

        total_time += time.time() - epoch_start

        if n_batches > 0:
            final_train_loss = epoch_loss / n_batches
            final_train_acc = epoch_acc / n_batches

    test_acc = 0.0
    if test_loader is not None:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device)
                output = eval_fn(structure, params, batch_x, batch_y)
                correct += (output.argmax(dim=-1) == batch_y).sum().item()
                total += batch_y.shape[0]
        test_acc = correct / total if total > 0 else 0.0

    return {
        "train_acc": final_train_acc,
        "test_acc": test_acc,
        "train_loss": final_train_loss,
        "time": total_time,
    }


def _collect_trainable(
    params: dict[str, dict[str, torch.Tensor]], requires_grad: bool
) -> list[torch.Tensor]:
    """Flatten node params, optionally forcing ``requires_grad`` on each."""
    param_list: list[torch.Tensor] = []
    for node_params in params.values():
        for p in node_params.values():
            if requires_grad:
                p.requires_grad_(True)
            param_list.append(p)
    return param_list


def _backprop_step(optimizer, structure, params, x, y):
    activities = _feedforward(structure, params, x)
    output = activities[structure.task_map.y.name]
    loss = F.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
    return output, loss


def _backprop_eval(structure, params, x, y):
    activities = _feedforward(structure, params, x)
    return activities[structure.task_map.y.name]


def train_backprop(
    structure: GraphStructure,
    params: dict[str, dict[str, torch.Tensor]],
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Train a GraphStructure using standard backpropagation.

    Uses topological feedforward + torch.autograd backward.

    Args:
        structure: The graph topology.
        params: Node parameters (will be updated in-place).
        train_loader: DataLoader for training.
        test_loader: Optional DataLoader for evaluation.
        epochs: Number of training epochs.
        lr: Learning rate (Adam).
        device: Torch device.

    Returns:
        Dict with keys "train_acc", "test_acc", "train_loss", "time".
    """
    param_list = _collect_trainable(params, requires_grad=True)
    optimizer = create_optimizer(
        param_list, OptimizerConfig(name="adam", lr=lr, weight_decay=0.0)
    )
    return _train_loop(
        optimizer,
        structure,
        params,
        train_loader,
        test_loader,
        epochs,
        device,
        _backprop_step,
        _backprop_eval,
    )


def _compute_pc_weight_gradients(
    structure: GraphStructure,
    params: dict[str, dict[str, torch.Tensor]],
    activities: dict[str, torch.Tensor],
    targets: torch.Tensor | None = None,
) -> None:
    """Compute PC weight gradients using local prediction errors.

    For each node with parameters:
        1. Find node_input (predecessor activity feeding into this node).
        2. Find target for this node (from supervision for output, or parent).
        3. Compute weight gradient = ∂||forward(node_input, params) - target||² / ∂params.

    This is a LOCAL gradient (depends only on node's input, output, and target).
    Gradients are accumulated into each parameter's .grad field.
    """
    # Compute expected activity for each node
    expected: dict[str, torch.Tensor] = {}

    # For output node, use supervised target
    if targets is not None:
        out_name = structure.task_map.y.name
        if targets.dim() == 1:
            expected[out_name] = F.one_hot(
                targets, num_classes=activities[out_name].shape[-1]
            ).float()
        else:
            expected[out_name] = targets.float()

    # For hidden nodes, use parent's forward to compute expected activity
    for edge in structure.edges:
        child_name = edge.target.owner.name
        if child_name in expected:
            continue  # Already set (e.g., output node)
        parent_name = edge.source.name
        parent_p = params.get(parent_name, {})
        parent_act = activities[parent_name]

        if not parent_p:
            continue  # Parent has no params (e.g., ReLU)

        try:
            fwd_args: dict[str, torch.Tensor] = dict(parent_p)
            fwd_args["input"] = parent_act
            prediction = edge.source.forward(**fwd_args)
            expected[child_name] = prediction
        except RuntimeError:
            pass  # Dimension mismatch, skip

    # Compute weight gradients
    for node in structure.nodes:
        node_p = params.get(node.name, {})
        if not node_p:
            continue

        node_act = activities[node.name]
        exp_act = expected.get(node.name)
        if exp_act is None:
            continue

        # Get node input (from first predecessor)
        preds = structure.get_predecessors(node)
        if not preds:
            continue

        node_input = activities[preds[0][0].name]

        # Ensure exp_act shape matches node_act
        if exp_act.shape != node_act.shape:
            if (
                exp_act.dim() >= 2
                and node_act.dim() >= 2
                and exp_act.shape[0] == node_act.shape[0]
            ):
                min_dim = min(exp_act.shape[-1], node_act.shape[-1])
                exp_act_proj = torch.zeros_like(node_act)
                exp_act_proj[..., :min_dim] = exp_act[..., :min_dim]
                exp_act = exp_act_proj
            else:
                continue

        error = node_act - exp_act

        # Compute gradients using simple Hebbian-like rule
        for k in node_p:
            if k == "weight":
                # ΔW ∝ error ⊗ input  (outer product, averaged over batch)
                g = torch.mm(error.detach().T, node_input.detach())
                if node_p[k].grad is None:
                    node_p[k].grad = torch.zeros_like(node_p[k])
                node_p[k].grad = node_p[k].grad + g
            elif k == "bias":
                g = error.detach().sum(dim=0)
                if node_p[k].grad is None:
                    node_p[k].grad = torch.zeros_like(node_p[k])
                node_p[k].grad = node_p[k].grad + g


def train_pcn(
    structure: GraphStructure,
    params: dict[str, dict[str, torch.Tensor]],
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
    infer_steps: int = 1,
    eta_infer: float = 0.05,
) -> dict[str, float]:
    """Train a GraphStructure using Predictive Coding.

    Per batch:
        1. Run inference settle (feedforward for feedforward graphs).
        2. Compute PC weight gradients using local prediction errors.
        3. Update weights with Adam.

    Args:
        structure: The graph topology.
        params: Node parameters (updated in-place).
        train_loader: DataLoader for training.
        test_loader: Optional DataLoader for evaluation.
        epochs: Number of training epochs.
        lr: Learning rate (Adam).
        device: Torch device.
        infer_steps: Number of inference settling steps.
        eta_infer: Inference learning rate (used for cyclic graphs).

    Returns:
        Dict with keys "train_acc", "test_acc", "train_loss", "time".
    """
    from computronium.graph.inference import InferenceSGD

    infer = InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps)
    structure.inference = infer

    param_list = _collect_trainable(params, requires_grad=False)
    optimizer = create_optimizer(
        param_list, OptimizerConfig(name="adam", lr=lr, weight_decay=0.0)
    )

    def _pcn_step(optimizer, structure, params, x, y):
        settled = infer.settle(structure, params, x, y=y)
        for node_p in params.values():
            for p in node_p.values():
                if p.grad is not None:
                    p.grad.zero_()
        _compute_pc_weight_gradients(structure, params, settled, targets=y)
        optimizer.step()
        output = settled[structure.task_map.y.name]
        loss = F.cross_entropy(output, y)
        return output, loss

    def _pcn_eval(structure, params, x, y):
        settled = infer.settle(structure, params, x, y=y)
        return settled[structure.task_map.y.name]

    return _train_loop(
        optimizer,
        structure,
        params,
        train_loader,
        test_loader,
        epochs,
        device,
        _pcn_step,
        _pcn_eval,
    )
