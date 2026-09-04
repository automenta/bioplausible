"""Deep-linear transition schedule lock (TODO.md R3.1).

``extract_layered_params`` used to return weights/biases/activations as
separate tuples, losing interleaving — the first ePC port re-walked modules
privately and silently degenerated on deep-linear stacks (the paper's own
testbed). This lock proves the shared ``transitions`` schedule is
depth-structure correct: a deep-linear geometry (no activation modules)
yields one transition per Linear with empty activation chains, and mixed
stacks attach each activation to the transition it follows.
"""

import torch
from torch import nn

from computronium.ontology._settle_kernel import extract_layered_params
from computronium.ontology.geometry import FeedforwardGeometry, GeometryConfig


def _deep_linear_geometry(dims: tuple[int, ...]) -> FeedforwardGeometry:
    return FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=dims[0], hidden_dims=dims[1:-1], output_dim=dims[-1]
        )
    )


def test_deep_linear_transitions_are_one_per_linear() -> None:
    geometry = _deep_linear_geometry((4, 8, 8, 8, 2))
    # A deep-linear stack: strip the builder's ReLU modules (the paper's
    # own testbed — depth without activation nonlinearity).
    geometry._layers = nn.ModuleList([
        m for m in geometry._layers if isinstance(m, nn.Linear)
    ])
    params = extract_layered_params(geometry)
    assert params is not None
    # No activation modules in a deep-linear stack: 4 Linears, empty chains.
    assert params.activations == ()
    assert len(params.transitions) == len(params.weights) == 4
    for i, (weight, bias, chain) in enumerate(params.transitions):
        assert chain == (), f"transition {i} must be activation-free"
        assert weight.data_ptr() == params.weights[i].data_ptr()
        assert bias is params.biases[i]


def test_activation_modules_attach_to_their_transition() -> None:
    torch.manual_seed(0)
    geometry = _deep_linear_geometry((4, 8, 2))
    geometry._layers = nn.ModuleList([
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.ReLU(),
        nn.Linear(8, 2),
    ])
    params = extract_layered_params(geometry)
    assert params is not None
    assert len(params.transitions) == 3
    _, _, chain0 = params.transitions[0]
    _, _, chain1 = params.transitions[1]
    _, _, chain2 = params.transitions[2]
    assert len(chain0) == 1 and isinstance(chain0[0], nn.ReLU)
    assert len(chain1) == 2 and isinstance(chain1[0], nn.Tanh)
    assert chain2 == ()


def test_error_injection_positions_match_transitions() -> None:
    """ePC injects εᵢ after each non-final transition's activation chain."""
    geometry = _deep_linear_geometry((4, 8, 8, 2))
    params = extract_layered_params(geometry)
    assert params is not None
    hidden_transitions = params.transitions[:-1]
    assert len(hidden_transitions) == 2  # ε₀ after layer 0, ε₁ after layer 1
    # Each transition's weight maps from the previous layer's width.
    assert [t[0].shape for t in params.transitions] == [
        (8, 4),
        (8, 8),
        (2, 8),
    ]
