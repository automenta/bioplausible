"""P0a — RULE_SPACES↔constructor integrity gate, and R3 knob-efficacy.

Covers the validator, the phantom detection, the finder gate, the KB surface
emitter, and the "no advertised dimension is a no-op" (R3) checks.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import bioplausible.zoo  # ruff: ignore[unused-import]  (populate the model registry)
from bioplausible.core.exceptions import SpaceSignatureMismatchError
from bioplausible.hyperopt.rule_frontier import RuleFrontierFinder
from bioplausible.hyperopt.search_space import (
    RULE_SPACES,
    emit_rule_space_surfaces,
    surface_for_rule,
    validate_all_rule_spaces,
    validate_rule_space,
)


class _FakeDriver:
    """Records calls; nothing is actually trained."""

    def __init__(self):
        self.calls = 0

    def train(self, *, model, task, config, seed, epochs, device):
        self.calls += 1
        return {
            "final_acc": 0.9,
            "forward_flops": 10,
            "backward_flops": 5,
            "peak_memory_mb": 4.0,
            "wall_time_s": 1.0,
        }


def test_all_rule_spaces_are_honest():
    """After the P0a fix every advertised space matches its constructor."""
    surfaces = validate_all_rule_spaces()
    assert set(surfaces) == set(RULE_SPACES)
    for rule, surface in surfaces.items():
        assert not surface.phantoms, f"{rule} has phantoms: {surface.phantoms}"
        assert all(sink != "phantom" for _key, sink in surface.sinks)


def test_neural_cube_keeps_only_real_knobs():
    """The flagship space is honest: only constructor/training knobs remain."""
    surface = surface_for_rule("neural_cube")
    assert sorted(surface.model) == sorted("neural_cube")
    # phantom knobs from §0.1 are gone; real dims remain.
    assert not surface.phantoms
    sinks = dict(surface.sinks)
    assert sinks["cube_size"] == "constructor"
    assert sinks["max_steps"] == "constructor"
    # NeuralCube now accepts learning_rate/beta as constructor params (plan 7.5)
    assert sinks["learning_rate"] == "constructor"
    assert sinks["weight_decay"] == "training"


@given(rule=st.sampled_from(sorted(RULE_SPACES)))
def test_every_advertised_key_is_consumed_somewhere(rule):
    """Property: no advertised dimension is silently dropped (P0a contract)."""
    surface = surface_for_rule(rule)
    sinks = dict(surface.sinks)
    for key in RULE_SPACES[rule]:
        assert sinks[key] in {
            "constructor",
            "kwargs",
            "training",
        }, f"{rule}.{key} is a phantom"


def test_phantom_space_raises_descriptive_error(monkeypatch):
    """Re-introducing a phantom fails loudly with the offending keys named."""
    monkeypatch.setitem(
        RULE_SPACES,
        "neural_cube",
        {
            **RULE_SPACES["neural_cube"],
            "damping": (0.0, 0.9, "linear"),
            "hidden_dim": (32, 1024, "log"),
        },
    )
    with pytest.raises(SpaceSignatureMismatchError) as ei:
        validate_rule_space("neural_cube")
    assert ei.value.rule == "neural_cube"
    assert {"damping", "hidden_dim"} <= ei.value.phantoms
    assert "damping" in str(ei.value)


def test_finder_gate_blocks_probes_on_phantom_space(tmp_path, monkeypatch):
    """No probe budget is spent on a broken space (P0a gate)."""
    monkeypatch.setitem(
        RULE_SPACES,
        "neural_cube",
        {
            **RULE_SPACES["neural_cube"],
            "tol": (1e-6, 1e-2, "log"),
        },
    )
    driver = _FakeDriver()
    finder = RuleFrontierFinder(
        driver,
        rule="neural_cube",
        task="mnist",
        budget_probes=4,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    with pytest.raises(SpaceSignatureMismatchError):
        finder.find()
    assert driver.calls == 0  # the gate fired BEFORE any probe


def test_finder_runs_on_honest_space(tmp_path):
    """On an honest space the finder proceeds to probe (no gate trip)."""
    driver = _FakeDriver()
    finder = RuleFrontierFinder(
        driver,
        rule="neural_cube",
        task="mnist",
        budget_probes=3,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    decision = finder.find()
    assert driver.calls == 3
    assert decision.rule == "neural_cube"


def test_emitter_writes_honest_surface_records_to_kb(tmp_path):
    """P0a emitter persists a queryable surface record per validated family."""
    from bioplausible.knowledge.kb import KnowledgeBase

    kb = KnowledgeBase(db_path=f"{tmp_path}/kb.db", auto_embed=False)
    ids = emit_rule_space_surfaces(kb)
    assert set(ids) == set(RULE_SPACES)

    records = kb.query(topic="rule_space_surface", source="validator")
    assert len(records) == len(RULE_SPACES)
    for rec in records:
        assert rec.finding == "honest"
        assert "surface" in rec.extra


# --- R3: knob-efficacy — signature-valid ≠ effect-valid ---------------------


def test_convergence_knob_is_a_wired_lever():
    """Perturbing convergence_threshold changes settling (no-op-free knob)."""
    from bioplausible.zoo._settling import settle_state
    from bioplausible.zoo.models.eqprop.neural_cube import NeuralCube

    x = torch.randn(3, 8)
    eager = NeuralCube(
        cube_size=3,
        input_dim=8,
        output_dim=4,
        max_steps=20,
        convergence_threshold=1.0,
        convergence_start=2,
    )
    strict = NeuralCube(
        cube_size=3,
        input_dim=8,
        output_dim=4,
        max_steps=20,
        convergence_threshold=1e-9,
        convergence_start=2,
    )
    _, steps_eager, conv_eager = settle_state(eager, x)
    _, steps_strict, conv_strict = settle_state(strict, x)
    # A deliberately-loose threshold must terminate early and converge; a tight
    # one runs to the ceiling. The advertised knob has a measurable effect.
    assert steps_eager < steps_strict
    assert conv_eager and not conv_strict


@given(
    cube_a=st.integers(min_value=3, max_value=5),
    cube_b=st.integers(min_value=3, max_value=5),
)
def test_cube_size_perturbation_changes_architecture(cube_a, cube_b):
    """Property: perturbing cube_size (a real dimension) changes the model.

    The honest, deterministic effect of a dimension — not float noise — is the
    architectural one: neuron count and parameter count must scale with it.
    """
    from bioplausible.zoo.models.eqprop.neural_cube import NeuralCube

    a = NeuralCube(cube_size=cube_a, input_dim=8, output_dim=4, max_steps=10)
    b = NeuralCube(cube_size=cube_b, input_dim=8, output_dim=4, max_steps=10)
    params_a = sum(p.numel() for p in a.parameters())
    params_b = sum(p.numel() for p in b.parameters())
    if cube_a != cube_b:
        assert a.n_neurons != b.n_neurons
        assert params_a != params_b
    else:
        assert a.n_neurons == b.n_neurons
        assert params_a == params_b
