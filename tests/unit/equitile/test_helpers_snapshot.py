"""Snapshot tests for extracted EquiTile helpers.

Golden values computed with fixed seed -> fixed tensor output.
Guards refactors of _relax -> (_step_with_tolerance, _measure_change,
_check_convergence) and _apply_hebbian_updates -> (_propagate_errors_backward,
_compute_weight_updates, _apply_weight_updates) and match/case dispatch
(_get_activation, train_step).
"""

import pytest
import torch

from bioplausible.equitile import EquiTile


def _make_model(**overrides) -> EquiTile:
    """Tiny deterministic EquiTile (3 tiles: input/hidden/output)."""
    kwargs: dict = {
        "neurons_per_tile": 4,
        "num_layers": 3,
        "tiles_per_layer": 1,
        "input_dim": 4,
        "output_dim": 2,
        "mode": "pc",
        "dropout": 0.0,
        "inference_steps": 5,
        "step_size": 0.1,
        "relaxation_tolerance": 1e-6,
    }
    kwargs.update(overrides)
    model = EquiTile(**kwargs)
    model.eval()
    return model


_X = torch.tensor([[0.5, -0.2, 0.8, 0.1], [-0.6, 0.3, -0.1, 0.7]], dtype=torch.float32)
_Y = torch.tensor([0, 1], dtype=torch.long)
_BATCH = 2


def _prepare(model: EquiTile):
    """Run init + predictions + errors so the model is ready for snapshot calls."""
    input_proj = model.W_in(_X)
    model._init_activities(input_proj, _BATCH, _X.device)
    model._compute_predictions(_BATCH, _X.device)
    model._compute_errors()
    return input_proj


# ---- _relax trio  (task 1.7) ----


def test_step_with_tolerance_snapshot() -> None:
    """_step_with_tolerance: one relax step gives deterministic activities."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    model._step_with_tolerance(input_proj, 0.1, True, None)

    act = model.graph.tiles[1].activity
    assert act is not None
    expected = torch.tensor([
        [0.04193415, 0.03051286, -0.04526899, 0.00280806],
        [-0.00071170, 0.00965601, -0.04151358, 0.04185685],
    ])
    torch.testing.assert_close(act, expected, rtol=1e-5, atol=1e-7)


def test_step_with_tolerance_keeps_input_pinned() -> None:
    """Input tile activities stay unchanged after _step_with_tolerance."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    before = model.graph.tiles[0].activity.clone()
    model._step_with_tolerance(input_proj, 0.1, True, None)
    after = model.graph.tiles[0].activity
    torch.testing.assert_close(before, after)


def test_measure_change_zero_when_no_step() -> None:
    """_measure_change returns 0 when prev == current activities."""
    torch.manual_seed(7)
    model = _make_model()
    _prepare(model)
    prev = {
        t.id: (t.activity.clone() if t.activity is not None else None)
        for t in model.graph.all_tiles
    }
    assert model._measure_change(prev) == pytest.approx(0.0, abs=1e-12)


def test_measure_change_positive_after_step() -> None:
    """_measure_change returns >0 after a relaxation step."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    prev = {
        t.id: (t.activity.clone() if t.activity is not None else None)
        for t in model.graph.all_tiles
    }
    model._step_with_tolerance(input_proj, 0.1, True, None)
    change = model._measure_change(prev)
    assert change > 0.0
    # Golden value for this exact seed + config
    assert change == pytest.approx(0.013391388580203056, abs=1e-12)


def test_check_convergence_edge_cases() -> None:
    """_check_convergence guards: None tolerance/prev/early steps->False."""
    torch.manual_seed(7)
    model = _make_model()
    _prepare(model)
    prev = {
        t.id: (t.activity.clone() if t.activity is not None else None)
        for t in model.graph.all_tiles
    }

    assert not model._check_convergence(prev, None, 10)
    assert not model._check_convergence(None, 1e-4, 10)
    assert not model._check_convergence(prev, 1e-4, 1)
    assert not model._check_convergence(prev, 1e-4, 2)
    assert model._check_convergence(prev, 1e-4, 3)


def test_full_relax_convergence_snapshot() -> None:
    """_relax with tolerance: final activities are deterministic."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    model._relax(input_proj, 5, tolerance=1e-6)

    t0 = model.graph.tiles[0].activity
    expected_t0 = torch.tensor([
        [0.61077821, -0.11780813, 0.52571166, 0.47322416],
        [0.62278008, 0.27237332, 0.10581949, 0.76629573],
    ])
    torch.testing.assert_close(t0, expected_t0, rtol=1e-5, atol=1e-7)

    t1 = model.graph.tiles[1].activity
    expected_t1 = torch.tensor([
        [0.18965235, 0.13821819, -0.19715063, 0.01118154],
        [-0.01583163, 0.05159291, -0.18731004, 0.18600160],
    ])
    torch.testing.assert_close(t1, expected_t1, rtol=1e-5, atol=1e-7)

    t2 = model.graph.tiles[2].activity
    expected_t2 = torch.tensor([[0.00398423, 0.01772728], [-0.01590117, 0.00744427]])
    torch.testing.assert_close(t2, expected_t2, rtol=1e-5, atol=1e-7)


# ---- _apply_hebbian_updates trio  (task 1.8) ----


def test_propagate_errors_backward_snapshot() -> None:
    """_propagate_errors_backward: deterministic error tensors from output_delta."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    model._relax(input_proj, 3)

    out_acts = torch.cat(
        [model.graph.tiles[tid].activity for tid in model.graph.output_tile_ids],
        dim=-1,
    )
    _, output_delta = model._compute_loss_and_delta(model.W_out(out_acts), _Y)
    tile_errors = model._propagate_errors_backward(output_delta)

    # Output tile error = output_delta slice
    assert 2 in tile_errors  # ruff: ignore[magic-value-comparison] -- magic tile id
    expected_out = torch.tensor([[0.31460840, -0.12759131], [-0.31534389, 0.12788957]])
    torch.testing.assert_close(tile_errors[2], expected_out, rtol=1e-5, atol=1e-7)

    # Hidden tile error = backprop through edge (1 -> 2) weights
    hidden_id: int = 1
    assert hidden_id in tile_errors
    expected_hid = torch.tensor([
        [0.28175700, -0.21215147, 0.21775761, -0.16483884],
        [-0.28241569, 0.21264744, -0.21826667, 0.16522419],
    ])
    torch.testing.assert_close(
        tile_errors[hidden_id], expected_hid, rtol=1e-5, atol=1e-7
    )


def test_compute_weight_updates_snapshot() -> None:  # ruff: ignore[too-many-locals] -- many locals from 4 edge tests
    """_compute_weight_updates: deterministic weight/bias update tensors."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    model._relax(input_proj, 3)

    out_acts = torch.cat(
        [model.graph.tiles[tid].activity for tid in model.graph.output_tile_ids],
        dim=-1,
    )
    _, output_delta = model._compute_loss_and_delta(model.W_out(out_acts), _Y)
    tile_errors = model._propagate_errors_backward(output_delta)

    # Edge (0,1): 4x4 weight update
    upd_01 = model._compute_weight_updates(0, 0, 1, tile_errors, _BATCH)
    assert upd_01 is not None
    wu_01, bu_01 = upd_01
    expected_wu = torch.tensor([
        [-0.00126537, 0.00095277, -0.00097795, 0.00074029],
        [-0.02257426, 0.01699749, -0.01744665, 0.01320682],
        [0.03200273, -0.02409674, 0.02473351, -0.01872285],
        [-0.02832623, 0.02132849, -0.02189210, 0.01657195],
    ])
    torch.testing.assert_close(wu_01, expected_wu, rtol=1e-5, atol=1e-7)

    # Edge (1,2): 4x2 weight update
    upd_12 = model._compute_weight_updates(1, 1, 2, tile_errors, _BATCH)
    assert upd_12 is not None
    wu_12, bu_12 = upd_12
    expected_wu12 = torch.tensor([
        [0.00787892, -0.00319534],
        [0.00358312, -0.00145316],
        [-0.00037675, 0.00015279],
        [-0.00702288, 0.00284817],
    ])
    torch.testing.assert_close(wu_12, expected_wu12, rtol=1e-5, atol=1e-7)

    # Bias updates
    expected_bu_01 = torch.tensor([-0.00012039, 0.00009065, -0.00009304, 0.00007043])
    torch.testing.assert_close(bu_01, expected_bu_01, rtol=1e-5, atol=1e-7)

    expected_bu_12 = torch.tensor([-0.00013442, 0.00005451])
    torch.testing.assert_close(bu_12, expected_bu_12, rtol=1e-5, atol=1e-7)


def test_compute_weight_updates_none_paths() -> None:
    """_compute_weight_updates returns None when src-dst is unavailable."""
    torch.manual_seed(3)
    model = _make_model(inference_steps=2)
    model.eval()
    x2 = torch.tensor(
        [[0.1, -0.5, 0.3, 0.9], [0.4, 0.2, -0.7, 0.6]], dtype=torch.float32
    )
    ip = model.W_in(x2)
    model._init_activities(ip, 2, x2.device)
    model._compute_predictions(2, x2.device)
    model._compute_errors()
    od = model._compute_loss_and_delta(
        model.W_out(
            torch.cat(
                [
                    model.graph.tiles[tid].activity
                    for tid in model.graph.output_tile_ids
                ],
                dim=-1,
            )
        ),
        torch.tensor([0, 1]),
    )[1]
    te = model._propagate_errors_backward(od)

    # src activity is None -> None
    model.graph.tiles[0].activity = None
    assert model._compute_weight_updates(0, 0, 1, te, 2) is None

    # dst missing from tile_errors -> None
    assert model._compute_weight_updates(1, 1, 2, {}, 2) is None


def test_apply_hebbian_updates_snapshot() -> None:
    """_apply_hebbian_updates: weight tensors change deterministically."""
    torch.manual_seed(7)
    model = _make_model()
    input_proj = _prepare(model)
    model._relax(input_proj, 3)

    out_acts = torch.cat(
        [model.graph.tiles[tid].activity for tid in model.graph.output_tile_ids],
        dim=-1,
    )
    _, output_delta = model._compute_loss_and_delta(model.W_out(out_acts), _Y)

    w01_before = model.edge_weights["edge_0_1"].data.clone()
    w12_before = model.edge_weights["edge_1_2"].data.clone()

    model._apply_hebbian_updates(output_delta, _BATCH)

    w01_after = model.edge_weights["edge_0_1"].data
    w12_after = model.edge_weights["edge_1_2"].data

    expected_w01_after = torch.tensor([
        [0.25709933, -0.22320126, -0.37622640, 0.21151762],
        [-1.00417912, -0.11574991, 0.09438914, 0.12894689],
        [1.13956797, 1.10836363, -0.66410613, -0.87448078],
        [-0.04395852, 0.31763756, -0.62613368, 0.84625322],
    ])
    torch.testing.assert_close(w01_after, expected_w01_after, rtol=1e-5, atol=1e-7)

    expected_w12_after = torch.tensor([
        [1.07098794, 0.43274078],
        [-0.42962652, 0.60349160],
        [0.53413028, -0.38965741],
        [-0.49713400, 0.06591684],
    ])
    torch.testing.assert_close(w12_after, expected_w12_after, rtol=1e-5, atol=1e-7)

    assert not torch.allclose(w01_before, w01_after, rtol=1e-8, atol=1e-10)
    assert not torch.allclose(w12_before, w12_after, rtol=1e-8, atol=1e-10)

    expected_b01_after = torch.tensor([
        1.20385414e-06,
        -9.06458297e-07,
        9.30369765e-07,
        -7.04299850e-07,
    ])
    torch.testing.assert_close(
        model.edge_biases["edge_0_1"].data, expected_b01_after, rtol=1e-5, atol=1e-7
    )
    expected_b12_after = torch.tensor([1.34421839e-06, -5.45116848e-07])
    torch.testing.assert_close(
        model.edge_biases["edge_1_2"].data, expected_b12_after, rtol=1e-5, atol=1e-7
    )


# ---- _get_activation dispatch  (task 1.9) ----


def test_get_activation_dispatch() -> None:
    """_get_activation maps each known name to the correct activation class."""
    model = _make_model()
    assert isinstance(model._get_activation("tanh"), torch.nn.Tanh)
    assert isinstance(model._get_activation("relu"), torch.nn.ReLU)
    assert isinstance(model._get_activation("gelu"), torch.nn.GELU)
    assert isinstance(model._get_activation("silu"), torch.nn.SiLU)
    assert isinstance(model._get_activation("nope"), torch.nn.GELU)


# ---- train_step dispatch  (task 1.9) ----


def test_train_step_backprop_dispatch() -> None:
    """train_step(mode='backprop') calls backprop-specific code path."""
    torch.manual_seed(7)
    model = _make_model(mode="backprop", inference_steps=2)
    stats = model.train_step(_X, _Y)
    assert stats["mode"] == "backprop"
    assert stats["loss"] == pytest.approx(0.6933987140655518, abs=1e-6)
    assert stats["accuracy"] == pytest.approx(0.5, abs=1e-6)


def test_train_step_ep_dispatch() -> None:
    """train_step(mode='ep') calls EP-specific code path."""
    torch.manual_seed(7)
    model = _make_model(
        mode="ep",
        inference_steps=2,
        inference_steps_free=2,
        inference_steps_nudged=2,
    )
    stats = model.train_step(_X, _Y)
    assert stats["mode"] == "ep"
    assert stats["beta"] == pytest.approx(0.1, abs=1e-6)
    assert stats["loss"] == pytest.approx(0.7059271335601807, abs=1e-6)
    assert stats["accuracy"] == pytest.approx(0.5, abs=1e-6)


def test_train_step_pc_dispatch() -> None:
    """train_step(mode='pc') calls PC-specific code path."""
    torch.manual_seed(7)
    model = _make_model(mode="pc", inference_steps=2)
    stats = model.train_step(_X, _Y)
    assert stats["mode"] == "pc"
    assert stats["loss"] == pytest.approx(0.6933987140655518, abs=1e-6)
    assert stats["accuracy"] == pytest.approx(0.5, abs=1e-6)
    assert stats["active_tiles"] == 2  # ruff: ignore[magic-value-comparison] -- expected count
