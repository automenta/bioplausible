"""Widget-tree descriptor tests (Sprint 3.2)."""

from dataclasses import dataclass, field

from widgets import build_widget_tree

from computronium.core.trainer import TrainerConfig


@dataclass
class Nested:
    sparsity: float = 0.5


@dataclass
class Outer:
    lr: float = 0.001
    epochs: int = 5
    nested: Nested = field(default_factory=Nested)


class TestWidgetTree:
    def test_builds_field_descriptors(self):
        group = build_widget_tree(Outer())
        names = {f.name for f in group.fields}
        assert names == {"lr", "epochs"}

    def test_nested_dataclass_recurse(self):
        group = build_widget_tree(Outer())
        assert len(group.groups) == 1
        assert group.groups[0].label == "Nested"
        assert group.groups[0].fields[0].name == "sparsity"

    def test_trainer_config_tree(self):
        cfg = TrainerConfig(model="backprop_mlp", task="mnist", epochs=3)
        group = build_widget_tree(cfg)
        names = {f.name for f in group.fields}
        assert "epochs" in names
        assert "task" in names

    def test_apply_to_frozen_dataclass(self):
        outer = Outer()
        group = build_widget_tree(outer)
        lr_field = next(f for f in group.fields if f.name == "lr")
        updated = lr_field.apply(outer, 0.01)
        assert updated.lr == 0.01


class TestDictKnobs:
    def test_dict_of_scalars_expands_to_group(self):
        cfg = TrainerConfig(
            model="backprop_mlp",
            task="mnist",
            epochs=3,
            optimizer_kwargs={"lr": 0.001},
            model_kwargs={"hidden_dim": 32},
        )
        group = build_widget_tree(cfg)
        # optimizer_kwargs ({"lr": ...}) and model_kwargs should be knob groups.
        labels = {g.label for g in group.groups}
        assert "Optimizer Kwargs" in labels
        optim = next(g for g in group.groups if g.label == "Optimizer Kwargs")
        assert any(
            f.name == "lr" and f.path == ("optimizer_kwargs", "lr")
            for f in optim.fields
        )

    def test_dict_knob_apply_writes_through_path(self):
        cfg = TrainerConfig(
            model="tile_pc",
            task="mnist",
            epochs=3,
            optimizer_kwargs={"lr": 0.001},
            model_kwargs={"hidden_dim": 32},
        )
        group = build_widget_tree(cfg)
        model_group = next(g for g in group.groups if g.label == "Model Kwargs")
        hidden = next(f for f in model_group.fields if f.name == "hidden_dim")
        hidden.apply(cfg, 64)
        assert cfg.model_kwargs["hidden_dim"] == 64
