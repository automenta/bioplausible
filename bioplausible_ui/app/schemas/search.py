from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef, PlotDef
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from bioplausible_ui.core.widgets.dataset_picker import DatasetPicker
from bioplausible_ui.core.widgets.model_selector import ModelSelector
from bioplausible_ui.core.widgets.hyperparam_editor import HyperparamEditor

SEARCH_TAB_SCHEMA = TabSchema(
    name="Search",
    widgets=[
        WidgetDef("task_selector", TaskSelector),
        WidgetDef("dataset_picker", DatasetPicker, bindings={"task": "@task_selector.value"}),
        WidgetDef("model_selector", ModelSelector, bindings={"task": "@task_selector.value"}),
        WidgetDef("search_space", HyperparamEditor, bindings={"model": "@model_selector.value"}),
    ],
    actions=[
        ActionDef("start_search", "🔍", "_start_search", style="success"),
        ActionDef("stop_search", "⏹", "_stop_search", style="danger"),
    ],
    plots=[
        PlotDef("search_progress", xlabel="Trial", ylabel="Best Metric"),
    ]
)
