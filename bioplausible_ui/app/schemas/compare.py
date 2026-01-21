from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef, PlotDef
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from bioplausible_ui.core.widgets.dataset_picker import DatasetPicker
from bioplausible_ui.core.widgets.model_selector import ModelSelector

COMPARE_TAB_SCHEMA = TabSchema(
    name="Compare",
    widgets=[
        WidgetDef("task_selector", TaskSelector),
        WidgetDef("dataset_picker", DatasetPicker, bindings={"task": "@task_selector.value"}),
        WidgetDef("model_selector_1", ModelSelector, params={"task": "vision"}, bindings={"task": "@task_selector.value"}),
        WidgetDef("model_selector_2", ModelSelector, params={"task": "vision"}, bindings={"task": "@task_selector.value"}),
    ],
    actions=[
        ActionDef("compare", "📊", "_start_comparison", style="success"),
    ],
    plots=[
        PlotDef("comparison_plot", xlabel="Epoch", ylabel="Accuracy"),
    ]
)
