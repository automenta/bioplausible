from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef, PlotDef, LayoutDef
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from bioplausible_ui.core.widgets.dataset_picker import DatasetPicker
from bioplausible_ui.core.widgets.model_selector import ModelSelector
from bioplausible_ui.core.widgets.run_selector import RunSelector

COMPARE_TAB_SCHEMA = TabSchema(
    name="Compare",
    widgets=[
        WidgetDef("run_selector_1", RunSelector),
        WidgetDef("run_selector_2", RunSelector),
    ],
    actions=[
        ActionDef("compare_runs", "📉 Compare Runs", "_compare_saved_runs"),
    ],
    plots=[
        PlotDef("comparison_plot", xlabel="Epoch", ylabel="Accuracy"),
        PlotDef("loss_plot", xlabel="Epoch", ylabel="Loss"),
    ]
)
