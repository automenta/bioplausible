from bioplausible_ui.core.schema import ActionDef, TabSchema, WidgetDef
from bioplausible_ui.core.widgets.dataset_picker import DatasetPicker
from bioplausible_ui.core.widgets.hyperparam_editor import HyperparamEditor
from bioplausible_ui.core.widgets.log_output import LogOutput
from bioplausible_ui.core.widgets.model_selector import ModelSelector
from bioplausible_ui.core.widgets.radar_view import RadarView
from bioplausible_ui.core.widgets.results_table import ResultsTable
from bioplausible_ui.core.widgets.task_selector import TaskSelector

SEARCH_TAB_SCHEMA = TabSchema(
    name="Search",
    widgets=[
        WidgetDef("task_selector", TaskSelector),
        WidgetDef(
            "dataset_picker", DatasetPicker, bindings={"task": "@task_selector.value"}
        ),
        WidgetDef(
            "model_selector", ModelSelector, bindings={"task": "@task_selector.value"}
        ),
        WidgetDef(
            "search_space",
            HyperparamEditor,
            bindings={"model": "@model_selector.value"},
        ),
        # Add visual components
        WidgetDef("radar_view", RadarView),
        WidgetDef("results_table", ResultsTable),
        WidgetDef("log_output", LogOutput),
    ],
    actions=[
        ActionDef("start", "🔍", "_start_search", style="success"),
        ActionDef("stop", "⏹", "_stop_search", style="danger"),
    ],
    plots=[],  # Using custom RadarView instead of standard plot
)
